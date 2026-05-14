#!/usr/bin/env python3
"""Audit which raw AraCyc pathways were excluded from PathwayML-Ath and why.

Run from the repository root:
    python3 check_missing_aracyc_pathways.py

Uses DuckDB SQL to cross-reference the raw AraCyc table against the saved
positive samples. The pipeline keeps only AraCyc pathways with >=5 unique
Arabidopsis (AT) genes; this script categorizes every excluded pathway as
either 'no valid AT gene' or 'fewer than 5 valid AT genes', and flags any
unexpected omissions. AraCyc IDs in samples.json carry an `AC_` prefix.
"""

from pathlib import Path

import duckdb


PROJECT_DIR = Path(__file__).resolve().parent


def sql_path(path: Path) -> str:
    """Return a path string escaped for a single-quoted DuckDB SQL literal."""
    return str(path).replace("'", "''")


def main() -> None:
    """Run the AraCyc coverage audit and print categorized missing pathways."""
    aracyc_path = sql_path(PROJECT_DIR / "data/aracyc_pathways.20251021")
    samples_path = sql_path(PROJECT_DIR / "tables/reproducibility/samples.json")

    # SQL CTEs mirror the Python pipeline:
    # 1. raw_aracyc reads the tab-delimited AraCyc table.
    # 2. valid_at_genes keeps Gene-id values that are real Arabidopsis AT genes.
    # 3. pathway_summary counts valid AT genes per raw AraCyc pathway.
    # 4. used_aracyc reads saved positive samples and removes the AC_ prefix.
    # 5. missing is raw AraCyc minus AraCyc pathways actually used by the model.
    base_sql = f"""
    WITH
    raw_aracyc AS (
        -- AraCyc columns include Pathway-id, Pathway-name, and Gene-id.
        SELECT
            "Pathway-id" AS pathway_id,
            "Pathway-name" AS pathway_name,
            upper("Gene-id") AS gene_id
        FROM read_csv_auto(
            '{aracyc_path}',
            delim = '\t',
            header = true
        )
        WHERE "Pathway-id" IS NOT NULL
          AND "Pathway-id" <> ''
    ),
    valid_at_genes AS (
        -- The pipeline keeps only real Arabidopsis gene IDs, not NIL rows.
        SELECT pathway_id, gene_id
        FROM raw_aracyc
        WHERE gene_id <> 'NIL'
          AND gene_id LIKE 'AT%'
    ),
    valid_counts AS (
        -- Count unique valid AT genes per pathway.
        SELECT
            pathway_id,
            count(DISTINCT gene_id) AS valid_at_genes,
            string_agg(DISTINCT gene_id, ',' ORDER BY gene_id) AS valid_gene_list
        FROM valid_at_genes
        GROUP BY pathway_id
    ),
    raw_gene_values AS (
        -- Keep raw Gene-id values so NIL-only pathways can be shown explicitly.
        SELECT
            pathway_id,
            string_agg(DISTINCT gene_id, ',' ORDER BY gene_id) AS raw_gene_values
        FROM raw_aracyc
        GROUP BY pathway_id
    ),
    pathway_names AS (
        -- One pathway name per pathway ID.
        SELECT pathway_id, any_value(pathway_name) AS pathway_name
        FROM raw_aracyc
        GROUP BY pathway_id
    ),
    pathway_summary AS (
        -- One row per raw AraCyc pathway, with valid AT gene count attached.
        SELECT
            names.pathway_id,
            names.pathway_name,
            coalesce(counts.valid_at_genes, 0) AS valid_at_genes,
            coalesce(counts.valid_gene_list, '') AS valid_gene_list,
            raw_values.raw_gene_values
        FROM pathway_names AS names
        LEFT JOIN valid_counts AS counts USING (pathway_id)
        LEFT JOIN raw_gene_values AS raw_values USING (pathway_id)
    ),
    used_aracyc AS (
        -- samples.json stores AraCyc positives as AC_<original Pathway-id>.
        SELECT replace(id, 'AC_', '') AS pathway_id
        FROM read_json_auto('{samples_path}')
        WHERE label = 1
          AND type = 'curated_pathway'
          AND id LIKE 'AC_%'
    ),
    expected_used AS (
        -- This is the code's inclusion rule: at least 5 unique valid AT genes.
        SELECT pathway_id
        FROM valid_counts
        WHERE valid_at_genes >= 5
    ),
    missing AS (
        -- Raw AraCyc pathways absent from used_aracyc are filtered out upstream.
        SELECT
            pathway_summary.*,
            CASE
                WHEN valid_at_genes = 0 THEN 'no valid AT gene'
                WHEN valid_at_genes < 5 THEN 'fewer than 5 valid AT genes'
                ELSE 'unexpected missing'
            END AS reason
        FROM pathway_summary
        ANTI JOIN used_aracyc USING (pathway_id)
    ),
    no_valid_at_gene AS (
        SELECT *
        FROM missing
        WHERE valid_at_genes = 0
    ),
    too_few_genes AS (
        SELECT *
        FROM missing
        WHERE valid_at_genes BETWEEN 1 AND 4
    ),
    unexpected_missing AS (
        -- Should be empty. If not, samples.json and the >=5 rule disagree.
        SELECT expected_used.pathway_id
        FROM expected_used
        ANTI JOIN used_aracyc USING (pathway_id)
    )
    """

    summary_sql = base_sql + """
    SELECT
        (SELECT count(*) FROM pathway_summary) AS raw_aracyc_pathways,
        (SELECT count(*) FROM used_aracyc) AS used_aracyc_pathways,
        (SELECT count(*) FROM missing) AS missing_aracyc_pathways,
        (SELECT count(*) FROM no_valid_at_gene) AS missing_with_no_valid_at_gene,
        (SELECT count(*) FROM too_few_genes) AS missing_with_1_to_4_valid_at_genes,
        (SELECT count(*) FROM expected_used) AS expected_used_by_gene_rule,
        (SELECT count(*) FROM unexpected_missing) AS unexpected_missing_after_gene_rule;
    """

    no_valid_sql = base_sql + """
    SELECT pathway_id, raw_gene_values
    FROM no_valid_at_gene
    ORDER BY pathway_id;
    """

    too_few_sql = base_sql + """
    SELECT pathway_id, valid_at_genes, valid_gene_list
    FROM too_few_genes
    ORDER BY pathway_id;
    """

    unexpected_sql = base_sql + """
    SELECT
        unexpected_missing.pathway_id,
        pathway_summary.valid_at_genes,
        pathway_summary.valid_gene_list
    FROM unexpected_missing
    JOIN pathway_summary USING (pathway_id)
    ORDER BY pathway_id;
    """

    all_missing_sql = base_sql + """
    SELECT pathway_id, pathway_name, valid_at_genes, reason
    FROM missing
    ORDER BY pathway_id;
    """

    con = duckdb.connect()
    summary = con.sql(summary_sql).fetchone()
    no_valid_rows = con.sql(no_valid_sql).fetchall()
    too_few_rows = con.sql(too_few_sql).fetchall()
    unexpected_rows = con.sql(unexpected_sql).fetchall()
    all_missing_rows = con.sql(all_missing_sql).fetchall()

    print("Raw AraCyc pathways:", summary[0])
    print("Used AraCyc pathways:", summary[1])
    print("Missing AraCyc pathways:", summary[2])
    print("Missing with no valid AT gene:", summary[3])
    print("Missing with 1-4 valid AT genes:", summary[4])
    print("Expected used by >=5 AT gene rule:", summary[5])
    print("Unexpected missing after >=5 AT gene rule:", summary[6])
    print()

    print("Missing with no valid AT gene:")
    for pathway_id, raw_gene_values in no_valid_rows:
        print(f"{pathway_id}\t0 AT genes\traw Gene-id values: {raw_gene_values}")

    print()
    print("Missing with 1-4 valid AT genes:")
    for pathway_id, valid_at_genes, valid_gene_list in too_few_rows:
        print(f"{pathway_id}\t{valid_at_genes} AT genes\t{valid_gene_list}")

    if unexpected_rows:
        print()
        print("Unexpected missing pathways with >=5 AT genes:")
        for pathway_id, valid_at_genes, valid_gene_list in unexpected_rows:
            print(f"{pathway_id}\t{valid_at_genes} AT genes\t{valid_gene_list}")

    print()
    print("All missing AraCyc pathways:")
    print("Pathway-id\tPathway-name\tvalid_AT_genes\treason")
    for pathway_id, pathway_name, valid_at_genes, reason in all_missing_rows:
        print(f"{pathway_id}\t{pathway_name}\t{valid_at_genes}\t{reason}")


if __name__ == "__main__":
    main()
