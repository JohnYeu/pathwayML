#!/usr/bin/env python3
"""Audit which raw KEGG pathways were excluded from PathwayML-Ath and why.

Run from the repository root:
    python3 check_missing_kegg_pathways.py

Uses DuckDB SQL to cross-reference the raw KEGG pathway-gene file against
the saved positive samples. The pipeline keeps only pathways with >=5 genes,
so any raw KEGG ID with fewer genes will be absent from samples.json.
"""

from pathlib import Path

import duckdb


PROJECT_DIR = Path(__file__).resolve().parent


def sql_path(path: Path) -> str:
    """Return a path string escaped for a single-quoted DuckDB SQL literal."""
    return str(path).replace("'", "''")


def main() -> None:
    """Run the KEGG coverage audit and print missing pathways with gene counts."""
    kegg_path = sql_path(PROJECT_DIR / "data/kegg_pathway_genes.txt")
    samples_path = sql_path(PROJECT_DIR / "tables/reproducibility/samples.json")

    # SQL CTEs mirror the Python pipeline:
    # 1. raw_kegg reads all KEGG pathway-gene rows from the tab-delimited file.
    # 2. raw_kegg_counts collapses each raw pathway to unique genes.
    # 3. used_kegg reads saved positive samples and keeps only KEGG IDs.
    # 4. missing is raw KEGG minus KEGG pathways actually used by the pipeline.
    base_sql = f"""
    WITH
    raw_kegg AS (
        -- Raw file has two tab-separated columns: path:athXXXXX and ath:AT...
        SELECT
            replace(pathway_raw, 'path:', '') AS pathway_id,
            upper(replace(gene_raw, 'ath:', '')) AS gene_id
        FROM read_csv(
            '{kegg_path}',
            delim = '\t',
            header = false,
            columns = {{
                'pathway_raw': 'VARCHAR',
                'gene_raw': 'VARCHAR'
            }}
        )
    ),
    raw_kegg_counts AS (
        -- Count unique genes per raw KEGG pathway.
        SELECT
            pathway_id,
            count(DISTINCT gene_id) AS n_genes,
            string_agg(DISTINCT gene_id, ',' ORDER BY gene_id) AS genes
        FROM raw_kegg
        GROUP BY pathway_id
    ),
    used_kegg AS (
        -- samples.json stores all pipeline samples; label=1 curated pathways
        -- with IDs starting ath are KEGG positives used by the model.
        SELECT id AS pathway_id
        FROM read_json_auto('{samples_path}')
        WHERE label = 1
          AND type = 'curated_pathway'
          AND id LIKE 'ath%'
    ),
    missing AS (
        -- Raw KEGG pathways absent from used_kegg are filtered out upstream.
        SELECT raw_kegg_counts.*
        FROM raw_kegg_counts
        ANTI JOIN used_kegg USING (pathway_id)
    )
    """

    summary_sql = base_sql + """
    SELECT
        (SELECT count(*) FROM raw_kegg_counts) AS raw_kegg_pathways,
        (SELECT count(*) FROM used_kegg) AS used_kegg_pathways,
        (SELECT count(*) FROM missing) AS missing_kegg_pathways;
    """

    missing_sql = base_sql + """
    SELECT pathway_id, n_genes, genes
    FROM missing
    ORDER BY pathway_id;
    """

    con = duckdb.connect()
    summary = con.sql(summary_sql).fetchone()
    missing_rows = con.sql(missing_sql).fetchall()

    print("Raw KEGG pathways:", summary[0])
    print("Used KEGG pathways:", summary[1])
    print("Missing KEGG pathways:", summary[2])
    print()

    for pathway_id, n_genes, genes in missing_rows:
        print(f"{pathway_id}\t{n_genes} genes\t{genes}")


if __name__ == "__main__":
    main()
