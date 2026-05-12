# Claude Handoff: Corrected No-Embedding Code

> Superseded note, 2026-05-11: for the latest recomputed code/data package,
> use `RECOMPUTED_RESULTS_HANDOFF.md` and
> `tables/paper_reproducibility_manifest.csv` as the source of truth. The
> current seed-42 XGBoost test AUROC is 0.948; the original handoff text below
> has been corrected where it previously showed an older value.

This zip fixes the inconsistency in the previous handoff.

## What Was Wrong

- `PathwayML_Ath_Analysis.ipynb` was an embedding model:
  `72 GO + 4 Jaccard + 20 SVD + 2 Size = D=98`.
- The manuscript described a no-embedding model:
  `85 GO + 4 Jaccard + 2 Size = D=91`.
- Old tables still contained `svd_*`, `Embedding only`, and D=98 outputs.
- `final_no_emb.json` had no reproducible code path.

## What Is Fixed Here

- Canonical code is `run_no_embedding_reproducible.py`.
- The main notebook is now only a clean wrapper for that script.
- Final features contain no embedding terms.
- Feature selection uses training split only.
- Selected GO terms, negatives, candidates, and splits are saved.
- Output tables are regenerated from one run.

## Corrected Run Summary

```text
Selected GO terms: 74
D = 80 = 74 GO + 4 Jaccard + 2 Size
XGBoost CV AUROC = 0.952 +/- 0.002 SE
XGBoost Test AUROC = 0.948
```

## Important Manuscript Change

The corrected deterministic candidate run does **not** support the previous
C1 novelty claim:

```text
C1 score = 0.883
max overlap fraction = 0.412
ORA significant = true
novel = false
```

Please update the manuscript from `tables/results_no_embedding.json` and do
not reuse the old D=91 / C1-novel text.

## How To Reproduce

```bash
pip install -r requirements.txt
python run_no_embedding_reproducible.py
```

Primary results:

```text
tables/results_no_embedding.json
tables/table1_performance.csv
tables/table2_shap_importance.csv
tables/table3_ablation.csv
tables/candidate_results.csv
```
