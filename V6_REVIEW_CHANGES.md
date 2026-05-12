# V6 Local Replacement and Codex Review Changes

> Superseded note, 2026-05-11: after the full recompute audit, use
> `RECOMPUTED_RESULTS_HANDOFF.md` and
> `tables/paper_reproducibility_manifest.csv` as the current source of truth.
> The LaTeX draft was intentionally not updated.

Date: 2026-05-11

## Source and backup

- Source package: `/Users/yuzhuoye/Downloads/PathwayML_Ath_Paper_V6_final_package.zip`
- Local target: `/Users/yuzhuoye/PathwayML-Ath`
- Pre-replacement backup: `/Users/yuzhuoye/Downloads/PathwayML-Ath_preV6_backup_20260511_151324.zip`
- Original V6 paper files were preserved under `paper/original_v6/`.

## Replacement policy

- The local project was clean-synced from the V6 code archive.
- Preserved local-only paths: `.venv/`, `.gitignore`, and `paper/`.
- Cache/build noise was excluded or removed: `__pycache__/`, `*.pyc`, and `.DS_Store`.
- `paper/` remains ignored by `.gitignore` and should not be pushed.

## Code fixes

- `generalization_fast.py` now computes per-negative-type performance from the canonical seed-42 artifacts:
  - `tables/reproducibility/samples.json`
  - `tables/reproducibility/splits.json`
  - `tables/reproducibility/selected_go_terms.json`
  - `run_no_embedding_reproducible.py::make_models(42)["XGBoost"]`
- This makes the "All mixed negatives" row match the seed-42 XGBoost result exactly:
  - AUROC = 0.9480024005486968
  - AUPRC = 0.8924071220796562
- `generalization_fast.py` now recomputes LOFO by default. A cached LOFO table can be read only with `--use-cached-lofo`.
- The recomputed LOFO result is written to both `tables/table8_lofo_generalization.csv` and `tables/table8_lofo_generalization_training_only.csv`.
- `latex/README.md` was corrected to state that `latex/main.tex` is legacy and not the current V6 manuscript source. Its older corrected-value note was replaced with the canonical `0.948` seed-42 result.

## Paper files

- Original V6 paper:
  - `paper/original_v6/PathwayML_Ath_Paper_V6_final_revised.docx`
  - `paper/original_v6/PathwayML_Ath_Paper_V6_final_revised.pdf`
- Codex-checked paper:
  - `paper/PathwayML_Ath_Paper_V6_Codex_checked.docx`
  - `paper/PathwayML_Ath_Paper_V6_Codex_checked.pdf`

The checked paper keeps the V6 manuscript structure and only changes the per-negative-type values tied to the corrected canonical seed-42 analysis.

Updated paper values:

| Comparison | n pos | n neg | AUROC | AUPRC | Negative median | Negative IQR |
|---|---:|---:|---:|---:|---:|---|
| All mixed negatives | 108 | 216 | 0.948 | 0.892 | 0.023 | 0.004-0.179 |
| jaccard matched | 108 | 48 | 0.997 | 0.999 | 0.002 | 0.001-0.010 |
| co annotation | 108 | 48 | 0.979 | 0.990 | 0.006 | 0.002-0.041 |
| chimera | 108 | 66 | 0.876 | 0.911 | 0.217 | 0.071-0.565 |
| shuffled | 108 | 54 | 0.965 | 0.982 | 0.024 | 0.007-0.137 |

Figure 3 in the checked paper was replaced with the regenerated canonical plot from `figures/fig11_negative_type_performance.png`.

## Verification summary

Commands run:

```bash
python -m py_compile *.py
python run_no_embedding_reproducible.py --seed 42 --no-figures
python run_no_embedding_reproducible.py --seeds 1-20 --no-figures
python analyze_stability.py --repo .
python generalization_fast.py
```

Confirmed values:

| Check | Result |
|---|---|
| Seed-42 feature dimension | D = 80 |
| Seed-42 selected GO terms | 74 |
| Seed-42 XGBoost AUROC / AUPRC / F1 | 0.948 / 0.892 / 0.831 |
| 20-seed XGBoost AUROC | 0.955 +/- 0.010 SD |
| 20-seed XGBoost AUPRC | 0.909 +/- 0.024 SD |
| GO selected range | 54-157 |
| Unique selected GO terms | 184 |
| Core GO terms >=75% seeds | 57 |
| Recurrent GO terms >=50% seeds | 127 |
| Corr(selected GO count, test AUROC) | -0.036 |
| LOFO AUROC range after recompute audit | 0.861-0.957 |

DOCX/PDF QA:

- LibreOffice rendering was unavailable because `soffice` is not installed.
- Microsoft Word was used to export the checked DOCX to PDF.
- The PDF has 11 pages.
- The PDF was rendered to PNG pages with macOS PDFKit for visual inspection.
- Page 5 was checked after the Table 4/Figure 3 update; the table and figure show the corrected canonical values without visible overlap or clipping.

## Remaining notes

- `latex/main.tex` remains a legacy draft and should not be used as the current V6 paper source.
- Several exploratory LOFO scripts are retained from the V6 package for transparency. The checked workflow uses `generalization_fast.py` recomputation for the manuscript LOFO table.
- No GitHub push was performed.
