# V4 thesis-level fixes and added analyses

This revision keeps the no-embedding model as the canonical model and adds analyses that improve scientific defensibility rather than inflating the claims.

## 1. Stability-aware GO feature analysis

Problem: the number of selected GO terms changes from seed to seed.

Cause: each seed regenerates hard negatives, train/test splits, and mutual-information-based feature selection on the training split. The selected GO count is therefore a property of the stochastic training protocol, not a stable biological quantity.

Added code: `analyze_stability.py`.

Outputs:

- `tables/go_selection_stability.csv`
- `tables/go_selection_pairwise_jaccard.csv`
- `tables/go_selection_stability_summary.json`
- `figures/fig8_go_selection_stability.png`

Key result from the 20-seed artifacts:

- selected GO terms per seed: mean 106.6, SD 39.0, range 54-157;
- 184 unique GO terms were selected at least once;
- 57 terms were selected in at least 75% of seeds;
- 127 terms were selected in at least 50% of seeds;
- pairwise Jaccard similarity among selected-term sets: mean 0.524, SD 0.143.

Interpretation: the exact GO list is not fixed, but a recurrent core exists. Therefore, the manuscript should report both the canonical seed-42 configuration and the multi-seed distribution.

## 2. Feature-count sensitivity analysis

Problem: if the selected GO term count changes, readers may suspect that performance depends on a lucky dimensionality.

Added outputs:

- `tables/go_count_performance_correlation.csv`
- `figures/fig10_performance_vs_go_count.png`

Key result:

- Pearson correlation between selected GO count and XGBoost test AUROC: -0.036.

Interpretation: changing the number of selected GO terms across seeds does not explain test AUROC. This supports the claim that the signal is robust, even if the selected GO list is variable.

## 3. Candidate-score uncertainty

Problem: candidate examples should not be interpreted from a single seed only.

Added outputs:

- `tables/candidate_uncertainty_summary.csv`
- `figures/fig9_candidate_score_uncertainty.png`

Key result:

- C1: 0.877 +/- 0.100 across seeds; ORA significant in all runs; not novel.
- C2: 0.823 +/- 0.150; one seed falls below 0.5, so it is less stable than C3.
- C3: 0.958 +/- 0.034; consistently high.
- C4: 0.002 +/- 0.002; consistently low despite ORA significance.

Interpretation: C3 is the clean positive control, C4 is the clean incoherent control, and C1 is a known stress/glutathione-related module rather than a discovery claim.

## 4. Manuscript-level corrections

- Removed the old D=91 / 85-GO-term manuscript numbers from the main result.
- Used the corrected seed-42 no-embedding run: D=80 = 74 GO + 4 Jaccard + 2 size.
- Used multi-seed model results as the primary robustness result: XGBoost test AUROC 0.955 +/- 0.010 SD across 20 independent runs.
- Removed the false C1 novelty claim.
- Removed the fake GitHub placeholder from Data Availability. The manuscript now says that the repository URL should be supplied after deposition.
- Removed or corrected uncertain references; every retained reference has a DOI and/or open readable path listed in `REFERENCE_AUDIT.md`.

## How to regenerate the added analysis

From the repository root:

```bash
python analyze_stability.py --repo .
```

This script expects the existing multi-seed artifacts under `tables/multiseed/reproducibility/`.
