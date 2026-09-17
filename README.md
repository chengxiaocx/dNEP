# dNEP

dNEP is a research repository for analyzing behavioral and neuroimaging data related to lexicality and emotional prosody processing in native Mandarin speakers and second language (L2) learners.

## Overview

Participants completed word and sentence tasks across four functional runs (two word runs, two sentence runs). The pipeline covers:

- behavioral preprocessing and mixed-effects modeling
- fMRI run quality control and best-run selection
- connectivity-based hyperalignment to a common representational space
- whole-brain GLM with 14 contrasts (smoothed at 10mm FWHM)
- ROI-based group comparisons using the HCP-MMP 180-parcel atlas

Groups:
- Native speakers: 3102–3116 (n=15)
- L2 learners: 3202–3216 (n=15, split into high- and low-proficiency subgroups)

## Repository Contents

| File | Description |
|------|-------------|
| `dnep_behavioral_results.Rmd` | Accuracy and RT preprocessing, mixed-effects models, and Figure 1 |
| `dnep_quality_control.py` | Run quality control (within- vs between-task similarity), best-run selection, and group reliability comparison |
| `dnep_ha.py` | Connectivity-based hyperalignment: compute per-subject transforms and apply to test runs |
| `dnep_ha_glm_wholebrain.py` | Design matrix construction, smoothing, GLM, and whole-brain figure generation (Figs 2–5, S2–S3) |
| `dnep_ha_roi.py` | HCP-MMP parcel-level t-tests (Native vs L2) and ANOVA (Native vs L2-High vs L2-Low) with FDR correction and Tukey post-hoc |
| `glasser_parcellation_information.tsv` | HCP-MMP parcellation metadata |
| `nep_timing_wots.zip` | Trial-level timing files for GLM design matrix construction |

## Pipeline Order

The behavioral analysis (`dnep_behavioral_results.Rmd`) is independent. The neuroimaging pipeline runs in the following order:

```
dnep_quality_control.py → dnep_ha.py → dnep_ha_glm_wholebrain.py → dnep_ha_roi.py
```

1. `dnep_quality_control.py` — computes 4×4 inter-run correlation matrices, runs QC t-test, selects best two runs per subject, saves `best_runs.csv`
2. `dnep_ha.py` — reads `best_runs.csv`, computes hyperalignment transforms on training runs, applies transforms to test runs, saves aligned `.npy` files
3. `dnep_ha_glm_wholebrain.py` — builds HRF-convolved design matrices from timing files, applies 10mm smoothing, runs GLM, saves `.npz` outputs and brain figures
4. `dnep_ha_roi.py` — reads GLM outputs, averages vertex-wise t-values within HCP-MMP parcels, runs group comparisons, saves CSVs

## Dependencies

**Python:** numpy, scipy, pandas, statsmodels, neuroboros, hyperalignment

**R:** tidyverse, ggplot2, lme4, lmerTest, emmeans, Rmisc

## License

All materials in this repository (code, data, and documentation) are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the materials for any purpose, provided you give appropriate credit to the original authors.
