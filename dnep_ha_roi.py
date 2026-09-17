"""
nep_ha_roi.py
=============
HCP-MMP 180-parcel ROI analysis.
Reads GLM output from nep_ha_glm_smooth.py.

Analyses:
  - Native vs L2 (two-sample t-test, FDR corrected)
  - Native vs L2-High vs L2-Low (one-way ANOVA, FDR + Tukey post-hoc)
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import neuroboros as nb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = '/work/cxiao/NEP/HA/GLM/prosem_contrasts_14_smoothed10'
SPACE    = 'onavg-ico32'

native_sids  = [str(s) for s in range(3102, 3117)]
l2_high_sids = ['3212', '3214', '3205', '3213', '3211', '3203', '3206', '3207']
l2_low_sids  = ['3210', '3204', '3209', '3216', '3215', '3202', '3208']
all_sids     = native_sids + l2_high_sids + l2_low_sids

contrast_names = {
    0: 'positive_vs_negative_prosody', 1: 'real_vs_fake',
    2: 'positive_vs_negative_semantics', 3: 'congruent_vs_incongruent',
    4: 'congruent_vs_neutral', 5: 'incongruent_vs_neutral',
    6: 'real_vs_rest', 7: 'fake_vs_rest',
    8: 'real_vs_fake_in_positive', 9: 'real_vs_fake_in_negative',
    10: 'real_vs_rest_in_positive', 11: 'fake_vs_rest_in_positive',
    12: 'real_vs_rest_in_negative', 13: 'fake_vs_rest_in_negative',
}
selected_contrasts = [6, 7, 1, 8, 9, 10, 11, 12, 13]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_parcellation():
    mask = nb.mask('lr', space=SPACE)
    mmp  = np.concatenate([nb.parcellation('HCP_MMP', lr) for lr in 'lr'])[mask]
    return mmp

def vertex_to_parcel(ts_vertex, mmp, n_parcels=180):
    mmp_s  = np.where(mmp < 0, 0, mmp)
    sums   = np.bincount(mmp_s, weights=ts_vertex, minlength=n_parcels + 1)[1:]
    counts = np.bincount(mmp_s, minlength=n_parcels + 1)[1:]
    return sums / counts

def mean_sd_se_ci(vals):
    n  = len(vals)
    m  = vals.mean()
    sd = vals.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n)  if n > 1 else np.nan
    ci = stats.t.ppf(0.975, df=n-1) * se if n > 1 else np.nan
    return m, sd, se, m - ci, m + ci

def load_subject_ts(sids):
    subject_ts = {}
    for sid in sids:
        files = sorted(glob.glob(os.path.join(DATA_DIR, f'{sid}_*.npz')))
        if not files: continue
        subject_ts[sid] = np.mean(
            [np.load(f, allow_pickle=True)['ts'] for f in files], axis=0)
    return subject_ts

# ---------------------------------------------------------------------------
# Analysis 1: Native vs L2 t-test
# ---------------------------------------------------------------------------
def run_parcel_ttest():
    mmp        = load_parcellation()
    parcel_ids = np.arange(1, 181)
    group_map  = {s: 'Native' for s in native_sids}
    group_map.update({s: 'L2' for s in l2_high_sids + l2_low_sids})
    subject_ts = load_subject_ts(all_sids)

    all_results = {}
    for c_idx in selected_contrasts:
        c_name = contrast_names[c_idx]
        rows = [{'sid': sid, 'group': group_map[sid],
                 **{f'p{pid}': vertex_to_parcel(ts[c_idx], mmp)[pid-1]
                    for pid in parcel_ids}}
                for sid, ts in subject_ts.items()]
        pdf = pd.DataFrame(rows)
        nat = pdf[pdf['group'] == 'Native']
        l2  = pdf[pdf['group'] == 'L2']

        results = []
        for pid in parcel_ids:
            col = f'p{pid}'
            nv, lv = nat[col].dropna(), l2[col].dropna()
            if len(nv) < 2 or len(lv) < 2: continue
            t, p = stats.ttest_ind(nv, lv, equal_var=False)
            nm, nsd, nse, nlo, nhi = mean_sd_se_ci(nv)
            lm, lsd, lse, llo, lhi = mean_sd_se_ci(lv)
            results.append({'contrast': c_name, 'parcel': pid,
                            't_stat': t, 'p_value': p,
                            'native_mean': nm, 'native_sd': nsd,
                            'native_se': nse, 'native_ci_lower': nlo, 'native_ci_upper': nhi,
                            'l2_mean': lm, 'l2_sd': lsd,
                            'l2_se': lse, 'l2_ci_lower': llo, 'l2_ci_upper': lhi,
                            'n_native': len(nv), 'n_l2': len(lv)})

        rdf = pd.DataFrame(results)
        reject, p_fdr, _, _ = multipletests(rdf['p_value'], alpha=0.05, method='fdr_bh')
        rdf['p_fdr'], rdf['significant_fdr'] = p_fdr, reject
        rdf = rdf.sort_values('p_value').reset_index(drop=True)
        all_results[c_name] = rdf
        rdf.to_csv(os.path.join(DATA_DIR, f'parcel_ttest_{c_name}.csv'), index=False)
        print(f'{c_name}: {reject.sum()} / 180 significant (FDR < 0.05)')

    pd.concat(all_results.values()).to_csv(
        os.path.join(DATA_DIR, 'parcel_ttest_all_contrasts.csv'), index=False)

# ---------------------------------------------------------------------------
# Analysis 2: Native vs L2-High vs L2-Low ANOVA + Tukey
# ---------------------------------------------------------------------------
def run_parcel_anova():
    mmp         = load_parcellation()
    parcel_ids  = np.arange(1, 181)
    group_order = ['Native', 'L2_High', 'L2_Low']
    group_map   = {s: 'Native'  for s in native_sids}
    group_map.update({s: 'L2_High' for s in l2_high_sids})
    group_map.update({s: 'L2_Low'  for s in l2_low_sids})
    subject_ts = load_subject_ts(all_sids)

    all_results, all_posthoc = {}, {}
    for c_idx in selected_contrasts:
        c_name = contrast_names[c_idx]
        rows = [{'sid': sid, 'group': group_map[sid],
                 **{f'p{pid}': vertex_to_parcel(ts[c_idx], mmp)[pid-1]
                    for pid in parcel_ids}}
                for sid, ts in subject_ts.items()]
        pdf  = pd.DataFrame(rows)
        gdfs = {g: pdf[pdf['group'] == g] for g in group_order}

        results, posthoc_rows = [], []
        for pid in parcel_ids:
            col  = f'p{pid}'
            vals = {g: gdfs[g][col].dropna() for g in group_order}
            if any(len(v) < 2 for v in vals.values()): continue
            f, p = stats.f_oneway(*[vals[g] for g in group_order])
            desc = {}
            for g in group_order:
                m, sd, se, lo, hi = mean_sd_se_ci(vals[g])
                k = g.lower()
                desc.update({f'{k}_mean': m, f'{k}_sd': sd, f'{k}_se': se,
                              f'{k}_ci_lower': lo, f'{k}_ci_upper': hi})
            results.append({'contrast': c_name, 'parcel': pid,
                            'f_stat': f, 'p_value': p, **desc,
                            'n_native': len(vals['Native']),
                            'n_l2_high': len(vals['L2_High']),
                            'n_l2_low':  len(vals['L2_Low'])})

        rdf = pd.DataFrame(results)
        reject, p_fdr, _, _ = multipletests(rdf['p_value'], alpha=0.05, method='fdr_bh')
        rdf['p_fdr'], rdf['significant_fdr'] = p_fdr, reject
        rdf = rdf.sort_values('p_value').reset_index(drop=True)
        all_results[c_name] = rdf
        rdf.to_csv(os.path.join(DATA_DIR, f'parcel_anova_{c_name}.csv'), index=False)
        print(f'{c_name}: {reject.sum()} / 180 significant (ANOVA FDR < 0.05)')

        for pid in rdf.loc[reject, 'parcel']:
            col = f'p{pid}'
            sub = pdf[['group', col]].dropna()
            tukey = pairwise_tukeyhsd(sub[col], sub['group'], alpha=0.05)
            tdf = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tdf.insert(0, 'parcel', pid); tdf.insert(0, 'contrast', c_name)
            posthoc_rows.append(tdf)

        if posthoc_rows:
            phdf = pd.concat(posthoc_rows, ignore_index=True)
            all_posthoc[c_name] = phdf
            phdf.to_csv(os.path.join(DATA_DIR, f'parcel_tukey_{c_name}.csv'), index=False)

    pd.concat(all_results.values()).to_csv(
        os.path.join(DATA_DIR, 'parcel_anova_all_contrasts.csv'), index=False)
    if all_posthoc:
        pd.concat(all_posthoc.values()).to_csv(
            os.path.join(DATA_DIR, 'parcel_tukey_all_contrasts.csv'), index=False)

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_parcel_ttest()
    run_parcel_anova()
