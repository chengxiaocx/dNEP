"""
nep_ha_glm_smooth.py
====================
Steps:
  1. Build HRF-convolved design matrices from timing files
  2. Smooth + denoise + GLM on aligned data
  3. Whole-brain plots for figures
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import zscore, gamma
from scipy.ndimage import convolve1d
import neuroboros as nb
from neuroboros import smooth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TIMING_DIR    = '/work/cxiao/nep_timing_re/nep_timing_wots'
ALIGNED_DIR   = '/work/cxiao/NEP/HA/aligned'
CONFOUNDS_DIR = '/work/cxiao/NEP/nb-data/25.1.4/confounds'
BEST_RUNS_CSV = '/work/cxiao/NEP/GLM/prosem_contrasts_revised/prosem_contrasts_mat/best_runs.csv'
OUTPUT_DIR    = '/work/cxiao/NEP/HA/GLM/prosem_contrasts_14_smoothed10'
PLOT_DIR      = os.path.join(OUTPUT_DIR, 'figures')

TR    = 1.5
N_DIV = 256
FWHM  = 10.0
SPACE = 'onavg-ico32'

native_sids  = [str(s) for s in range(3102, 3117)]
l2_high_sids = ['3212', '3214', '3205', '3213', '3211', '3203', '3206', '3207']
l2_low_sids  = ['3210', '3204', '3209', '3216', '3215', '3202', '3208']

conditions = [
    'positive_positive', 'positive_negative', 'positive_neutral',
    'negative_negative', 'negative_positive', 'negative_neutral'
]

full_contrasts = [
    [ 1,  1,  1, -1, -1, -1],  # 0.  positive vs negative prosody
    [ 1,  1, -2,  1,  1, -2],  # 1.  real vs fake
    [ 1, -1,  0, -1,  1,  0],  # 2.  positive vs negative semantics
    [ 1, -1,  0,  1, -1,  0],  # 3.  congruent vs incongruent
    [ 1,  0, -1,  1,  0, -1],  # 4.  congruent vs neutral
    [ 0,  1, -1,  0,  1, -1],  # 5.  incongruent vs neutral
    [ 1,  1,  0,  1,  1,  0],  # 6.  real vs rest
    [ 0,  0,  1,  0,  0,  1],  # 7.  fake vs rest
    [ 1,  1, -2,  0,  0,  0],  # 8.  real vs fake in positive prosody
    [ 0,  0,  0,  1,  1, -2],  # 9.  real vs fake in negative prosody
    [ 1,  1,  0,  0,  0,  0],  # 10. real vs rest in positive prosody
    [ 0,  0,  1,  0,  0,  0],  # 11. fake vs rest in positive prosody
    [ 0,  0,  0,  1,  1,  0],  # 12. real vs rest in negative prosody
    [ 0,  0,  0,  0,  0,  1],  # 13. fake vs rest in negative prosody
]

# ---------------------------------------------------------------------------
# Step 1: Design matrices
# ---------------------------------------------------------------------------
def build_hrf():
    tt   = np.arange(20 * N_DIV) * TR / N_DIV
    bold = gamma.pdf(tt, 6) - gamma.pdf(tt, 16) / 6.0
    return bold / bold.sum()

def build_design_matrices():
    bold   = build_hrf()
    kwargs = {'mode': 'nearest', 'origin': -len(bold) // 2}
    df_best = pd.read_csv(BEST_RUNS_CSV)
    design_matrices = {}

    for _, row in df_best.iterrows():
        sub_id = str(row['subject'])
        for run_col in ['run_A', 'run_B']:
            run_str = row[run_col]
            task, run_num = run_str.split('_')
            tsv = os.path.join(TIMING_DIR, f'sub-{sub_id}_task-{task}_run-{run_num}.tsv')
            if not os.path.exists(tsv):
                print(f'[skip] {tsv}'); continue

            nt        = 440 if task == 'wd' else 454
            t_highres = np.arange(nt * N_DIV) * TR / N_DIV
            df_tsv    = pd.read_csv(tsv, sep='\t')

            cols = []
            for cond in conditions:
                blocks = df_tsv[df_tsv['trial_type'] == cond]
                boxcar = np.zeros_like(t_highres)
                for _, b in blocks.iterrows():
                    boxcar[(t_highres >= b['onset']) &
                           (t_highres < b['onset'] + b['duration'])] = 1.0
                reg = convolve1d(boxcar, bold, axis=0, **kwargs)
                cols.append(reg.reshape(-1, N_DIV).mean(axis=1))

            key = f'sub-{sub_id}_task-{task}_run-{run_num}'
            design_matrices[key] = pd.DataFrame(np.stack(cols, axis=1), columns=conditions)
            print(f'Design matrix: {key} | shape {design_matrices[key].shape}')

    return design_matrices

# ---------------------------------------------------------------------------
# Step 2: Smooth + GLM
# ---------------------------------------------------------------------------
def run_glm(design_matrices):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    M = smooth('lr', fwhm=FWHM, space=SPACE, mask=True)

    for key, dm_df in design_matrices.items():
        parts   = key.split('_')
        sub_id  = parts[0].replace('sub-', '')
        task    = parts[1].replace('task-', '')
        run_num = int(parts[2].replace('run-', ''))

        aligned_path = os.path.join(ALIGNED_DIR, f'{sub_id}_{task}-{run_num:02d}_aligned.npy')
        conf_path    = os.path.join(CONFOUNDS_DIR,
                           f'sub-{sub_id}_task-{task}_run-{run_num:02d}_desc-confounds_timeseries.npy')

        if not os.path.exists(aligned_path):
            print(f'[skip] missing aligned: {aligned_path}'); continue
        if not os.path.exists(conf_path):
            print(f'[skip] missing confounds: {conf_path}'); continue

        dm = np.load(aligned_path) @ M
        confounds = np.load(conf_path)
        dm = dm - confounds @ np.linalg.lstsq(confounds, dm, rcond=None)[0]
        dm = np.nan_to_num(zscore(dm, axis=0))

        valid = dm_df.columns[(dm_df != 0).any(axis=0)]
        X     = dm_df[valid].values
        cvecs = [pd.Series(c, index=conditions)[valid].values for c in full_contrasts]

        betas, ts, R2s = nb.glm(dm, X, None, contrasts=cvecs, return_r2=True)
        out = os.path.join(OUTPUT_DIR, f'{sub_id}_{task}_{run_num:02d}.npz')
        nb.save(out, {'beta': betas, 'ts': ts, 'R2s': R2s})
        print(f'Saved: {out}')

# ---------------------------------------------------------------------------
# Step 3: Whole-brain plots
# ---------------------------------------------------------------------------
def load_group_means(sids, c_idx):
    arrays = []
    for sub in sids:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f'{sub}_*.npz')))
        if not files: continue
        runs = [np.load(f, allow_pickle=True)['ts'] for f in files]
        arrays.append(np.mean(runs, axis=0))
    if not arrays: return None
    return np.nanmean(np.array(arrays)[:, c_idx, :], axis=0)

def plot_map(data, title, filename, vmin=-3, vmax=3):
    lr_masks = [nb.mask(lr, SPACE) for lr in 'lr']
    img = nb.plot(np.nan_to_num(data), space=SPACE, cmap='RdBu_r',
                  bar_title='$t$', mask=lr_masks, vmin=vmin, vmax=vmax)
    if hasattr(img, 'save'):
        img.save(os.path.join(PLOT_DIR, filename))
        print(f'Saved: {filename}')
    return img

def plot_figures():
    os.makedirs(PLOT_DIR, exist_ok=True)
    learner_sids = l2_high_sids + l2_low_sids

    # Fig 2: Native and Learner for real_vs_fake
    for sids, label in [(native_sids, 'Native'), (learner_sids, 'Learner')]:
        plot_map(load_group_means(sids, 1), f'{label}: Real_vs_Fake',
                 f'fig2_{label}_Real_vs_Fake.png')

    # Fig 3a: Native and Learner for real_vs_fake_in_positive
    for sids, label in [(native_sids, 'Native'), (learner_sids, 'Learner')]:
        plot_map(load_group_means(sids, 8), f'{label}: Real_vs_Fake_Positive',
                 f'fig3a_{label}_Real_vs_Fake_Positive.png')

    # Fig 3b: Native and Learner for real_vs_fake_in_negative
    for sids, label in [(native_sids, 'Native'), (learner_sids, 'Learner')]:
        plot_map(load_group_means(sids, 9), f'{label}: Real_vs_Fake_Negative',
                 f'fig3b_{label}_Real_vs_Fake_Negative.png')

    # Fig 4: Native - Learner difference for real_vs_rest and fake_vs_rest
    for label, c_idx, fname in [('Real_vs_Rest', 6, 'fig4a_diff_Real_vs_Rest.png'),
                                 ('Fake_vs_Rest', 7, 'fig4b_diff_Fake_vs_Rest.png')]:
        nm = load_group_means(native_sids,  c_idx)
        lm = load_group_means(learner_sids, c_idx)
        plot_map(nm - lm, f'Native > Learner: {label}', fname)

    # Fig 5: Native, L2-High, L2-Low for real_vs_rest
    for sids, label in [(native_sids, 'Native'), (l2_high_sids, 'L2_High'), (l2_low_sids, 'L2_Low')]:
        plot_map(load_group_means(sids, 6), f'{label}: Real_vs_Rest',
                 f'fig5_{label}_Real_vs_Rest.png')

    # Fig S2: Native, L2-High, L2-Low for fake_vs_rest
    for sids, label in [(native_sids, 'Native'), (l2_high_sids, 'L2_High'), (l2_low_sids, 'L2_Low')]:
        plot_map(load_group_means(sids, 7), f'{label}: Fake_vs_Rest',
                 f'figS2_{label}_Fake_vs_Rest.png')

    # Fig S3: Native, L2-High, L2-Low for real_vs_fake
    for sids, label in [(native_sids, 'Native'), (l2_high_sids, 'L2_High'), (l2_low_sids, 'L2_Low')]:
        plot_map(load_group_means(sids, 1), f'{label}: Real_vs_Fake',
                 f'figS3_{label}_Real_vs_Fake.png')

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    design_matrices = build_design_matrices()
    run_glm(design_matrices)
    plot_figures()
