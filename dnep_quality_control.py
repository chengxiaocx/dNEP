import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = './NEP/GLM/prosem_contrasts_revised'
MAT_DIR  = os.path.join(DATA_DIR, 'prosem_contrasts_mat')

native      = list(range(3102, 3117))
learner     = list(range(3202, 3217))
run_configs = [('wd','01'), ('wd','02'), ('sen','01'), ('sen','02')]
runs        = ['wd_01', 'wd_02', 'sen_01', 'sen_02']
contrasts   = ['PP_vs_NP', 'Real_vs_Fake', 'PS_vs_NS',
               'Congruent_vs_Incongruent', 'Congruent_vs_Neutral', 'Incongruent_vs_Neutral']

# ---------------------------------------------------------------------------
# Step 1: Compute 4x4 correlation matrices per subject and save
# ---------------------------------------------------------------------------
def compute_corr_matrices():
    os.makedirs(MAT_DIR, exist_ok=True)
    for sub in native + learner:
        ts_runs, beta_runs = [], []
        for task, run in run_configs:
            f = os.path.join(DATA_DIR, f'{sub}_{task}_{run}.npz')
            if not os.path.exists(f):
                print(f'[skip] {sub}: missing {task}_{run}'); break
            with np.load(f) as a:
                ts_runs.append(a['ts'])
                beta_runs.append(a['beta'])
        if len(ts_runs) != 4:
            continue
        result = {'subject_id': sub,
                  'group': 'native' if sub in native else 'learner',
                  'ts':   {c: np.corrcoef(np.vstack([r[i] for r in ts_runs]))   for i, c in enumerate(contrasts)},
                  'beta': {c: np.corrcoef(np.vstack([r[i] for r in beta_runs])) for i, c in enumerate(contrasts)}}
        out = os.path.join(MAT_DIR, f'sub_{sub}_corr_matrices.npy')
        np.save(out, result)
        print(f'Saved {sub} -> {out}')

# ---------------------------------------------------------------------------
# Step 2: QC — within- vs between-task similarity (Real_vs_Fake, ts only)
# Within : mean Fisher-z of r(wd_01, wd_02) and r(sen_01, sen_02)
# Between: mean Fisher-z across all word x sentence pairings
# ---------------------------------------------------------------------------
def run_qc_ttest():
    within_z, between_z = [], []
    for sub in native + learner:
        f = os.path.join(MAT_DIR, f'sub_{sub}_corr_matrices.npy')
        if not os.path.exists(f): continue
        mat = np.load(f, allow_pickle=True).item()['ts']['Real_vs_Fake']
        within_z.append(np.mean([np.arctanh(mat[0,1]), np.arctanh(mat[2,3])]))
        between_z.append(np.mean(np.arctanh([mat[0,2], mat[0,3], mat[1,2], mat[1,3]])))
    t, p = stats.ttest_rel(within_z, between_z)
    print(f'QC: N={len(within_z)}  within={np.mean(within_z):.3f}  between={np.mean(between_z):.3f}  t={t:.3f}  p={p:.4f}')

# ---------------------------------------------------------------------------
# Step 3: Select two best runs per subject
# Highest Fisher-z-averaged correlation across all 6 contrasts
# ---------------------------------------------------------------------------
def get_best_runs():
    records = []
    for f in sorted(glob.glob(os.path.join(MAT_DIR, 'sub_*_corr_matrices.npy'))):
        d = np.load(f, allow_pickle=True).item()
        mean_r = np.tanh(np.mean([np.arctanh(np.clip(d['ts'][c], -0.9999, 0.9999))
                                   for c in contrasts], axis=0))
        upper = np.triu(mean_r, k=1)
        i, j  = np.unravel_index(np.argmax(upper), upper.shape)
        records.append({'subject': d['subject_id'], 'group': d['group'],
                        'run_A': runs[i], 'run_B': runs[j],
                        'corr': round(float(upper[i, j]), 3)})
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Step 4: Compare inter-run correlation between native and learner groups
# ---------------------------------------------------------------------------
def run_group_ttest(df):
    native_r  = df[df['group'] == 'native']['corr']
    learner_r = df[df['group'] == 'learner']['corr']
    t, p = stats.ttest_ind(native_r, learner_r, equal_var=False)
    print(f'Native  (n={len(native_r)}): M={native_r.mean():.3f}  SD={native_r.std():.3f}')
    print(f'Learner (n={len(learner_r)}): M={learner_r.mean():.3f}  SD={learner_r.std():.3f}')
    print(f'Group t-test: t={t:.3f}  p={p:.4f}')

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
compute_corr_matrices()
run_qc_ttest()
df = get_best_runs()
print(df)
run_group_ttest(df)
