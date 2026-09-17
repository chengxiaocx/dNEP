"""
nep_ha.py
=========
Hyperalignment pipeline for the NEP dataset.

Steps:
  1. Load best-run assignments from QC output (nep_qc_wdsen.py)
  2. Compute per-subject hyperalignment transforms (searchlight Procrustes)
  3. Apply transforms to test runs and save aligned data

Dependencies:
    pip install git+https://github.com/feilong/neuroboros.git
    pip install git+https://github.com/feilong/hyperalignment.git
"""

import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import neuroboros as nb
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from hyperalignment import initialize_sparse_matrix, searchlight_weights
from hyperalignment.searchlight import searchlight_procrustes
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HA_DIR        = '/work/cxiao/NEP/HA'
TPL_DIR       = '/work/cxiao/sparta-gd_nll8.0_50000'
BEST_RUNS_CSV = '/work/cxiao/NEP/GLM/prosem_contrasts_revised/prosem_contrasts_mat/best_runs.csv'
RADIUS      = 20
TASK        = 'tb'
ALL_RUNS    = {'wd-01', 'wd-02', 'sen-01', 'sen-02'}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NEP(nb.datasets.Dataset):
    def __init__(self, root_dir='/work/cxiao/NEP/nb-data'):
        super().__init__(
            'NEP',
            dl_source=None,
            root_dir=root_dir,
            space=['onavg-ico32'],
            resample=['1step_pial_overlap'],
            prep='default',
            fp_version='25.1.4',
        )
        self.subjects = self.subject_sets['all']

    def rename_func(self, sid, task, run, suffix='.npy'):
        return f'sub-{sid}_task-{task}_run-{run:02d}{suffix}'

    def load_confounds(self, sid, task, run, fp_version):
        filename = f'task-{task}_run-{run:02d}_desc-confounds_timeseries.npy'
        path = os.path.join(self.root_dir, fp_version, 'confounds', filename)
        if not os.path.exists(path):
            alt = os.path.join(self.root_dir, fp_version, 'confounds', f'sub-{sid}_{filename}')
            if os.path.exists(alt):
                path = alt
            else:
                raise RuntimeError(f'Confounds not found: {path}')
        return [np.load(path)]

# ---------------------------------------------------------------------------
# Step 1: Parse best-run assignments
# ---------------------------------------------------------------------------
def standardize_run(run_str):
    """Converts 'wd_01', 'wd1', 'wd-1' -> 'wd-01'"""
    m = re.match(r'([a-zA-Z]+)[_-]?0*(\d+)', str(run_str).strip())
    return f'{m.group(1)}-0{m.group(2)}' if m else None

def load_run_assignments(dset):
    df = pd.read_csv(BEST_RUNS_CSV)
    assignments = {}
    for sid in dset.subjects:
        row = df[df['subject'].astype(str) == str(sid)]
        if row.empty:
            print(f'[skip] {sid}: not in CSV')
            continue
        best = {standardize_run(row['run_A'].values[0]),
                standardize_run(row['run_B'].values[0])}
        training = sorted(ALL_RUNS - best)
        assignments[sid] = {'best': list(best), 'training': training}
    return assignments

# ---------------------------------------------------------------------------
# Step 2: Compute hyperalignment transforms
# ---------------------------------------------------------------------------
def compute_transforms(dset, assignments, sls, tpl, mat0, w):
    total = len(dset.subjects)
    for i, sid in enumerate(dset.subjects, 1):
        print(f'\n[{i}/{total}] {sid}')
        out_dir = os.path.join(HA_DIR, 'xfms', str(sid))
        out_fn  = os.path.join(out_dir, f'{TASK}_{RADIUS}mm_to-tpl.npz')

        if os.path.exists(out_fn):
            print(f'  -> already exists, skipping')
            continue
        if sid not in assignments:
            print(f'  -> no run assignment, skipping')
            continue

        dm_list = []
        for run_str in assignments[sid]['training']:
            t_name, r_str = run_str.split('-')
            dm_list.append(dset.get_data(sid, t_name, int(r_str), 'lr'))
        dm = np.concatenate(dm_list, axis=0)

        targets = dm @ nb.mapping('lr', 'onavg-ico32', 'onavg-ico8', mask=True)
        conn    = 1 - cdist(targets.T, dm.T, 'correlation')
        conn    = np.nan_to_num(zscore(conn, axis=0))

        xfm = searchlight_procrustes(conn, tpl, sls, weights=w, mat0=mat0)
        os.makedirs(out_dir, exist_ok=True)
        nb.save(out_fn, xfm)
        print(f'  -> saved {out_fn}')

# ---------------------------------------------------------------------------
# Step 3: Apply transforms to test runs
# ---------------------------------------------------------------------------
def apply_transforms(dset, assignments):
    save_dir = os.path.join(HA_DIR, 'aligned')
    os.makedirs(save_dir, exist_ok=True)

    for sid in dset.subjects:
        if sid not in assignments:
            continue
        xfm_path = os.path.join(HA_DIR, 'xfms', str(sid), f'{TASK}_{RADIUS}mm_to-tpl.npz')
        try:
            xfm = sp.load_npz(xfm_path).toarray()
        except FileNotFoundError:
            print(f'[skip] {sid}: xfm not found')
            continue

        for run_str in assignments[sid]['best']:
            t_name, r_str = run_str.split('-')
            try:
                data = dset.get_data(sid, t_name, int(r_str), 'lr')
                if data.shape[1] != xfm.shape[0]:
                    data = data.T
                aligned = data @ xfm
                out = os.path.join(save_dir, f'{sid}_{run_str}_aligned.npy')
                np.save(out, aligned)
                print(f'{sid} | {run_str} | shape {aligned.shape} -> saved')
            except Exception as e:
                print(f'[error] {sid} {run_str}: {e}')

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dset = NEP()
    sls  = nb.sls('lr', RADIUS)

    mat0 = nb.record(
        os.path.join(HA_DIR, f'mat0_{RADIUS}mm.npz'),
        initialize_sparse_matrix,
        return_results=True,
    )(sls)

    tpl = np.concatenate([
        np.load(os.path.join(TPL_DIR, f'conn_zscored_{lr}h.npy')) for lr in 'lr'
    ], axis=1)
    tpl = tpl @ nb.mapping('lr', 'onavg-ico64', 'onavg-ico32')
    tpl = tpl[nb.mask('lr', 'onavg-ico8')]
    tpl = tpl[:, nb.mask('lr', 'onavg-ico32')]

    w = searchlight_weights(sls, radius=RADIUS)

    assignments = load_run_assignments(dset)

    # Save assignment log for reference
    log = pd.DataFrame([
        {'subject': sid, 'best_runs': str(v['best']), 'training_runs': str(v['training'])}
        for sid, v in assignments.items()
    ])
    log.to_csv(os.path.join(HA_DIR, 'training_run_assignments_log.csv'), index=False)

    compute_transforms(dset, assignments, sls, tpl, mat0, w)
    apply_transforms(dset, assignments)
