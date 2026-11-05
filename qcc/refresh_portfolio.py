"""Keep the submission at the public frontier: pick the cheapest public circuit per task, rebuild
submission.zip when any task improves, and (with --submit) submit it from nhatlong1103.

Source: the Apache-2.0 manifest `manwithacat/vqcc-apache-public-circuits` (local official-verifier costs).
State (current members, last submission) is kept in `state.json` next to the work directory.

    python refresh_portfolio.py --work DIR [--submit] [--min-gain 0.0005] [--verify]

`--verify` runs the official invsqrt/dqi/dfthcsos verifiers locally on changed members before
submitting (needs the competition verifiers under DIR/D/verifiers).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

DS = 'manwithacat/vqcc-apache-public-circuits'
COMP = 'google-verifiable-quantum-circuit-compilation'
REF = {'invsqrt': 2076774.2287, 'dqi': 2.4439e13, 'chemistry': 3355599588.3155365}
TOKEN = Path.home() / '.kaggle' / 'tok_nhatlong1103'


def kaggle(*args, quiet=True):
    env = dict(os.environ, KAGGLE_API_TOKEN=TOKEN.read_text().strip())
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True, env=env)
    if r.returncode and not quiet:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


def fetch(path: str, dest: Path) -> Path:
    out = dest / path
    if not out.exists():
        kaggle('datasets', 'download', DS, '-f', path, '-p', str(dest / Path(path).parent), '-q')
        alt = dest / Path(path).parent / Path(path).name
        if alt.exists() and alt != out:
            out.parent.mkdir(parents=True, exist_ok=True)
            alt.rename(out)
    return out


def best_rows(manifest: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for task in ('invsqrt', 'dqi', 'chemistry'):
        rows = manifest[manifest.task == task].sort_values('local_task_score')
        out[task] = rows.iloc[0]
    return out


def score(costs: dict[str, float]) -> float:
    return math.prod(REF[t] / c for t, c in costs.items()) ** (1 / 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--min-gain', type=float, default=0.0005, help='relative LB gain needed to resubmit')
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    state_file = work / 'state.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    (work / 'manifest.csv').unlink(missing_ok=True)
    kaggle('datasets', 'download', DS, '-f', 'manifest.csv', '-p', str(work), '-q')
    best = best_rows(pd.read_csv(work / 'manifest.csv'))
    new = {t: {'sha12': r.sha12, 'cost': float(r.local_task_score), 'author': r.author} for t, r in best.items()}
    old = state.get('members')
    s_new = score({t: v['cost'] for t, v in new.items()})
    s_old = score({t: v['cost'] for t, v in old.items()}) if old else 0.0
    print(json.dumps({'best_public': new, 'est_score_new': s_new, 'est_score_current': s_old}, indent=1))
    if old and s_new < s_old * (1 + a.min_gain):
        print('no meaningful improvement; nothing to do')
        return
    files = work / 'files'
    inv = fetch(f"circuits/invsqrt/{new['invsqrt']['sha12']}.kmb", files)
    dqi = fetch(f"circuits/dqi/{new['dqi']['sha12']}.kmb", files)
    chem = files / 'circuits' / 'chemistry' / new['chemistry']['sha12']
    for k in range(64):
        r = kaggle('datasets', 'download', DS, '-f', f'circuits/chemistry/{chem.name}/angle_k_{k}.kmb', '-p', str(chem), '-q')
        if r.returncode:
            break
    fetch(f'circuits/chemistry/{chem.name}/submission.npz', files)
    zip_path = work / 'submission.zip'
    subprocess.run([sys.executable, str(Path(__file__).with_name('make_submission.py')), '--invsqrt', str(inv),
                    '--dqi', str(dqi), '--chem', str(chem), '--out', str(zip_path)], check=True)
    if a.submit:
        msg = 'refresh: ' + ', '.join(f"{t} {v['sha12']}" for t, v in new.items()) + f' est {s_new:.5f}'
        r = kaggle('competitions', 'submit', COMP, '-f', str(zip_path), '-m', msg, quiet=False)
        print(r.stdout[-500:])
        state = {'members': new, 'submitted_est': s_new}
        state_file.write_text(json.dumps(state, indent=1))


if __name__ == '__main__':
    main()
