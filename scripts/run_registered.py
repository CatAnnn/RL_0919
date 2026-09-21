import concurrent.futures
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_seed(seed):
    path = ROOT / f'evidence/full_seed{seed}.log'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    with path.open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-u', 'train.py', '--seed', str(seed)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f'seed={seed} exit={result.returncode}', flush=True)
    return {'seed': seed, 'exit_code': result.returncode, 'log': str(path.relative_to(ROOT))}


if __name__ == '__main__':
    audit = json.loads((ROOT / 'evidence/smoke_audit.json').read_text(encoding='utf-8'))
    if not audit['passed']:
        raise RuntimeError('冒烟测试未通过，禁止完整训练')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_seed, [0, 1, 2, 3, 4]))
    (ROOT / 'evidence/registered_runs.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    sys.exit(0 if all(r['exit_code'] == 0 for r in results) else 1)
