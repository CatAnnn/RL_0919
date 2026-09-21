import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ALGORITHMS, json_dump, load_config
from train import train_one


if __name__ == '__main__':
    checks = {}
    for algorithm in ALGORITHMS:
        history = pd.read_csv(ROOT / 'logs/smoke_seed0' / algorithm / 'history.csv')
        checks[f'{algorithm}_8_iterations'] = len(history) == 24
        checks[f'{algorithm}_finite_updates'] = bool(np.isfinite(history.select_dtypes('number')).all().all())
        checks[f'{algorithm}_actor_changed'] = bool(history.actor_delta.sum() > 0)
    cfg = load_config()
    cfg.update(training_iterations=6, federation_interval=2, checkpoint_iterations=[1, 2, 4, 6])
    train_one(cfg, 'F-MADRL', 11, 'smoke_boundary11')
    folder = ROOT / 'checkpoints/smoke_boundary11/F-MADRL'
    original = torch.load(folder / 'iter_0006_post_fed.pt', weights_only=False)
    train_one(cfg, 'F-MADRL', 11, 'smoke_boundary11', folder / 'iter_0002_pre_fed.pt')
    resumed = torch.load(folder / 'iter_0006_post_fed.pt', weights_only=False)
    checks['resume_history_exact'] = original['history'] == resumed['history']
    checks['resume_parameters_exact'] = all(torch.equal(original['agents'][i][part][key], resumed['agents'][i][part][key])
        for i in range(3) for part in ['actor', 'critic'] for key in original['agents'][i][part])
    for boundary in [2, 4, 6]:
        pre = torch.load(folder / f'iter_{boundary:04d}_pre_fed.pt', weights_only=False)
        post = torch.load(folder / f'iter_{boundary:04d}_post_fed.pt', weights_only=False)
        checks[f'boundary_{boundary}_mean_exact'] = all(torch.equal(torch.stack([a[part][key] for a in pre['agents']]).mean(0), post['agents'][i][part][key])
            for part in ['actor', 'critic'] for key in pre['agents'][0][part] for i in range(3))
    json_dump(ROOT / 'evidence/smoke_audit.json', {'checks': checks, 'passed': all(checks.values())})
    print(json.dumps(checks, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)
