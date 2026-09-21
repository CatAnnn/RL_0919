import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ['F-MADRL', 'PPO-MADRL', 'A2C-MADRL', 'TRPO-MADRL']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_config(path='configs/main.yaml'):
    cfg = yaml.safe_load((ROOT / path).read_text(encoding='utf-8'))
    missing = [k for k, v in cfg.items() if v is None]
    if missing:
        raise ValueError('严格原文配置缺失参数，不能执行训练：' + ', '.join(missing))
    return cfg


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def provenance(cfg):
    inputs = list((ROOT / 'data').glob('*.csv'))
    source = [ROOT / x for x in ['train.py', 'common.py', 'assumptions.yaml']]
    source += list((ROOT / 'agents').glob('*.py'))
    source += list((ROOT / 'envs').glob('*.py'))
    source += list((ROOT / 'federated').glob('*.py'))
    return {
        'config_hash': config_hash(cfg), 'config': cfg,
        'python': sys.version, 'executable': sys.executable,
        'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(),
        'input_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in source},
        'paper_sha256': sha256(next(ROOT.glob('*.pdf'))),
        'scientific_status': 'assumption_labeled_reconstruction',
    }
