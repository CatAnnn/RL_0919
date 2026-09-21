"""执行已登记的反推候选验证与五种子训练。"""
import argparse
import concurrent.futures
import datetime
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ALGORITHMS, config_hash, json_dump, load_config, sha256
from envs.microgrid import MicrogridEnv
from train import train_one

CONFIG = 'configs/inferred_20260921.yaml'
OUT = ROOT / 'evidence/inferred_20260921'
PREFIX = 'inferred_20260921'


def smoke():
    cfg = load_config(CONFIG)
    OUT.mkdir(exist_ok=True, parents=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    with (OUT / 'unit_tests.txt').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('单元测试失败，禁止训练')
    checks = {'unit_tests': True}
    for algorithm in ALGORITHMS:
        short = dict(cfg, training_iterations=8)
        train_one(short, algorithm, 0, PREFIX + '_smoke')
        h = pd.read_csv(ROOT / 'logs' / (PREFIX + '_smoke') / algorithm / 'history.csv')
        checks[algorithm + '_complete_finite'] = len(h)==24 and bool(np.isfinite(h.select_dtypes('number')).all().all())
        checks[algorithm + '_real_updates'] = bool(h.actor_delta.sum()>0 and h.critic_delta.sum()>0)
        checks[algorithm + '_new_reward_identity'] = bool(np.allclose(h.episode_reward,-.02*(h.cg_cost+h.ba_cost)-.45*h.imbalance_cost,atol=1e-8))
    env = MicrogridEnv(cfg)
    env.reset(0)
    rng = np.random.default_rng(20260921)
    rows=[]
    for _ in range(24):
        _,_,_,r=env.step(rng.normal(0,4,(3,2)));rows.extend(r)
    df=pd.DataFrame(rows)
    checks['new_config_soc_balance'] = bool(df.soc_after.between(.1-1e-9,.9+1e-9).all() and df.balance_residual.abs().max()<1e-9)
    checks['new_config_hourly_reward'] = bool(np.allclose(df.reward,-.02*(df.cg_cost+df.ba_cost)-.45*df.imbalance_cost,atol=1e-8))
    boundary=dict(cfg,training_iterations=6,federation_interval=2,checkpoint_iterations=[1,2,4,6])
    run_id=PREFIX+'_boundary'
    train_one(boundary,'F-MADRL',11,run_id)
    cp=ROOT/'checkpoints'/run_id/'F-MADRL'
    before=torch.load(cp/'iter_0006_post_fed.pt',weights_only=False)
    train_one(boundary,'F-MADRL',11,run_id,cp/'iter_0002_pre_fed.pt')
    after=torch.load(cp/'iter_0006_post_fed.pt',weights_only=False)
    checks['boundary_resume_history_exact']=before['history']==after['history']
    checks['boundary_resume_parameters_exact']=all(torch.equal(before['agents'][i][p][k],after['agents'][i][p][k]) for i in range(3) for p in ['actor','critic'] for k in before['agents'][i][p])
    for step in [2,4,6]:
        pre=torch.load(cp/f'iter_{step:04d}_pre_fed.pt',weights_only=False)
        post=torch.load(cp/f'iter_{step:04d}_post_fed.pt',weights_only=False)
        checks[f'fedavg_{step}_exact']=all(torch.equal(torch.stack([a[p][k] for a in pre['agents']]).mean(0),post['agents'][i][p][k]) for p in ['actor','critic'] for k in pre['agents'][0][p] for i in range(3))
    json_dump(OUT/'smoke_audit.json',dict(passed=all(checks.values()),checks=checks,config_hash=config_hash(cfg),unit_tests_run=result.testsRun))
    print('Smoke checks:',checks,flush=True)
    if not all(checks.values()):
        raise RuntimeError('冒烟验证失败')


def run_seed(seed):
    path=OUT/f'seed{seed}.log'
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    command=[sys.executable,'-X','utf8','-u','train.py','--config',CONFIG,'--seed',str(seed),'--run-id',f'{PREFIX}_seed{seed}']
    with path.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        json_dump(OUT/f'seed{seed}_process.json',dict(pid=process.pid,command=command,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        code=process.wait()
    result=dict(seed=seed,exit_code=code,run_id=f'{PREFIX}_seed{seed}',log=str(path.relative_to(ROOT)))
    json_dump(OUT/f'seed{seed}_exit.json',result)
    print(result,flush=True)
    return result


def full():
    import json
    audit=json.loads((OUT/'smoke_audit.json').read_text(encoding='utf-8'))
    cfg=load_config(CONFIG)
    if not audit['passed'] or audit['config_hash']!=config_hash(cfg):
        raise RuntimeError('本配置冒烟未通过')
    json_dump(OUT/'launch.json',dict(config_hash=config_hash(cfg),config_file_sha256=sha256(ROOT/CONFIG),seeds=cfg['seeds'],main_seed=0,process_id=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run_seed,cfg['seeds']))
    json_dump(OUT/'registered_runs.json',results)
    if any(x['exit_code'] for x in results):
        raise RuntimeError('部分训练失败，检查各运行日志')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['smoke','full'])
    args=parser.parse_args()
    smoke() if args.stage=='smoke' else full()
