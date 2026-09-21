import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agents.on_policy import Agent
from common import ROOT, json_dump, sha256
from envs.microgrid import MicrogridEnv
from train import collect_episode


def evaluate_checkpoint(path, scenario=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config'].copy()
    if scenario:
        cfg['scenario'] = scenario
    torch.set_num_threads(cfg['threads'])
    agents = [Agent(cfg) for _ in range(3)]
    for agent, saved in zip(agents, checkpoint['agents']):
        agent.restore(saved)
    _, rows = collect_episode(MicrogridEnv(cfg), agents, cfg['evaluation_seed'], deterministic=True)
    frame = pd.DataFrame(rows)
    for key in ['run_id', 'seed', 'config_hash', 'iteration', 'stage', 'algorithm']:
        frame[key] = checkpoint[key]
    frame['scenario'] = cfg['scenario']
    frame['checkpoint_sha256'] = sha256(path)
    return frame, checkpoint


def evaluate_run(run_id, scenario=None):
    source = ROOT / 'checkpoints' / run_id / 'F-MADRL'
    out = ROOT / 'logs' / run_id / ('evaluation' if not scenario else f'evaluation_{scenario}')
    out.mkdir(parents=True, exist_ok=True)
    all_details, metrics, sources = [], [], []
    for iteration in [1, 50, 500, 700, 900, 1400, 1500]:
        path = source / f'iter_{iteration:04d}_pre_fed.pt'
        frame, checkpoint = evaluate_checkpoint(path, scenario)
        all_details.append(frame)
        sources.append({'iteration': iteration, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
        for mg, group in frame.groupby('mg'):
            metrics.append(dict(mg=int(mg), iteration=iteration, stage='pre_fed',
                                abs_deficit=float(group.p_de.abs().sum()),
                                abs_net_deficit=float(abs(group.p_de.sum())),
                                positive_deficit=float(group.p_de.clip(lower=0).sum()),
                                unbalanced_energy_kwh=float(group.p_de.abs().sum() * checkpoint['config']['dt_hours']),
                                cg_cost=float(group.cg_cost.sum()), ba_cost=float(group.ba_cost.sum()),
                                episode_reward=float(group.reward.sum()),
                                max_balance_residual=float(group.balance_residual.abs().max())))
        if iteration == 1500:
            frame.to_csv(out / 'schedule.csv', index=False)
    pd.concat(all_details).to_csv(out / 'checkpoint_details.csv', index=False)
    pd.DataFrame(metrics).to_csv(out / 'checkpoint_metrics.csv', index=False)
    json_dump(out / 'sources.json', dict(run_id=run_id, seed=checkpoint['seed'],
              config_hash=checkpoint['config_hash'], scenario=scenario or checkpoint['config']['scenario'],
              evaluation_action='tanh(latent mean)', checkpoints=sources,
              data=checkpoint['provenance']['input_hashes']))
    print(f'Evaluated {run_id} -> {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='main_seed0')
    parser.add_argument('--scenario', choices=['deterministic', 'gaussian_forecast_errors'])
    args = parser.parse_args()
    evaluate_run(args.run_id, args.scenario)
