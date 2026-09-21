import argparse
import csv
import datetime
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agents.on_policy import Agent
from common import ALGORITHMS, ROOT, config_hash, json_dump, load_config, provenance
from envs.microgrid import MicrogridEnv
from federated.average import fedavg


def collect_episode(env, agents, seed, deterministic=False):
    obs = env.reset(seed)
    storage = [[] for _ in range(6)]
    details = []
    for agent in agents:
        agent.sync_old()
    for _ in range(24):
        normalized = env.normalize(obs)
        latents, logps = [], []
        with torch.no_grad():
            for i, agent in enumerate(agents):
                state = torch.from_numpy(normalized[i])
                dist = agent.old_actor.distribution(state)
                action = dist.loc if deterministic else dist.sample()
                latents.append(action.numpy())
                logps.append(dist.log_prob(action).sum().item())
        latents = np.asarray(latents)
        next_obs, reward, done, rows = env.step(latents)
        values = [normalized, latents, reward, env.normalize(next_obs), np.full(3, float(done)), logps]
        for buffer, value in zip(storage, values):
            buffer.append(value)
        details.extend(rows)
        obs = next_obs
    return [np.asarray(x) for x in storage], details


def train_one(cfg, algorithm, seed, run_id, resume=None):
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agents = [Agent(cfg) for _ in range(3)]
    # Algorithm1从同一个全局初始化出发，四算法都使用同一初始化以公平比较。
    for agent in agents[1:]:
        agent.actor.load_state_dict(agents[0].actor.state_dict())
        agent.critic.load_state_dict(agents[0].critic.state_dict())
        agent.sync_old()
    env = MicrogridEnv(cfg)
    out = ROOT / 'logs' / run_id / algorithm
    checkpoint_dir = ROOT / 'checkpoints' / run_id / algorithm
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta = provenance(cfg)
    meta.update(run_id=run_id, algorithm=algorithm, seed=seed,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    history, events, start = [], [], 1
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['config_hash'] != config_hash(cfg) or state['algorithm'] != algorithm or state['seed'] != seed:
            raise ValueError('恢复配置、算法或种子不一致')
        if state['run_id'] != run_id:
            raise ValueError('恢复必须保持run_id，避免混合运行来源')
        for agent, saved in zip(agents, state['agents']):
            agent.restore(saved)
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        history, events = state['history'], state['events']
        start = state['iteration'] + 1
        if state['stage'] == 'pre_fed' and algorithm == 'F-MADRL' and state['iteration'] % cfg['federation_interval'] == 0:
            fedavg(agents)
            events.append({'iteration': state['iteration'], 'event': 'fedavg', 'replayed_on_resume': True})
    elif (out / 'history.csv').exists():
        raise FileExistsError(f'{out}: 已有结果，请使用新run_id或--resume')
    json_dump(out / 'metadata.json', meta)

    def save(iteration, stage):
        state = dict(iteration=iteration, stage=stage, run_id=run_id, seed=seed, algorithm=algorithm,
                     config=cfg, config_hash=config_hash(cfg), provenance=meta,
                     agents=[a.state() for a in agents], torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(), history=history, events=events)
        path = checkpoint_dir / f'iter_{iteration:04d}_{stage}.pt'
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    began = time.perf_counter()
    for iteration in range(start, cfg['training_iterations'] + 1):
        batch, details = collect_episode(env, agents, seed * 100000 + iteration)
        detail = pd.DataFrame(details)
        for i, agent in enumerate(agents):
            stats = agent.update([x[:, i] for x in batch], algorithm)
            rewards = batch[2][:, i]
            mg = detail[detail.mg == i + 1]
            history.append(dict(run_id=run_id, seed=seed, config_hash=config_hash(cfg), algorithm=algorithm,
                                iteration=iteration, mg=i + 1, phase=(iteration - 1) // cfg['federation_interval'] + 1,
                                episode_reward=float(rewards.sum()),
                                discounted_return=float((rewards * cfg['gamma'] ** np.arange(24)).sum()),
                                cg_cost=float(mg.cg_cost.sum()), ba_cost=float(mg.ba_cost.sum()),
                                imbalance_cost=float(mg.imbalance_cost.sum()),
                                abs_deficit=float(mg.p_de.abs().sum()), environment_steps=iteration * 24,
                                **stats))
        boundary = algorithm == 'F-MADRL' and iteration % cfg['federation_interval'] == 0
        save_now = iteration in cfg['checkpoint_iterations'] or iteration == cfg['training_iterations'] or boundary
        if save_now:
            save(iteration, 'pre_fed')
            detail.to_csv(out / f'trajectory_{iteration:04d}.csv', index=False)
        if boundary:
            fedavg(agents)
            events.append({'iteration': iteration, 'event': 'fedavg', 'weights': [1 / 3] * 3,
                           'actor_critic_log_std': True, 'optimizer_state': 'retained'})
            save(iteration, 'post_fed')
        if iteration % 50 == 0 or iteration == cfg['training_iterations']:
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            json_dump(out / 'federation_events.json', events)
            print(f'{run_id} {algorithm} {iteration}/{cfg["training_iterations"]} elapsed={time.perf_counter()-began:.1f}s', flush=True)
    if history:
        frame = pd.DataFrame(history)
        json_dump(out / 'completion.json', dict(run_id=run_id, seed=seed, algorithm=algorithm,
                  iterations=int(frame.iteration.max()), agent_transitions=int(frame.iteration.max()) * 72,
                  actor_updates=int(frame.actor_updates.sum()), critic_updates=int(frame.critic_updates.sum()),
                  elapsed_this_invocation_seconds=time.perf_counter() - began,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/main.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-id')
    parser.add_argument('--algorithm', choices=ALGORITHMS + ['all'], default='all')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--scenario', choices=['deterministic', 'gaussian_forecast_errors'])
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.iterations:
        cfg['training_iterations'] = args.iterations
    if args.scenario:
        cfg['scenario'] = args.scenario
    if args.resume and args.algorithm == 'all':
        parser.error('--resume需要明确一个算法')
    run_id = args.run_id or f'main_seed{args.seed}'
    for algorithm in ALGORITHMS if args.algorithm == 'all' else [args.algorithm]:
        train_one(cfg, algorithm, args.seed, run_id, args.resume)


if __name__ == '__main__':
    main()
