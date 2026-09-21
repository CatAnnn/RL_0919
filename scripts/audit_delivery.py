import json
import sys
import zipfile
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ALGORITHMS, json_dump, load_config, sha256
from envs.microgrid import MicrogridEnv
from evaluate import evaluate_checkpoint


def audit():
    checks = {}
    details = {}
    cfg = load_config()
    for seed in cfg['seeds']:
        run_id = f'main_seed{seed}'
        for algorithm in ALGORITHMS:
            folder = ROOT / 'logs' / run_id / algorithm
            frame = pd.read_csv(folder / 'history.csv')
            name = f'{run_id}/{algorithm}'
            checks[name + '/complete_history'] = len(frame) == 4500 and all(len(g) == 1500 and g.iteration.tolist() == list(range(1, 1501)) for _, g in frame.groupby('mg'))
            checks[name + '/finite_training'] = bool(np.isfinite(frame.select_dtypes('number')).all().all())
            checks[name + '/actor_and_critic_update'] = bool(frame.actor_delta.sum() > 0 and frame.critic_delta.sum() > 0)
            checks[name + '/reward_identity'] = bool(np.allclose(frame.episode_reward,
                -cfg['w_cost'] * (frame.cg_cost + frame.ba_cost) - cfg['w_deviation'] * frame.imbalance_cost, atol=1e-8))
            if algorithm == 'TRPO-MADRL':
                checks[name + '/kl_bound'] = bool((frame.kl <= cfg['trpo_max_kl'] + 1e-6).all())
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            checks[name + '/source_hashes'] = all(sha256(ROOT / path) == expected for path, expected in metadata['source_hashes'].items())
            checks[name + '/data_hashes'] = all(sha256(ROOT / path) == expected for path, expected in metadata['input_hashes'].items())
            for iteration in cfg['checkpoint_iterations']:
                checks[name + f'/checkpoint_{iteration}'] = (ROOT / 'checkpoints' / run_id / algorithm / f'iter_{iteration:04d}_pre_fed.pt').is_file()
            details[name] = {'actor_updates': int(frame.actor_updates.sum()), 'critic_updates': int(frame.critic_updates.sum()),
                             'agent_transitions': 108000, 'min_actor_delta': float(frame.actor_delta.min()),
                             'max_kl': float(frame.kl.max())}
        f = pd.read_csv(ROOT / 'logs' / run_id / 'F-MADRL/history.csv')
        p = pd.read_csv(ROOT / 'logs' / run_id / 'PPO-MADRL/history.csv')
        columns = ['episode_reward', 'cg_cost', 'ba_cost', 'actor_loss', 'critic_loss', 'actor_delta', 'critic_delta']
        checks[run_id + '/f_ppo_identical_before_federation'] = bool(np.array_equal(f[f.iteration <= 500][columns], p[p.iteration <= 500][columns]))
        for boundary in [500, 1000, 1500]:
            cp = ROOT / 'checkpoints' / run_id / 'F-MADRL'
            pre = torch.load(cp / f'iter_{boundary:04d}_pre_fed.pt', weights_only=False)
            post = torch.load(cp / f'iter_{boundary:04d}_post_fed.pt', weights_only=False)
            checks[run_id + f'/fedavg_{boundary}_exact'] = all(torch.equal(torch.stack([a[part][k] for a in pre['agents']]).mean(0), post['agents'][i][part][k])
                for part in ['actor', 'critic'] for k in pre['agents'][0][part] for i in range(3))
            checks[run_id + f'/adam_{boundary}_retained'] = all(torch.equal(pre['agents'][i][part]['state'][k][s], post['agents'][i][part]['state'][k][s])
                for i in range(3) for part in ['actor_opt', 'critic_opt'] for k in pre['agents'][i][part]['state'] for s in pre['agents'][i][part]['state'][k])

    out = ROOT / 'figures/main_seed0'
    checks['fig05_fig10_shared_csv_exact'] = (out / 'fig05_fmadrl_source.csv').read_bytes() == (out / 'fig10_fmadrl_source.csv').read_bytes()
    prov = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
    checks['all_figure_inputs_hash_match'] = all(sha256(ROOT / path) == expected for fig in prov.values() for path, expected in fig['inputs'].items())
    for number in range(5, 11):
        png, pdf, svg = [out / f'fig{number:02d}.{ext}' for ext in ['png', 'pdf', 'svg']]
        pix = fitz.Pixmap(str(png))
        with fitz.open(pdf) as document:
            checks[f'fig{number:02d}_formats'] = pix.width > 600 and pix.height > 600 and len(document) == 1 and '<svg' in svg.read_text(encoding='utf-8')
    schedule = pd.read_csv(ROOT / 'logs/main_seed0/evaluation/schedule.csv')
    checks['schedule_72_rows'] = len(schedule) == 72
    checks['schedule_soc_bounds'] = bool(schedule.soc_after.between(cfg['soc_min'] - 1e-9, cfg['soc_max'] + 1e-9).all())
    checks['schedule_balance'] = float(schedule.balance_residual.abs().max()) < 1e-9
    checks['schedule_sign'] = bool(np.allclose(schedule.p_de, -schedule.unbalanced, atol=1e-12))
    saved = pd.read_csv(ROOT / 'logs/main_seed0/evaluation/checkpoint_details.csv')
    replays = []
    for iteration in [1, 50, 500, 700, 900, 1400, 1500]:
        frame, _ = evaluate_checkpoint(ROOT / 'checkpoints/main_seed0/F-MADRL' / f'iter_{iteration:04d}_pre_fed.pt')
        old = saved[saved.iteration == iteration]
        numeric = frame.select_dtypes('number').columns
        error = float(np.max(np.abs(frame[numeric].to_numpy() - old[numeric].to_numpy())))
        replays.append({'iteration': iteration, 'maximum_numeric_error': error})
        checks[f'checkpoint_{iteration}_replay'] = error < 1e-8
    # 用独立重放环境核对实际潜动作、完整物理明细及交易，避免只核对图表文件存在。
    env = MicrogridEnv(cfg)
    env.reset(cfg['evaluation_seed'])
    physical = []
    for hour in range(1, 25):
        rows = schedule[schedule.hour == hour].sort_values('mg')
        _, _, _, detail = env.step(rows[['latent_cg', 'latent_ba']].to_numpy())
        physical.extend(detail)
    physical = pd.DataFrame(physical)
    columns = physical.select_dtypes('number').columns
    checks['schedule_physics_replay'] = bool(np.allclose(physical[columns], schedule[columns], atol=1e-8, rtol=0))
    utf8_files = [p for p in ROOT.rglob('*') if p.suffix in ['.py', '.md', '.yaml'] and not any(x in p.parts for x in ['.git', '.codex'])]
    bad = []
    for path in utf8_files:
        try:
            text = path.read_text(encoding='utf-8-sig')
            if '\ufffd' in text:
                bad.append(str(path.relative_to(ROOT)))
        except UnicodeDecodeError:
            bad.append(str(path.relative_to(ROOT)))
    checks['utf8_no_replacement_characters'] = not bad
    report = {'engineering_checks_passed': all(checks.values()), 'checks': checks,
              'training_details': details, 'replay': replays, 'bad_encoding_files': bad,
              'scope': 'These checks establish execution and invariants; they do not establish paper numerical reproduction.'}
    json_dump(ROOT / 'evidence/delivery_audit.json', report)
    manifest = {str(p.relative_to(ROOT)): sha256(p) for base in ['checkpoints', 'logs', 'figures', 'data', 'configs'] for p in (ROOT / base).rglob('*') if p.is_file()}
    json_dump(ROOT / 'evidence/artifact_sha256.json', manifest)
    print(json.dumps({'passed': report['engineering_checks_passed'], 'check_count': len(checks),
                      'failed': [k for k, v in checks.items() if not v]}, indent=2), flush=True)
    if not report['engineering_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    audit()
