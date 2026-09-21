import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ['F-MADRL', 'PPO-MADRL', 'A2C-MADRL', 'TRPO-MADRL']
COLORS = ['#1867a2', '#eb8b23', '#299148', '#c94146']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_curve(frame, window):
    # 每段单独平滑，不跨联邦边界；原始列不改写。
    return frame.groupby('phase', sort=False).episode_reward.transform(lambda values: values.rolling(window, min_periods=1).mean())


def plot_run(run_id):
    logs = ROOT / 'logs' / run_id
    out = ROOT / 'figures' / run_id
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: logs / name / 'history.csv' for name in ALGORITHMS}
    histories = {name: pd.read_csv(path) for name, path in paths.items()}
    meta = json.loads((logs / 'F-MADRL/metadata.json').read_text(encoding='utf-8'))
    sources = json.loads((logs / 'evaluation/sources.json').read_text(encoding='utf-8'))
    window = meta['config']['smoothing_window']
    schedule_path = logs / 'evaluation/schedule.csv'
    metrics_path = logs / 'evaluation/checkpoint_metrics.csv'
    schedule = pd.read_csv(schedule_path)
    metrics = pd.read_csv(metrics_path)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18,
                         'savefig.dpi': 220, 'svg.fonttype': 'none'})
    provenance = {}

    def save(fig, number, inputs, checkpoint):
        fig.savefig(out / f'fig{number:02d}.png', bbox_inches='tight')
        fig.savefig(out / f'fig{number:02d}.pdf', bbox_inches='tight')
        fig.savefig(out / f'fig{number:02d}.svg', bbox_inches='tight')
        plt.close(fig)
        provenance[f'fig{number:02d}'] = dict(run_id=run_id, seed=meta['seed'], config_hash=meta['config_hash'],
            data=meta['input_hashes'], checkpoint=checkpoint,
            inputs={str(p.relative_to(ROOT)): digest(p) for p in inputs},
            interpretation='Assumption-labeled reconstruction; not certified paper reproduction')

    fmadrl = histories['F-MADRL']
    shared = fmadrl[['run_id', 'seed', 'config_hash', 'iteration', 'mg', 'phase', 'episode_reward']]
    shared.to_csv(out / 'fig05_fmadrl_source.csv', index=False)
    shared.to_csv(out / 'fig10_fmadrl_source.csv', index=False)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True, layout='constrained')
    for i, ax in enumerate(axes):
        f = fmadrl[fmadrl.mg == i + 1]
        for phase, color in enumerate(['#fcf7dc', '#e6f5ef', '#f2eafa']):
            ax.axvspan(phase * 500 + 1, (phase + 1) * 500, color=color, zorder=0)
            ax.text((phase + .5) / 3, .94, f'Phase {phase + 1}', ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.plot(f.iteration, f.episode_reward, color=COLORS[i], alpha=.23, linewidth=.5)
        ax.plot(f.iteration, display_curve(f, window), color=COLORS[i], linewidth=1.2)
        for boundary in [500, 1000]:
            ax.axvline(boundary, color='#727272', linestyle='--', linewidth=.8)
        ax.set_ylabel('Reward')
        ax.set_title(f'({chr(97+i)}) MG{i+1}', loc='left', fontsize=10)
    axes[-1].set_xlabel('Local training iteration')
    axes[-1].set_xlim(1, 1500)
    save(fig, 5, [paths['F-MADRL']], 'online training episodes; FedAvg after 500 and 1000')

    for mg in range(1, 4):
        frame = schedule[schedule.mg == mg]
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, layout='constrained')
        right = axes[0].twinx()
        right.spines['right'].set_visible(True)
        cg_line = axes[0].plot(frame.time_index, frame.p_cg, '-o', color=COLORS[0], markersize=3, linewidth=1.1, label=r'$P_{CG}$')
        ba_line = right.plot(frame.time_index, frame.p_ba, '-s', color=COLORS[1], markersize=3, linewidth=1.1, label=r'$P_{BA}$')
        axes[0].set_ylabel(r'$P_{CG}$ (kW)', color=COLORS[0])
        right.set_ylabel(r'$P_{BA}$ (kW)', color=COLORS[1])
        right.grid(False)
        axes[0].legend(cg_line + ba_line, [x.get_label() for x in cg_line + ba_line],
                       loc='lower right', bbox_to_anchor=(1, 1.0), borderaxespad=0, ncols=2, frameon=False)
        axes[0].set_title(f'(a) MG{mg} scheduling', loc='left')
        axes[1].bar(frame.time_index, frame.unbalanced, color=COLORS[mg-1], width=.82)
        axes[1].axhline(0, color='black', linewidth=.7)
        axes[1].set_ylabel('Unbalanced demand (kW)')
        axes[1].set_title('(b) Before energy trading', loc='left')
        axes[1].set_xlabel('Time index (h); hour = index + 1')
        axes[1].set_xticks([0, 4, 8, 12, 16, 20, 23])
        axes[1].set_xlim(-.7, 23.7)
        save(fig, mg + 5, [schedule_path], sources['checkpoints'][-1])

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True, layout='constrained')
    metrics = metrics[metrics.iteration != 1500]
    for panel, (key, label) in enumerate([('abs_deficit', 'Unbalanced demand\n(sum of hourly kW)'), ('cg_cost', 'Cost of CG ($)'), ('ba_cost', 'Cost of BA ($)')]):
        for mg, marker in [(1, 's'), (2, 'o'), (3, '^')]:
            frame = metrics[metrics.mg == mg]
            axes[panel].plot(np.arange(len(frame)), frame[key], '-' + marker, color=COLORS[mg-1], markersize=4, linewidth=1.1, label=f'MG{mg}')
        axes[panel].set_ylabel(label)
        axes[panel].set_title(f'({chr(97+panel)})', loc='left')
    axes[0].legend(ncols=3, loc='best')
    axes[-1].set_xticks(range(6), ['1', '50', '500', '700', '900', '1400'])
    axes[-1].set_xlabel('Checkpoint iteration (equally spaced labels)')
    save(fig, 9, [metrics_path], sources['checkpoints'][:-1])

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True, layout='constrained')
    for i, ax in enumerate(axes):
        for name, color in zip(ALGORITHMS, COLORS):
            frame = histories[name][histories[name].mg == i + 1]
            ax.plot(frame.iteration, frame.episode_reward, color=color, alpha=.12, linewidth=.45)
            ax.plot(frame.iteration, display_curve(frame, window), color=color, linewidth=1.1, label=name)
        ax.set_ylabel('Reward')
        ax.set_title(f'({chr(97+i)}) MG{i+1}', loc='left')
    axes[0].legend(loc='best', ncols=2, fontsize=8)
    axes[-1].set_xlabel('Local training iteration')
    axes[-1].set_xlim(1, 1500)
    save(fig, 10, list(paths.values()), 'online training episodes, all algorithms 1500 iterations')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    cards = '\n'.join(f'<article><h2>Fig. {n}</h2><a href="fig{n:02d}.png"><img src="fig{n:02d}.png" alt="Figure {n}"></a><p><a href="fig{n:02d}.pdf">PDF</a> · <a href="fig{n:02d}.svg">SVG</a></p></article>' for n in range(5, 11))
    (out / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>图5—10真实训练重建</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#173047}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:28px}img{width:100%}article{border:1px solid #ddd;padding:16px}a{color:#1867a2}</style><h1>图5—10：真实训练的假设重建</h1><p>主运行：' + run_id + '。公开信息不足以认证严格复现；数值结论请阅读<a href="../../reproduction_report.md">复现报告</a>。所有曲线来自保存的训练日志和检查点评估。</p><main>' + cards + '</main></html>', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='main_seed0')
    plot_run(parser.parse_args().run_id)
