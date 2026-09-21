"""验证并汇总反推候选的最新训练、模型、图形及稳定性结果。"""
import json
import sys
import zipfile
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import ALGORITHMS, config_hash, json_dump, load_config, sha256
from envs.microgrid import MicrogridEnv
from evaluate import evaluate_checkpoint, evaluate_run

PREFIX='inferred_20260921'
EVIDENCE=ROOT/'evidence'/PREFIX
LATEST=ROOT/'latest_results'


def table(df):
    headers=list(df.columns)
    rows=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.3f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(rows)


def prepare_evaluations():
    cfg=load_config('configs/inferred_20260921.yaml')
    for seed in cfg['seeds']:
        evaluate_run(f'{PREFIX}_seed{seed}')


def deliver():
    cfg=load_config('configs/inferred_20260921.yaml')
    LATEST.mkdir(exist_ok=True)
    main=f'{PREFIX}_seed0'
    figures=ROOT/'figures'/main
    checks={};stability=[];final_details=[];histories={};optimizers=[]
    for seed in cfg['seeds']:
        run=f'{PREFIX}_seed{seed}'
        for alg in ALGORITHMS:
            folder=ROOT/'logs'/run/alg
            h=pd.read_csv(folder/'history.csv');histories[seed,alg]=h
            key=f'{run}/{alg}'
            metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            completion=json.loads((folder/'completion.json').read_text(encoding='utf-8'))
            checks[key+'/complete']=len(h)==4500 and all(g.iteration.tolist()==list(range(1,1501)) for _,g in h.groupby('mg'))
            checks[key+'/finite']=bool(np.isfinite(h.select_dtypes('number')).all().all())
            checks[key+'/candidate_config']=metadata['config']==cfg and metadata['config_hash']==config_hash(cfg)
            checks[key+'/new_reward_identity']=bool(np.allclose(h.episode_reward,-cfg['w_cost']*(h.cg_cost+h.ba_cost)-cfg['w_deviation']*h.imbalance_cost,rtol=0,atol=1e-7))
            checks[key+'/real_parameter_updates']=bool(h.actor_delta.sum()>0 and h.critic_delta.sum()>0)
            checks[key+'/completion']=completion['iterations']==1500 and completion['agent_transitions']==108000
            checks[key+'/sources']=all(sha256(ROOT/p)==v for p,v in metadata['source_hashes'].items())
            checks[key+'/data']=all(sha256(ROOT/p)==v for p,v in metadata['input_hashes'].items())
            if alg=='TRPO-MADRL':
                checks[key+'/kl_bound']=bool((h.kl<=cfg['trpo_max_kl']+1e-6).all())
            for it in cfg['checkpoint_iterations']:
                checks[key+f'/checkpoint_{it}']=(ROOT/'checkpoints'/run/alg/f'iter_{it:04d}_pre_fed.pt').is_file()
            final,_=evaluate_checkpoint(ROOT/'checkpoints'/run/alg/'iter_1500_pre_fed.pt')
            final_details.append(final)
            for mg in [1,2,3]:
                tail=h[(h.mg==mg)&(h.iteration>1450)];day=final[final.mg==mg]
                stability.append(dict(seed=seed,algorithm=alg,mg=mg,training_last50_mean=float(tail.episode_reward.mean()),
                    evaluation_reward=float(day.reward.sum()),abs_deficit=float(day.p_de.abs().sum()),cg_cost=float(day.cg_cost.sum()),ba_cost=float(day.ba_cost.sum())))
            optimizers.append(dict(run_id=run,algorithm=alg,agent_transitions=108000,actor_updates=int(h.actor_updates.sum()),critic_updates=int(h.critic_updates.sum()),elapsed_seconds=completion['elapsed_this_invocation_seconds']))
        f=histories[seed,'F-MADRL'];p=histories[seed,'PPO-MADRL']
        columns=['episode_reward','cg_cost','ba_cost','actor_loss','critic_loss','actor_delta','critic_delta']
        checks[run+'/f_ppo_before500_exact']=np.array_equal(f[f.iteration<=500][columns],p[p.iteration<=500][columns])
        for boundary in [500,1000,1500]:
            cp=ROOT/'checkpoints'/run/'F-MADRL'
            pre=torch.load(cp/f'iter_{boundary:04d}_pre_fed.pt',weights_only=False)
            post=torch.load(cp/f'iter_{boundary:04d}_post_fed.pt',weights_only=False)
            checks[run+f'/fedavg_{boundary}']=all(torch.equal(torch.stack([a[part][k] for a in pre['agents']]).mean(0),post['agents'][i][part][k]) for part in ['actor','critic'] for k in pre['agents'][0][part] for i in range(3))
            checks[run+f'/adam_retained_{boundary}']=all(torch.equal(pre['agents'][i][part]['state'][k][s],post['agents'][i][part]['state'][k][s]) for i in range(3) for part in ['actor_opt','critic_opt'] for k in pre['agents'][i][part]['state'] for s in pre['agents'][i][part]['state'][k])
    stability=pd.DataFrame(stability)
    stability.to_csv(LATEST/'stability_per_seed.csv',index=False)
    pd.concat(final_details).to_csv(LATEST/'final_evaluations_all_seeds.csv',index=False)
    summary=stability.groupby(['algorithm','mg']).agg(seed_count=('seed','count'),training_mean=('training_last50_mean','mean'),training_std=('training_last50_mean','std'),evaluation_mean=('evaluation_reward','mean'),evaluation_std=('evaluation_reward','std')).reset_index()
    summary.to_csv(LATEST/'stability_summary.csv',index=False)
    pd.DataFrame(optimizers).to_csv(LATEST/'training_budget.csv',index=False)
    # 图5和图10必须逐行共享原始曲线及显示值。
    checks['fig05_fig10_source_identical']=(figures/'fig05_fmadrl_source.csv').read_bytes()==(figures/'fig10_fmadrl_source.csv').read_bytes()
    shared=pd.read_csv(figures/'fig05_fmadrl_source.csv',float_precision='round_trip')
    all_curves=pd.read_csv(figures/'fig10_all_algorithms_source.csv',float_precision='round_trip')
    curve=all_curves[all_curves.algorithm=='F-MADRL'].sort_values(['mg','iteration'])
    common=shared.sort_values(['mg','iteration'])
    checks['fig05_fig10_plotted_values_identical']=bool(np.array_equal(curve[['episode_reward','display_reward']].to_numpy(),common[['episode_reward','display_reward']].to_numpy()))
    checks['exported_rewards_match_training']=bool(np.array_equal(shared.episode_reward,histories[0,'F-MADRL'].episode_reward))
    prov=json.loads((figures/'provenance.json').read_text(encoding='utf-8'))
    checks['figure_hashes']=all(sha256(ROOT/p)==h for f in prov.values() for p,h in f['inputs'].items())
    checks['figure_script_hash']=all(f['source_script_sha256']==sha256(ROOT/'scripts/plot_paper_style.py') for f in prov.values())
    for n in range(5,11):
        png=fitz.Pixmap(str(figures/f'fig{n:02d}.png'))
        with fitz.open(figures/f'fig{n:02d}.pdf') as pdf:
            checks[f'fig{n:02d}_formats']=len(pdf)==1 and png.width>1500 and '<svg' in (figures/f'fig{n:02d}.svg').read_text(encoding='utf-8')
    schedule=pd.read_csv(ROOT/'logs'/main/'evaluation/schedule.csv')
    checkpoints=pd.read_csv(ROOT/'logs'/main/'evaluation/checkpoint_details.csv')
    checks['schedule72']=len(schedule)==72
    checks['schedule_soc']=bool(schedule.soc_after.between(.1-1e-9,.9+1e-9).all())
    checks['schedule_balance']=bool(schedule.balance_residual.abs().max()<1e-9)
    checks['schedule_sign']=bool(np.allclose(schedule.p_de,-schedule.unbalanced,rtol=0,atol=1e-10))
    checks['schedule_reward']=bool(np.allclose(schedule.reward,-.02*(schedule.cg_cost+schedule.ba_cost)-.45*schedule.imbalance_cost,rtol=0,atol=1e-8))
    for it in [1,50,500,700,900,1400,1500]:
        replay,_=evaluate_checkpoint(ROOT/'checkpoints'/main/'F-MADRL'/f'iter_{it:04d}_pre_fed.pt')
        old=checkpoints[checkpoints.iteration==it]
        numeric=replay.select_dtypes('number').columns
        checks[f'checkpoint_replay_{it}']=bool(np.allclose(replay[numeric],old[numeric],rtol=0,atol=1e-8))
    env=MicrogridEnv(cfg);env.reset(cfg['evaluation_seed']);rows=[]
    for hour in range(1,25):
        g=schedule[schedule.hour==hour].sort_values('mg')
        _,_,_,detail=env.step(g[['latent_cg','latent_ba']].to_numpy());rows.extend(detail)
    replay=pd.DataFrame(rows);numeric=replay.select_dtypes('number').columns
    checks['physical_schedule_replay']=bool(np.allclose(replay[numeric],schedule[numeric],rtol=0,atol=1e-8))
    previous=json.loads((ROOT/'evidence/artifact_sha256.json').read_text(encoding='utf-8'))
    # 状态文档已在上一轮反推任务中更新，不属于不变的实验输入/模型/日志。
    previous={p:h for p,h in previous.items() if p!='reproduction_state.md'}
    old_bad=[p for p,h in previous.items() if sha256(ROOT/p)!=h]
    checks['previous_artifacts_unchanged']=not old_bad
    json_dump(EVIDENCE/'delivery_audit.json',dict(passed=all(checks.values()),checks=checks,old_artifact_mismatches=old_bad,old_artifacts_checked=len(previous),excluded_mutable_documents=['reproduction_state.md'],scope='工程真实性与一致性；不是论文数值成功证明'))
    if not all(checks.values()):
        raise RuntimeError([k for k,v in checks.items() if not v])

    f=histories[0,'F-MADRL'];compare=[]
    for mg,it,reference in [(1,956,-15059.85),(2,942,-3524.99),(3,938,-6164.79),(1,1328,-15059.85),(2,1437,-3415.94),(3,1406,-3743.82)]:
        actual=float(f[(f.mg==mg)&(f.iteration==it)].episode_reward.iloc[0])
        compare.append(dict(mg=mg,iteration=it,paper_reward=reference,actual_reward=actual,relative_error_percent=100*abs(actual-reference)/abs(reference)))
    comparison=pd.DataFrame(compare);comparison.to_csv(LATEST/'fig05_comparison.csv',index=False)
    ranking=stability.pivot(index=['seed','mg'],columns='algorithm',values='training_last50_mean')
    wins=ranking['F-MADRL']>ranking[[a for a in ALGORITHMS if a!='F-MADRL']].max(axis=1)
    eranking=stability.pivot(index=['seed','mg'],columns='algorithm',values='evaluation_reward')
    ewins=eranking['F-MADRL']>eranking[[a for a in ALGORITHMS if a!='F-MADRL']].max(axis=1)
    metrics=pd.read_csv(ROOT/'logs'/main/'evaluation/checkpoint_metrics.csv')
    last9=metrics[metrics.iteration==1400][['mg','abs_deficit','cg_cost','ba_cost']]
    sched_stats=schedule.groupby('mg').agg(cg_min=('p_cg','min'),cg_max=('p_cg','max'),ba_min=('p_ba','min'),ba_max=('p_ba','max'),max_abs_imbalance=('p_de',lambda s:s.abs().max())).reset_index()
    sched_stats.to_csv(LATEST/'schedule_summary.csv',index=False)
    scientific=dict(strict_paper_reproduction=False,reason='反推仅确定候选范围；作者网络、采样和统计口径未公开，不能认证同一配置。',
        fig05_all_six_points_within5percent=bool((comparison.relative_error_percent<=5).all()),
        fig09_mg1_1400_within10percent_of3000=bool(abs(float(last9[last9.mg==1].abs_deficit.iloc[0])-3000)<=300),
        training_win_count=int(wins.sum()),evaluation_win_count=int(ewins.sum()),seed_mg_comparisons=15)
    json_dump(LATEST/'acceptance.json',scientific)
    report=f'''# 反推候选重新训练结果

本轮20次完整训练已完成，最新图5—10来自同一预登记配置，主图统一使用种子0。采用论文布局与配色，保留真实曲线变化。**严格论文数值复现未获认证**；反推候选不能视作作者公开配置。

## 配置与执行

- w_C=0.02、w_de=0.45、电池100 kWh、初始SOC0.5。其余物理和训练约定见[配置](../configs/inferred_20260921.yaml)和[事前登记](../evidence/inferred_20260921/registration.md)。
- 4算法×5种子×1500次×24小时×3个MG，共2,160,000条智能体转移；全部从随机初始化重新训练。
- 原始训练奖励、分项成本、优化次数、检查点及随机数/优化器状态全部保留。图6—8使用第1500次聚合前模型，图9为指定六个聚合前检查点。
- 图5/10浅线为原始episode sum，实线为阶段内25点均值；图5标注来自相应迭代的真实原始奖励。图9横轴是等间距检查点标签；失衡为逐小时绝对功率之和，并另存kWh及其他汇总。图6—8横轴0—23对应数据hour=1—24。
- 本次用户明确要求依据反推候选训练，这是参数来源的改变，不是奖励数据的事后缩放。训练/环境不读取数字化参考曲线。网损0.02和正功率放电沿用物理实现；反推的图内符号冲突仍未解决。

## 最新图片和数据

[六图浏览](../figures/{main}/index.html) · [主运行训练目录](../logs/{main}/) · [主运行模型目录](../checkpoints/{main}/) · [逐小时调度CSV](../logs/{main}/evaluation/schedule.csv) · [图9指标CSV](../logs/{main}/evaluation/checkpoint_metrics.csv)

各图提供PNG 400dpi、PDF和SVG，版式契约为Times衬线字体、盒式坐标轴、向内刻度、论文对应的阶段底色/配色/标记、相同子图顺序。坐标范围覆盖本次全部数据，不照搬论文范围而隐藏不同结果。附带导出图5/10原始与显示曲线CSV、图5数值标注CSV，以及每图来源哈希。

## 图5：指定位置真实奖励与论文参考

{table(comparison)}

这些位置对应论文标签；跨文献训练随机性、汇总口径和未披露参数使逐点差异不能单独用来判断算法正确性，但差异不能隐藏。

## 图6—8：最新调度范围

{table(sched_stats)}

物理检查通过并不保证论文调度形状相同。原文MG1约110—155 kW、MG2/3电池多数时刻近零只是待比较的结果，本轮没有强制这些形状。

## 图9：第1400次的实际指标

{table(last9)}

论文MG1末期失衡约3000，本轮为{float(last9[last9.mg==1].abs_deficit.iloc[0]):.3f}。原文图9与图6—8存在不能直接联立的数值差异，详见[反推报告](../inverse_parameter_report.md)。

## 图10与五种子稳定性

F-MADRL在最后50次训练奖励均值上优于全部基线的比较数为 **{int(wins.sum())}/15**，固定最终模型确定性评估的胜出数为 **{int(ewins.sum())}/15**。不预设或重排算法排名。

{table(summary)}

[逐种子指标](stability_per_seed.csv) · [均值和样本标准差](stability_summary.csv) · [四算法全部最终逐小时评估](final_evaluations_all_seeds.csv) · [训练和优化预算](training_budget.csv)

## 验证与重绘

本轮单元测试和新配置冒烟已通过；{len(checks)}项工程检查通过，包括奖励分解、实际参数变化、TRPO KL约束、FedAvg、Adam状态保留、七检查点重放、独立物理重放，以及图5/10数据完全一致。旧交付清单排除持续更新的状态文档后，其余{len(previous)}个文件哈希全部保持一致。

```powershell
& 'F:\\Blackma\\Anaconda\\envs\\pytorch_env\\python.exe' -X utf8 evaluate.py --run-id {main}
& 'F:\\Blackma\\Anaconda\\envs\\pytorch_env\\python.exe' -X utf8 scripts/plot_paper_style.py --run-id {main}
```

以上命令只做检查点评估和重绘，不重新训练。完整新训练需使用未占用run_id；已有结果受到覆盖保护。新一轮使用同一配置、种子和训练入口可复跑，但本轮没有为改善图形而继续搜索。

[工程审计](../evidence/inferred_20260921/delivery_audit.json) · [数值状态](acceptance.json) · [图形来源](../figures/{main}/provenance.json)。历史输出保留原目录，最新入口只指向本轮完整的一组结果。
'''
    (LATEST/'reproduction_report.md').write_text(report,encoding='utf-8')
    run_links=[]
    for seed in cfg['seeds']:
        run=f'{PREFIX}_seed{seed}'
        links=' · '.join(f'<a href="../logs/{run}/{a}/history.csv">{a}</a>' for a in ALGORITHMS)
        run_links.append(f'<tr><td>{seed}</td><td>{links}</td><td><a href="../checkpoints/{run}/F-MADRL/iter_1500_pre_fed.pt">F-MADRL模型</a></td></tr>')
    cards=''.join(f'<article><h2>Fig. {n}</h2><a href="../figures/{main}/fig{n:02d}.png"><img src="../figures/{main}/fig{n:02d}.png" alt="Fig. {n}"></a><p><a href="../figures/{main}/fig{n:02d}.pdf">PDF</a> · <a href="../figures/{main}/fig{n:02d}.svg">SVG</a></p></article>' for n in range(5,11))
    (LATEST/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>最新训练数据与图5—10</title><style>body{font:16px system-ui;max-width:1240px;margin:40px auto;padding:0 24px;color:#222}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:24px}article{border:1px solid #ddd;padding:18px}img{width:100%}a{color:#125a94}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}</style><h1>最新训练数据与图5—10</h1><p>反推候选：w_C=0.02，w_de=0.45，电池100 kWh，初始SOC0.5。20次训练已完成；主图固定种子0。</p><p><a href="reproduction_report.md">本轮报告</a> · <a href="stability_summary.csv">五种子稳定性CSV</a> · <a href="../logs/'+main+'/evaluation/schedule.csv">调度CSV</a> · <a href="../logs/'+main+'/evaluation/checkpoint_metrics.csv">检查点指标CSV</a></p><p>图片按论文风格绘制，曲线保留本轮真实数值；严格论文数值复现尚未获认证。</p><table><tr><th>种子</th><th>原始训练数据</th><th>最终模型</th></tr>'+''.join(run_links)+'</table><main>'+cards+'</main></html>',encoding='utf-8')
    json_dump(LATEST/'latest.json',dict(main_run_id=main,run_ids=[f'{PREFIX}_seed{s}' for s in cfg['seeds']],config='configs/inferred_20260921.yaml',config_hash=config_hash(cfg),figures=f'figures/{main}',logs=[f'logs/{PREFIX}_seed{s}' for s in cfg['seeds']],checkpoints=[f'checkpoints/{PREFIX}_seed{s}' for s in cfg['seeds']],engineering_audit_passed=True,strict_paper_reproduction=False))
    source_paths=[ROOT/'train.py',ROOT/'common.py',ROOT/'evaluate.py',ROOT/'plot_figures.py',ROOT/'configs/inferred_20260921.yaml',EVIDENCE/'registration.md',ROOT/'scripts/run_inferred_training.py',ROOT/'scripts/plot_paper_style.py',Path(__file__)]
    source_paths += [p for folder in ['agents','envs','federated','data'] for p in (ROOT/folder).glob('*') if p.is_file()]
    with zipfile.ZipFile(EVIDENCE/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path,str(path.relative_to(ROOT)))
    with zipfile.ZipFile(EVIDENCE/'source_snapshot.zip') as archive:
        assert archive.testzip() is None
    files=list(LATEST.glob('*'))+list(figures.glob('*'))+source_paths
    for seed in cfg['seeds']:
        files += [p for base in ['logs','checkpoints'] for p in (ROOT/base/f'{PREFIX}_seed{seed}').rglob('*') if p.is_file()]
    json_dump(EVIDENCE/'artifact_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in files if p.is_file()})
    print(json.dumps(dict(engineering_checks=len(checks),passed=True,scientific=scientific),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','deliver'])
    args=parser.parse_args();prepare_evaluations() if args.stage=='evaluate' else deliver()
