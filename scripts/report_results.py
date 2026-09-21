import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ALGORITHMS, json_dump, load_config
from evaluate import evaluate_checkpoint


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    cfg = load_config()
    histories, stability, evaluations = {}, [], []
    for seed in cfg['seeds']:
        for algorithm in ALGORITHMS:
            history = pd.read_csv(ROOT / 'logs' / f'main_seed{seed}' / algorithm / 'history.csv')
            histories[seed, algorithm] = history
            evaluation, _ = evaluate_checkpoint(ROOT / 'checkpoints' / f'main_seed{seed}' / algorithm / 'iter_1500_pre_fed.pt')
            evaluations.append(evaluation)
            for mg in range(1, 4):
                last = history[(history.mg == mg) & (history.iteration > 1450)]
                daily = evaluation[evaluation.mg == mg]
                stability.append(dict(seed=seed, algorithm=algorithm, mg=mg,
                    training_last50_mean=last.episode_reward.mean(), evaluation_reward=daily.reward.sum(),
                    abs_deficit=daily.p_de.abs().sum(), cg_cost=daily.cg_cost.sum(), ba_cost=daily.ba_cost.sum()))
    values = pd.DataFrame(stability)
    values.to_csv(ROOT / 'logs/stability_per_seed.csv', index=False)
    pd.concat(evaluations).to_csv(ROOT / 'logs/final_evaluations_all_seeds.csv', index=False)
    summary = values.groupby(['algorithm', 'mg']).agg(
        seed_count=('seed', 'count'), training_mean=('training_last50_mean', 'mean'),
        training_std=('training_last50_mean', 'std'), evaluation_mean=('evaluation_reward', 'mean'),
        evaluation_std=('evaluation_reward', 'std')).reset_index()
    summary.to_csv(ROOT / 'logs/stability_summary.csv', index=False)
    rank = values.pivot(index=['seed', 'mg'], columns='algorithm', values='training_last50_mean')
    wins = (rank['F-MADRL'] > rank[[x for x in ALGORITHMS if x != 'F-MADRL']].max(axis=1))
    eval_rank = values.pivot(index=['seed', 'mg'], columns='algorithm', values='evaluation_reward')
    eval_wins = eval_rank['F-MADRL'] > eval_rank[[x for x in ALGORITHMS if x != 'F-MADRL']].max(axis=1)
    f = histories[0, 'F-MADRL']
    comparisons = []
    for mg, iteration, reference in [(1, 956, -15059.85), (2, 942, -3524.99)]:
        actual = f[(f.mg == mg) & (f.iteration == iteration)].episode_reward.iloc[0]
        comparisons.append(dict(mg=mg, iteration=iteration, paper=reference, actual=actual,
                                signed_difference=actual-reference, absolute_difference=abs(actual-reference),
                                relative_error_percent=100*abs(actual-reference)/abs(reference)))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(ROOT / 'logs/main_seed0/fig05_point_comparison.csv', index=False)
    schedule = pd.read_csv(ROOT / 'logs/main_seed0/evaluation/schedule.csv')
    schedule_stats = []
    for mg, g in schedule.groupby('mg'):
        schedule_stats.append(dict(mg=int(mg), cg_min=g.p_cg.min(), cg_max=g.p_cg.max(),
            ba_min=g.p_ba.min(), ba_max=g.p_ba.max(), ba_near_zero_hours=int((g.p_ba.abs() <= 1).sum()),
            deficit_hours=int((g.p_de > 0).sum()), max_abs_unbalanced=g.p_de.abs().max()))
    schedules = pd.DataFrame(schedule_stats)
    schedules.to_csv(ROOT / 'logs/main_seed0/schedule_summary.csv', index=False)
    metrics = pd.read_csv(ROOT / 'logs/main_seed0/evaluation/checkpoint_metrics.csv')
    final9 = metrics[metrics.iteration == 1400][['mg', 'abs_deficit', 'cg_cost', 'ba_cost']].copy()
    first9 = metrics[metrics.iteration == 1][['mg', 'abs_deficit', 'cg_cost', 'ba_cost']].copy()
    phase_rows = []
    for mg in range(1, 4):
        for end in [500, 1000, 1500]:
            phase_rows.append(dict(mg=mg, phase_end=end, last50_mean=f[(f.mg == mg) & f.iteration.between(end-49, end)].episode_reward.mean()))
    pd.DataFrame(phase_rows).to_csv(ROOT / 'logs/main_seed0/phase_summary.csv', index=False)
    main_compare = values[values.seed == 0][['algorithm', 'mg', 'training_last50_mean', 'evaluation_reward']]
    gaussian = pd.read_csv(ROOT / 'logs/main_seed0/evaluation_gaussian_forecast_errors/checkpoint_metrics.csv')
    gaussian_end = gaussian[gaussian.iteration == 1500][['mg', 'episode_reward', 'abs_deficit']]
    audit = json.loads((ROOT / 'evidence/delivery_audit.json').read_text(encoding='utf-8'))
    checks = {
        'fig05_reference_points_within_5_percent': bool((comparison.relative_error_percent <= 5).all()),
        'fig06_mg1_all_hours_deficit': bool((schedule[schedule.mg == 1].p_de > 0).all()),
        'fig06_mg1_cg_in_approximate_paper_range_with_5kw_margin': bool(schedule[schedule.mg == 1].p_cg.between(105,160).all()),
        'fig07_mg2_unbalanced_under_50kw': bool((schedule[schedule.mg == 2].p_de.abs() < 50).all()),
        'fig08_mg3_unbalanced_under_50kw': bool((schedule[schedule.mg == 3].p_de.abs() < 50).all()),
        'fig07_fig08_battery_near_zero_majority': all((schedule[schedule.mg == mg].p_ba.abs() <= 1).sum() > 12 for mg in [2,3]),
        'fig09_mg1_final_unbalanced_approx3000_within10percent': bool(abs(final9[final9.mg == 1].abs_deficit.iloc[0]-3000) <= 300),
        'fig09_mg2_mg3_final_under500': bool((final9[final9.mg.isin([2,3])].abs_deficit < 500).all()),
        'fig10_main_seed_f_wins_all_mgs': bool(wins.loc[0].all()),
    }
    # 对照容差是报告用的工程诊断阈值，不是作者给出的验收标准。
    acceptance = dict(strict_paper_reproduction=False, engineering_checks_passed=audit['engineering_checks_passed'],
        reason='Author configuration and exact metric/episode protocol unavailable; observed numerical discrepancies remain.',
        diagnostic_thresholds_not_author_criteria=True, diagnostic_checks=checks,
        training_win_count=int(wins.sum()), evaluation_win_count=int(eval_wins.sum()), seed_mg_comparisons=15)
    json_dump(ROOT / 'evidence/acceptance.json', acceptance)
    meta = json.loads((ROOT / 'logs/main_seed0/F-MADRL/metadata.json').read_text(encoding='utf-8'))
    report = f'''# 图5—10复现报告

## 结论

已完成真实训练、指定检查点评估和六图输出；**严格论文复现未通过**。本次结果是公开方法与显式工程假设下的重建，不能称为作者实验的等价重现。论文缺少奖励权重、电池参数、网络/动作分布和统计协议，实际数值也存在显著差异。没有根据原图调节输入或奖励，没有伪造曲线，没有挑选种子或拼接模型。

- 主图固定 `main_seed0`，一套预登记配置，种子0—4全部报告。
- 4算法 × 5种子 × 1500次 × 24小时 × 3个MG = **2,160,000条智能体训练转移**（不含独立冒烟与评估）。
- 工程审计：{len(audit['checks'])}项，全部通过状态：{audit['engineering_checks_passed']}；单元测试13项；四算法冒烟与边界恢复均通过。参数变化、KL约束和重放核验不等同论文数值验收。
- 实际环境：`{meta['executable']}`，Python `{meta['python'].split()[0]}`，PyTorch `{meta['torch']}`；CPU每个训练进程1线程，最多3个独立种子进程。与作者Python3.6.8/PyTorch1.7.1不同。
- 作者给出的参数、补充参数和公式修正见 [规格](paper_spec.md)、[假设](assumptions.yaml)、[歧义登记](discrepancy_register.md)。

## 图5：训练奖励

数据：[F-MADRL原始历史](logs/main_seed0/F-MADRL/history.csv)。主指标为24步奖励之和，另存折扣回报。实线仅作25次尾随平滑，各阶段分别计算；浅色线为原始记录。500、1000次之后实际聚合，未人工制造跃迁。

下表将论文标签中的指定迭代与同迭代原始episode sum比较；原文未公开奖励汇总和平滑，故误差也可能包含统计口径差异。

{markdown_table(comparison)}

阶段末50次均值：

{markdown_table(pd.DataFrame(phase_rows))}

## 图6—8：同一检查点的调度

统一采用种子0、第1500次聚合前模型，表II确定性场景、初始SOC0.5、tanh后的潜变量均值动作。数据：[schedule.csv](logs/main_seed0/evaluation/schedule.csv)。上图保留CG/BA双纵轴；下图使用交易前U=-Pde；hour=1—24与time_index=0—23同时保存。

{markdown_table(schedules)}

原文图6的MG1 CG约110—155kW且全天缺电；图7/8声称MG2/3电池多数时间接近零、失衡绝对值低于50kW。上述实际范围是训练所得，不强制这些特征。近零统计采用|PBA|≤1kW，属于报告阈值。三幅图的具体失败项列于文末验收表。所有外部购售电与内部交易单独保存；交易后平衡为零不能证明交易前失衡与原文一致。

## 图9：真实检查点的物理指标

使用1、50、500、700、900、1400次保存的模型；500取pre_fed。所有模型在相同确定性日曲线和SOC初值下重新评估。采用sum abs(Pde)、逐小时CG/BA成本求和；另保存abs(sum Pde)、正缺额之和和kWh。横轴是等距检查点标签。

第1次：

{markdown_table(first9)}

第1400次：

{markdown_table(final9)}

论文描述MG1失衡约5500降至3000、MG2/3降至500以下，MG1 BA成本约6000升至25000。本次应以表内实际数值比较，不能将不同统计口径或不同检查点混入图9。原文汇总式未公开，不能据此断定论文错误。

## 图10：四算法比较

F-MADRL直接读取图5同一CSV，导出的两个来源表逐字节相同。所有算法具有相同网络容量、状态动作、环境、奖励和采样预算；优化次数随算法而不同，详见 [审计记录](evidence/delivery_audit.json)。A2C执行不带clip的策略梯度，TRPO执行KL/Hessian向量积/共轭梯度/线搜索。

主种子最后50次训练均值与统一确定性最终评估：

{markdown_table(main_compare)}

不能预设F-MADRL最好。本次5种子×3个MG的15次比较中，F-MADRL训练末50次均值胜过全部基线 **{int(wins.sum())}/15**，最终确定性评估胜过全部基线 **{int(eval_wins.sum())}/15**。Table III不用于图10训练终值，本次也未复制作者未公开的Table III测试场景。

## 五种子稳定性（增强实验）

以下std是五个预定种子间样本标准差，不是单个训练过程内部波动，也不是论文原始统计；全部种子均保留。没有选择最优种子重绘主图。

{markdown_table(summary)}

详细数据：[逐种子](logs/stability_per_seed.csv)、[均值标准差](logs/stability_summary.csv)、[最终评估逐小时](logs/final_evaluations_all_seeds.csv)。

## 预测误差对照与范围

主结果选择确定性表II场景，原因是V-A与结论矛盾。另将同一组F-MADRL检查点置于固定高斯误差场景评估，风光15%、负荷3%，种子20260920；不是重新训练，也不代表统计稳健性已充分验证。第1500次结果：

{markdown_table(gaussian_end)}

高斯误差下完整训练**未执行**；命令入口已提供。作者原始代码/模型/完整超参数**未取得**；未开展未知参数搜索或30次试错，也未用既往仓库的模型替代本轮训练。

## 验收与差异原因

{markdown_table(pd.DataFrame([{'诊断项': k, '通过': v} for k,v in checks.items()]))}

5%/10%等容差只是报告诊断，不是论文规定标准，且不能覆盖所有曲线点。严格验收false同时基于未公开协议与实际差异，不将这些抽样检查当作完整逐点重现。已核实：数据表、物理恒等式、联邦平均、恢复、初始500次F/PPO同轨迹、真实参数更新及模型重放。尚未确定：奖励权重、电池参数、优化内循环、动作分布与原文统计口径各自造成多少差异。它们是待验证解释，不能断言某个单一因素导致全部偏差。

## 证据与重放

- [六图入口](figures/main_seed0/index.html)，每图PNG/PDF/SVG；[来源索引](figures/main_seed0/provenance.json)。
- [单元测试](evidence/unit_tests.txt)、[冒烟/精确恢复](evidence/smoke_audit.json)、[完整执行记录](evidence/registered_runs.json)。
- [工程审计](evidence/delivery_audit.json)、[科学验收](evidence/acceptance.json)、[产物哈希](evidence/artifact_sha256.json)。
- [运行命令](README.md)，绘图脚本只读取已保存CSV/JSON；改变样式无需重训。

开发期测试发现的SOC下界自放电投影错误已在完整训练前修复并回归验证。Conda包装命令长时间无输出，改为调用已核实的同一pytorch_env解释器，没有切换环境。论文阅读技能缺失的共享参考文件未作为科学依据，所有必需公式和表格以PDF图像及仓库模块核对。
'''
    (ROOT / 'reproduction_report.md').write_text(report, encoding='utf-8')
    state = f'''# 复现状态

## 当前任务
- 范围：图5—10，指定pytorch_env，真实实验与严格验收。
- 执行状态：已完成代码、13项单元测试、冒烟、20次完整训练、评估、六图和五种子报告。
- 科学状态：严格论文复现未通过；工程真实性与一致性审计通过。不能将任务目标标为已实现。

## 产物索引
| 产物 | 路径 | 状态 |
|---|---|---|
| 规格和公式 | paper_spec.md、discrepancy_register.md | PDF图像已核对 |
| 参数假设 | assumptions.yaml、configs/ | 工程参数明确标注 |
| 表I/II | data/ | 原表24点保留 |
| 测试 | evidence/unit_tests.txt、evidence/smoke_audit.json | 通过 |
| 完整训练 | logs/main_seed0至main_seed4、checkpoints/ | 4算法各1500次 |
| 主调度与检查点评估 | logs/main_seed0/evaluation/ | 7个真实检查点 |
| 六图 | figures/main_seed0/index.html | PNG/PDF/SVG |
| 稳定性 | logs/stability_summary.csv | 五种子均值与样本标准差 |
| 审计/验收 | evidence/delivery_audit.json、evidence/acceptance.json | 工程通过，科学未通过 |
| 差异报告 | reproduction_report.md | 逐图实际结果与限制 |

## 已确认决策
- 一套预登记配置，种子0—4；主图始终种子0，无图形拟合、无最优种子挑选。
- 确定性表II场景，初始SOC0.5，局部5维历史状态；策略潜均值经tanh映射。
- 图5/10共用原始episode sum；平滑25且不跨500/1000边界。
- 图6—8为1500 pre_fed；图9为1/50/500/700/900/1400，500 pre_fed。
- 完整训练共2,160,000条智能体转移；配置哈希在各日志metadata中，模型含优化器和随机数状态。

## 缺口与后续
- 论文未公开的奖励权重、电池容量/SOC/效率、网络、动作分布、采样与统计口径尚未取得。
- 图9 MG1第1400次失衡为{final9[final9.mg == 1].abs_deficit.iloc[0]:.3f}，论文约3000；逐图失败项见报告。
- 五种子F-MADRL训练排序胜出{int(wins.sum())}/15，不能复述为全面优于基线。
- 高斯误差的固定检查点评估已执行；高斯完整训练未执行；未开展额外参数搜索。
- 继续工作应先取得作者配置/代码/指标定义，或明确新的假设敏感性实验；不能修改表II或奖励去拟合图片。
'''
    (ROOT / 'reproduction_state.md').write_text(state, encoding='utf-8')
    print(json.dumps(acceptance, indent=2), flush=True)
