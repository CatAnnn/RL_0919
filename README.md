# F-MADRL 图5—10可追溯重建

论文：*Federated Multiagent Deep Reinforcement Learning Approach via Physics-Informed Reward for Multimicrogrid Energy Management*。

本项目按照公开表格、公式和算法实现真实实验；作者未公开参数集中声明为工程假设。是否达到论文数值结果，以 [复现报告](reproduction_report.md) 和 [验收JSON](evidence/acceptance.json) 为准，不能从图形文件存在推断复现成功。

## 查看交付

- [六图浏览入口](figures/main_seed0/index.html)：PNG、PDF、SVG。
- [规格](paper_spec.md)、[假设](assumptions.yaml)、[公式歧义](discrepancy_register.md)。
- `logs/main_seed0/F-MADRL/history.csv`：图5/10共用的原始奖励。
- `logs/main_seed0/evaluation/schedule.csv`：图6—8的同一检查点评估。
- `logs/main_seed0/evaluation/checkpoint_metrics.csv`：图9；`checkpoint_details.csv`保留逐小时指标。
- `logs/stability_per_seed.csv`、`logs/stability_summary.csv`：五种子稳定性。
- `checkpoints/main_seed*/`：模型、优化器、随机数状态、配置与历史。
- `evidence/`：单元/冒烟测试、完整执行、哈希、真实性审计。

## 环境和依赖

在仓库根目录PowerShell执行。所有Python操作必须使用 `pytorch_env`。

```powershell
# 仅在该环境尚不存在时创建，已有环境不重复创建。
conda create -n pytorch_env python=3.10 -y
conda activate pytorch_env
python -c "import sys,torch; print(sys.executable); print(torch.__version__)"
python -m pip install -r requirements.txt
```

本轮已验证的解释器为 `F:\Blackma\Anaconda\envs\pytorch_env\python.exe`，没有安装或更改已有依赖。Conda包装命令曾长时间停留，实际执行用该环境的解释器绝对路径；这不改变环境。单元测试使用标准库unittest，无需pytest。以下命令中的python可统一替换为PowerShell形式 `& 'F:\Blackma\Anaconda\envs\pytorch_env\python.exe'`。

## 测试、冒烟和完整训练

```powershell
python -X utf8 scripts/run_tests.py
python -X utf8 train.py --run-id smoke_seed0 --iterations 8
python -X utf8 scripts/smoke_audit.py
python -X utf8 scripts/run_registered.py
```

`run_registered.py`先检查冒烟通过，再以最多3个独立进程运行种子0—4、各四算法1500次。此命令用于全新输出目录；已存在history的run_id会拒绝覆盖。现有交付无需重新训练。若需要单独复跑，可使用新名称：

```powershell
python -X utf8 train.py --run-id rerun_seed0 --seed 0
python -X utf8 evaluate.py --run-id rerun_seed0
python -X utf8 plot_figures.py --run-id rerun_seed0
```

主配置是 `configs/main.yaml`。`configs/strict.yaml`中的null未知项会触发错误，不能静默补为“论文参数”。每次迭代是每个MG一条24步本地轨迹，四算法采样量一致；PPO四次actor更新、A2C一次、TRPO一次约束更新尝试，critic均四次，实际成功次数见completion与history。

## 检查点评估与仅重绘

```powershell
python -X utf8 evaluate.py --run-id main_seed0
python -X utf8 evaluate.py --run-id main_seed0 --scenario gaussian_forecast_errors
python -X utf8 plot_figures.py --run-id main_seed0
python -X utf8 scripts/audit_delivery.py
python -X utf8 scripts/report_results.py
```

绘图只读取保存的CSV/JSON；样式变化不需要训练。图5/10各保存一份同源表用于逐字节验证。图6—8使用第1500次pre_fed，图9的500也使用pre_fed；500/1000/1500均有pre/post两份。评估策略为潜高斯均值经tanh映射，不等同有界随机动作的数学期望。

图9按全天sum abs(Pde)计算失衡，原图标kW，本实现同时另存kWh。表II价格保留原文数值和印刷单位，不除1000。主要时序、终止bootstrap、奖励与交易口径见假设文件。

## 恢复与高斯场景入口

```powershell
# 同一run_id、配置、算法与种子；恢复后重建该检查点之后的日志。
python -X utf8 train.py --run-id main_seed0 --seed 0 --algorithm F-MADRL --resume checkpoints/main_seed0/F-MADRL/iter_1000_post_fed.pt

# 可选对照实验：本轮没有执行高斯场景的完整重训。
python -X utf8 train.py --run-id gaussian_seed0 --seed 0 --scenario gaussian_forecast_errors
python -X utf8 evaluate.py --run-id gaussian_seed0
python -X utf8 plot_figures.py --run-id gaussian_seed0
```

恢复会重新计算后续轨迹，适合中断任务；不应只为查看结果而运行。边界pre_fed恢复时先执行应有聚合，post_fed恢复不重复聚合。固定场景每次迭代使用可重建的外生种子，Torch/NumPy/Python随机数状态与优化器状态一起保存，已用短程逐参数/逐日志精确一致测试验证。

## 项目目录

`envs/`物理模型与交易；`agents/`PPO/A2C/TRPO；`federated/`等权聚合；`tests/`单元测试；`configs/`配置；`data/`表I/II；`reference_digitized/`仅训练后论文读数；`scripts/`运行、恢复测试、审计与报告。原始PDF、prompt.md及prompts/未修改。
