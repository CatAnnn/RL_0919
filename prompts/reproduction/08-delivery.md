# 八、项目交付与验收

> 按需模块：按当前任务范围检查交付物和证据；读取本模块本身不触发完整训练。
> 与根目录 `prompt.md` 的全程约束一起使用。

交付完整、可运行项目，至少包括：

configs/
data/
envs/
agents/
federated/
tests/
checkpoints/
logs/
figures/

train.py
evaluate.py
plot_figures.py
requirements.txt
README.md
paper_spec.md
assumptions.yaml
discrepancy_register.md
reproduction_report.md

输出：
fig05.png/pdf/svg
fig06.png/pdf/svg
fig07.png/pdf/svg
fig08.png/pdf/svg
fig09.png/pdf/svg
fig10.png/pdf/svg

图内使用与论文对应的英文标签和(a)(b)(c)编号。
图5/9/10为三个纵向面板；
图6/7/8为两个面板，调度上面板包含双纵轴。

绘图程序只能读取已经保存的CSV/JSON。
不得在绘图脚本中填入手工曲线或用随机函数制造结果。

每张图关联：
run_id、seed、config_hash、数据来源、checkpoint。

先运行短程冒烟测试，再进行完整训练。
至少验证：

1. 表I/II数据长度、关键数值与单位；
2. 动作区间、充放电符号和SOC连续性；
3. P_de = -U_t；
4. 奖励与成本/失衡分解恒等式；
5. 交易量不超过供需能力；
6. 联邦平均与广播后参数一致；
7. PPO old policy、GAE与终止掩码；
8. 图5与图10的F-MADRL数据逐条相同。

提供完整命令：
环境创建 → 安装依赖 → 单元测试 → 短程训练 →
完整训练 → 检查点评估 → 仅用日志重新绘制六图。

支持从检查点恢复。
调整图形样式不能要求重复训练。

reproduction_report.md逐图说明：
原文结论、实际结果、数值差异、
未明确的设置、采用的解释、失败项和可能原因。

原图数字化读数仅可作为reference_digitized参考数据，
不得冒充原始训练日志。

严格禁止：
用指数函数加噪声伪造收敛曲线；
在500/1000处手动制造奖励跃迁；
强制F-MADRL优于所有基线；
为了匹配图形修改输入数据；
将“看起来像”当作成功复现。

如果资源不足或尚未执行完整训练，
明确区分：
已写代码、已运行测试、已完成训练、未执行实验。
