# 最新训练结果

[打开最新数据与图5—10](latest_results/index.html) · [最新结果报告](latest_results/reproduction_report.md) · [六图PDF合辑](latest_results/figures_05_10.pdf)

本轮采用反推候选 `w_C=0.02、w_de=0.45、电池100 kWh、SOC0=0.5`，已完成四算法、五种子、各1500次完整训练。主图固定种子0，按照论文的布局、字体和配色输出PNG/PDF/SVG，图片保留最新真实训练结果。

- 原始数据：`logs/inferred_20260921_seed0/`至`logs/inferred_20260921_seed4/`。
- 模型：`checkpoints/inferred_20260921_seed0/`至`checkpoints/inferred_20260921_seed4/`。
- 最新六图：`figures/inferred_20260921_seed0/`。
- 配置：[inferred_20260921.yaml](configs/inferred_20260921.yaml)。
- 五种子统计：[stability_summary.csv](latest_results/stability_summary.csv)。

样式对齐和工程检查通过不等于论文数值复现成功，实际差异已在本轮报告中保留。根目录原README及原复现报告对应旧轮，继续保留作为历史记录。

仅重绘最新六图：

```powershell
& 'F:\Blackma\Anaconda\envs\pytorch_env\python.exe' -X utf8 scripts/plot_paper_style.py --run-id inferred_20260921_seed0
```
