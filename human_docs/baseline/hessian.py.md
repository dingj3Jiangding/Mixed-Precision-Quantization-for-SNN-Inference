# 文件：`baseline/hessian.py`

## 作用
- 实现 Hessian 敏感度分析主流程（采用梯度平方近似 Hessian 对角/Fisher 风格估计）。
- 生成层级敏感度排序，并在给定平均 bit 预算下做 layer-wise mixed-precision 分配。
- 对比 `FP32`、`Uniform`、`HessianMixed` 三种设置并导出结果。

## 如何使用
- 主函数：
  - `run_hessian_sensitivity_analysis(cfg, checkpoint_path, bits_list, target_avg_bits, output_dir)`
- 必要输入：
  - `checkpoint_path`：已训练 FP32 权重（例如 `outputs/baseline/fp32_last.pt`）
  - `bits_list`：可选位宽（例如 `[8, 4, 2]`）
  - `target_avg_bits`：目标平均位宽（例如 `4.0`）

## 输出文件
- `layer_sensitivity.csv`：每层敏感度、密度、排名、分配位宽
- `bit_allocation.csv`：最终 layer-wise 位宽分配表
- `comparison.csv`：`FP32 / Uniform / HessianMixed` 指标对比
- `summary.json`：本次运行摘要
- `sensitivity_ranking.png`：敏感度排名图（若 matplotlib 可用）
