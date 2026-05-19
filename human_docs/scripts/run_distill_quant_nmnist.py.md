# 文件：`scripts/run_distill_quant_nmnist.py`

## 作用
- 调用 `baseline_nmnist/distill_quant.py`，在 `baseline_nmnist` 上运行蒸馏辅助 mixed-precision 量化。
- 默认流程是：
  1. 读取 FP32 checkpoint
  2. 读取或计算 Hessian bit allocation
  3. 用 FP32 teacher 对量化 student 做 KD fine-tuning
  4. 输出 FP32 / Uniform / HessianKD 对比

## 如何运行
```bash
python scripts/run_distill_quant_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_last.pt \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_nmnist_hessian_sensitivity/bit_allocation.csv \
  --device cuda \
  --batch-size-train 8 \
  --batch-size-test 16 \
  --max-test-batches 10 \
  --quant-epochs 5 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0
```

## 推荐高精度命令
下面这条更适合作为 N-MNIST 上蒸馏辅助量化的正式实验配置。它是**偏向高精度**的推荐命令，不等于已经验证过的全局最优：

```bash
python scripts/run_distill_quant_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_distill_quant \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_nmnist_hessian_sensitivity/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 10 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0 \
  --seed 42 \
  --device cuda
```

## 关键参数
- `--weight-allocation-csv`：复用已有 Hessian allocation，避免重复做高显存 Hessian 估计。
- `--distill-alpha`：KD loss 占比，`0` 表示只用 CE，`1` 表示只用 KD。
- `--distill-temperature`：teacher-student soft target 温度。
- `--quant-epochs`：蒸馏 fine-tuning 轮数。

## 输出
- mixed-precision allocation
- epoch-level KD fine-tuning metrics
- FP32 / Uniform / HessianKD comparison
