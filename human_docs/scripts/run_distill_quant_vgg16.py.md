# 文件：`scripts/run_distill_quant_vgg16.py`

## 作用
- 调用 `baseline_vgg16/distill_quant.py`，在 `baseline_vgg16` 上运行蒸馏辅助 mixed-precision 量化。
- 默认流程是：
  1. 读取 FP32 checkpoint
  2. 读取或计算 Hessian bit allocation
  3. 用 FP32 teacher 对量化 student 做 KD fine-tuning
  4. 输出 FP32 / Uniform / HessianKD 对比

## 如何运行
```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_last.pt \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv \
  --device cuda \
  --batch-size-train 8 \
  --batch-size-test 16 \
  --max-test-batches 10 \
  --quant-epochs 5 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0
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
