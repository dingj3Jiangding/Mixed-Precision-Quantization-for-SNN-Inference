# 文件：`baseline_vgg16/distill_quant.py`

## 作用
- 在 `baseline_vgg16` 上实现一条新的量化验证路径：先做 Hessian-guided layer-wise bit allocation，再用 FP32 teacher 对量化 student 做 logits distillation fine-tuning。
- 目标不是替换原有 `hessian.py`，而是在相同 bit allocation 基础上测试蒸馏是否能减少量化精度损失。

## 方法说明
- teacher：由 `fp32_last.pt` 加载得到的 FP32 模型，训练时冻结。
- student：从同一 checkpoint 初始化，再按 layer-wise bit allocation 在 forward 前量化权重。
- loss：
  - hard label cross-entropy
  - teacher-student KL distillation loss
  - 总损失为 `(1 - alpha) * CE + alpha * KD`

当前版本只做 **logits distillation**，还没有加入 spike-level 或 membrane/state trajectory distillation。

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

## 输出文件
- `outputs/baseline_vgg16_distill_quant/layer_sensitivity.csv`
- `outputs/baseline_vgg16_distill_quant/bit_allocation.csv`
- `outputs/baseline_vgg16_distill_quant/comparison.csv`
- `outputs/baseline_vgg16_distill_quant/distill_quant_epoch_metrics.csv`
- `outputs/baseline_vgg16_distill_quant/summary.json`
- `outputs/baseline_vgg16_distill_quant/sensitivity_ranking.png`
