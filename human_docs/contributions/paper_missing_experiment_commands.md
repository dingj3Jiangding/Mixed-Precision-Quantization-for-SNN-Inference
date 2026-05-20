# Paper Missing Experiment Commands

## 作用

- 汇总当前论文还建议补做的实验命令。
- 只保留与论文主结果直接相关的实验。
- 默认围绕 `baseline_vgg16` 主线组织。

---

## 一、实验优先级

建议按下面顺序执行：

1. CIFAR-10 `HessianMixed` 严格主对照重跑
2. CIFAR-10 `HessianMixed + KD` 严格主对照重跑
3. CIFAR-10 KD 小规模超参数消融
4. 训练成本记录
5. 可选：N-MNIST 的严格 KD supporting 对照

---

## 二、CIFAR-10 主线正式补实验

## 2.1 HessianMixed 严格主对照

目的：

- 作为 KD 主对照
- 与 `HessianMixed + KD` 保持完全一致协议

建议命令：

```bash
python scripts/run_hessian_sensitivity_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_hessian_sensitivity_formal \
  --bits 8,4 \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-hessian-batches 10 \
  --max-test-batches 0 \
  --trace-probes 1 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --allocation-policy rank-map \
  --seed 42 \
  --device cuda
```

输出重点：

- `outputs/baseline_vgg16_hessian_sensitivity_formal/comparison.csv`
- `outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv`

---

## 2.2 HessianMixed + KD 严格主对照

目的：

- 固定同一份 Hessian allocation
- 固定同一 teacher / student 初始化
- 形成论文主表最关键对照

建议命令：

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_formal \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0 \
  --seed 42 \
  --device cuda
```

输出重点：

- `outputs/baseline_vgg16_distill_quant_formal/comparison.csv`
- `outputs/baseline_vgg16_distill_quant_formal/distill_quant_epoch_metrics.csv`

---

## 三、CIFAR-10 KD 小规模超参数消融

## 3.1 说明

这里的目的不是做大规模调参，而是证明：

- KD 的提升不是偶然超参数命中
- `alpha = 0.5, T = 2.0` 是一个合理选择

建议最小集合：

1. `alpha = 0.3, T = 2`
2. `alpha = 0.5, T = 1`
3. `alpha = 0.5, T = 4`
4. `alpha = 0.7, T = 2`

这些命令都默认复用：

- `outputs/baseline_vgg16/fp32_best.pt`
- `outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv`

---

## 3.2 `alpha = 0.3, T = 2`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a03_t2 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.3 \
  --distill-temperature 2.0 \
  --seed 42 \
  --device cuda
```

## 3.3 `alpha = 0.5, T = 1`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a05_t1 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 1.0 \
  --seed 42 \
  --device cuda
```

## 3.4 `alpha = 0.5, T = 4`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a05_t4 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 4.0 \
  --seed 42 \
  --device cuda
```

## 3.5 `alpha = 0.7, T = 2`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a07_t2 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.7 \
  --distill-temperature 2.0 \
  --seed 42 \
  --device cuda
```

---

## 四、训练成本记录

## 4.1 说明

论文里如果要写：

- KD 增加训练成本
- 但不增加部署期结构成本

那么建议至少记录：

- HessianMixed fine-tuning 总耗时
- HessianMixed + KD fine-tuning 总耗时

最简单的做法是使用 shell 的 `time`。

---

## 4.2 记录 HessianMixed 训练时间

```bash
/usr/bin/time -p python scripts/run_hessian_sensitivity_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_hessian_sensitivity_timed \
  --bits 8,4 \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-hessian-batches 10 \
  --max-test-batches 0 \
  --trace-probes 1 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --allocation-policy rank-map \
  --seed 42 \
  --device cuda
```

记录：

- `real`
- `user`
- `sys`

建议手动抄进表格或实验日志。

---

## 4.3 记录 HessianMixed + KD 训练时间

```bash
/usr/bin/time -p python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_timed \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0 \
  --seed 42 \
  --device cuda
```

论文里可以比较：

- HessianMixed total fine-tuning time
- HessianMixed + KD total fine-tuning time

---

## 五、可选：N-MNIST supporting KD 严格对照

优先级低于 CIFAR-10 主线。

## 5.1 N-MNIST HessianMixed

```bash
python scripts/run_hessian_sensitivity_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_hessian_sensitivity_formal \
  --bits 8,4 \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-hessian-batches 10 \
  --max-test-batches 0 \
  --trace-probes 1 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --allocation-policy rank-map \
  --seed 42 \
  --device cuda
```

## 5.2 N-MNIST HessianMixed + KD

```bash
python scripts/run_distill_quant_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_distill_quant_formal \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_nmnist_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0 \
  --seed 42 \
  --device cuda
```

---

## 六、跑完后你至少应该整理出的表

### 表 1：CIFAR-10 主结果表

- FP32
- UniformW4
- HessianMixed
- StateAwareHessian
- HessianMixed + KD

### 表 2：KD 严格主对照表

- HessianMixed
- HessianMixed + KD

### 表 3：KD 超参数消融表

- `alpha / T`
- `test_acc`
- `avg_weight_bits`
- `avg_batch_infer_ms`
- `sop_proxy`

### 表 4：训练成本表

- HessianMixed total time
- HessianMixed + KD total time

---

## 七、一句话总结

如果只按论文当前最需要补的实验来跑，最关键的就是：

1. `HessianMixed` 严格主对照
2. `HessianMixed + KD` 严格主对照
3. KD 小规模超参数消融
4. KD 训练成本记录

这四项补完，论文的实验可信度会明显提高。
