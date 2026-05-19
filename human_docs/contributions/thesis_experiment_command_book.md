# Thesis Experiment Command Book

## 作用
- 汇总当前项目为了形成论文主结果，还需要补做的实验。
- 按执行顺序给出一套可直接运行的命令清单。
- 把 CIFAR-10 `baseline_vgg16` 主线和 N-MNIST 支持性实验分开整理。

## 一、你现在还缺哪些实验

基于当前讨论，距离“论文主结果可写”还缺的实验主要有以下几类。

### A. `baseline_vgg16` 主结果线还缺的实验

1. **统一协议下的 HessianMixed 正式结果**
   - 使用 `fp32_best.pt`
   - full test evaluation
   - 更完整的 Hessian batches

2. **统一协议下的 StateAwareHessian 正式结果**
   - 不再使用 quick validation 风格的小样本测试
   - 和 HessianMixed 在同一协议下比较

3. **KD 主对照实验**
   - `HessianMixed`
   - `HessianMixed + KD`
   - 固定同一个 `bit_allocation.csv`

4. **KD 小规模超参数消融**
   - `alpha ∈ {0.3, 0.5, 0.7}`
   - `T ∈ {1, 2, 4}`

5. **统一主结果汇总**
   - 至少形成：
     - FP32
     - UniformW4
     - HessianMixed
     - StateAwareHessian
     - HessianMixed + KD

### B. `baseline_nmnist` 支持性实验还缺的实验

1. `baseline_nmnist` FP32 正式训练结果
2. `baseline_nmnist` UniformW4 / W8
3. `baseline_nmnist` HessianMixed
4. 可选：`baseline_nmnist` Hessian + KD

### C. 训练成本 / 部署成本分析实验

为了支撑蒸馏的论文表述，还建议补：

1. HessianMixed fine-tune 时间
2. HessianMixed + KD fine-tune 时间
3. 如果可能，记录训练峰值显存

## 二、推荐执行顺序

建议严格按下面顺序执行，不要交叉乱跑。

1. `baseline_vgg16` FP32 baseline 确认
2. `baseline_vgg16` uniform quant
3. `baseline_vgg16` HessianMixed 正式结果
4. `baseline_vgg16` StateAwareHessian 正式结果
5. `baseline_vgg16` KD 主对照实验
6. `baseline_vgg16` KD 超参数消融
7. `baseline_nmnist` supporting experiments
8. 最后统一做结果汇总与表格整理

---

## 三、CIFAR-10 / `baseline_vgg16` 主结果命令

## 3.1 FP32 baseline

```bash
python scripts/run_baseline_vgg16.py \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16 \
  --epochs 100 \
  --t-steps 16 \
  --batch-size-train 128 \
  --batch-size-test 256 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --seed 42 \
  --device cuda
```

说明：
- 正式主线建议使用 `fp32_best.pt` 作为后续 teacher 和量化初始化基线。

## 3.2 Uniform quantization

```bash
python scripts/run_uniform_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_uniform_quant \
  --bits 8,4 \
  --t-steps 16 \
  --batch-size-test 256 \
  --seed 42 \
  --device cuda
```

## 3.3 HessianMixed 正式结果

```bash
python scripts/run_hessian_sensitivity_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_hessian_sensitivity \
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

## 3.4 StateAwareHessian 正式结果

```bash
python scripts/run_state_aware_hessian_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_state_aware_hessian \
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
  --state-aware-alpha 0.75 \
  --seed 42 \
  --device cuda
```

## 3.5 KD 主对照实验

前提：
- 已经有 `outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv`

### 3.5.1 Hessian + KD（主配置）

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv \
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

### 3.5.2 HessianMixed 对照

如果你想让 KD 主对照更干净，建议再单独重跑一遍 HessianMixed，并使用与 KD 相同的：
- `fp32_best.pt`
- `bit_allocation.csv`
- `quant_epochs`
- `batch-size-test`

当前脚本没有“复用 allocation CSV”的单独入口，所以最稳的办法是保持 `run_hessian_sensitivity_vgg16.py` 使用同样的正式配置，并把该结果作为 KD 主对照。

## 3.6 KD 超参数消融

下面这些命令建议按需跑，不一定都进入主表，但至少要形成小规模 ablation。

### `alpha = 0.3, T = 2`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a03_t2 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv \
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

### `alpha = 0.7, T = 2`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a07_t2 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv \
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

### `alpha = 0.5, T = 1`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a05_t1 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv \
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

### `alpha = 0.5, T = 4`

```bash
python scripts/run_distill_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_distill_quant_a05_t4 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv \
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

## 3.7 Weight + State mixed precision（探索性）

```bash
python scripts/run_weight_state_mixed_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_best.pt \
  --data-root baseline/data \
  --output-dir outputs/baseline_vgg16_weight_state_mixed \
  --weight-bits 8,4 \
  --state-bits 8,6 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-hessian-batches 5 \
  --max-train-batches 0 \
  --max-test-batches 0 \
  --trace-probes 1 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --seed 42 \
  --device cuda
```

说明：
- 这条线建议放在附加实验或 future-work 风格的位置，不建议在主结果里和 KD 同等级主打。

---

## 四、N-MNIST supporting experiments

## 4.1 FP32 baseline

```bash
python scripts/run_baseline_nmnist.py \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist \
  --epochs 20 \
  --t-steps 16 \
  --batch-size-train 64 \
  --batch-size-test 128 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --seed 42 \
  --device cuda
```

## 4.2 Uniform quantization

```bash
python scripts/run_uniform_quant_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_uniform_quant \
  --bits 8,4 \
  --t-steps 16 \
  --batch-size-test 128 \
  --seed 42 \
  --device cuda
```

## 4.3 HessianMixed

```bash
python scripts/run_hessian_sensitivity_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_hessian_sensitivity \
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

## 4.4 StateAwareHessian（可选）

```bash
python scripts/run_state_aware_hessian_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_state_aware_hessian \
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
  --state-aware-alpha 0.75 \
  --seed 42 \
  --device cuda
```

## 4.5 KD（可选 supporting experiment）

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
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0 \
  --seed 42 \
  --device cuda
```

---

## 五、训练成本记录建议

如果你要在论文里写 KD 增加训练成本、但不增加部署成本，建议至少补以下运行记录：

1. `run_hessian_sensitivity_vgg16.py` 正式运行时间
2. `run_distill_quant_vgg16.py` 正式运行时间
3. 若可行，记录训练峰值显存

最简单做法：

```bash
time python scripts/run_hessian_sensitivity_vgg16.py ...
time python scripts/run_distill_quant_vgg16.py ...
```

---

## 六、最终主表建议使用哪些结果

建议最终主表优先放：

1. FP32
2. UniformW4
3. HessianMixed
4. StateAwareHessian
5. HessianMixed + KD

`WeightStateMixed` 建议放到：

- 附录
- 补充实验
- 或未来工作讨论部分

## 七、一句话总结

如果你现在想最快形成论文主结果，最关键要跑完的是：

1. `baseline_vgg16` HessianMixed 正式结果
2. `baseline_vgg16` StateAwareHessian 正式结果
3. `baseline_vgg16` Hessian + KD 主对照实验
4. `baseline_vgg16` KD 小规模超参数消融

N-MNIST 则作为支持性验证主线，用来增强 SNN-specific 的说服力。
