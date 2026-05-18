# 云服务器部署指南

本文档用于把当前 SNN mixed-precision quantization 项目部署到云服务器，并运行 baseline / uniform quantization / Hessian sensitivity 实验。

---

## 1. 推荐云服务器配置

当前项目主要依赖 GPU 跑 PyTorch + SpikingJelly。

推荐配置：

- GPU：RTX 3080 Ti / 3090 / 4090 / A5000 均可
- 显存：至少 12GB，24GB 更舒服
- CPU：8 核以上
- 内存：32GB 以上，64GB 更稳
- 系统盘：100GB SSD 以上
- 系统：Ubuntu 20.04 / Ubuntu 22.04
- 框架镜像：PyTorch 2.x + CUDA 11.8 或 CUDA 12.8

如果平台让你选择框架：

```text
框架名称：PyTorch
框架版本：PyTorch 2.x + CUDA 11.8 / 12.8
Python：3.9
```

---

## 2. 登录服务器

一般云平台会提供 SSH 命令，例如：

```bash
ssh root@服务器IP
```

如果有端口：

```bash
ssh -p 端口号 root@服务器IP
```

登录后先检查 GPU：

```bash
nvidia-smi
```

如果能看到 GPU 型号、显存和 CUDA driver，说明 GPU 可用。

---

## 3. 上传项目代码

如果仓库是私有仓库，建议先配置 Git SSH 私钥；如果你用 `scp` 上传代码，可以跳过本节。

### 3.1 配置 Git SSH 私钥

在服务器上生成 SSH key：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

一路回车即可。默认会生成：

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

启动 ssh-agent 并添加私钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

查看公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出内容，然后添加到 GitHub / GitLab：

```text
GitHub:
Settings -> SSH and GPG keys -> New SSH key

GitLab:
Preferences -> SSH Keys -> Add new key
```

测试连接：

```bash
ssh -T git@github.com
```

如果看到类似：

```text
Hi 用户名! You've successfully authenticated...
```

说明 SSH key 配置成功。

如果你用 GitLab：

```bash
ssh -T git@gitlab.com
```

首次连接时可能会问：

```text
Are you sure you want to continue connecting?
```

输入：

```bash
yes
```

### 3.2 使用已有私钥

如果你已经有本地私钥，也可以把私钥复制到服务器：

```bash
mkdir -p ~/.ssh
nano ~/.ssh/id_ed25519
```

把私钥内容粘贴进去后保存，然后设置权限：

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

注意：私钥内容应类似：

```text
-----BEGIN OPENSSH PRIVATE KEY-----
...
-----END OPENSSH PRIVATE KEY-----
```

不要把私钥提交到 Git 仓库，也不要发给别人。

### 3.3 配置 Git 用户信息

如果后面要在服务器上 commit：

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

### 方式 A：用 Git

如果你的项目已经上传到 GitHub / GitLab：

```bash
git clone 仓库地址
cd Mixed-Precision-Quantization-for-SNN-Inference
```

### 方式 B：用 scp 从本地上传

在本地终端执行：

```bash
scp -r Mixed-Precision-Quantization-for-SNN-Inference root@服务器IP:/root/
```

然后在服务器上进入项目：

```bash
cd /root/Mixed-Precision-Quantization-for-SNN-Inference
```

### 方式 C：压缩后上传

本地压缩：

```bash
zip -r snn_project.zip Mixed-Precision-Quantization-for-SNN-Inference
```

上传：

```bash
scp snn_project.zip root@服务器IP:/root/
```

服务器解压：

```bash
cd /root
unzip snn_project.zip
cd Mixed-Precision-Quantization-for-SNN-Inference
```

---

## 4. 创建 Python 环境

如果服务器已有 PyTorch 镜像，可以先检查：

```bash
python -V
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

如果需要自己建 conda 环境：

```bash
conda create -n spiking_jelly python=3.9 -y
conda activate spiking_jelly
```

安装 PyTorch。保持 `Python 3.9` 不变，只根据显卡切换 CUDA wheel：

如果是较新的 RTX 5090 / Blackwell 卡，建议使用 CUDA 12.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果是旧一些的 CUDA 11.8 环境，仍然可以使用：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果云镜像已经装好 PyTorch，可以跳过这一步。

安装完成后建议立即确认：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

---

## 5. 安装项目依赖

进入项目根目录后安装依赖：

```bash
pip install spikingjelly matplotlib pandas numpy tqdm
```

检查关键依赖：

```bash
python -c "import torch, torchvision, spikingjelly; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda', torch.version.cuda); print('spikingjelly ok')"
```

如果输出 `cuda available True`，说明 PyTorch 能使用 GPU。

---

## 6. 准备数据和 checkpoint

项目默认使用 CIFAR-10。

如果服务器没有数据，可以让脚本自动下载：

```bash
python scripts/run_baseline.py --download --device cuda --max-train-batches 1 --max-test-batches 1
```

如果你已经有训练好的 checkpoint，需要确认文件存在：

```bash
ls outputs/baseline/fp32_last.pt
```

如果没有 checkpoint，需要先训练 baseline：

```bash
python scripts/run_baseline.py --epochs 15 --device cuda --download
```

训练完成后应生成：

```text
outputs/baseline/fp32_last.pt
outputs/baseline/epoch_metrics.csv
outputs/baseline/summary.json
```

---

## 7. 先跑 Smoke Test

正式跑 Hessian 前，先用很小 batch 数确认流程没问题。

```bash
python scripts/run_hessian_sensitivity.py \
  --checkpoint-path outputs/baseline/fp32_last.pt \
  --bits 8,4,2 \
  --device cuda \
  --trace-probes 1 \
  --max-hessian-batches 1 \
  --max-train-batches 1 \
  --max-test-batches 1 \
  --quant-epochs 1
```

成功后应生成：

```text
outputs/hessian_sensitivity/layer_sensitivity.csv
outputs/hessian_sensitivity/bit_allocation.csv
outputs/hessian_sensitivity/comparison.csv
outputs/hessian_sensitivity/quant_finetune_epoch_metrics.csv
outputs/hessian_sensitivity/summary.json
```

---

## 8. 正式运行实验

### 8.1 FP32 baseline

```bash
python scripts/run_baseline.py \
  --epochs 15 \
  --device cuda \
  --download
```

### 8.2 Uniform quantization 对照实验

```bash
python scripts/run_uniform_quant.py \
  --checkpoint-path outputs/baseline/fp32_last.pt \
  --bits 8,4,2 \
  --device cuda
```

输出：

```text
outputs/uniform_quant/uniform_comparison.csv
outputs/uniform_quant/summary.json
```

### 8.3 Hessian trace mixed precision

建议先不要把 `trace-probes` 设太大，因为 Hessian-vector product 很慢。

中等规模运行：

```bash
python scripts/run_hessian_sensitivity.py \
  --checkpoint-path outputs/baseline/fp32_last.pt \
  --bits 8,4,2 \
  --device cuda \
  --trace-probes 1 \
  --max-hessian-batches 10 \
  --quant-epochs 1 \
  --allocation-policy rank-map
```

更完整运行：

```bash
python scripts/run_hessian_sensitivity.py \
  --checkpoint-path outputs/baseline/fp32_last.pt \
  --bits 8,4,2 \
  --device cuda \
  --trace-probes 2 \
  --max-hessian-batches 20 \
  --quant-epochs 3 \
  --allocation-policy rank-map
```

---

## 9. 查看结果

Hessian 主要看：

```text
outputs/hessian_sensitivity/layer_sensitivity.csv
outputs/hessian_sensitivity/bit_allocation.csv
outputs/hessian_sensitivity/comparison.csv
outputs/hessian_sensitivity/quant_finetune_epoch_metrics.csv
outputs/hessian_sensitivity/summary.json
outputs/hessian_sensitivity/sensitivity_ranking.png
```

`comparison.csv` 里看三组结果：

```text
FP32
UniformW...
HessianMixed
```

`HessianMixed` 是 mixed-precision fine-tuning 后的结果。

---

## 10. 下载结果到本地

在本地终端执行：

```bash
scp -r root@服务器IP:/root/Mixed-Precision-Quantization-for-SNN-Inference/outputs ./outputs_from_server
```

如果 SSH 有端口：

```bash
scp -P 端口号 -r root@服务器IP:/root/Mixed-Precision-Quantization-for-SNN-Inference/outputs ./outputs_from_server
```

---

## 11. 常见问题

### 11.1 `torch.cuda.is_available()` 是 False

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 正常但 PyTorch CUDA 不可用，通常是装了 CPU 版 PyTorch，需要重新安装 CUDA 版 PyTorch。

### 11.2 `No module named 'spikingjelly'`

安装：

```bash
pip install spikingjelly
```

### 11.3 Hessian 跑得很慢

优先降低：

```bash
--max-hessian-batches 5
--trace-probes 1
--batch-size-train 64
```

不要一开始就跑完整训练集 Hessian trace。

### 11.4 CUDA deterministic warning

如果看到类似：

```text
CUBLAS_WORKSPACE_CONFIG
```

这通常是 warning，不是 error。可以忽略，或者运行前设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

也可以关闭 deterministic：

```bash
python scripts/run_hessian_sensitivity.py ... --no-deterministic
```

### 11.5 显存不够

降低 batch size：

```bash
--batch-size-train 64
--batch-size-test 128
```

Hessian trace 仍然慢或爆显存时，继续降低：

```bash
--max-hessian-batches 3
--trace-probes 1
```

---

## 12. 推荐首次完整流程

```bash
cd /root/Mixed-Precision-Quantization-for-SNN-Inference

python scripts/run_baseline.py \
  --epochs 15 \
  --device cuda \
  --download

python scripts/run_uniform_quant.py \
  --checkpoint-path outputs/baseline/fp32_last.pt \
  --bits 8,4,2 \
  --device cuda

python scripts/run_hessian_sensitivity.py \
  --checkpoint-path outputs/baseline/fp32_last.pt \
  --bits 8,4,2 \
  --device cuda \
  --trace-probes 1 \
  --max-hessian-batches 10 \
  --quant-epochs 1 \
  --allocation-policy rank-map
```
