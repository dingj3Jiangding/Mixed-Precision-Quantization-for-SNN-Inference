import time
import torch

@torch.no_grad()        # turn off gradient calculation
def eval_fp32_snn(model, test_loader, T=16, encoder="direct", device="cuda"):
    # encoder = "direct".  这对应 encoder="direct"：direct/constant 编码 
    # 含义：同一张静态图片在每个时间步都输入同样的模拟值（不做 Poisson 随机采样）。
    model.eval()
    model = model.to(device).float()

    correct, total = 0, 0
    t_sum = 0.0

    for x, y in test_loader:
        x = x.to(device).float()
        y = y.to(device)                # to.(device) put computing task on specific device

        # 1) reset SNN states (必须)
        
        ########
        # SNN 与 ANN 的最大不同之一：神经元有状态（例如膜电位 V、refractory 等）。
        # 如果不 reset，上一批数据的膜电位会“残留”到下一批，导致：
        # accuracy 不稳定/不可信
        # 结果不可复现
        # 所以每个 batch 必须清空状态。
        #########

        if hasattr(model, "reset"):
            model.reset()
        else:
            raise RuntimeError("Model must implement reset() for SNN state reset.")

        # 2) build time sequence [T,B,C,H,W]

        ##########
        # 原始 x 是 [B, C, H, W]
        # unsqueeze(0) 变成 [1, B, C, H, W]
        # repeat(T, ...) 复制 T 次，得到 [T, B, C, H, W]
        # 这对应 encoder="direct"：direct/constant 编码
        # 含义：同一张静态图片在每个时间步都输入同样的模拟值（不做 Poisson 随机采样）。
        ##########
        if encoder == "direct":
            x_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        else:
            raise NotImplementedError("Implement poisson/rate encoder if needed.")

        # 3) forward
        t0 = time.perf_counter()                # 高精度计时
        logits_seq = model(x_seq)               # expect [T,B,10] -- 让模型跑 T steps
        t_sum += time.perf_counter() - t0       # 累加所有 batch 的推理时间
        # 4) aggregate over time
        logits = logits_seq.mean(dim=0)  # [B,10] 取 T steps 的avg
        pred = logits.argmax(dim=1) #在index-1（0-index）维度上取最大值的索引 -- 获得每个batch 最大值的索引

        correct += (pred == y).sum().item()
        total += y.numel()

    acc = correct / total
    avg_batch_time = t_sum / len(test_loader)
    return acc, avg_batch_time
