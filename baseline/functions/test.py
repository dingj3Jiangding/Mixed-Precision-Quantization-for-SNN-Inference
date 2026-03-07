import time
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

def quick_verify(base_model: nn.Module, test_loader, T=16, device="cuda"):
    base_model = base_model.to(device).float().eval()

    wrapped = SNNTimeWrapper(base_model).to(device).float().eval()

    # 取一个 batch 做形状验证
    x, y = next(iter(test_loader))
    x = x.to(device).float()
    y = y.to(device)
    B = x.shape[0]

    x_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [T,B,C,H,W]

    # (A) shape check
    wrapped.reset()
    out = wrapped(x_seq)
    print("[Shape Check] out.shape =", tuple(out.shape), " (expect:", (T, B, 10), "or", (T, B, "num_classes"), ")")

    # (B) reset correctness check
    wrapped.reset()
    pred1 = out.mean(0).argmax(1)

    wrapped.reset()
    out2 = wrapped(x_seq)
    pred2 = out2.mean(0).argmax(1)

    same_with_reset = (pred1 == pred2).float().mean().item()
    print(f"[Reset Check] same predictions ratio WITH reset = {same_with_reset:.4f} (expect ~1.0)")

    # 可选：不 reset 再跑一次，通常会不同（不保证一定不同，但若不同说明 reset 必须）
    out3 = wrapped(x_seq)
    pred3 = out3.mean(0).argmax(1)
    same_without_reset = (pred1 == pred3).float().mean().item()
    print(f"[Reset Check] same predictions ratio WITHOUT reset = {same_without_reset:.4f} (often < 1.0)")

    # (C) run eval on a few batches
    acc, avg_t = eval_fp32_snn(wrapped, test_loader, T=T, encoder="direct", device=device, max_batches=5)
    print(f"[Eval Check] acc={acc:.4f}, avg_batch_time={avg_t:.4f}s (over 5 batches)")

    return wrapped


# ---------- 4) 使用方式（把这里换成你的模型） ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 这里换成你的 SpikingJelly ResNet-19 SNN（单步输入 [B,C,H,W] -> 输出 [B,10]）
    base_model = ResNet19SNN(...)
    # 可选：加载权重
    # ckpt = torch.load("fp32_snn.pth", map_location="cpu")
    # base_model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    # 注意：你需要已经准备好 test_loader
    wrapped_model = quick_verify(base_model, test_loader, T=16, device=device)