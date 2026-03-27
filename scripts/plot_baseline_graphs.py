# save as: scripts/plot_baseline_graphs.py
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path("outputs/baseline/epoch_metrics.csv")
    out_dir = Path("outputs/baseline/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 1) Train/Test Accuracy vs Epoch
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="train_acc")
    plt.plot(df["epoch"], df["test_acc"], marker="s", label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy vs Epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_epoch.png", dpi=200)
    plt.close()

    # 2) Train/Test Loss vs Epoch
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    plt.plot(df["epoch"], df["test_loss"], marker="s", label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss vs Epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_vs_epoch.png", dpi=200)
    plt.close()

    # 3) Spike Rate vs Epoch
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["spike_rate"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Spike Rate")
    plt.title("Spike Rate vs Epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "spike_rate_vs_epoch.png", dpi=200)
    plt.close()

    # 4) Avg Inference Time vs Epoch
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["avg_batch_infer_ms"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Inference Time (ms/batch)")
    plt.title("Avg Inference Time vs Epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "infer_time_vs_epoch.png", dpi=200)
    plt.close()

    # 5) Accuracy vs Resource (baseline内部每个epoch点)
    # 可选：后续做 FP32/W8/W4/W2 时也能沿用同逻辑
    if "sop_proxy" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.scatter(df["sop_proxy"], df["test_acc"])
        for _, row in df.iterrows():
            plt.annotate(f"E{int(row['epoch'])}", (row["sop_proxy"], row["test_acc"]))
        plt.xlabel("SOP Proxy")
        plt.ylabel("Test Accuracy")
        plt.title("Accuracy vs Resource (SOP Proxy)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy_vs_sop_proxy.png", dpi=200)
        plt.close()

    print(f"Done. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
