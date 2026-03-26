# Command
# python scripts/plot_baseline_line.py --csv "outputs/baseline/base_result E10/epoch_metrics.csv" --out-dir "outputs/baseline/base_result E10"


import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def read_metrics(csv_path: Path):
    epochs = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    infer_ms = []
    sop_proxy = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_acc.append(float(row["train_acc"]))
            test_acc.append(float(row["test_acc"]))
            train_loss.append(float(row["train_loss"]))
            test_loss.append(float(row["test_loss"]))
            infer_ms.append(float(row["avg_batch_infer_ms"]))
            sop_proxy.append(float(row["sop_proxy"]))

    if not epochs:
        raise ValueError(f"No data rows found in: {csv_path}")

    return {
        "epochs": epochs,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "avg_batch_infer_ms": infer_ms,
        "sop_proxy": sop_proxy,
    }


def plot_accuracy(epochs, train_acc, test_acc, out_path: Path, show: bool = False):
    plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(epochs, train_acc, marker="o", linewidth=2, label="train_acc")
    plt.plot(epochs, test_acc, marker="s", linewidth=2, label="test_acc")

    plt.title("Baseline Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)

    if show:
        plt.show()

    plt.close()


def plot_loss(epochs, train_loss, test_loss, out_path: Path):
    plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(epochs, train_loss, marker="o", linewidth=2, label="train_loss")
    plt.plot(epochs, test_loss, marker="s", linewidth=2, label="test_loss")

    plt.title("Baseline Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_tradeoff(test_acc, x_values, x_label: str, out_path: Path):
    plt.figure(figsize=(7, 5), dpi=140)
    plt.scatter(x_values, test_acc, s=45)
    for i, (xv, yv) in enumerate(zip(x_values, test_acc), start=1):
        plt.annotate(str(i), (xv, yv), textcoords="offset points", xytext=(5, 4), fontsize=8)

    plt.title("Accuracy vs Cost")
    plt.xlabel(x_label)
    plt.ylabel("test_acc")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def read_confusion_matrix(csv_path: Path):
    matrix = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            matrix.append([float(x) for x in row])

    if not matrix:
        raise ValueError(f"No rows found in confusion matrix CSV: {csv_path}")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("Confusion matrix must be square (N x N).")
    return matrix


def plot_confusion_matrix(matrix, class_names, out_path: Path):
    n = len(matrix)
    if len(class_names) != n:
        raise ValueError(f"class_names length ({len(class_names)}) must match matrix size ({n}).")

    plt.figure(figsize=(8, 7), dpi=140)
    im = plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    ticks = list(range(n))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    max_val = max(max(row) for row in matrix) if matrix else 0.0
    threshold = max_val * 0.5
    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            color = "white" if value > threshold else "black"
            plt.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Export compact baseline charts in one command.")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/baseline/base_result E10/epoch_metrics.csv",
        help="Path to epoch_metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/baseline/base_result E10",
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--tradeoff-x",
        choices=["avg_batch_infer_ms", "sop_proxy"],
        default="avg_batch_infer_ms",
        help="X-axis metric for tradeoff scatter.",
    )
    parser.add_argument(
        "--cm-csv",
        type=str,
        default="",
        help="Optional confusion matrix CSV path (NxN).",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names for confusion matrix.",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure window")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    metrics = read_metrics(csv_path)

    acc_path = out_dir / "accuracy_curve.png"
    loss_path = out_dir / "loss_curve.png"
    tradeoff_path = out_dir / f"acc_vs_{args.tradeoff_x}.png"

    plot_accuracy(
        metrics["epochs"],
        metrics["train_acc"],
        metrics["test_acc"],
        out_path=acc_path,
        show=args.show,
    )
    plot_loss(metrics["epochs"], metrics["train_loss"], metrics["test_loss"], out_path=loss_path)

    x_label = "avg_batch_infer_ms" if args.tradeoff_x == "avg_batch_infer_ms" else "sop_proxy"
    plot_tradeoff(metrics["test_acc"], metrics[args.tradeoff_x], x_label=x_label, out_path=tradeoff_path)

    print(f"Saved: {acc_path}")
    print(f"Saved: {loss_path}")
    print(f"Saved: {tradeoff_path}")

    cm_csv = Path(args.cm_csv) if args.cm_csv else None
    if cm_csv is not None:
        if not cm_csv.exists():
            raise FileNotFoundError(f"Confusion matrix CSV not found: {cm_csv}")
        class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
        matrix = read_confusion_matrix(cm_csv)
        cm_path = out_dir / "confusion_matrix.png"
        plot_confusion_matrix(matrix, class_names=class_names, out_path=cm_path)
        print(f"Saved: {cm_path}")
    else:
        print("Skipped confusion_matrix.png (provide --cm-csv to generate it).")


if __name__ == "__main__":
    main()
