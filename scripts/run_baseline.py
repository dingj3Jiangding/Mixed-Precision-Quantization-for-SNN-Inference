import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is importable when executing "python scripts/run_baseline.py".
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline import BaselineConfig, run_baseline


def _none_if_non_positive(value: int) -> Optional[int]:
    return None if value <= 0 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible CIFAR-10 SNN baseline.")
    parser.add_argument("--data-root", default="baseline/data")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--t-steps", type=int, default=16)
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BaselineConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_download=args.download,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        num_workers=args.num_workers,
        epochs=args.epochs,
        t_steps=args.t_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        max_train_batches=_none_if_non_positive(args.max_train_batches),
        max_test_batches=_none_if_non_positive(args.max_test_batches),
    )
    summary = run_baseline(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
