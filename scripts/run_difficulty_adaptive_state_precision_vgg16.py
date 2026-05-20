import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _none_if_non_positive(value: int) -> Optional[int]:
    return None if value <= 0 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run difficulty-adaptive state precision verification on baseline_vgg16."
    )
    parser.add_argument("--checkpoint-path", default="outputs/baseline_vgg16/fp32_last.pt")
    parser.add_argument("--data-root", default="baseline/data")
    parser.add_argument("--output-dir", default="outputs/difficulty_adaptive_state_precision_vgg16")
    parser.add_argument("--low-state-bits", type=int, default=4)
    parser.add_argument("--high-state-bits", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--difficulty-metric", choices=["confidence", "margin"], default="confidence")
    parser.add_argument("--easy-threshold", type=float, default=0.7)
    parser.add_argument("--t-steps", type=int, default=16)
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--max-hessian-batches", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--quant-epochs", type=int, default=1)
    parser.add_argument("--quant-lr", type=float, default=1e-4)
    parser.add_argument("--quant-weight-decay", type=float, default=5e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from baseline_vgg16.config import BaselineVGG16Config
    from difficulty_adaptive_state_precision_vgg16.runner import (
        run_difficulty_adaptive_state_precision_analysis,
    )

    cfg = BaselineVGG16Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_download=args.download,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        num_workers=args.num_workers,
        t_steps=args.t_steps,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        max_train_batches=_none_if_non_positive(args.max_train_batches),
    )
    summary = run_difficulty_adaptive_state_precision_analysis(
        cfg=cfg,
        checkpoint_path=args.checkpoint_path,
        low_state_bits=args.low_state_bits,
        high_state_bits=args.high_state_bits,
        warmup_steps=args.warmup_steps,
        difficulty_metric=args.difficulty_metric,
        easy_threshold=args.easy_threshold,
        output_dir=args.output_dir,
        max_hessian_batches=_none_if_non_positive(args.max_hessian_batches),
        max_test_batches=_none_if_non_positive(args.max_test_batches),
        quant_epochs=args.quant_epochs,
        quant_lr=args.quant_lr,
        quant_weight_decay=args.quant_weight_decay,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
