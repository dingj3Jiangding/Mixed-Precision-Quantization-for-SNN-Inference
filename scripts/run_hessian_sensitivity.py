import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline import BaselineConfig, run_hessian_sensitivity_analysis


def _parse_bits(bits_text: str) -> list:
    values = [item.strip() for item in bits_text.split(",") if item.strip()]
    if not values:
        raise ValueError("bits list cannot be empty, e.g. '8,4,2'")
    return [int(item) for item in values]


def _none_if_non_positive(value: int) -> Optional[int]:
    return None if value <= 0 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hessian sensitivity analysis and mixed-precision allocation.")
    parser.add_argument("--checkpoint-path", default="outputs/baseline/fp32_last.pt")
    parser.add_argument("--data-root", default="baseline/data")
    parser.add_argument("--output-dir", default="outputs/hessian_sensitivity")
    parser.add_argument("--bits", default="8,4,2")
    parser.add_argument("--target-avg-bits", type=float, default=4.0)
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
    parser.add_argument("--max-test-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bits_list = _parse_bits(args.bits)

    cfg = BaselineConfig(
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
    )
    summary = run_hessian_sensitivity_analysis(
        cfg=cfg,
        checkpoint_path=args.checkpoint_path,
        bits_list=bits_list,
        target_avg_bits=args.target_avg_bits,
        output_dir=args.output_dir,
        max_hessian_batches=_none_if_non_positive(args.max_hessian_batches),
        max_test_batches=_none_if_non_positive(args.max_test_batches),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
