import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_bits(bits_text: str) -> list:
    values = [item.strip() for item in bits_text.split(",") if item.strip()]
    if not values:
        raise ValueError("bits list cannot be empty, e.g. '8,4'")
    return [int(item) for item in values]


def _none_if_non_positive(value: int) -> Optional[int]:
    return None if value <= 0 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run allocation-aware KD on mixed-precision baseline_nmnist."
    )
    parser.add_argument("--checkpoint-path", default="outputs/baseline_nmnist/fp32_last.pt")
    parser.add_argument("--data-root", default="baseline_nmnist/data")
    parser.add_argument("--output-dir", default="outputs/allocation_aware_kd_nmnist")
    parser.add_argument("--bits", default="8,4")
    parser.add_argument("--weight-allocation-csv", default=None)
    parser.add_argument("--target-avg-bits", type=float, default=None)
    parser.add_argument("--t-steps", type=int, default=16)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--max-hessian-batches", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--trace-probes", type=int, default=1)
    parser.add_argument("--quant-epochs", type=int, default=5)
    parser.add_argument("--quant-lr", type=float, default=1e-4)
    parser.add_argument("--quant-weight-decay", type=float, default=5e-4)
    parser.add_argument("--allocation-policy", choices=["rank-map", "tiered"], default="rank-map")
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--feature-distill-beta", type=float, default=0.1)
    parser.add_argument("--severity-power", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from allocation_aware_kd_nmnist.runner import run_allocation_aware_distillation_analysis
    from baseline_nmnist.config import BaselineNMNISTConfig

    bits_list = _parse_bits(args.bits)
    cfg = BaselineNMNISTConfig(
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
        max_test_batches=_none_if_non_positive(args.max_test_batches),
    )
    summary = run_allocation_aware_distillation_analysis(
        cfg=cfg,
        checkpoint_path=args.checkpoint_path,
        bits_list=bits_list,
        target_avg_bits=args.target_avg_bits,
        output_dir=args.output_dir,
        weight_allocation_csv=args.weight_allocation_csv,
        max_hessian_batches=_none_if_non_positive(args.max_hessian_batches),
        max_test_batches=_none_if_non_positive(args.max_test_batches),
        trace_probes=args.trace_probes,
        quant_epochs=args.quant_epochs,
        quant_lr=args.quant_lr,
        quant_weight_decay=args.quant_weight_decay,
        allocation_policy=args.allocation_policy,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        feature_distill_beta=args.feature_distill_beta,
        severity_power=args.severity_power,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
