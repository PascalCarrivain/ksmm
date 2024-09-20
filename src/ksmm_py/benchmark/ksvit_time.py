import argparse
from ksmm_py.model.get_vit import get_submodel_vit, get_input_shape
import torch
from pathlib import Path
import torch.utils.benchmark as benchmark
import pandas as pd

from ksmm_py.benchmark.utils import (
    parse_patterns,
    get_dtype,
    set_device_and_get_device_name,
    get_path_submodules_vit,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "simple_vit_s16_in1k",
            "simple_vit_b16_in1k",
            "simple_vit_l16_in1k",
            "simple_vit_h14_in1k",
        ],
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=[
            "gelu",
            "linear",
            "linear_up",
            "linear_down",
            "linear_bias",
            "linear_up_bias",
            "linear_down_bias",
            "layernorm",
            "custom_layernorm",
            "ffn",
            "attention",
            "ffn_residual",
            "attention_residual",
            "block",
            "vit",
            "vit_surgery_ffn_only",
            "only_linear_in_ffn",
            "only_linear_in_attn",
            "only_linear_in_block",
            "only_linear_in_vit",
            "vit_only_linear_in_ffn",
        ],
    )

    parser.add_argument(
        "--sdpa-version",
        choices=["flash", "equivalent", "default"],
        help="Whether to use flash attention, an equivalent torch implementation, or the default torch implementation.",
    )

    parser.add_argument(
        "--split-qkv",
        type=int,
        choices=[0, 1],
        help="Whether to use store the query, key and value matrices as three distinct matrices (useful when replacing each one of them by a Kronecker-sparse layer) or as a single dense matrix qkv (more efficient but not suited for Kronecker-sparse replacement).",
    )

    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        help="e.g. '4,192,192,1' '1,192,192,4' for first factor with pattern (4,192,192,1) and second factor with pattern (1,192,192,4)",
    )
    parser.add_argument("--patterns_up", type=str, nargs="+")
    parser.add_argument("--patterns_down", type=str, nargs="+")

    # Experimental settings
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"])
    parser.add_argument(
        "--bs-last",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 for batch-size-first, 1 for batch-size-last",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--min-run-time", type=float, default=5)
    parser.add_argument("--saving-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--algo",
        choices=[
            "sparse",
            "dense",
            "bmm",
            "kernel",
            "nn_linear",
            "unfused_linear",
            "bsr",
            "einsum",
        ],
    )

    args = parser.parse_args()

    args.bs_last = args.bs_last == 1
    args.split_qkv = None if args.split_qkv is None else args.split_qkv == 1
    args.patterns = parse_patterns(args.patterns)
    args.patterns_up = parse_patterns(args.patterns_up)
    args.patterns_down = parse_patterns(args.patterns_down)

    print("\n", args)

    return args


def save_results(
    args,
    m,
    device_name,
):

    results_df = pd.DataFrame(
        [
            {
                "arch": args.arch,
                "granularity": args.granularity,
                "patterns": args.patterns,
                "patterns_up": args.patterns_up,
                "patterns_down": args.patterns_down,
                "batch_size": args.batch_size,
                "algo": args.algo,
                "precision": args.precision,
                "bs-last": args.bs_last,
                "batch-size": args.batch_size,
                "min-run-time": args.min_run_time,
                "mean": m.mean,
                "median": m.median,
                "iqr": m.iqr,
            }
        ]
    )

    if args.saving_dir is not None:
        saving_path = args.saving_dir / get_path_submodules_vit(
            args.arch,
            args.granularity,
            args.algo,
            "bs_last" if args.bs_last else "bs_first",
            args.precision,
            args.batch_size,
            device_name,
            args.patterns,
            args.patterns_down,
            args.patterns_up,
            args.sdpa_version,
            args.split_qkv,
        )
        saving_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(saving_path)


def main(args):
    # SET DEVICE
    device_name = set_device_and_get_device_name(args.device_id, args.device)

    # GET PRECISION

    dtype = get_dtype(args.precision)

    # GET MODEL
    model = get_submodel_vit(
        args.arch,
        args.bs_last,
        args.granularity,
        args.patterns,
        args.patterns_down,
        args.patterns_up,
        args.algo,
        dtype=dtype,
        device=args.device,
        sdpa_version=args.sdpa_version,
        split_qkv=args.split_qkv,
    )
    model = model.to(device=args.device, dtype=dtype)
    model.eval()

    # GET INPUT
    x_shape = get_input_shape(
        args.arch, args.granularity, args.bs_last, args.batch_size
    )
    x = torch.randn(*x_shape, dtype=dtype, device=args.device)

    # BENCHMARK
    t = benchmark.Timer(
        stmt=f"forward_pass(model, x)",
        setup=f"from ksmm_py.benchmark.utils import forward_pass",
        globals={"model": model, "x": x},
        num_threads=torch.get_num_threads(),
        label=f"{args.arch}, {args.granularity}",
        sub_label=f"{args.algo}, {args.precision}, {args.bs_last}",
        description=f"{x.dtype}, x.shape={x.shape}",
    )
    m = t.blocked_autorange(min_run_time=args.min_run_time)

    # when not enough time to measure, increase the min_run_time to 11.0 * max(m.mean, m.median)
    if m.number_per_run <= 10:
        t = benchmark.Timer(
            stmt=f"forward_pass(model, x)",
            setup=f"from ksmm_py.benchmark.utils import forward_pass",
            globals={"model": model, "x": x},
            num_threads=torch.get_num_threads(),
            label=f"{args.arch}, {args.granularity}",
            sub_label=f"{args.algo}, {args.precision}, {args.bs_last}",
            description=f"{x.dtype}, x.shape={x.shape}",
        )
        m = t.blocked_autorange(min_run_time=11.0 * max(m.mean, m.median))
    print(m)

    # SAVE RESULTS
    save_results(
        args,
        m,
        device_name,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
