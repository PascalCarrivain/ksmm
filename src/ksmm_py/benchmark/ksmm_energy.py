import argparse
import os
from pathlib import Path
import torch.backends.cuda
from ksmm_py.benchmark.utils import (
    parse_patterns,
    forward_pass,
    get_dtype
)

from ksmm_py.layer.get_linear_layer import get_linear_layer

import pandas as pd
from ksmm_py.benchmark.utils import (
    get_dtype,
    get_in_size_out_size,
    set_device_and_get_device_name,
)

from pyJoules.energy_meter import measure_energy
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.pandas_handler import PandasHandler


def get_args():
    parser = argparse.ArgumentParser()
    # Path to save csv result
    parser.add_argument("--saving-csv", type=Path)
    parser.add_argument("--device-id", type=int, default=0)

    # Parameters for butterfly parameterization
    parser.add_argument("--patterns", type=str, nargs="+", default="", )
    parser.add_argument("--algo",
                        choices=["sparse", "dense", "bmm", "kernel", "einsum", "bsr"],
                        help="Choice of the KSMM algorithm. See the paper, or look directly into the functions forward_bs_first and forward_bs_last (file ksmm_py/layer/kronecker_sparse/interface.py) for details about the algorithms sparse, dense, bmm, kernel, einsum and bsr.",
                        )
    parser.add_argument("--bs-last", type=int, choices=[0, 1], default=0,
                        help="0 for batch-size-first, 1 for batch-size-last",
                        )

    # Experimental setting
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)

    # Check correctness
    parser.add_argument("--correctness", action="store_true")

    args = parser.parse_args()

    args.bs_last = args.bs_last == 1

    print("\n", args)
    return args


def save_results(args, m_df, device_name, nruns):
    # Ensure the directory exists
    args.saving_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load the CSV if it exists, otherwise create an empty DataFrame
    if os.path.exists(args.saving_csv):
        # Load existing data
        df = pd.read_csv(args.saving_csv)
        expected_columns = ["pattern", "batch-size", "algo", "bs_position", "precision", "device_name", "median", "nruns", "ecs"]
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in the CSV file. Please provide a valid CSV file.")
    else:
        # Define the columns of the DataFrame
        df = pd.DataFrame(columns=["pattern", "batch-size", "algo", "bs_position", "precision", "device_name", "median", "nruns", "ecs"])

    # Define the new entry
    bs_position ="bs_last" if args.bs_last else "bs_first"
    results_df = pd.DataFrame(
        [
            {
                "pattern": str(args.patterns),
                "batch-size": args.batch_size,
                "algo": args.algo,
                "bs_position": bs_position,
                "precision": args.precision,
                "device_name": device_name,
                "median": m_df['nvidia_gpu_0'].median(),
                "nruns": nruns,
                "ecs": str(m_df['nvidia_gpu_0'].tolist()),
            }
        ]
    )

    # Check if the CSV already contains a row with the same configuration
    if not df.empty:
        existing_row = df[
            (df["pattern"] == str(args.patterns)) &
            (df["batch-size"] == args.batch_size) &
            (df["algo"] == args.algo) &
            (df["bs_position"] == bs_position) &
            (df["precision"] == args.precision) &
            (df["device_name"] == device_name)
        ]
    else:
        existing_row = pd.DataFrame()


    if not existing_row.empty:
        # Get the index of the existing row
        index = existing_row.index
        #Check there is only one measurement for this configuration
        assert len(index) == 1, "The CSV file should not contain multiple rows with the same configuration (pattern, batch-size, algo, bs_position, precision, and device_name)."

        # Update the existing row in df with the values from results_df
        df.loc[index, results_df.columns] = results_df.iloc[0].values
    else:
        # Append the new row
        df = pd.concat([df, results_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    df.to_csv(args.saving_csv, index=False)


def main(args):
    patterns = parse_patterns(args.patterns)

    # SET DEVICE

    device_name = set_device_and_get_device_name(args.device_id, args.device)

    in_size, out_size = get_in_size_out_size(patterns)

    # GET DTYPE
    dtype = get_dtype(args.precision)

    # GET LAYER
    layer = get_linear_layer(
        patterns=patterns,
        algo=args.algo,
        dtype=dtype,
        bs_last=args.bs_last,
        device=args.device
    )

    # GET INPUT
    if args.bs_last:
        x = torch.randn(in_size, args.batch_size, dtype=dtype, device=args.device)
    else:
        x = torch.randn(args.batch_size, in_size, dtype=dtype, device=args.device)

    # CHECK CORRECTNESS
    if args.correctness:
        # correct output
        dense = layer.get_dense_product().to("cpu")
        y = x.clone().to("cpu")
        if args.bs_last:
            correct_output = torch.mm(dense, y)
        else:
            correct_output = torch.mm(dense, y.t()).t()
        # layer output
        layer.eval()
        with torch.no_grad():
            output = layer(x).to("cpu")

        # comparison
        rel_err = torch.norm(correct_output - output) / torch.norm(correct_output)
        print("Relative error = {0:e}".format(rel_err))

    # Energy consumption
    layer.eval()

    def _repeats(nruns):
        pandas_handler = PandasHandler()
        @measure_energy(domains=[NvidiaGPUDomain(0)], handler=pandas_handler)
        def _run(model, x, nruns: int=10):
            for n in range(nruns):
                forward_pass(model, x)
                torch.cuda.synchronize()
        for _ in range(10):
            _run(layer, x, nruns)
        return pandas_handler.get_dataframe()

    nruns = 2
    while True:
        df = _repeats(nruns)
        q1 = df["nvidia_gpu_0"].quantile(0.25)
        q3 = df["nvidia_gpu_0"].quantile(0.75)
        iqr = q3 - q1
        median = df["nvidia_gpu_0"].median()
        if iqr < (median / 10) and (df["nvidia_gpu_0"] != 0.0).sum() == 10:
            break
        else:
            nruns *= 2

    save_results(args, df, device_name, nruns)


if __name__ == "__main__":
    args = get_args()
    main(args)
