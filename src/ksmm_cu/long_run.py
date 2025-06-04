# example of use: python3 long_run.py "kernel_bs_first_half4" 0 full 100 4.512.512.4
# python3 long_run.py "kernel_bs_first_half4" 0 full 100 0.0.0.0 --tuples_file selected_tuples.in
import argparse
import glob
import itertools
import os
import subprocess
import re
import shutil


def is_gpu_available(gpu_name):
    """Check if a specific GPU model is available using nvidia-smi."""
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi -L", shell=True, text=True
        )
        return gpu_name in gpu_info
    except subprocess.CalledProcessError:
        return False


def determine_architecture():
    """Determine the CUDA architecture based on available GPU."""
    if is_gpu_available("A100"):
        print("A100 found")
        return "sm_80"
    elif is_gpu_available("RTX"):
        return "sm_75" if is_gpu_available("2080") else "sm_86"
    elif is_gpu_available("V100"):
        print("V100 found")
        return "sm_70"
    return None


def compile_and_run(kernel_name, device_id, nrepeats):
    """Compile the CUDA file with NVCC and run it with the specified parameters."""
    arch = determine_architecture()
    if arch is None:
        print("Unsupported GPU architecture")
        return

    # Remove previous executable if it exists
    if os.path.exists("sparse_mm"):
        os.remove("sparse_mm")

    # Compile CUDA code
    compile_command = [
        "nvcc",
        "-Xcompiler",
        "-fopenmp",
        "src/sparse_mm.cu",
        "-lcublas",
        "-lcurand",
        "-o",
        "sparse_mm",
        f"-arch={arch}",
        "-src-in-ptx",
        "--generate-line-info",
        # "--ptxas-options=-v,--opt-level=0", "--keep"
        "--ptxas-options=-v,--opt-level=3",
        "--keep",
    ]
    subprocess.run(compile_command, check=True)

    # Run the compiled program
    run_command = [
        "./sparse_mm",
        "-n",
        kernel_name,
        "-d",
        str(device_id),
        "-r",
        str(nrepeats),
    ]
    subprocess.run(run_command, check=True)


def is_valid_configuration_sanity_check(TILEY, b, c, TILEX, TILEK, TX, TY):
    # Check TILEY condition
    TILEYs = [16, 18, 24, 26, 32, 48, 52, 64, 128, 256]
    if TILEY not in TILEYs:
        return False

    # Define TILEXs and TILEKs based on values of B and C
    if b == 48 or c == 48:
        TILEXs = [16, 24, 48]
        TILEKs = [16, 24]
    elif b == 80 or c == 80:
        TILEXs = [8, 10, 20, 40]
        TILEKs = [8, 10, 16, 20]
    elif b == 96 or c == 96:
        TILEXs = [32, 64]
        TILEKs = [16, 32]
    elif b == 160 or c == 160:
        TILEXs = [16, 20, 40, 80]
        TILEKs = [16, 20, 40]
    elif b == 906 or c == 906:
        TILEXs = [252]
        TILEKs = [16, 24, 32]
    else:
        TILEXs = [16, 18, 24, 26, 32, 48, 52, 64, 128, 256]
        TILEKs = [4, 8, 12, 16, 18, 24, 26, 32, 48, 52, 64]

    # Check if TILEX and TILEK are within TILEXs and TILEKs
    if TILEX not in TILEXs or TILEK not in TILEKs:
        return False

    # Check TX and TY conditions
    if TX not in [4, 8] or TY not in [4, 8]:
        return False

    # If all conditions are met, the configuration is valid
    return True


def get_input_size(a, b, c, d):
    return a * c * d


def get_output_size(a, b, c, d):
    return a * b * d


def is_valid_pattern(batch, a, b, c, d):
    # Only b=c, b=4c, c=4b, corresponding to squared matrices and up/down projections in attention layers
    if b != c and b != 4 * c and c != 4 * b:
        return False
    # Skip the next ones as the ratios (b+c)/(bc) are too close to the ratios of other patterns already in the benchmark
    bc_to_skip = [(1024, 256), (128, 512), (65, 256)]
    if (b, c) in bc_to_skip or (c, b) in bc_to_skip:
        return False
    # Skip patterns associated to dimensions too large wrt greatest int that can be represented
    in_dim = get_input_size(a, b, c, d)
    out_dim = get_output_size(a, b, c, d)
    if (
        batch * in_dim >= 2147483647
        or batch * out_dim >= 2147483647
        or a * b * c * d >= 2147483647
    ):
        return False
    return True


def replace_constants_in_template(template_file, output_file, replacements):
    """Replace placeholders in template file and save to output file."""
    with open(template_file, "r") as f:
        content = f.read()
    for placeholder, value in replacements.items():
        content = re.sub(placeholder, str(value), content)
    with open(output_file, "w") as f:
        f.write(content)


def configure_templates(
    kernel_name,
    batch,
    a,
    b,
    c,
    d,
    WMMA_X,
    WMMA_K,
    WMMA_Y,
    TILEX,
    TILEK,
    TILEY,
    TX,
    TY,
    x,
    y,
    nwx,
    nwy,
    VSIZE,
    dtype,
    USE_LDG,
    USE_LDG_RC,
    USE_STCG,
    USE_SHFL_SYNC,
):
    """For each file f, copy the template_f.cuh into a f.cu, and replaces the strings xCSTx in the template by its actual value provided here as argument CST."""
    files = [
        "sparse_mm",
        "kernels_float2",
        "kernels_float4",
        "kernels_half2",
        "kernels_half4",
        "kernels_half8",
        "kernels_fp8e4m3",
        "kernels_tc",
    ]
    for f in files:
        template_file = (
            f"src/template_{f}.{'cu' if f == 'sparse_mm' else 'cuh'}"
        )
        output_file = f"src/{f}.{'cu' if f == 'sparse_mm' else 'cuh'}"
        shutil.copyfile(template_file, output_file)

        # Define directive to handle #include kernels_*.cuh, based on kernel_name and half precision flag
        define_statement = (
            "TC"
            if "_tc" in kernel_name
            else (
                f"FLOAT{VSIZE}"
                if dtype == "f32"
                else f"HALF{VSIZE}"
                if dtype == "f16"
                else "FP8E4M3"
            )
        )
        # Replacement values for placeholders
        replacements = {
            r"xBATCHSIZEx": batch,
            r"xINPUTSIZEx": get_input_size(a, b, c, d),
            r"xOUTPUTSIZEx": get_output_size(a, b, c, d),
            r"xax": a,
            r"xbx": b,
            r"xcx": c,
            r"xdx": d,
            r"xWMMA_Yx": WMMA_Y,
            r"xWMMA_Xx": WMMA_X,
            r"xWMMA_Kx": WMMA_K,
            r"xNWARPSYx": nwx,
            r"xNWARPSXx": nwy,
            r"xTILEXx": TILEX,
            r"xTILEKx": TILEK,
            r"xTILEYx": TILEY,
            r"xTXx": TX,
            r"xTYx": TY,
            r"xXx": x,
            r"xYx": y,
            r"xNTHREADSx": x * y,
            r"xCUBLAS_GEMM_ALGOx": "CUBLAS_GEMM_DEFAULT",
            r"xCUBLAS_GEMM_ALGO_TENSOR_OPx": "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
            r"defineFP": f"define {define_statement}",
            r"xUSE_LDGx": USE_LDG,
            r"xUSE_LDG_RCx": USE_LDG_RC,
            r"xUSE_STCGx": USE_STCG,
            r"xUSE_SHFL_SYNCx": USE_SHFL_SYNC,
            r"xSIZE_SHFL_SYNCx": x,
        }

        replace_constants_in_template(output_file, output_file, replacements)


def set_header_outfile(kernel_name):
    if "_tc" in kernel_name:
        header = "batch_size a b c d TILEX TILEK TILEY TX TY nwarpsX nwarpsY WMMA_X WMMA_K WMMA_Y ms std mse"
    else:
        header = (
            "batch_size a b c d TILEX TILEK TILEY TX TY TILEX TILEY ms std mse"
        )

    # Write header in output file
    output_file = f"{kernel_name}.out"
    with open(output_file, "w") as f:
        f.write(header + "\n")


def remove_previous_files(kernel_name):
    for file in [
        f"{kernel_name}.out",
        f"wrong_mse_{kernel_name}.out",
        "log.out",
    ]:
        if os.path.exists(file):
            os.remove(file)

    # Remove files matching the pattern 'assert_failed*.out'
    for file in glob.glob("assert_failed*.out"):
        os.remove(file)


def get_dtype_info(kernel_name):
    # determine data type and its size
    if "half" in kernel_name or "_tc" in kernel_name:
        sizeofdtype = 2
        dtype = "f16"
        strdtype = "half"
    elif "float" in kernel_name:
        sizeofdtype = 4
        dtype = "f32"
        strdtype = "float"
    elif "fp8e4m3" in kernel_name:
        sizeofdtype = 1
        dtype = "f8e4m3"
        strdtype = "fp8e4m3"
    else:
        raise ValueError(f'type not detected from kernel name "{kernel_name}"')

    # determine vector size
    if f"{strdtype}8" in kernel_name:
        VSIZE = 8
    elif f"{strdtype}4" in kernel_name:
        VSIZE = 4
    elif f"{strdtype}2" in kernel_name:
        VSIZE = 2
    else:
        VSIZE = 1
    return dtype, sizeofdtype, VSIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kernel and device setup script"
    )

    # Define command-line arguments
    parser.add_argument(
        "kernel_name",
        nargs="?",
        default="kernel_bs_first_float4",
        help="Name of the kernel, default: kernel_bs_first_float4.",
    )
    parser.add_argument(
        "device_id",
        nargs="?",
        type=int,
        default=0,
        help="Device ID, default: 0.",
    )
    parser.add_argument(
        "flag_check",
        nargs="?",
        default="full",
        help="Flags to determine various checks (e.g., 'sanity_check', 'small', 'full', or 'hyper_params...'), default: full.",
    )
    parser.add_argument(
        "nrepeats",
        nargs="?",
        type=int,
        default=100,  # Replace with your default value for `nrepeats`
        help="Number of repetitions, default: 100.",
    )
    parser.add_argument(
        "abcd",
        nargs="?",
        type=str,
        default="",
        help="A pattern (a,b,c,d) can be given in the format 'a.b.c.d'",
    )

    parser.add_argument(
        "--tuples_file",
        type=str,
        default=None,
        help="Path to a file containing a list of tuples (a, b, c, d, batch size) to process.",
    )

    return parser.parse_args()


def load_tuples(file_path):
    """Load tuples from a file."""
    with open(file_path, "r") as f:
        tuples = []
        for line in f:
            try:
                sl = line.split(",")
                tuples.append(list(map(int, sl)))
            except ValueError:
                print(f"Skipping invalid line: {line}")
        return tuples


def is_valid_configuration(
    kernel_name,
    a,
    b,
    c,
    d,
    x,
    k,
    y,
    TX,
    TY,
    TILEX,
    TILEK,
    TILEY,
    nwx,
    nwy,
    WMMA_X,
    WMMA_K,
    WMMA_Y,
    batch,
    sizeofdtype,
    VSIZE,
):
    # conditions related to tensorcores
    if "_tc" not in kernel_name and (nwx != 1 or nwy != 1):
        return False
    if "_tc" in kernel_name:
        if TILEY * TILEK != nwx * nwy * WMMA_Y * WMMA_K:
            return False
        if TILEK * TILEX != nwx * nwy * WMMA_K * WMMA_X:
            return False
        if TILEX * TILEY != nwx * nwy * WMMA_X * WMMA_Y:
            return False
        warp_size = 32
        if nwx * nwy * warp_size != x * y:
            return False
    # Compute shared memory (double-buffering)
    double_buffering = 2
    smem = sizeofdtype * (
        double_buffering * TILEY * TILEK
        + double_buffering * TILEK * TILEX
        + TILEY * TILEX
    )
    # check shared mem do not exceed max hardware capacity, and check also number of threads (=x*y)
    if smem >= 49152 or (x * y) > 1024:
        return False
    # tiles divide the dimensions of diagonal sub-blocks
    if (b * d) % (d * TILEX) != 0:
        return False
    if (c * d) % (d * TILEK) != 0:
        return False
    # for def of stride values
    if (VSIZE * x * y) % TILEX != 0:
        return False
    # for def of stride input
    if (VSIZE * x * y) % TILEY != 0:
        return False

    stride_values = (VSIZE * x * y) // TILEX
    stride_input = (VSIZE * x * y) // TILEY
    if TILEK % stride_input != 0:
        return False
    if TILEK % stride_values != 0:
        return False
    if batch % TILEY != 0:
        return False

    if TX < VSIZE or TY < VSIZE or TILEK < VSIZE:
        return False
    if TILEK > TILEX or TILEK > TILEY:
        return False

    return True


if __name__ == "__main__":
    # Define parameter ranges
    WMMA_X = 16
    WMMA_K = 16
    WMMA_Y = 16
    USE_LDG = 1
    USE_LGC_RC = 1
    USE_STCG = 1
    USE_SHFL_SYNC = 1

    # xs=[i for i in range(1, 64 + 1)]
    xs = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    ys = xs
    TXs = [i for i in range(2, 16 + 1)]
    TYs = TXs
    ks = TXs
    nwxs = [1, 2, 4, 8, 16]
    nwys = nwxs

    args = parse_args()

    if args.tuples_file:
        # Load tuples from the file
        abcd_batch = load_tuples(args.tuples_file)
        print(f"Loaded {len(abcd_batch)} tuples from {args.tuples_file}.")
    elif args.abcd:
        # Parse single tuple from argument
        try:
            a, b, c, d, batch_size = map(int, args.abcd.split("."))
            abcd_batch = list(
                itertools.product([batch_size], [a], [b], [c], [d])
            )
        except ValueError:
            raise ValueError(
                "The abcd argument must be in 'a.b.c.d' format with integer values."
            )
    else:
        # Generate tuples automatically
        as_values = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        bs_values = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        cs_values = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        ds_values = [1, 4, 16, 64, 128]
        abcd_batch = list(
            itertools.product(
                as_values, bs_values, cs_values, ds_values, [25088]
            )
        )

    # Process the tuples
    kernel_name = args.kernel_name
    device_id = args.device_id
    flag_check = args.flag_check
    nrepeats = args.nrepeats

    # Remove previous files
    remove_previous_files(kernel_name)

    # Set header of results file
    set_header_outfile(kernel_name)

    # Get data type info

    dtype, sizeofdtype, VSIZE = get_dtype_info(kernel_name)

    small = (
        True if "small" == flag_check else False
    )  # do only patterns with a=b=c=d
    sanity_check = True if "sanity_check" == flag_check else False
    full = True if "full" == flag_check else False
    hyper_params = True if flag_check.startswith("hyper_params") else False
    if hyper_params:
        out = flag_check.replace("hyper_params", "").split(".")
        out = list(map(int, out))

    # Loop through all patterns
    for a, b, c, d, batch in abcd_batch:
        # if not is_valid_pattern(batch, a, b, c, d):
        #     continue
        if small and (a != b or b != c or c != d):
            continue

        at_least_one_config = False

        # Loop through all hyper-parameters

        for x, k, y, TX, TY, nwx, nwy in itertools.product(
            xs, ks, ys, TXs, TYs, nwxs, nwys
        ):
            TILEX = x * TX
            TILEK = k * VSIZE
            TILEY = y * TY

            if not is_valid_configuration(
                kernel_name,
                a,
                b,
                c,
                d,
                x,
                k,
                y,
                TX,
                TY,
                TILEX,
                TILEK,
                TILEY,
                nwx,
                nwy,
                WMMA_X,
                WMMA_K,
                WMMA_Y,
                batch,
                sizeofdtype,
                VSIZE,
            ):
                continue

            if hyper_params and (
                TILEX != out[0]
                or TILEK != out[1]
                or TILEY != out[2]
                or TX != out[3]
                or TY != out[4]
            ):
                continue

            if sanity_check and not is_valid_configuration_sanity_check(
                TILEY, b, c, TILEX, TILEK, TX, TY
            ):
                continue

            configure_templates(
                kernel_name,
                batch,
                a,
                b,
                c,
                d,
                WMMA_X,
                WMMA_K,
                WMMA_Y,
                TILEX,
                TILEK,
                TILEY,
                TX,
                TY,
                x,
                y,
                nwx,
                nwy,
                VSIZE,
                dtype,
                USE_LDG,
                USE_LGC_RC,
                USE_STCG,
                USE_SHFL_SYNC,
            )

            compile_and_run(kernel_name, device_id, nrepeats)

            at_least_one_config = True

        if not at_least_one_config:
            with open(
                f"assert_failed_{a}.{b}.{c}.{d}_{kernel_name}.out", "w"
            ) as f:
                f.write(
                    f"{get_input_size(a,b,c,d)} {get_output_size(a,b,c,d)}\n"
                )
