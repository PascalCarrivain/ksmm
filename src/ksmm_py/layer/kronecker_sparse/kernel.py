import torch
import os
import inspect
from torch.utils.cpp_extension import load
from typing import Any, Tuple
import warnings
from warnings import warn
warnings.simplefilter(action="always")


ksmm_compiled_kernel = {}


def add_lines_to_instantiate_kernel_with_hp(lines_to_add, kernel_name, batch_size, a, b, c, d, precision, TILEX, TILEK, TILEY, TX, TY, VSIZE):
    output_size = a * b * d
    lines_to_add.append(
        f"\t\tif (batch_size == {batch_size} && a == {a} && b == {b} && c == {c} && d == {d}) {{\n"
    )
    lines_to_add.append(
        f"\t\t\tthreadsPerBlock.x = {(TILEX // TX) * (TILEY // TY)};\n"
    )
    lines_to_add.append(
        f"\t\t\tblockGrid.x = {(output_size + TILEX - 1) // TILEX};\n"
    )
    lines_to_add.append(
        f"\t\t\tblockGrid.y = {(batch_size + TILEY - 1) // TILEY};\n"
    )
    if precision == 'half' and VSIZE == 4:
        lines_to_add.append(
            f"\t\t\t{kernel_name}4<bool, {TILEX}, {TILEK}, {TILEY}, {TX}, {TY}, {VSIZE}, false, {TILEX // TX}><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);\n"
        )
    elif precision == 'half' and VSIZE == 2:
        lines_to_add.append(
            f"\t\t\t{kernel_name}2<{precision + str(VSIZE)}, {TILEX}, {TILEK}, {TILEY}, {TX}, {TY}, {VSIZE}, true, {TILEX // TX}><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);\n"
        )
    elif precision == 'float' and VSIZE in [2, 4]:
        lines_to_add.append(
            f"\t\t\t{kernel_name}<{precision + str(VSIZE)}, {TILEX}, {TILEK}, {TILEY}, {TX}, {TY}, {VSIZE}, true, {TILEX // TX}><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);\n"
        )
    elif precision == 'fp8e4m3' and VSIZE in [2, 4]:
        lines_to_add.append(
            "\t\t\t{0:s}<{1:s}, {2:s}, {3:s}, {4:d}, {5:d}, {6:d}, {7:d}, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:s}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
            kernel_name.replace(precision,"fp8"), '__nv_fp8_e4m3', f'__nv_fp8x{VSIZE}_e4m3', '__NV_E4M3', TILEX, TILEK, TILEY, TX, TY, 'batch_size', 'a', 'b', 'c', 'd'
            )
        )
    elif precision == 'fp8e5m2' and VSIZE in [2, 4]:
        lines_to_add.append(
            "\t\t\t{0:s}<{1:s}, {2:s}, {3:s}, {4:d}, {5:d}, {6:d}, {7:d}, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:s}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
                kernel_name.replace(precision,"fp8"), '__nv_fp8_e5m2', f'__nv_fp8x{VSIZE}_e5m2', '__NV_E5M2', TILEX, TILEK, TILEY, TX, TY, 'batch_size', 'a', 'b', 'c', 'd'
            )
        )
    else:
        raise Exception(
            f"ERROR: Unsupported precision {precision} with VSIZE {VSIZE}. Supported precisions are 'half' and 'float'."
        )
    lines_to_add.append("\t\t\tbreak;\n")
    lines_to_add.append("\t\t}\n")

    return lines_to_add

class Kernel(torch.autograd.Function):
    @staticmethod
    def forward(
        input,
        factor,
        bs_last,
        a,
        b,
        c,
        d,
        dtype,
        vsize=4
    ):
        filename = inspect.getframeinfo(
            inspect.currentframe()).filename
        directory = os.path.dirname(os.path.abspath(filename))

        global ksmm_compiled_kernel

        shape = input.shape
        name = f"{a}.{b}.{c}.{d}.{shape[1] if bs_last else shape[0]}.{vsize}"
        target_batch_size = input.shape[-1] if bs_last else input.shape[0]


        # Find already compiled kernel.
        kernel = ksmm_compiled_kernel.get(name)

        if kernel is None:
            print(f"-------Current Python process hasn't compiled yet the kernel for the target dimensions (batch-size={target_batch_size}, a={a}, b={b}, c={c}, d={d}), compiling it now...---------------------")
            # Did not find a compiled kernel,
            # therefore create and compile one.
            for bsl in [False, True]:
                for precision in ['half', 'float']:
                    # for precision in ['fp8e4m3', 'fp8e5m2', 'half', 'float']:
                    # if we only do a given precision, we should in kernel.cu include only the .cuh corresponding to that precision (the other might not exist if never created)
                    # if dtype == torch.float32 or dtype == torch.float:
                    #     precision = 'float'
                    # elif dtype == torch.float16 or dtype == torch.half:
                    #     precision = 'half'
                    # elif dtype == torch.float8_e4m3fn:
                    #     precision = 'fp8'

                    # Best VSIZE is in the file *.best.
                    if bsl:
                        filename = f"kernel_bs_last_{precision}.cuh"
                    else:
                        filename = f"kernel_bs_first_{precision}.cuh"

                    
                    kernel_name = filename.replace(".cuh", "")

                    filename = directory + "/" + filename

                    # First check if the .cuh file exists, if not create it with empty declaration of the kernel for now
                    if not os.path.exists(filename):
                        with open(filename, "w") as out_file:
                            # Header
                            out_file.write("// -*- c -*-\n\n")
                            out_file.write("#ifndef {0:s}\n".format(kernel_name.upper()))
                            out_file.write("#define {0:s}\n\n".format(kernel_name.upper()))
                            out_file.write('#include "template_kernels_{0:s}.cuh"\n\n'.format(precision))
                            # kernel header
                            out_file.write("void best_{0:s}({1:s} *input, {1:s} *values, {1:s} *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){{\n".format(kernel_name, precision))
                            # out_file.write("void best_kernel({0:s} *input, {0:s} *values, {0:s} *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){{\n".format(precision))
                            out_file.write("\twhile (1) {\n")
                            out_file.write("\t\tthreadsPerBlock.y = 1;\n")

                            # Add a case in the .cuh to catch cases that would not correspond to any of the hyperparameters combinations that will be added (throw an error in such cases)
                            out_file.write("\t\tassert(1 == 0);\n")
                            out_file.write("\t\tbreak;\n")
                            out_file.write("\t}\n")
                            out_file.write("}\n\n")
                            out_file.write("#endif\n")


                    # Open the existing .cuh file, get all lines
                    with open(filename, "r") as out_file:
                        file_lines = out_file.readlines()


                    # Check if the file already contains a case corresponding to target (batch_size, a, b, c, d)
                    # if so, do nothing (the kernel has been compiled previously, we will set ksmm_compiled_kernel[name] = kernel at the end for current process)
                    instance_exists = any(f"batch_size == {target_batch_size} && a == {a} && b == {b} && c == {c} && d == {d}" in line for line in file_lines)
                    if instance_exists:
                        continue
                    
                    # target (batch_size, a, b, c, d) is not in the .cuh file, we will add it
                    # Let's first check if we can find corresponding hyper-parameters in the .best file (collected during a finetuning run)
                    lines_to_add = []
                    found_hyperparameters = False
                    with open(filename.replace('.cuh', '.best'), "r") as in_file:
                        lines = in_file.readlines()
                        for l in lines[1:]:
                            sl = l.split(' ')
                            batch_size = int(sl[0])
                            abcd = (int(sl[1]), int(sl[2]), int(sl[3]), int(sl[4]))
                            VSIZE = int(sl[10])
                            # for we find target (batch_size, a, b, c, d) in the .best file, write the corresponding kernel instance in the .cuh file
                            if abcd[0] == a and abcd[1] == b and abcd[2] == c and abcd[3] == d and batch_size == target_batch_size and vsize == VSIZE:
                                TILEX = int(sl[5])
                                TILEK = int(sl[6])
                                TILEY = int(sl[7])
                                TX = int(sl[8])
                                TY = int(sl[9])
                                # VSIZE = int(sl[10])
                                lines_to_add = add_lines_to_instantiate_kernel_with_hp(lines_to_add, kernel_name, batch_size, a, b, c, d, precision, TILEX, TILEK, TILEY, TX, TY, VSIZE)
                                found_hyperparameters = True
                                print(f"Found hyper-parameters for {batch_size},{a},{b},{c},{d}, bs last = {bsl}, {precision}, vsize = {VSIZE} OK.")
                                break
                    if not found_hyperparameters:
                        # In the case we do not find the hyper-parameters in the .best file, we will try to find some compatible ones
                        # for that we first look at what is the greatest divisor of the batch size among the elements of tmp
                        # and try to add a corresponding kernel instance
                        tmp = [256, 128, 96, 64, 48, 32, 24, 16, 12]
                        # tmp = [256, 128, 64, 32, 16]
                        # tmp = [256, 128]
                        for TILEY in tmp:
                            if found_hyperparameters:
                                break
                            for yy in [4, 3, 2, 1]:
                                if found_hyperparameters:
                                    break
                                TY = yy * vsize
                                if TILEY % TY != 0:
                                    continue
                                y = TILEY // TY
                                for TILEX in tmp:
                                    if found_hyperparameters:
                                        break
                                    for xx in [4, 3, 2, 1]:
                                        if found_hyperparameters:
                                            break
                                        TX = xx * vsize
                                        if TILEX % TX != 0:
                                            continue
                                        x = TILEX // TX
                                        if (x * y) > 1024:
                                            continue
                                        for k in range(16, 0, -1):
                                            TILEK = k * vsize
                                            if TILEK > 64:
                                                continue
                                            if TILEK > TILEX or TILEK > TILEY:
                                                continue
                                            if (vsize * x * y) % TILEX != 0:
                                                continue
                                            if (vsize * x * y) % TILEY != 0:
                                                continue
                                            stride_values = (vsize * x * y) // TILEX
                                            stride_input = (vsize * x * y) // TILEY
                                            if TILEK % stride_input != 0:
                                                continue
                                            if TILEK % stride_values != 0:
                                                continue
                                            smem = dtype.itemsize * 2 * (TILEY * TILEK + TILEK * TILEX)
                                            if smem > 49152:
                                                continue
                                            if TILEX > b:
                                                continue
                                            if ((b * d) % (d * TILEX)) != 0:
                                                continue
                                            if TILEK > c:
                                                continue
                                            if (c % TILEK) != 0:
                                                continue
                                            
                                            lines_to_add = add_lines_to_instantiate_kernel_with_hp(lines_to_add, kernel_name, target_batch_size, a, b, c, d, precision, TILEX, TILEK, TILEY, TX, TY, VSIZE)

                                            found_hyperparameters = True
                                            print(f"Found default hyper-parameters for {target_batch_size},{a},{b},{c},{d}, bs last = {bsl}, {precision}, vsize = {VSIZE} OK.")
                                            break
                                        if found_hyperparameters:
                                            break
                                    if found_hyperparameters:
                                        break
                                if found_hyperparameters:
                                    break
                            if found_hyperparameters:
                                break

                    if not found_hyperparameters:
                        raise Exception(
                            f"ERROR: Could not find hyperparameters for the kernel instance corresponding to the target (batch_size, a, b, c, d, precision, VSIZE) = ({target_batch_size}, {a}, {b}, {c}, {d}, {precision}, {vsize}) in the .best file, nor could we find default compatible ones. Please suggest new hyperparameters in the .best file or modify the code to add a new kernel instance with the desired hyperparameters."
                        )
                    # we will write the new kernel instances at the top of the .cuh file, after the `threadsPerBlock.y = 1;` line
                    try:
                        idx = file_lines.index("\t\tthreadsPerBlock.y = 1;\n")
                    except ValueError:
                        print(file_lines)
                        raise Exception(f"ERROR: threadsPerBlock.y = 1; not found in the {filename} file. Please check the file content.")


                    # Insert the new lines after the `threadsPerBlock.y = 1;` line{
                    new_code = file_lines[:idx + 1] + lines_to_add + file_lines[idx + 1:]
                    # Write the new lines to the file
                    with open(filename, "w") as out_file:
                        out_file.writelines(new_code)

            # Now let's compile the kernel with the new instances
            
            path = directory + "/kernel"

            # Check if the build directory exists, if not create it
            if not os.path.exists("build/"):
                os.makedirs("build/")
                warn("Created 'build/' directory to store compilation files for the kernel.", UserWarning)


            # load() is lazy and will not re-compile if the .cu and .cpp have not changed, even though we changed the .cuh file above
            # so change the timestamp of the .cu to force recompilation
            cpp_path = path + ".cpp"
            # Read the current content of the file.
            with open(cpp_path, "r") as f:
                content = f.read()

            # Define a dummy comment that you will add.
            dummy_comment = "\n// Force rebuild dummy comment\n"

            # Check if the dummy comment is already present.
            # You can design this toggle mechanism in a couple of ways:
            if "// Force rebuild dummy comment" in content:
                # Option 1: Remove the dummy comment if already present.
                new_content = content.replace(dummy_comment, "")
            else:
                # Option 2: Append the dummy comment to force a rebuild.
                new_content = content + dummy_comment

            # Write back the modified content.
            with open(cpp_path, "w") as f:
                f.write(new_content)

            kernel = load(
                name="kernel",
                sources=[path + ".cpp", path + ".cu"],
                verbose=True,
                extra_cflags=["-g"],
                build_directory="build/",
            )
            ksmm_compiled_kernel[name] = kernel

        # else:
        #     # Do nothing, kernel has been compiled previously.
        #     raise Exception("WARNING: do nothing, kernel has been compiled previously.")

        output = kernel.forward(
            input,
            factor,
            bs_last,
            a,
            b,
            c,
            d,
            str(dtype),
            vsize,
        )

        return output

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        pass

    @staticmethod
    def backward(ctx, grad_output):
        input, matrix, output = ctx.saved_tensors
        grad_input, grad_rest = kernel.backward(
            grad_output, input, matrix
        )
        return grad_input, *grad_rest
