#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import gc
import getopt
import json
import numpy as np
import sys
import time


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="file name", default='tuning/kernel_bs_first_float4.out', type=str)
    args = parser.parse_args()
    name = args.n

    if 'float' in name:
        dtype = np.dtype('float32')
        precision = 'float'
    if 'half' in name:
        dtype = np.dtype('float16')
        precision = 'half'
    if 'bs_first' in name:
        bs_last = 0
    if 'bs_last' in name:
        bs_last = 1

    fname = name.replace('.out', '.cuh')
    kname = name.replace('tuning/', '').replace('.out', '')

    with open(name.replace('.out', '.best'), 'w') as best_out:
        best_out.write("batch_size a b c d TILEX TILEK TILEY TX TY VSIZE ms std mse\n")

    with open(fname, "w") as out_file:
        out_file.write("// -*- c -*-\n\n")
        out_file.write("#ifndef {0:s}\n".format(kname.upper()))
        out_file.write("#define {0:s}\n\n".format(kname.upper()))
        out_file.write('#include "template_kernels_{0:s}.cuh"\n\n'.format(precision))
        out_file.write("void best_{0:s}({1:s} *input, {1:s} *values, {1:s} *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){{\n".format(kname, precision))
        out_file.write("\twhile (1) {\n")
        out_file.write("\t\tthreadsPerBlock.y = 1;\n")

    # Read patterns.
    patterns = []
    with open("patterns.in", "r") as fpatterns:
        lines = fpatterns.readlines()
        for l in lines:
            sl = l.split(',')
            a, b, c, d, batch_size = int(sl[0]), int(sl[1]), int(sl[2]), int(sl[3]), int(sl[4])
            patterns.append((a, b, c, d, batch_size))

    for p in patterns:
        a, b, c, d, batch_size = p
        with open(name, "r") as in_file:
            # batch_size a b c d TILEX TILEK TILEY TX TY VSIZE ms std mse
            lines = in_file.readlines()
            tmin = 1e9
            std_min = 1e9
            mse_min = 1e9
            find_min = False
            for line in lines:
                sl = line.split()
                if sl[0] == 'batch_size':
                    continue
                if (int(sl[0]) == batch_size
                    and int(sl[1]) == a
                    and int(sl[2]) == b
                    and int(sl[3]) == c
                    and int(sl[4]) == d):
                    if float(sl[11]) < tmin:
                        TILEX = int(sl[5])
                        TILEK = int(sl[6])
                        TILEY = int(sl[7])
                        TX = int(sl[8])
                        TY = int(sl[9])
                        VSIZE = int(sl[10])
                        tmin = float(sl[11])
                        std_min = float(sl[12])
                        mse_min = float(sl[13])
                        find_min = True
            if tmin != 0.0 and find_min:
                with open(name.replace('.out', '.best'), 'a') as best_out:
                    best_out.write(f"{batch_size} {a} {b} {c} {d} {TILEX} {TILEK} {TILEY} {TX} {TY} {VSIZE} {tmin} {std_min} {mse_min}" + "\n")
                input_size = a * c * d
                output_size = a * b * d
                x = TILEX // TX
                with open(
                        fname, "a"
                ) as out_file:
                    out_file.write(
                        "\t\tif (batch_size == {0:d} && a == {1:d} && b == {2:d} && c == {3:d} && d == {4:d}) {{\n".format(
                            batch_size, a, b, c, d
                        )
                    )
                    out_file.write(
                        "\t\t\tthreadsPerBlock.x = {0:d};\n".format(
                            (TILEX // TX) * (TILEY // TY)
                        )
                    )
                    out_file.write(
                        "\t\t\tblockGrid.x = {0:d};\n".format(
                            (output_size + TILEX - 1) // TILEX
                        )
                    )
                    out_file.write(
                        "\t\t\tblockGrid.y = {0:d};\n".format(
                            (batch_size + TILEY - 1) // TILEY
                        )
                    )
                    if precision == 'half' and VSIZE == 4:
                        out_file.write(
                            "\t\t\t{0:s}4<{1:s}, {2:d}, {3:d}, {4:d}, {5:d}, {6:d}, {7:d}, false, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:d}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
                                kname, 'bool', TILEX, TILEK, TILEY, TX, TY, VSIZE, x, batch_size, 'a', 'b', 'c', 'd'
                            )
                        )
                    elif precision == 'half' and VSIZE == 2:
                        out_file.write(
                            "\t\t\t{0:s}2<{1:s}, {2:d}, {3:d}, {4:d}, {5:d}, {6:d}, {7:d}, true, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:d}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
                                kname, precision + str(VSIZE), TILEX, TILEK, TILEY, TX, TY, VSIZE, x, batch_size, 'a', 'b', 'c', 'd'
                            )
                        )
                    else:
                        out_file.write(
                            "\t\t\t{0:s}<{1:s}, {2:d}, {3:d}, {4:d}, {5:d}, {6:d}, {7:d}, true, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:d}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
                                kname, precision + str(VSIZE), TILEX, TILEK, TILEY, TX, TY, VSIZE, x, batch_size, 'a', 'b', 'c', 'd'
                            )
                        )
                    out_file.write("\t\t\tbreak;\n")
                    out_file.write("\t\t}\n")


    # Default
    with open(
            fname, "a"
    ) as out_file:
        VSIZE = 4
        tmp = [128, 64]
        for TILEY in tmp:
            for yy in [4, 3, 2, 1]:
                TY = yy * VSIZE
                if TILEY % TY != 0:
                    continue
                y = TILEY // TY
                for TILEX in tmp:
                    for xx in [4, 3, 2, 1]:
                        TX = xx * VSIZE
                        if TILEX % TX != 0:
                            continue
                        x = TILEX // TX
                        if (x * y) > 1024:
                            continue
                        for k in range(16, 0, -1):
                            TILEK = k * VSIZE
                            if TILEK > 64:
                                continue
                            if TILEK > TILEX or TILEK > TILEY:
                                continue
                            if (VSIZE * x * y) % TILEX != 0:
                                continue
                            if (VSIZE * x * y) % TILEY != 0:
                                continue
                            stride_values = (VSIZE * x * y) // TILEX
                            stride_input = (VSIZE * x * y) // TILEY
                            if TILEK % stride_input != 0:
                                continue
                            if TILEK % stride_values != 0:
                                continue
                            smem = dtype.itemsize * 2 * (TILEY * TILEK + TILEK * TILEX)
                            if smem >= 49152:
                                continue
                            # input_size = a * c * d
                            # output_size = a * b * d
                            out_file.write(
                                "\t\tif ((batch_size % {2:d}) == 0 && b > {0:d} && ((b * d) % (d * {0:d})) == 0 && c > {1:d} && ((c * d) % (d * {1:d})) == 0) {{\n".format(
                                    TILEX, TILEK, TILEY
                                )
                            )
                            out_file.write(
                                "\t\t\tthreadsPerBlock.x = {0:d};\n".format(x * y)
                            )
                            out_file.write(
                                "\t\t\tblockGrid.x = (a * b * d + {0:d} - 1) / {0:d};\n".format(TILEX)
                            )
                            out_file.write(
                                "\t\t\tblockGrid.y = (batch_size + {0:d} - 1) / {0:d};\n".format(TILEY)
                            )
                            if precision == 'half' and VSIZE == 4:
                                out_file.write(
                                    "\t\t\t{0:s}4<{1:s}, {2:d}, {3:d}, {4:d}, {5:d}, {6:d}, {7:d}, false, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:s}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
                                        kname, 'bool', TILEX, TILEK, TILEY, TX, TY, VSIZE, x, 'batch_size', 'a', 'b', 'c', 'd'
                                    )
                                )
                            if precision == 'half' and VSIZE == 2:
                                out_file.write(
                                    "\t\t\t{0:s}2<{1:s}, {2:d}, {3:d}, {4:d}, {5:d}, {6:d}, {7:d}, true, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:s}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
                                        kname, precision + str(VSIZE), TILEX, TILEK, TILEY, TX, TY, VSIZE, x, 'batch_size', 'a', 'b', 'c', 'd'
                                    )
                                )
                            else:
                                out_file.write(
                                    "\t\t\t{0:s}<{1:s}, {2:d}, {3:d}, {4:d}, {5:d}, {6:d}, {7:d}, true, {8:d}><<<blockGrid, threadsPerBlock>>>(input, values, {9:s}, output, {10:s}, {11:s}, {12:s}, {13:s});\n".format(
                                        kname, precision + str(VSIZE), TILEX, TILEK, TILEY, TX, TY, VSIZE, x, 'batch_size', 'a', 'b', 'c', 'd'
                                    )
                                )
                            out_file.write("\t\t\tbreak;\n")
                            out_file.write("\t\t}\n")


    with open(fname, "a") as out_file:
        out_file.write("\t\tassert(1 == 0);\n")
        out_file.write("\t\tbreak;\n")
        out_file.write("\t}\n")
        out_file.write("}\n\n")
        out_file.write("#endif\n")


if __name__ == "__main__":
    main(sys.argv[1:])
