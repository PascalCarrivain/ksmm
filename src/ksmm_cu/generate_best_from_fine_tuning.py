#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import gc
import getopt
import json
import numpy as np
import sys
import time


a = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
b = [48, 64, 80, 96, 128, 192, 256, 384, 512, 640, 768, 1024]
c = [48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024]
d = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="file name", default='tuning/kernel_bs_first_float4.out', type=str)
    args = parser.parse_args()
    name = args.n

    if 'float4' in name:
        fp = 'float4'
    if 'half2' in name:
        fp = 'half2'
    index = -1
    if 'kernel0' in name:
        index = 0
    if 'kernel1' in name:
        index = 1
    if 'bs_first' in name:
        bs_last = 0
    if 'bs_last' in name:
        bs_last = 1
    vtype = fp.replace('float4', 'float').replace('half2', 'half')

    fname = name.replace('.out', '.cuh')
    kname = name.replace('tuning/', '').replace('_factor0', '').replace('_factor1', '').replace('.out', '')

    with open(fname, "w") as out_file:
        out_file.write("// -*- c -*-\n\n")
        out_file.write("#ifndef {0:s}\n".format(kname.upper()))
        out_file.write("#define {0:s}\n\n".format(kname.upper()))
        out_file.write('#include "template_kernels_{0:s}.cuh"\n\n'.format(fp))
        # out_file.write("template <const int TILEX, const int TILEK, const int TILEY, const int TX, const int TY>\n")
        # # out_file.write("__global__ __launch_bounds__(xNTHREADS_FLOAT4x) void {0:s}(\n".format(kname))
        # out_file.write("__global__ void {0:s}(\n".format(kname))
        # out_file.write("{0:s} *input, {0:s} *values, const int output_size, const int batch_size,\n".format(vtype))
        # out_file.write("const int input_size, {0:s} *output, const int {1:s});\n\n".format(vtype, 'a' if index == 0 else 'd'))
        out_file.write("void best_{0:s}({1:s} *input, {1:s} *values, {1:s} *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){{\n".format(kname, vtype))
        out_file.write("\twhile (1) {\n")
        out_file.write("\t\tthreadsPerBlock.y = 1;\n")

    with open(name, "r") as in_file:
        # batch_size a b c d TILEX TILEK TILEY TX TY WX WY CUDA time (in ms) std mse
        lines = in_file.readlines()
        for i in a:
            if "kernel1" in name and i > 1:
                continue
            for j in b:
                for k in c:
                    for l in d:
                        if "kernel0" in name and l > 1:
                            continue
                        for batch_size in [196, 25088]:
                            tmin = 1e9
                            find_min = False
                            offset = 1
                            for line in lines:
                                sl = line.split()
                                if sl[0] == 'batch_size':
                                    continue
                                if (
                                    int(sl[0]) == batch_size
                                    and int(sl[1]) == i
                                    and int(sl[2]) == j
                                    and int(sl[3]) == k
                                    and int(sl[4]) == l
                                ):
                                    if float(sl[13 - offset]) < tmin:
                                        TILEX = int(sl[6 - offset])
                                        TILEK = int(sl[7 - offset])
                                        TILEY = int(sl[8 - offset])
                                        TX = int(sl[9 - offset])
                                        TY = int(sl[10 - offset])
                                        tmin = float(sl[13 - offset])
                                        find_min = True
                            if tmin != 0.0 and find_min:
                                input_size = i * k * l
                                output_size = i * j * l
                                with open(
                                    fname, "a"
                                ) as out_file:
                                    out_file.write(
                                        "\t\tif (batch_size == {0:d} && a == {1:d} && b == {2:d} && c == {3:d} && d == {4:d}) {{\n".format(
                                            batch_size, i, j, k, l
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
                                    if index == -1:
                                        out_file.write(
                                            "\t\t\t{5:s}<{0:d}, {1:d}, {2:d}, {3:d}, {4:d}><<<blockGrid, threadsPerBlock>>>(input, values, {6:d}, output, {7:s}, {8:s}, {9:s}, {10:s});\n".format(
                                                TILEX, TILEK, TILEY, TX, TY, kname, batch_size, 'a', 'b', 'c', 'd'
                                            )
                                        )
                                    else:
                                        out_file.write(
                                            "\t\t\t{5:s}<{0:d}, {1:d}, {2:d}, {3:d}, {4:d}><<<blockGrid, threadsPerBlock>>>(input, values, {6:d}, {7:d}, {8:d}, output, {9:s});\n".format(
                                                TILEX, TILEK, TILEY, TX, TY, kname, output_size, batch_size, input_size, 'a' if index == 0 else 'd'
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
