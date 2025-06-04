#!/bin/bash

to_pytorch=../ksmm_py/layer/kronecker_sparse/;

for f in kernel_bs_first_float.out\
	     kernel_bs_last_float.out\
	     kernel_bs_first_half.out\
	     kernel_bs_last_half.out;
do
    python3 generate_best_from_fine_tuning.py -n tuning/${f};
    tmp=$(echo ${f} | sed s/".out"//);
    # echo ${tmp}".cuh";
    # cp tuning/${tmp}".cuh" ${to_pytorch};
    echo ${tmp}".best";
    cp tuning/${tmp}".best" ${to_pytorch};
done;
