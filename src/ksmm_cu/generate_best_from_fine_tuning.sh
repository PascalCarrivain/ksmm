#!/bin/bash

to_pytorch=../fkmd/layer/kronecker_sparse/;

for f in kernel_bs_first_float4.out\
	     kernel_bs_first_half2.out\
	     kernel_bs_last_float4.out\
	     kernel_bs_last_half2.out;
do
    python3 print_best.py -n tuning/${f};
    tmp=$(echo ${f} | sed s/".out"//);
    echo ${tmp}".cuh";
    cp tuning/${tmp}".cuh" ${to_pytorch};
done;

# cp src/template_kernels_half2.cuh ${to_pytorch};
# cp src/template_kernels_float4.cuh ${to_pytorch};
