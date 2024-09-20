#!/bin/bash

############ To change if you run your own benchmark and don't want to erase the furnished benchmark ############
saving_csv=results/0_ksmm_time_your_own_benchmark.csv

# Get the GPU device id from user input (default = 0)
if [[ $1 == "" ]]; then
	echo "By default, the GPU device id is set to 0."
	echo "If you want to change it, please run the script with the desired device id as an argument."
	device_id=0
else
	device_id=$1
fi

############# To change if you want to run the benchmark in another configuration ############
device=cuda
batch_size=25088

# to run with all chains:
a_list=(1 2 3 4 6 8 12 16 24 32 48 64 96 128)
b_list=(48 64 96 128 192 256 384 512 768 1024)
c_list=(48 64 96 128 192 256 384 512 768 1024)
d_list=(4 16 64)
# d_list=(1 2 3 4 6 8 12 16 24 32 48 64 96 128)

# # to run with a single chain, uncomment what follows with your chain
# a_list=(1)
# b_list=(48)
# c_list=(192)
# d_list=(4)

# We make sure that the kernel is recompiled before running the benchmark
if [[ -d "build" ]]; then
	rm -f build/*
else
	mkdir build
fi
# Compile the kernel before running the benchmark
python3 src/ksmm_py/layer/kronecker_sparse/kernel.py

for bs_last in 0 1; do
	for precision in fp32 fp16; do
		for a in "${a_list[@]}"; do
			for b in "${b_list[@]}"; do
				for c in "${c_list[@]}"; do
					for d in "${d_list[@]}"; do
						input_size=$((${a} * ${c} * ${d}))
						output_size=$((${a} * ${b} * ${d}))
						for algo in "kernel" "bmm" "bsr" "dense" "sparse" "einsum"; do
							# Skip some patterns to keep the benchmark time reasonable, see appendix B.1 of the paper for more details
							# skip if (b == 1024 and c == 256) or (b == 256 and c == 1024):
							if [[ ${b} -eq 1024 ]] && [[ ${c} -eq 256 ]]; then
								continue
							fi
							if [[ ${b} -eq 256 ]] && [[ ${c} -eq 1024 ]]; then
								continue
							fi
							# skip if (b == 128 and c == 512) or (b == 512 and c == 128):
							if [[ ${b} -eq 128 ]] && [[ ${c} -eq 512 ]]; then
								continue
							fi
							if [[ ${b} -eq 512 ]] && [[ ${c} -eq 128 ]]; then
								continue
							fi
							# skip if (b == 64 and c == 256) or (b == 256 and c == 64):
							if [[ ${b} -eq 64 ]] && [[ ${c} -eq 256 ]]; then
								continue
							fi
							if [[ ${b} -eq 256 ]] && [[ ${c} -eq 64 ]]; then
								continue
							fi

							# Skip patterns whose associated matrix sizes are greater than int max (2^31 - 1)
							size=$((${batch_size} * ${input_size}))
							if [[ ${size} -ge 2147483647 ]]; then
								continue
							fi
							size=$((${batch_size} * ${output_size}))
							if [[ ${size} -ge 2147483647 ]]; then
								continue
							fi
							size=$((${a} * ${b} * ${c} * ${d}))
							if [[ ${size} -ge 2147483647 ]]; then
								continue
							fi

							# Skip sparsity patterns with b !=c and b != 4c and c != 4b, see appendix B.1 of the paper for more details
							if [[ ${b} -ne ${c} ]] && [[ ${b} -ne $((4 * ${c})) ]] && [[ ${c} -ne $((4 * ${b})) ]]; then
								continue
							fi

							pattern="${a},${b},${c},${d}"

							python3 src/ksmm_py/benchmark/ksmm_time.py \
								--saving-csv $saving_csv \
								--algo $algo \
								--device $device \
								--device-id $device_id \
								--precision $precision \
								--batch-size $batch_size \
								--patterns ${pattern} \
								--bs-last ${bs_last} \
								--correctness
						done
					done
				done
			done
		done
	done
done
