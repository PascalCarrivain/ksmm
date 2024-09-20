batch_size=128
min_run_time=10
device=cuda
saving_dir=results/2_benchmark_ksvit_s_16/
bs_last=0
sdpa_version=default
split_qkv=1

arch="simple_vit_s16_in1k"
dim=384
hidden_dim=$((dim * 4))

# Get the GPU device id from user input (default = 0)
if [[ $1 == "" ]]; then
	echo "By default, the GPU device id is set to 0."
	echo "If you want to change it, please run the script with the desired device id as an argument."
	device_id=0
else
	device_id=$1
fi

for precision in "fp32" "fp16"; do
    for granularity in "linear" "linear_bias" "linear_down" "linear_down_bias" "linear_up" "linear_up_bias" "ffn" "ffn_residual" "attention" "attention_residual" "block" "vit"; do
        # benchmark dense algorithm with usual dense matrices
        for algo in "nn_linear"; do
            python3 src/ksmm_py/benchmark/ksvit_time.py \
                --arch $arch \
                --granularity $granularity \
                --patterns "1,${dim},${dim},1" \
                --patterns_up "1,${hidden_dim},${dim},1" \
                --patterns_down "1,${dim},${hidden_dim},1" \
                --precision $precision \
                --bs-last $bs_last \
                --batch-size $batch_size \
                --min-run-time $min_run_time \
                --saving-dir $saving_dir \
                --device $device \
                --device-id $device_id \
                --algo $algo \
                --sdpa-version $sdpa_version \
                --split-qkv $split_qkv
        done

        # benchmark bmm and kernel algorithms replacing dense matrices with Kronecker-sparse layers
        for algo in "bmm" "kernel"; do
            python3 src/ksmm_py/benchmark/ksvit_time.py \
                --arch $arch \
                --granularity $granularity \
                --patterns "2,48,192,1" "1,192,48,2" \
                --patterns_up "6,64,64,1" "1,768,192,2" \
                --patterns_down "6,64,256,1" "1,128,128,3" \
                --precision $precision \
                --bs-last $bs_last \
                --batch-size $batch_size \
                --min-run-time $min_run_time \
                --saving-dir $saving_dir \
                --device $device \
                --device-id $device_id \
                --algo $algo \
                --sdpa-version $sdpa_version \
                --split-qkv $split_qkv

        done
    done
done
