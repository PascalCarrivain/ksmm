#!/bin/bash

if [[ ${5} == "" ]] && [[ ${6} == "" ]];
then
    echo "./long_run.sh kernel_bs_first_float 0 hyper_params64.16.32.8.8.4 100 test_patterns.in no_nvprof";
    exit -1;
fi;

rm -f ${1}.out;
rm -f log.out;
rm -f assert_failed*.out;

echo "batch_size a b c d TILEX TILEK TILEY TX TY VSIZE" > wrong_mse_${1}.out;

if [[ ${1} == "" ]];
then
    kernel_name="kernel_bs_first_float";
else
    kernel_name=${1};
fi;

if [[ ${2} == "" ]];
then
    device_id=0;
else
    device_id=${2};
fi;

if [[ ${3} == *"sanity_check"* ]];
then
    sanity_check=1;
else
    sanity_check=0;
fi;

if [[ ${3} == *"full"* ]];
then
    sanity_check=0;
fi;

if [[ ${3} == "hyper_params"* ]];
then
    sanity_check=0;
    hyper_params=1;
    tmp=${3}
    tmp=${tmp//hyper_params/};
    readarray -d "." -t out <<< "${tmp}"
else
    hyper_params=0;
    out=(0 0 0 2 2 2);
fi;

if [[ ${4} == "" ]];
then
    nrepeats=100;
else
    nrepeats=${4};
fi;

if [[ ${kernel_name} == *"cublas"* ]];
then
    vsize=4;
    kernel="cublas";
    if [[ ${kernel_name} == *"cublas_factor0"* ]];
    then
	kernel=0;
    fi;
    if [[ ${kernel_name} == *"cublas_factor1"* ]];
    then
	kernel=1;
    fi;
    # sizeof as a function of the precision
    if [[ ${kernel_name} == *"fp32"* ]];
    then
	hp=0;
	fp=4;
    else
	hp=1;
	fp=2;
    fi;
    kernel=0;
    echo "batch_size a b c d ms std mse" > ${kernel_name}.out;
else
    # Check kernel index from kernel name
    if [[ ${kernel_name} == *"_tc"* ]];
    then
	# Tensor-cores
	# Header of the file where to store best hyper-parameters
	echo "batch_size a b c d TILEX TILEK TILEY TX TY nwarpsX nwarpsY ms std mse" > ${kernel_name}.out;
	# sizeof as a function of the precision
	hp=1;
	fp=2;
	vsize=2;
    else
	# Home-made kernel
	# Header of the file where to store best hyper-parameters
	echo "batch_size a b c d TILEX TILEK TILEY TX TY VSIZE ms std mse" > ${kernel_name}.out;
	if [[ ${kernel_name} == *"float"* ]];
	then
	    # sizeof as a function of the precision
	    hp="FLOAT";
	    fp=4;
	else
	    if [[ ${kernel_name} == *"half"* ]];
	    then
		# sizeof as a function of the precision
		hp="HALF";
		fp=2;
	    else
		if [[ ${kernel_name} == *"e4m3"* ]];
		then
		    # sizeof as a function of the precision
		    hp="FP8E4M3";
		    fp=1;
		    vsize=4;
		fi;
		if [[ ${kernel_name} == *"e5m2"* ]];
		then
		    # sizeof as a function of the precision
		    hp="FP8E5M2";
		    fp=1;
		    vsize=4;
		fi;
	    fi;
	fi;
    fi;
fi;


# Loop over tuple (a,b,c,d)
while IFS='' read -r line;
do
    readarray -d "," -t abcd <<< "${line},";
    A=${abcd[0]};
    B=${abcd[1]};
    C=${abcd[2]};
    D=${abcd[3]};
    batch=${abcd[4]};
    if [[ $6 == "nvprof" ]];
    then
	if [[ ${kernel_name} == *"cublas"* ]];
	then
            echo "pattern $A,$B,$C,$D,$batch" > nvprof_cublas.out;
	else
            echo "pattern $A,$B,$C,$D,$batch" > nvprof.out;
	fi;
    fi;
    # Multiplication X @ K^T
    # X in R^{batch x (a * c * d)}
    # K in R^{a * b * d, a * c * d}
    is=$((${A}*${C}*${D}));
    os=$((${A}*${B}*${D}));
    # # Check required memory
    # mem=$((${fp}*${batch}*${is}+2*${fp}*${batch}*${os}+${fp}*${B}*${os}));
    # if [[ ${mem} -gt 20000000000 ]];
    # then
    #     continue;
    # fi;
    # echo ${fp} ${batch} ${is} $((${fp}*${batch}*${is}));
    # Check size of the arrays
    length=$((${batch}*${is}));
    if [[ ${length} -ge 2147483647 ]];
    then
	continue;
    fi;
    length=$((${batch}*${os}));
    if [[ ${length} -ge 2147483647 ]];
    then
	continue;
    fi;
    length=$((${A}*${B}*${C}*${D}));
    if [[ ${length} -ge 2147483647 ]];
    then
	continue;
    fi;
    run="false";
    cublas=0;
    for vsize in 2 4;#1 2 3 4;
    do
	# TILEX is x * TX
	for x in {1..64};
	do
	    # sub-tile along X
	    for TX in {1..16};
	    do
		# Sub-tile must be greater or equal to vsize
		if [[ ${TX} -lt ${vsize} ]] || [[ $((${TX}%${vsize})) -ne 0 ]];
		then
		    continue;
		fi;
		# Size of the tile along X
		TILEX=$((${x}*${TX}));
		# D * TILEX must divide number of columns of a super-block
		if [[ $(($((${B}*${D}))%$((${D}*${TILEX})))) -ne 0 ]];
		then
		    continue;
		fi;
		# TILEK is k * vsize
		for k in {1..16};
		do
		    # Size of the tile along K
		    TILEK=$((${k}*${vsize}));
		    # # D * TILEK must divide C * D
		    # if [[ $(($((${C}*${D}))%$((${D}*${TILEK})))) -ne 0 ]];
		    # TILEK must divide C
		    if [[ $((${C}%${TILEK})) -ne 0 ]];
                    then
			continue;
      	            fi;
		    # TILEY is y * TY
		    for y in {1..64};
		    do
			# stride(values) = vsize * nthreads / TILEX
			# stride(values) = vsize * y / TX
			strideValues=$(($((${vsize}*${x}*${y}))/${TILEX}));
			if [[ $(($((${vsize}*${x}*${y}))%${TILEX})) -ne 0 ]];
			then
			    continue;
			fi;
			# sub-tile along Y
			for TY in {1..16};
			do
			    # Sub-tile must be greater or equal to vsize
			    if [[ ${TY} -lt ${vsize} ]] || [[ $((${TY}%${vsize})) -ne 0 ]];
			    then
				continue;
			    fi;
			    # Size of the tile along Y
			    TILEY=$((${y}*${TY}));
			    # Check only one set of hyper parameters.
			    if [[ ${hyper_params} -eq 1 ]] && ([[ ${TILEX} -ne ${out[0]} ]] || [[ ${TILEK} -ne ${out[1]} ]] || [[ ${TILEY} -ne ${out[2]} ]] || [[ ${TX} -ne ${out[3]} ]] || [[ ${TY} -ne ${out[4]} ]] || [[ ${vsize} -ne ${out[5]} ]]);
			    then
				continue;
			    fi;
			    # stride(input) = vsize * nthreads / TILEY
			    # stride(input) = vsize * x / TY
			    strideInput=$(($((${vsize}*${x}*${y}))/${TILEY}));
			    if [[ $(($((${vsize}*${x}*${y}))%${TILEY})) -ne 0 ]];
			    then
				continue;
			    fi;
			    # Check TILEK <= TILEX, TILEY
			    if [[ ${TILEK} -gt ${TILEX} ]] || [[ ${TILEK} -gt ${TILEY} ]];
			    then
				continue;
			    fi;
			    # If sanity check, test small number of hyper-parameters only.
			    # See the following conditions for more details
			    if [[ ${sanity_check} -eq 1 ]];
			    then
				if [[ ${TILEY} -ne 32 ]] && [[ ${TILEY} -ne 64 ]] && [[ ${TILEY} -ne 128 ]] && [[ ${TILEY} -ne 256 ]];
				then
				    continue;
				fi;
				if [[ ${B} -eq 48 ]] || [[ ${C} -eq 48 ]];
				then
                                    TILEXs=".12.16.24.48.";
                                    TILEKs=".12.16.24.";
				else
				    if [[ ${B} -eq 80 ]] || [[ ${C} -eq 80 ]];
                                    then
					TILEXs=".8.10.20.40.";
					TILEKs=".8.10.16.20.";
                                    else
					if [[ ${B} -eq 96 ]] || [[ ${C} -eq 96 ]];
					then
                                            TILEXs=".32.48.64.";
                                            TILEKs=".16.32.48.";
					else
                                            if [[ ${B} -eq 160 ]] || [[ ${C} -eq 160 ]];
                                            then
						TILEXs=".16.20.40.60.80.";
						TILEKs=".16.20.40.";
                                            else
						TILEXs=".32.64.128.256.";
						TILEKs=".16.32.64.128.";
                                            fi;
					fi;
                                    fi;
				fi;
				if [[ ${TILEXs} != *".${TILEX}."* ]] || [[ ${TILEKs} != *".${TILEK}."* ]];
				then
                                    continue;
				fi;
			    fi;
			    # Number of warps along x (tensor-cores only)
			    for nwx in 1 2 4 8 16;
			    do
				# if [[ $((nwx*16)) -ne ${TILEX} ]];
				# then
				# 	continue;
				# fi;
				# Number of warps along y (tensor-cores only)
				for nwy in 1 2 4 8 16;
				do
				    # if [[ $((nwy*16)) -ne ${TILEY} ]];
				    # then
				    #     continue;
				    # fi;
				    # Tensor cores: check number of threads
				    # Check if the number of threads is equal to the number
				    # of warps times warp size (only for tensor cores).
				    # nthreads = nwx * nwy * 32
				    # nthreads = x * y
				    if [[ ${kernel_name} == *"_tc"* ]];
				    then
					# Store input
					if [[ $((${TILEY}*${TILEK})) -ne $((${nwx}*${nwy}*256)) ]];
					then
					    continue;
					fi;
					# Store values
					if [[ $((${TILEK}*${TILEX})) -ne $((${nwx}*${nwy}*256)) ]];
					then
					    continue;
					fi;
					# Store output
					if [[ $((${TILEY}*${TILEX})) -ne $((${nwx}*${nwy}*256)) ]];
					then
					    continue;
					fi
					if [[ $((${nwx}*${nwy}*32)) -ne $((${x}*${y})) ]];
					then
					    continue;
					fi;
				    fi;
				    # If not tensor-cores no need to run many number of warps values
				    if [[ ${kernel_name} != *"_tc"* ]] && ([[ ${nwx} -ne 1 ]] || [[ ${nwy} -ne 1 ]]);
				    then
					continue;
				    fi;
				    # TILEY must divide batch size
				    if [[ $((${batch}%${TILEY})) -ne 0 ]];
				    then
					continue;
				    fi;
				    # Compute shared memory (double-buffering)
				    # smem=$((${fp}*$((2*${TILEY}*${TILEK}+2*${TILEK}*${TILEX}+${TILEY}*${TILEX}))));
				    smem=$((${fp}*$((2*${TILEY}*${TILEK}+2*${TILEK}*${TILEX}))));
				    # Check shared memory
				    if [[ ${smem} -ge 49152 ]];
				    then
					continue;
				    fi;
				    # Check number of threads
				    if [[ $((${x}*${y})) -gt 1024 ]];
				    then
					continue;
				    fi;
				    # strides must divide TILEK
				    if [[ $((${TILEK}%${strideInput})) -ne 0 ]] || [[ $((${TILEK}%${strideValues})) -ne 0 ]];
				    then
					continue;
				    fi;
				    # Replace all the xNAMEx by a constant value
				    # All the template_*.cuh files become *.cuh
				    for f in "sparse_mm" "kernels_float" "kernels_half" "kernels_half8" "kernels_fp8" "kernels_tc";
				    do
					if [[ ${f} == "sparse_mm" ]];
					then
					    new_f=src/${f}.cu;
					    cp src/template_${f}.cu ${new_f};
					else
					    new_f=src/${f}.cuh;
					    cp src/template_${f}.cuh ${new_f};
					fi;
					# Add a define directive to handle #include kernels_*.cuh
					if [[ ${kernel_name} == *"_tc"* ]];
					then
					    sed -i "s/defineFP/define TC/g" ${new_f};
					else
					    sed -i "s/defineFP/define ${hp}/g" ${new_f};
					fi;
					sed -i -e "s/xBATCHSIZEx/${batch}/g" \
					    -e "s/xINPUTSIZEx/${is}/g" \
					    -e "s/xOUTPUTSIZEx/${os}/g" \
					    -e "s/xax/${A}/g" \
					    -e "s/xbx/${B}/g" \
					    -e "s/xcx/${C}/g" \
					    -e "s/xdx/${D}/g" \
					    -e 's/xWMMA_Yx/16/g' \
					    -e 's/xWMMA_Xx/16/g' \
					    -e 's/xWMMA_Kx/16/g' \
					    -e "s/xNWARPSYx/${nwx}/g" \
					    -e "s/xNWARPSXx/${nwy}/g" \
					    -e "s/xVSIZEx/${vsize}/g" \
					    -e "s/xTILEXx/${TILEX}/g" \
					    -e "s/xTILEKx/${TILEK}/g" \
					    -e "s/xTILEYx/${TILEY}/g" \
					    -e "s/xTXx/${TX}/g" \
					    -e "s/xTYx/${TY}/g" \
					    -e "s/xXx/${x}/g" \
					    -e "s/xYx/${y}/g" \
					    -e "s/xNTHREADSx/$((x*y))/g" \
					    -e "s/xCUBLAS_GEMM_ALGOx/CUBLAS_GEMM_DEFAULT/g" \
					    -e "s/xCUBLAS_GEMM_ALGO_TENSOR_OPx/CUBLAS_GEMM_DEFAULT_TENSOR_OP/g" ${new_f};
				    done;
				    # Find arch depending on the GPU
				    # You can edit to add more GPU
				    if [[ $(nvidia-smi -L) == *"A100"* ]];
				    then
					echo "A100 found";
					arch=sm_80;
				    fi;
				    if [[ $(nvidia-smi -L) == *"RTX"* ]];
				    then
					if [[ $(nvidia-smi -L) == *"2080"* ]];
					then
					    arch=sm_75;
					    echo "RTX found";
					else
					    arch=sm_86;
					    echo "RTX found" $arch;
					fi;
				    fi;
				    if [[ $(nvidia-smi -L) == *"V100"* ]];
				    then
					echo "V100 found";
					arch=sm_70;
				    fi;
				    if [[ $(nvidia-smi -L) == *"P100"* ]];
				    then
					echo "P100 found";
					arch=sm_60;
				    fi;
				    # Compile and run
				    rm -f sparse_mm;
				    nvcc -Xcompiler -fopenmp src/sparse_mm.cu -lcublas -lcurand -o sparse_mm -arch=${arch} \
					 -src-in-ptx --generate-line-info --ptxas-options=-v,--opt-level=3 --keep;
				    # /usr/local/cuda-12.6/bin/nvcc --std=c++17 -Xcompiler -fopenmp src/sparse_mm.cu -lcublas -lcurand -o sparse_mm -arch=${arch} \
				    # 				  -src-in-ptx --generate-line-info --ptxas-options=-v,--opt-level=3 --keep;
				    if [[ ${kernel_name} == *"cublas"* ]];
				    then
					CUBLASLT_LOG_LEVEL=5 ./sparse_mm -n ${kernel_name} -d ${device_id} -r ${nrepeats};
					if [[ ${6} == "nvprof" ]];
					then
					    (nvprof --metrics global_hit_rate ./sparse_mm -n ${kernel_name} -d ${device_id} -r ${nrepeats}) 2>&1 |grep "global_hit_rate" |tee -a nvprof_cublas.out;
					fi;
					cublas=1;
				    else
					if [[ ${6} == "nvprof" ]];
					then
					    echo "TILEX=$TILEX TILEK=$TILEK TILEY=$TILEY TX=$TX TY=$TY VSIZE=$vsize" >> nvprof.out;
					    metrics="global_hit_rate,local_hit_rate,shared_utilization,shared_load_throughput,shared_store_throughput,shared_efficiency,sm_efficiency,achieved_occupancy,warp_execution_efficiency,local_memory_overhead,l2_utilization";
					    (/usr/bin/nvprof --metrics $metrics --devices ${device_id} ./sparse_mm -n ${kernel_name} -d ${device_id} -r 3) 2>&1 |grep -E ${metrics//,/|} |tee -a nvprof.out;
					    # nsys profile --trace cuda,cudnn,nvtx,osrt,oshmem --cudabacktrace all --cuda-memory-usage true -o "$A.$B.$C.$D.${batch}_${TILEX}.${TILEK}.${TILEY}.${TX}.${TY}.out" ./sparse_mm -n ${kernel_name} -d ${device_id} -r ${nrepeats};
					fi;
					./sparse_mm -n ${kernel_name} -d ${device_id} -r ${nrepeats};
					cublas=0;
				    fi;
				    run="true";
				    # If cublas no need to loop over all the hyper-parameters
				    if [[ ${cublas} -eq 1 ]];
				    then
					break;
				    fi;
				done;
				# If cublas no need to loop over all the hyper-parameters
				if [[ ${cublas} -eq 1 ]];
				then
				    break;
				fi;
			    done;
			    # If cublas no need to loop over all the hyper-parameters
			    if [[ ${cublas} -eq 1 ]];
			    then
				break;
			    fi;
			done;
			# If cublas no need to loop over all the hyper-parameters
			if [[ ${cublas} -eq 1 ]];
			then
			    break;
			fi;
		    done;
		    # If cublas no need to loop over all the hyper-parameters
		    if [[ ${cublas} -eq 1 ]];
		    then
			break;
		    fi;
		done;
		# If cublas no need to loop over all the hyper-parameters
		if [[ ${cublas} -eq 1 ]];
		then
		    break;
		fi;
	    done;
	    # If cublas no need to loop over all the hyper-parameters
	    if [[ ${cublas} -eq 1 ]];
	    then
		break;
	    fi;
	done;
	# If cublas no need to loop over all the hyper-parameters
	if [[ ${cublas} -eq 1 ]];
	then
	    break;
	fi;
    done;
    if [[ ${run} == "false" ]];
    then
	echo ${is} ${os} > "assert_failed_"${A}"."${B}"."${C}"."${D}_${kernel_name}.out;
    fi;
done < ${5}
