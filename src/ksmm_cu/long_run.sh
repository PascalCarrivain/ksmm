#!/bin/bash

rm -f ${1}.out;
rm -f log.out;
rm -f assert_failed*.out;

# Possible tuples (a,b,c,d) are the following
# Cartesian product as x bs x cs x ds.

# as=(1 2 3 4 5 6 8 12 16 24 32 48 64 96 128);
# bs=(48 64 96 128 192 256 384 512 768 1024);
# cs=(48 64 96 128 192 256 384 512 768 1024);
# ds=(1 2 3 4 5 6 8 12 16 24 32 48 64 96 128);

as=(1 2 3 4 5 6 8 10 12 16 20 24 32 40 48 64 80 96 128);
bs=(48 64 80 96 128 160 192 256 320 384 512 640 768 1024 1280);
cs=(48 64 80 96 128 160 192 256 320 384 512 640 768 1024 1280);
ds=(1 2 3 4 5 6 8 10 12 16 20 24 32 40 48 64 80 96 128);

# as=(1);
# bs=(48);
# cs=(48);
# ds=(1);

as=(1 2 3 4 6 8 12 16 24 32 48 64 96 128);
bs=(48 64 96 128 192 256 384 512 768 1024);
cs=(48 64 96 128 192 256 384 512 768 1024);
ds=(4 16 64);


echo "batch_size a b c d TILEX TILEK TILEY TX TY" > wrong_mse_${1}.out;

if [[ ${1} == "" ]];
then
    kernel_name="kernel_bs_first_float4";
else
    kernel_name=${1};
fi;

if [[ ${2} == "" ]];
then
    device_id=0;
else
    device_id=${2};
fi;

if [[ ${3} == "sanity_check" ]];
then
    sanity_check=1;
else
    sanity_check=0;
fi;

if [[ ${4} == "" ]];
then
    nrepeats=100;
else
    nrepeats=10;
fi;

if [[ ${5} == "" ]];
then
    count_hyper=0;
else
    count_hyper=${5};
fi;

if [[ ${kernel_name} == *"cublas"* ]];
then
    kernel="cublas";
    if [[ ${kernel_name} == *"cublas0"* ]];
    then
	kernel=0;
    fi;
    if [[ ${kernel_name} == *"cublas1"* ]];
    then
	kernel=1;
    fi;
    # sizeof as a function of the precision
    # hp=0, fp=2 half-precision
    # hp=1, fp=4 float
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
    else
	# Home-made kernel
	# Header of the file where to store best hyper-parameters
	echo "batch_size a b c d TILEX TILEK TILEY TX TY WX WY ms std mse" > ${kernel_name}.out;
	if [[ ${kernel_name} == *"float4"* ]];
	then
	    # sizeof as a function of the precision
	    hp=0;
	    fp=4;
	else
	    # sizeof as a function of the precision
	    hp=1;
	    fp=2;
	fi;
    fi;
fi;


# Loop over batch size
for batch in 25088;#50176;#196;
do
    # Loop over tuple (a,b,c,d)
    for (( a=0; a<${#as[@]}; a=$(($a+1)) ));
    do
	A=${as[$a]};
	for (( b=0; b<${#bs[@]}; b=$(($b+1)) ));
	do
	    B=${bs[$b]};
	    for (( c=0; c<${#cs[@]}; c=$(($c+1)) ));
	    do
		C=${cs[$c]};
		# Study only b=c, b=4c and c=4b
		if [[ ${B} -ne ${C} ]] && [[ ${B} -ne $((4*${C})) ]] && [[ ${C} -ne $((4*${B})) ]];
		then
		    continue;
		fi;
		for (( d=0; d<${#ds[@]}; d=$(($d+1)) ));
		do
		    D=${ds[$d]};
		    if [[ ${count_hyper} -eq 1 ]];
		    then
			echo ${A} ${B} ${C} ${D};
		    fi;
		    # ???
		    if [[ 1 -eq 1 ]];
		    then
			if [[ ${A} -eq 1 ]] || [[ ${D} -eq 1 ]];
			then
			    continue;
			fi;
			# skip if (b == 1024 and c == 256) or (b == 256 and c == 1024):
			if [[ ${B} -eq 1024 ]] && [[ ${C} -eq 256 ]];
			then
                            continue
			fi
			if [[ ${B} -eq 256 ]] && [[ ${C} -eq 1024 ]];
			then
                            continue
			fi
			# skip if (b == 128 and c == 512) or (b == 512 and c == 128):
			if [[ ${B} -eq 128 ]] && [[ ${C} -eq 512 ]];
			then
                            continue
			fi
			if [[ ${B} -eq 512 ]] && [[ ${C} -eq 128 ]];
			then
                            continue
			fi
			# skip if (b == 64 and c == 256) or (b == 256 and c == 64):
			if [[ ${B} -eq 64 ]] && [[ ${C} -eq 256 ]];
			then
                            continue
			fi
			if [[ ${B} -eq 256 ]] && [[ ${C} -eq 64 ]];
			then
                            continue
			fi
		    fi;
		    # ???
		    # Multiplication X @ B^T
		    # X in R^{batch x (a * c * d)}
		    # B in R^{a * b * d, a * c * d}
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
		    # TILEX is x * TX
		    for x in {1..3} 4 6 8 10 12 16 32 64;
		    do
			# sub-tile along X
			for TX in {2..10};
			do
			    # Size of the tile along X
			    TILEX=$((${x}*${TX}));
			    # D * TILEX must divide number of columns of a super-block
			    if [[ $(($((${B}*${D}))%$((${D}*${TILEX})))) -ne 0 ]];
			    then
				continue;
			    fi;
			    # TILEK is k * fp = k * (2 or 4)
			    for k in {2..10};
			    do
				# Size of the tile along K
				TILEK=$((${fp}*${k}));
				# D * TILEK must divide C * D
				if [[ $(($((${C}*${D}))%$((${D}*${TILEK})))) -ne 0 ]];
                                then
				    continue;
      	                        fi;
				# TILEY is y * TY
				for y in {1..3} 4 6 8 10 12 16 32 64;
				do
				    # stride(values) = fp * nthreads / TILEX
				    strideValues=$(($((${fp}*${x}*${y}))/${TILEX}));
				    if [[ $(($((${fp}*${x}*${y}))%${TILEX})) -ne 0 ]];
				    then
					continue;
				    fi;
				    # sub-tile along Y
				    for TY in {2..10};
				    do
					# Size of the tile along Y
					TILEY=$((${y}*${TY}));
					# stride(input) = fp * nthreads / TILEY
					strideInput=$(($((${fp}*${x}*${y}))/${TILEY}));
					if [[ $(($((${fp}*${x}*${y}))%${TILEY})) -ne 0 ]];
					then
					    continue;
					fi;
					# Sub-tile must be greater or equal to fp
					if [[ ${TX} -lt ${fp} ]] || [[ ${TY} -lt ${fp} ]];
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
					    if [[ ${TILEY} -ne 32 ]] && [[ ${TILEY} -ne 64 ]] && [[ ${TILEY} -ne 128 ]];
                                            then
						continue;
                                            fi;
					    if [[ ${B} -eq 48 ]] || [[ ${C} -eq 48 ]];
                                            then
                                                TILEXs="16.24.48";
                                                TILEKs="16.24";
                                            else
						if [[ ${B} -eq 80 ]] || [[ ${C} -eq 80 ]];
                                                then
                                                    TILEXs="8.10.20.40";
                                                    TILEKs="8.10.16.20";
                                                else
                                                    if [[ ${B} -eq 96 ]] || [[ ${C} -eq 96 ]];
                                                    then
                                                        TILEXs="32.64";
                                                        TILEKs="16.32";
                                                    else
                                                        if [[ ${B} -eq 160 ]] || [[ ${C} -eq 160 ]];
                                                        then
                                                            TILEXs="16.20.40.80";
                                                            TILEKs="16.20.40";
                                                        else
                                                            TILEXs="64";
                                                            TILEKs="16.32";
                                                        fi;
                                                    fi;
                                                fi;
                                            fi;
					    if [[ ${TILEXs} != *"${TILEX}"* ]] || [[ ${TILEKs} != *"${TILEK}"* ]];
                                            then
                                                continue;
                                            fi;
					    if [[ ${TX} -ne 4 ]] && [[ ${TX} -ne 8 ]];
                                            then
						continue;
                                            fi;
					    if [[ ${TY} -ne 4 ]] && [[ ${TY} -ne 8 ]];
                                            then
						continue;
                                            fi;
					fi;
					# Number of warps along x (tensor-cores only)
					for nwx in 1;#1 2 3 4;
					do
					    # Number of warps along y (tensor-cores only)
					    for nwy in 1;#1 2 3 4;
					    do
						# If not tensor-cores no need to run many number of warps values
						if [[ ${kernel_name} != *"_tc"* ]] && ([[ ${nwx} -ne 1 ]] || [[ ${nwy} -ne 1 ]]);
						then
						    continue;
						fi;
						# Tensor cores: check number of threads
						# nthreads = nwx * nwy * 32
						# nthreads = x * y
						if [[ ${kernel_name} == *"_tc"* ]];
						then
						    if [[ $((${nwx}*${nwy}*32)) -ne $((${x}*${y})) ]];
						    then
							continue;
						    fi;
						fi;
						# TILEY must divide batch size
						if [[ $((${batch}%${TILEY})) -ne 0 ]];
						then
						    continue;
						fi;
						# Check if the number of threads is equal to the number
						# of warps times warp size (only for tensor cores).
						if [[ ${kernel_name} == *"_tc"* ]];
						then
						    if [[ ${TILEX} -ne 16 ]] || [[ ${TILEK} -ne 16 ]] || [[ ${TILEY} -ne 16 ]] || [[ $((${x}*${y})) -ne $((${nwx}*${nwy}*32)) ]];
						    then
							continue;
						    fi;
						fi;
						# Compute shared memory (double-buffering)
						smem=$((${fp}*$((2*${TILEY}*${TILEK}+2*${TILEK}*${TILEX}+${TILEY}*${TILEX}))));
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
						# Check TILEY, TILEK, TILEX < 4 (float4) or 2 (half2)
						if [[ ${TILEX} -lt ${fp} ]] || [[ ${TILEK} -lt ${fp} ]] || [[ ${TILEY} -lt ${fp} ]];
						then
						    continue;
						fi;
						# strides must divide TILEK
						if [[ $((${TILEK}%${strideInput})) -ne 0 ]];
						then
						    continue;
						fi;
						if [[ $((${TILEK}%${strideValues})) -ne 0 ]];
						then
						    continue;
						fi;
						if [[ ${count_hyper} -eq 0 ]];
						then
						    # Replace all the xNAMEx by a constant value
						    # All the template_*.cuh files become *.cuh
						    for f in "sparse_mm" "kernels_float4" "kernels_half2";
						    do
							if [[ ${f} == "sparse_mm" ]];
							then
							    new_f=src/${f}.cu;
							    cp src/template_${f}.cu ${new_f};
							fi;
							if [[ ${f} == "kernels_float4" ]] || [[ ${f} == "kernels_half2" ]];
							then
							    new_f=src/${f}.cuh;
							    cp src/template_${f}.cuh ${new_f};
							fi;
							# Add a define directive to handle half and float
							if [[ ${hp} -eq 0 ]];
							then
							    sed -i "s/defineFLOAT4orHALF2/define FLOAT4/g" ${new_f};
							fi;
							if [[ ${hp} -eq 1 ]];
							then
							    sed -i "s/defineFLOAT4orHALF2/define HALF2/g" ${new_f};
							fi;
							sed -i -e "s/xBATCHSIZEx/${batch}/g" \
							    -e "s/xINPUTSIZEx/${is}/g" \
							    -e "s/xOUTPUTSIZEx/${os}/g" \
							    -e "s/xax/${A}/g" \
							    -e "s/xbx/${B}/g" \
							    -e "s/xcx/${C}/g" \
							    -e "s/xdx/${D}/g" \
							    -e "s/xWMMA_Yx/16/g" \
							    -e "s/xWMMA_Xx/16/g" \
							    -e "s/xWMMA_Kx/16/g" \
							    -e "s/xNWARPSYx/${nwx}/g" \
							    -e "s/xNWARPSXx/${nwy}/g" \
							    -e "s/xTILEXx/${TILEX}/g" \
							    -e "s/xTILEKx/${TILEK}/g" \
							    -e "s/xTILEYx/${TILEY}/g" \
							    -e "s/xTXx/${TX}/g" \
							    -e "s/xTYx/${TY}/g" \
							    -e "s/xNTHREADS_FLOAT4x/$((x*y))/g" \
							    -e "s/xNTHREADS_HALF2x/$((x*y))/g" \
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
						    # Compile and run
						    rm -f sparse_mm;
						    nvcc -Xcompiler -fopenmp src/sparse_mm.cu -lcublas -lcurand -o sparse_mm -arch=${arch};# --ptxas-options=-v;# --maxrregcount=128;
						    if [[ ${kernel_name} == *"cublas"* ]];
						    then
							CUBLASLT_LOG_LEVEL=5 ./sparse_mm -n ${kernel_name} -d ${device_id} -r ${nrepeats};
							cublas=1;
						    else
							./sparse_mm -n ${kernel_name} -d ${device_id} -r ${nrepeats};
							cublas=0;
						    fi;
						    # nvprof --print-gpu-trace ./sparse_mm -n ${kernel_name};# 2>&1 |tee -a log.out;
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
		    if [[ ${run} == "false" ]];
		    then
			echo ${is} ${os} > "assert_failed_"${A}"."${B}"."${C}"."${D}_${kernel_name}.out;
		    fi;
		done;
	    done;
	done;
    done;
done;
