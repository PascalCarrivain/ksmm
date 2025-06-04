// -*- c -*-

#ifndef KERNEL_BS_FIRST_HALF4
#define KERNEL_BS_FIRST_HALF4

#include "template_kernels_half.cuh"

void best_kernel_bs_first_half4(half *input, half *values, half *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		assert(1 == 0);
		break;
	}
}

#endif
