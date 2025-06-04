// -*- c -*-

#ifndef KERNEL_BS_FIRST_HALF
#define KERNEL_BS_FIRST_HALF

#include "template_kernels_half.cuh"

void best_kernel_bs_first_half(half *input, half *values, half *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 3136;
			kernel_bs_first_half2<half2, 16, 8, 8, 2, 2, 2, true, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		assert(1 == 0);
		break;
	}
}

#endif
