// -*- c -*-

#ifndef KERNEL_BS_LAST_HALF
#define KERNEL_BS_LAST_HALF

#include "template_kernels_half.cuh"

void best_kernel_bs_last_half(half *input, half *values, half *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 196;
			kernel_bs_last_half4<bool, 16, 16, 128, 8, 4, 4, false, 2><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		assert(1 == 0);
		break;
	}
}

#endif
