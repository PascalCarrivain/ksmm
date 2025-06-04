// -*- c -*-

#ifndef KERNEL_BS_LAST_FLOAT
#define KERNEL_BS_LAST_FLOAT

#include "template_kernels_float.cuh"

void best_kernel_bs_last_float(float *input, float *values, float *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 392;
			kernel_bs_last_float<float4, 16, 16, 64, 4, 4, 4, true, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		assert(1 == 0);
		break;
	}
}

#endif
