// -*- c -*-

#ifndef KERNEL_BS_FIRST_FLOAT
#define KERNEL_BS_FIRST_FLOAT

#include "template_kernels_float.cuh"

void best_kernel_bs_first_float(float *input, float *values, float *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 8) {
			threadsPerBlock.x = 24;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_float<float4, 48, 24, 32, 8, 8, 4, true, 6><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		assert(1 == 0);
		break;
	}
}

#endif
