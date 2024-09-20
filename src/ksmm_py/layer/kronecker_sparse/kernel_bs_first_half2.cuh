// -*- c -*-

#ifndef KERNEL_BS_FIRST_HALF2
#define KERNEL_BS_FIRST_HALF2

#include "template_kernels_half2.cuh"

void best_kernel_bs_first_half2(half *input, half *values, half *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 3) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 6) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 48) {
			threadsPerBlock.x = 16;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 96) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 392;
			kernel_bs_first_half2<16, 16, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 3) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 6) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 96) {
			threadsPerBlock.x = 16;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 192 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 1;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 2;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 3) {
			threadsPerBlock.x = 128;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 4;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 5) {
			threadsPerBlock.x = 256;
			blockGrid.x = 5;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 6) {
			threadsPerBlock.x = 128;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 12) {
			threadsPerBlock.x = 256;
			blockGrid.x = 12;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 24) {
			threadsPerBlock.x = 128;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 96;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 64 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 2;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 5;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 8) {
			threadsPerBlock.x = 128;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 12) {
			threadsPerBlock.x = 128;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 48) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 64;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 96) {
			threadsPerBlock.x = 512;
			blockGrid.x = 96;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 64 && c == 256 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 128;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 6) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 12) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 24) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 64) {
			threadsPerBlock.x = 128;
			blockGrid.x = 192;
			blockGrid.y = 196;
			kernel_bs_first_half2<32, 16, 128, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 96) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 96 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 3) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 12) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 64) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 96) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 96 && c == 384 && d == 128) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 2;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 10;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 8) {
			threadsPerBlock.x = 128;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 12) {
			threadsPerBlock.x = 128;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 24) {
			threadsPerBlock.x = 512;
			blockGrid.x = 48;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 48) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 128;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 96) {
			threadsPerBlock.x = 512;
			blockGrid.x = 192;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 128 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 256;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 2;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 10;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 6) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 8) {
			threadsPerBlock.x = 128;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 12) {
			threadsPerBlock.x = 128;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 128;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 192;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 256;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 3) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 60;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 6) {
			threadsPerBlock.x = 16;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 96) {
			threadsPerBlock.x = 16;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 48 && d == 128) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1536;
			blockGrid.y = 392;
			kernel_bs_first_half2<16, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 3) {
			threadsPerBlock.x = 128;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 12) {
			threadsPerBlock.x = 128;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 24) {
			threadsPerBlock.x = 128;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 64) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 288;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 192 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 3) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 6) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 24) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 48) {
			threadsPerBlock.x = 256;
			blockGrid.x = 144;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 192;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 192 && c == 768 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 288;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 2) {
			threadsPerBlock.x = 128;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 3) {
			threadsPerBlock.x = 128;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 5) {
			threadsPerBlock.x = 64;
			blockGrid.x = 20;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 8) {
			threadsPerBlock.x = 128;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 12) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 24) {
			threadsPerBlock.x = 128;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 48) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 96) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 64 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 512;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 3) {
			threadsPerBlock.x = 128;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 5) {
			threadsPerBlock.x = 64;
			blockGrid.x = 20;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 512;
			blockGrid.x = 64;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 24) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 48) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 64) {
			threadsPerBlock.x = 64;
			blockGrid.x = 256;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 384;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 256 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 512;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 5) {
			threadsPerBlock.x = 64;
			blockGrid.x = 20;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 6) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 16) {
			threadsPerBlock.x = 128;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 48) {
			threadsPerBlock.x = 256;
			blockGrid.x = 192;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 256 && c == 1024 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 256;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 5) {
			threadsPerBlock.x = 64;
			blockGrid.x = 30;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 48) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 96) {
			threadsPerBlock.x = 64;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 96 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 3) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 30;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 12) {
			threadsPerBlock.x = 128;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 16) {
			threadsPerBlock.x = 128;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 48) {
			threadsPerBlock.x = 256;
			blockGrid.x = 288;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 384;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 576;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 384 && c == 384 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 3) {
			threadsPerBlock.x = 128;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 40;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 8) {
			threadsPerBlock.x = 128;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 24) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 48) {
			threadsPerBlock.x = 256;
			blockGrid.x = 384;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 64) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 96) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 128 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 3) {
			threadsPerBlock.x = 128;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 40;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 48) {
			threadsPerBlock.x = 256;
			blockGrid.x = 384;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 512;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 5) {
			threadsPerBlock.x = 128;
			blockGrid.x = 60;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 8) {
			threadsPerBlock.x = 512;
			blockGrid.x = 96;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 12) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 24) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 32) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 48) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 192 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1152;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 5) {
			threadsPerBlock.x = 64;
			blockGrid.x = 60;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 12) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 24) {
			threadsPerBlock.x = 256;
			blockGrid.x = 288;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 32) {
			threadsPerBlock.x = 256;
			blockGrid.x = 384;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 48) {
			threadsPerBlock.x = 512;
			blockGrid.x = 576;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 768 && c == 768 && d == 96) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1152;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 3) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 80;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 6) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 8) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 12) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 24) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 48) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 256 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 3) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 5) {
			threadsPerBlock.x = 32;
			blockGrid.x = 80;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 6) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 8) {
			threadsPerBlock.x = 128;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 12) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 24) {
			threadsPerBlock.x = 256;
			blockGrid.x = 384;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 48) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 1024 && c == 1024 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 48 && c == 192 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 2;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 128;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 2;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 96 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 384 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 96 && c == 384 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 128;
			blockGrid.x = 64;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 128 && c == 128 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 256;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 192 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 384;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 192 && c == 768 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 256 && c == 256 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 512;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 96 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 384 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 384 && c == 384 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 512 && c == 512 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 512 && c == 512 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 768 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 768 && c == 768 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 2 && b == 1024 && c == 1024 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 48 && c == 192 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 3;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 392;
			kernel_bs_first_half2<32, 16, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 96 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 384 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 96 && c == 384 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 256;
			blockGrid.x = 96;
			blockGrid.y = 196;
			kernel_bs_first_half2<64, 16, 128, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 128 && c == 128 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 128;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 192 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 576;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 9;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 192 && c == 768 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 128;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 256 && c == 256 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 96 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 384 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 384 && c == 384 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1152;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 512 && c == 512 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 768 && c == 192 && d == 16) {
			threadsPerBlock.x = 128;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 768 && c == 768 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 3 && b == 1024 && c == 1024 && d == 16) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 48 && c == 192 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 64;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 4;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 96 && c == 96 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 96 && c == 384 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 128 && c == 128 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 512;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 192 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 192 && c == 768 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 256 && c == 256 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 384 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 384 && c == 384 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 512 && c == 512 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 768 && c == 192 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 768 && c == 768 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 4 && b == 1024 && c == 1024 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 5;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 5;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 10;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 10;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 60;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 15;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 20;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 20;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 20;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 30;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 30;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 40;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 40;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 60;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 60;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 80;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 5 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 80;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 48 && c == 192 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 6;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 96 && c == 96 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 96 && c == 384 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 128 && c == 128 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 768;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 192 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1152;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 18;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 192 && c == 768 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 384 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 384 && c == 384 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 512 && c == 512 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 768 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 768 && c == 768 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 6 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 8;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 96 && c == 96 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 96 && c == 384 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 128 && c == 128 && d == 64) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 384 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 384 && c == 384 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 512 && c == 512 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 8 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 12;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 96 && c == 96 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 2304;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 96 && c == 384 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 96;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 2304;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 36;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 384 && c == 96 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 384 && c == 384 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 12 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 64 && c == 64 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 16;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 256 && c == 256 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 16 && b == 1024 && c == 1024 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 48 && c == 48 && d == 64) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 48 && c == 192 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 24;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 256;
			blockGrid.x = 24;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 192 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 72;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 192 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 256;
			blockGrid.x = 576;
			blockGrid.y = 392;
			kernel_bs_first_half2<64, 16, 64, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 768 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 768 && c == 768 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 24 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 32;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 128 && c == 128 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 512 && c == 512 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 32 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 48;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 96 && c == 96 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 2304;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 96 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 144;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 384 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 384 && c == 384 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 48 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 128;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 64 && c == 64 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 256 && c == 256 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 256 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 1024 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 64 && b == 1024 && c == 1024 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 48 && c == 48 && d == 16) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 48 && c == 192 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 96;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 8, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 192;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 192 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 192 && c == 192 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 192 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 288;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 576;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 768 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 96 && b == 768 && c == 768 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1152;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 48 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 48 && c == 48 && d == 4) {
			threadsPerBlock.x = 48;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<48, 6, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 48 && c == 192 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 4, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 64 && c == 64 && d == 4) {
			threadsPerBlock.x = 128;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 64 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 96 && c == 96 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 96 && c == 96 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 96 && c == 384 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<32, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 128 && c == 128 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 128 && c == 128 && d == 4) {
			threadsPerBlock.x = 64;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 128 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 192 && c == 48 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1536;
			blockGrid.y = 784;
			kernel_bs_first_half2<16, 16, 32, 4, 4><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 192 && c == 192 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 384;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 256 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 256 && c == 256 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 512;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 384 && c == 96 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 4, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 384 && c == 384 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 768;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 512 && c == 128 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 128 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 1024;
			blockGrid.y = 784;
			kernel_bs_first_half2<64, 16, 32, 8, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		assert(1 == 0);
		break;
	}
}

#endif
