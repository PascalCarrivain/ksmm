// -*- c++ -*-
#ifndef KERNELS_FLOAT
#define KERNELS_FLOAT

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdio.h>

#define WARP_SIZE 32

// using namespace nvcuda;


template <typename T, const bool uss, const int sss, const int vsize>
__device__ inline void smem_load(float *reg, float *shared_mem, int idx,
				 int tx);

template <typename T, const bool uss, const int sss, const int vsize>
__device__ inline void smem_load(float *reg, float *shared_mem, int idx,
				 int tx) {
  unsigned xx[11];
  xx[0] = 0x3F;
  xx[1] = 0xFC0;
  xx[2] = 0x3F000;
  xx[3] = 0xFC0000;
  xx[4] = 0x3F000000;
  xx[5] = 0xC0000000;
  xx[6] = 0x00000FFF;
  xx[7] = 0x00FFF000;
  xx[8] = 0xFF000000;
  xx[9] = 0x00FFFFFF;
  xx[10] = 0xFF000000;

  if constexpr (uss && (sss == 32 || sss == 16 || sss == 8 || sss == 4 || sss == 2)) {
    // Reminder: int thready = threadIdx.x / (TILEX / TX);
    // Let say TILEX / TX = 32. Therefore, we have thready = 0 for all threadIdx.x in [0,...,31] (warp 0).
    //                                                     = 1 for all threadIdx.x in [32,...,63] (warp 1).
    //                                                     = 2 for all threadIdx.x in [64,...,95] (warp 2).
    // For threadIdx.x % (TILEX / TX) == 0 and a given y, it access to the same value of shared_input[load][...].
    // It means that for threadIdx.x % (TILEX / TX) == 0 we can call `__shfl_sync()`.
    // Do we conclude that TILEX / TX must less or equal to WARP_SIZE?
    // 6, 12 and 24 do not divide 32.
    if((tx % 4) == 0)
      reinterpret_cast<T *>(&reg[0])[0] = reinterpret_cast<T *>(&shared_mem[idx])[0];
#pragma unroll
    for (int v = 0; v < vsize; v++)
      reg[v] = __shfl_sync(4 - 1, reg[v], 0, 4);
  } else if constexpr (uss && (sss == 24 || sss == 12 || sss == 6)) {
    // Case sss=12:
    // `__shfl_sync()` to the first group of 12 threads of the warp,
    // `__shfl_sync()` to the second group of 12 threads of the warp,
    // `__shfl_sync()` to the last group of 8 threads of the warp.
    int lane, warp, diff;
    warp = WARP_SIZE * (tx / WARP_SIZE);
    if((tx - warp) % 4 == 0) {
      lane = 4 * (tx / 4);
      diff = (tx - warp) / 4;
      if constexpr (sss == 12) {
	diff += 6;
      }
      if constexpr (sss == 24) {
	diff += 6 + 3;
      }
      reinterpret_cast<T *>(&reg[0])[0] = reinterpret_cast<T *>(&shared_mem[idx])[0];
    }
#pragma unroll
    for (int v = 0; v < vsize; v++)
      reg[v] = __shfl_sync(xx[diff], reg[v], lane, WARP_SIZE);
  } else {
    reinterpret_cast<T *>(&reg[0])[0] =
      reinterpret_cast<T *>(&shared_mem[idx])[0];
  }
}


template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_float(
__global__ void kernel_bs_first_float(float *input, float *valuesT,
                                      int batch_size, float *output, int a,
                                      int b, int c, int d);

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_float(
__global__ void kernel_bs_first_float(float *input, float *valuesT,
                                      int batch_size, float *output, int a,
                                      int b, int c, int d) {
  // Butterfly factor
  // tuple (a, b, c, d)
  // B = kron(Id_{a,a}, kron(1_{b,c}, Id_{d,d}))
  // There is 'a' super-blocks of shape (b * d, c * d).
  // Number of non-zero per super-block is
  // b per column and c per row.
  // We would like to compute X @ B^T.
  // X shape is (batch, a * c * d).
  // B^T shape is (a * c * d, a * b * d).
  int input_size = a * c * d;
  int output_size = a * b * d;
  // TILEX / TX threads per column
  // Get the current thread
  int threadx = threadIdx.x % (TILEX / TX);
  int thready = threadIdx.x / (TILEX / TX);
  // To store input in shared memory
  __shared__ float shared_input[2][TILEY * TILEK];
  // To store sparse matrix in shared memory
  __shared__ float shared_values[2][TILEK * TILEX];
  // // To store output in shared memory
  // __shared__ float shared_output[TILEY * TILEX];
  float acc[TY * TX] = {0.0f};
  float regY[TY] = {0.0f};
  float regX[TX] = {0.0f};

  // Current super-block
  int sb_id = (blockIdx.x * TILEX) / (b * d);
  // Move to current super-block
  valuesT = &valuesT[b * c * d * sb_id];
  // Move blockIdx.y * TILEY rows
  input = &input[blockIdx.y * TILEY * input_size];
  // Move blockIdx.y * TILEY rows
  output = &output[blockIdx.y * TILEY * output_size];

  // Group index in current super-block
  int grp_id = (blockIdx.x * TILEX - b * d * sb_id) / b;
  int id_in_grp = ((blockIdx.x * TILEX - b * d * sb_id) % b) / TILEX;

  // Move input to column that correspond to current super-block
  input += c * d * sb_id + grp_id;
  // Inside the current super-block handle the value of d
  valuesT += c * b * grp_id + (blockIdx.x * TILEX - b * d * sb_id) % b;
  // Move output to the current super-column
  output += b * d * sb_id;
  // Inside the current super-block handle the value of d
  output += TILEX * d * id_in_grp + grp_id;

  // Indices to load (input matrix)
  int InputSubRow = threadIdx.x % (TILEY / VSIZE);
  int InputSubCol = threadIdx.x / (TILEY / VSIZE);

  // Indices to load (Butterfly factor)
  int ValuesSubRow = threadIdx.x / (TILEX / VSIZE);
  int ValuesSubCol = threadIdx.x % (TILEX / VSIZE);

  // Use stride to load from GMEM to SMEM
  const int nthreads = (TILEX / TX) * (TILEY / TY);
  const int StrideInput = VSIZE * nthreads / TILEY;
  const int StrideValues = VSIZE * nthreads / TILEX;

  int tmp_col, tmp_row;

  // Load the first batch of input from global to shared memory TILEY * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput) {
#pragma unroll
    for (int v = 0; v < VSIZE; v++)
      shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + v] =
        input[d * (InputSubCol + s) + (InputSubRow * VSIZE + v) * input_size];
  }
  // Load the first batch of the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideValues) {
    // Reconstruct row and column indices
    tmp_row = ValuesSubRow + s;
    tmp_col = ValuesSubCol * VSIZE;
    reinterpret_cast<T *>(&shared_values[0][tmp_row * TILEX + tmp_col])[0] =
        reinterpret_cast<T *>(&valuesT[b * tmp_row + tmp_col % b])[0];
  }

  int load = 0;
  int write = 1;

  // Loop over non-zero entries by TILEK
  for (int k = 0; k < c; k += TILEK) {
    __syncthreads();
    // Load smem in register and compute accumulation
#pragma unroll
    for (int i = 0; i < TILEK; i++) {
#pragma unroll
      for (int y = 0; y < TY; y += VSIZE) {
        // Since thready = threadIdx.x / (TILEX / TX), consecutive threads may
        // access the same values so only one of them can load it and share it
        // with the others via __shfl_sync.
	// smem_load<T, uss, sss, VSIZE>(&regY[y], &shared_input[load][0], i * TILEY + thready * TY + y, threadIdx.x);
        reinterpret_cast<T*>(&regY[y])[0] =
	  reinterpret_cast<T*>(&shared_input[load][i * TILEY + thready * TY + y])[0];
      }
#pragma unroll
      for (int x = 0; x < TX; x += VSIZE) {
        // Since threadx = threadIdx.x % (TILEX / TX), consecutive threads in a
        // warp do not access the same values so no __shfl_sync can be used in
        // general.
        reinterpret_cast<T *>(&regX[x])[0] = reinterpret_cast<T *>(
            &shared_values[load][i * TILEX + threadx * TX + x])[0];
      }
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++)
          acc[y * TX + x] += regY[y] * regX[x];
      }
    }

    load = load ^ 1;
    input += d * TILEK;
    valuesT += TILEK * b;

    // Load next batch of the input from global to shared memory TILEY * TILEK
    // Condition on row of valuesT
    if ((k + TILEK) < c) {
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideInput) {
        // Use T to load input into shared memory
        // input must use the row-major format
#pragma unroll
	for (int v = 0; v < VSIZE; v++)
	  shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + v] =
            input[d * (InputSubCol + s) + (InputSubRow * VSIZE + v) * input_size];
      }
      // Load next batch of the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideValues) {
        // Reconstruct row and column indices
        tmp_row = ValuesSubRow + s;
        tmp_col = ValuesSubCol * VSIZE;
        reinterpret_cast<T *>(
            &shared_values[write][tmp_row * TILEX + tmp_col])[0] =
            reinterpret_cast<T *>(&valuesT[b * tmp_row + tmp_col % b])[0];
      }
      write = write ^ 1;
    }
  }

  //   // Store accumulation to shared memory
  // #pragma unroll
  //   for (int y = 0; y < TY; y++)
  //     {
  //       for (int x = 0; x < TX; x += VSIZE)
  // 	{
  // 	  reinterpret_cast<T*>(
  // 				     &shared_output[(thready * TY + y) * TILEX + threadx * TX
  // + x])[0] = 	    reinterpret_cast<T*>(&acc[y * TX + x])[0];
  // 	}
  //     }

  //   // Write out the accumulation (from shared to global memory)
  // #pragma unroll
  //   for (int y = 0; y < TY; y++)
  //     {
  // #pragma unroll
  //       for (int x = 0; x < TX; x += VSIZE)
  // 	{
  // 	  output[(thready * TY + y) * output_size + d * (threadx * TX + x + 0)]
  // = 	    shared_output[(thready * TY + y) * TILEX + threadx * TX + x + 0];
  // if constexpr (VSIZE > 1) { 	    output[(thready * TY + y) *
  // output_size + d *
  // (threadx * TX + x + 1)] = 	      shared_output[(thready * TY + y) * TILEX +
  // threadx * TX + x + 1];
  // 	  }
  // 	  if constexpr (VSIZE > 2) {
  // 	    output[(thready * TY + y) * output_size + d * (threadx * TX + x +
  // 2)] = 	      shared_output[(thready * TY + y) * TILEX + threadx * TX + x
  // + 2];
  // 	  }
  // 	  if constexpr (VSIZE > 3) {
  // 	    output[(thready * TY + y) * output_size + d * (threadx * TX + x +
  // 3)] = 	      shared_output[(thready * TY + y) * TILEX + threadx * TX + x
  // + 3];
  // 	  }
  // 	}
  //     }

  // Write out the accumulation (from shared to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int v = 0; v < VSIZE; v++)
	output[(thready * TY + y) * output_size + d * (threadx * TX + x + v)] =
          acc[y * TX + x + v];
    }
  }
}


template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_float(
__global__ void kernel_bs_last_float(float *inputT, float *valuesT,
                                     int batch_size, float *outputT, int a,
                                     int b, int c, int d);

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_float(
__global__ void kernel_bs_last_float(float *inputT, float *valuesT,
                                     int batch_size, float *outputT, int a,
                                     int b, int c, int d) {
  // Butterfly factor
  // tuple (a, b, c, d)
  // B = kron(Id_{a,a}, kron(1_{b,c}, Id_{d,d}))
  // There is 'a' super-blocks of shape (b * d, c * d).
  // Number of non-zero per super-block is
  // b per column and c per row.
  // We would like to compute X @ B^T.
  // X shape is (batch, a * c * d).
  // B^T shape is (a * c * d, a * b * d).
  // TILEX / TX threads per column
  // Get the current thread
  int threadx = threadIdx.x % (TILEX / TX);
  int thready = threadIdx.x / (TILEX / TX);
  // To store input in shared memory
  __shared__ float shared_input[2][TILEY * TILEK];
  // To store sparse matrix in shared memory
  __shared__ float shared_values[2][TILEK * TILEX];
  // // To store output in shared memory
  // __shared__ float shared_output[TILEY * TILEX];
  float acc[TY * TX] = {0.0f};
  float regY[TY] = {0.0f};
  float regX[TX] = {0.0f};

  // Current super-block
  int sb_id = (blockIdx.x * TILEX) / (b * d);
  // Move to current super-block
  valuesT = &valuesT[b * c * d * sb_id];
  // Move blockIdx.y * TILEY rows
  inputT = &inputT[blockIdx.y * TILEY];
  // Move blockIdx.y * TILEY rows
  outputT = &outputT[blockIdx.y * TILEY];

  // Group index in current super-block
  int grp_id = (blockIdx.x * TILEX - b * d * sb_id) / b;
  int id_in_grp = ((blockIdx.x * TILEX - b * d * sb_id) % b) / TILEX;

  // Move input to column that correspond to current super-block
  inputT += (c * d * sb_id + grp_id) * batch_size;
  // Inside the current super-block handle the value of d
  valuesT += c * b * grp_id + (blockIdx.x * TILEX - b * d * sb_id) % b;
  // // Move output to the current super-column
  outputT += b * d * sb_id * batch_size;
  // Inside the current super-block handle the value of d
  outputT += (TILEX * d * id_in_grp + grp_id) * batch_size;

  // Indices to load (input matrix)
  int InputSubRow = threadIdx.x % (TILEY / VSIZE);
  int InputSubCol = threadIdx.x / (TILEY / VSIZE);

  // Indices to load (Butterfly factor)
  int ValuesSubRow = threadIdx.x / (TILEX / VSIZE);
  int ValuesSubCol = threadIdx.x % (TILEX / VSIZE);

  // Use stride to load from GMEM to SMEM
  const int nthreads = (TILEX / TX) * (TILEY / TY);
  const int StrideInput = VSIZE * nthreads / TILEY;
  const int StrideValues = VSIZE * nthreads / TILEX;

  int tmp_col, tmp_row;

  // Load the first batch of input from global to shared memory TILEY * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput) {
    // Use T to load input into shared memory
    // inputT must use the col-major format
    reinterpret_cast<T *>(
        &shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE])[0] =
        reinterpret_cast<T *>(&inputT[d * (InputSubCol + s) * batch_size +
                                      InputSubRow * VSIZE])[0];
  }
  // Load the first batch of Butterfly factor from global to shared memory TILEK
  // * TILEX
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideValues) {
    // Reconstruct row and column indices
    tmp_row = ValuesSubRow + s;
    tmp_col = ValuesSubCol * VSIZE;
    reinterpret_cast<T *>(&shared_values[0][tmp_row * TILEX + tmp_col])[0] =
        reinterpret_cast<T *>(&valuesT[b * tmp_row + tmp_col % b])[0];
  }

  int load = 0;
  int write = 1;

  // Loop over non-zero entries by TILEK
  for (int k = 0; k < c; k += TILEK) {
    __syncthreads();
    // Load smem to register and compute accumulation
#pragma unroll
    for (int i = 0; i < TILEK; i++) {
#pragma unroll
      for (int y = 0; y < TY; y += VSIZE) {
        // Since thready = threadIdx.x / (TILEX / TX), consecutive threads may
        // access the same values so only one of them can load it and share it
        // with the others via __shfl_sync.
        // smem_load<T, uss, sss, VSIZE>(&regY[y], &shared_input[load][0], i * TILEY + thready * TY + y, threadIdx.x);
        reinterpret_cast<T*>(&regY[y])[0] =
	  reinterpret_cast<T*>(&shared_input[load][i * TILEY + thready * TY + y])[0];
      }
#pragma unroll
      for (int x = 0; x < TX; x += VSIZE) {
        // Since threadx = threadIdx.x % (TILEX / TX), consecutive threads in a
        // warp do not access the same values so no __shfl_sync can be used in
        // general.
        reinterpret_cast<T *>(&regX[x])[0] = reinterpret_cast<T *>(
            &shared_values[load][i * TILEX + threadx * TX + x])[0];
      }
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++)
          acc[y * TX + x] += regY[y] * regX[x];
      }
    }

    load = load ^ 1;
    inputT += d * TILEK * batch_size;
    valuesT += TILEK * b;

    // Condition on row of valuesT
    if ((k + TILEK) < c) {
      // Load next batch from global to shared memory TILEY x TILEK
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideInput) {
        // Use T to load input into shared memory
        // input must use the col-major format
        reinterpret_cast<T *>(&shared_input[write][(InputSubCol + s) * TILEY +
                                                   InputSubRow * VSIZE])[0] =
            reinterpret_cast<T *>(&inputT[d * (InputSubCol + s) * batch_size +
                                          InputSubRow * VSIZE])[0];
      }
      // Load the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideValues) {
        // Reconstruct row and column indices
        tmp_row = ValuesSubRow + s;
        tmp_col = ValuesSubCol * VSIZE;
        reinterpret_cast<T *>(
            &shared_values[write][tmp_row * TILEX + tmp_col])[0] =
            reinterpret_cast<T *>(&valuesT[b * tmp_row + tmp_col % b])[0];
      }
      write = write ^ 1;
    }
  }

  //   // Store accumulation to shared memory
  // #pragma unroll
  //   for (int y = 0; y < TY; y++)
  //     {
  //       for (int x = 0; x < TX; x += VSIZE)
  // 	{
  // 	  shared_output[thready * TY + y + (threadx * TX + x + 0) * TILEY] =
  // 	    acc[y * TX + x + 0];
  // 	  shared_output[thready * TY + y + (threadx * TX + x + 1) * TILEY] =
  // 	    acc[y * TX + x + 1];
  // 	  shared_output[thready * TY + y + (threadx * TX + x + 2) * TILEY] =
  // 	    acc[y * TX + x + 2];
  // 	  shared_output[thready * TY + y + (threadx * TX + x + 3) * TILEY] =
  // 	    acc[y * TX + x + 3];
  // 	}
  //     }

  //   // Write out the accumulation (from shared to global memory)
  // #pragma unroll
  //   for (int y = 0; y < TY; y += VSIZE)
  //     {
  // #pragma unroll
  //       for (int x = 0; x < TX; x++)
  // 	{
  // 	  reinterpret_cast<T*>(
  // 				     &outputT[thready * TY + y + d * (threadx * TX + x)
  // * batch_size])[0] = 	    reinterpret_cast<T*>(
  // &shared_output[thready * TY + y + (threadx * TX + x) * TILEY])[0];
  // 	}
  //     }

  // Write out the accumulation (from registers to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int v = 0; v < VSIZE; v++)
	outputT[thready * TY + y + d * (threadx * TX + x + v) * batch_size] =
	  acc[y * TX + x + v];
    }
  }
}

#endif
