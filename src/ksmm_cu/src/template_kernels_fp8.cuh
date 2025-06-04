// -*- c++ -*-
#ifndef KERNELS_FP8
#define KERNELS_FP8

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdio.h>

// using namespace nvcuda;

template <typename T, typename Tx, __nv_fp8_interpretation_t fp8_i, const int TILEX, const int TILEK,
          const int TILEY, const int TX, const int TY, const int VSIZE>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_fp8(
__global__ void kernel_bs_first_fp8(T *input, T *valuesT, int batch_size,
                                    T *output, int a, int b, int c, int d);

template <typename T, typename Tx, __nv_fp8_interpretation_t fp8_i, const int TILEX, const int TILEK,
          const int TILEY, const int TX, const int TY, const int VSIZE>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_fp8e4m3(
__global__ void kernel_bs_first_fp8(T *input, T *valuesT, int batch_size,
                                    T *output, int a, int b, int c, int d) {
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
  __shared__ T shared_input[2][TILEY * TILEK];
  // To store sparse matrix in shared memory
  __shared__ T shared_values[2][TILEK * TILEX];
  // To store output in shared memory
  __shared__ T shared_output[TILEY * TILEX];
  T tmp_acc[TY * TX] = {static_cast<T>(0.0f)};
  T regY[TY] = {static_cast<T>(0.0f)};
  T regX[TX] = {static_cast<T>(0.0f)};
  __nv_fp8_storage_t storage1_t, storage2_t, storage3_t;

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
    reinterpret_cast<Tx *>(&shared_values[0][tmp_row * TILEX + tmp_col])[0] =
        reinterpret_cast<Tx *>(&valuesT[b * tmp_row + tmp_col % b])[0];
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
      for (int y = 0; y < TY; y += VSIZE)
        reinterpret_cast<Tx *>(&regY[y])[0] = reinterpret_cast<Tx *>(
            &shared_input[load][i * TILEY + thready * TY + y])[0];
#pragma unroll
      for (int x = 0; x < TX; x += VSIZE)
        reinterpret_cast<Tx *>(&regX[x])[0] = reinterpret_cast<Tx *>(
            &shared_values[load][i * TILEX + threadx * TX + x])[0];
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++) {
          storage1_t = __nv_fp8_storage_t(tmp_acc[y * TX + x]);
          storage2_t = __nv_fp8_storage_t(regY[y]);
          storage3_t = __nv_fp8_storage_t(regX[x]);
          tmp_acc[y * TX + x] = T(__nv_cvt_halfraw_to_fp8(
              __hadd(__nv_cvt_fp8_to_halfraw(storage1_t, fp8_i),
                     __hmul(__nv_cvt_fp8_to_halfraw(storage2_t, fp8_i),
                            __nv_cvt_fp8_to_halfraw(storage3_t, fp8_i))),
              __NV_NOSAT, fp8_i));
        }
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
        // input must use the row-major format
#pragma unroll
        for (int v = 0; v < VSIZE; v++)
          shared_input[write]
                      [(InputSubCol + s) * TILEY + InputSubRow * VSIZE + v] =
                          input[d * (InputSubCol + s) +
                                (InputSubRow * VSIZE + v) * input_size];
      }
      // Load next batch of the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideValues) {
        // Reconstruct row and column indices
        tmp_row = ValuesSubRow + s;
        tmp_col = ValuesSubCol * VSIZE;
        reinterpret_cast<Tx *>(
            &shared_values[write][tmp_row * TILEX + tmp_col])[0] =
            reinterpret_cast<Tx *>(&valuesT[b * tmp_row + tmp_col % b])[0];
      }
      write = write ^ 1;
    }
  }

  // Store accumulation to shared memory
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
      reinterpret_cast<Tx *>(
          &shared_output[(thready * TY + y) * TILEX + threadx * TX + x])[0] =
          reinterpret_cast<Tx *>(&tmp_acc[y * TX + x])[0];
    }
  }

  // Write out the accumulation (from shared to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int v = 0; v < VSIZE; v++)
        output[(thready * TY + y) * output_size + d * (threadx * TX + x + v)] =
            shared_output[(thready * TY + y) * TILEX + threadx * TX + x + v];
    }
  }
}

template <typename T, typename Tx, __nv_fp8_interpretation_t fp8_i, const int TILEX, const int TILEK,
          const int TILEY, const int TX, const int TY, const int VSIZE>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_fp8(
__global__ void kernel_bs_last_fp8(T *inputT, T *valuesT, int batch_size,
                                   T *outputT, int a, int b, int c, int d);

template <typename T, typename Tx, __nv_fp8_interpretation_t fp8_i, const int TILEX, const int TILEK,
          const int TILEY, const int TX, const int TY, const int VSIZE>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_fp8(
__global__ void kernel_bs_last_fp8(T *inputT, T *valuesT, int batch_size,
                                   T *outputT, int a, int b, int c, int d) {
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
  __shared__ T shared_input[2][TILEY * TILEK];
  // To store sparse matrix in shared memory
  __shared__ T shared_values[2][TILEK * TILEX];
  // To store output in shared memory
  __shared__ T shared_output[TILEY * TILEX];
  T tmp_acc[TY * TX] = {static_cast<T>(0.0f)};
  T regY[TY] = {static_cast<T>(0.0f)};
  T regX[TX] = {static_cast<T>(0.0f)};
  __nv_fp8_storage_t storage1_t, storage2_t, storage3_t;

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
    // inputT must use the col-major format
    reinterpret_cast<Tx *>(
        &shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE])[0] =
        reinterpret_cast<Tx *>(&inputT[d * (InputSubCol + s) * batch_size +
                                       InputSubRow * VSIZE])[0];
  }
  // Load the first batch of Butterfly factor from global to shared memory TILEK
  // * TILEX
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideValues) {
    // Reconstruct row and column indices
    tmp_row = ValuesSubRow + s;
    tmp_col = ValuesSubCol * VSIZE;
    reinterpret_cast<Tx *>(&shared_values[0][tmp_row * TILEX + tmp_col])[0] =
        reinterpret_cast<Tx *>(&valuesT[b * tmp_row + tmp_col % b])[0];
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
      for (int y = 0; y < TY; y += VSIZE)
        reinterpret_cast<Tx *>(&regY[y])[0] = reinterpret_cast<Tx *>(
            &shared_input[load][i * TILEY + thready * TY + y])[0];
#pragma unroll
      for (int x = 0; x < TX; x += VSIZE)
        reinterpret_cast<Tx *>(&regX[x])[0] = reinterpret_cast<Tx *>(
            &shared_values[load][i * TILEX + threadx * TX + x])[0];
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++) {
          storage1_t = __nv_fp8_storage_t(tmp_acc[y * TX + x]);
          storage2_t = __nv_fp8_storage_t(regY[y]);
          storage3_t = __nv_fp8_storage_t(regX[x]);
          tmp_acc[y * TX + x] = T(__nv_cvt_halfraw_to_fp8(
              __hadd(__nv_cvt_fp8_to_halfraw(storage1_t, fp8_i),
                     __hmul(__nv_cvt_fp8_to_halfraw(storage2_t, fp8_i),
                            __nv_cvt_fp8_to_halfraw(storage3_t, fp8_i))),
              __NV_NOSAT, fp8_i));
        }
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
        // input must use the col-major format
        reinterpret_cast<Tx *>(&shared_input[write][(InputSubCol + s) * TILEY +
                                                    InputSubRow * VSIZE])[0] =
            reinterpret_cast<Tx *>(&inputT[d * (InputSubCol + s) * batch_size +
                                           InputSubRow * VSIZE])[0];
      }
      // Load Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideValues) {
        // Reconstruct row and column indices
        tmp_row = ValuesSubRow + s;
        tmp_col = ValuesSubCol * VSIZE;
        reinterpret_cast<Tx *>(
            &shared_values[write][tmp_row * TILEX + tmp_col])[0] =
            reinterpret_cast<Tx *>(&valuesT[b * tmp_row + tmp_col % b])[0];
      }
      write = write ^ 1;
    }
  }

  // Store accumulation to shared memory
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int v = 0; v < VSIZE; v++)
        shared_output[thready * TY + y + (threadx * TX + x + v) * TILEY] =
            tmp_acc[y * TX + x + v];
    }
  }

  // Write out the accumulation (from shared to global memory)
#pragma unroll
  for (int y = 0; y < TY; y += VSIZE) {
#pragma unroll
    for (int x = 0; x < TX; x++) {
      reinterpret_cast<Tx *>(
          &outputT[thready * TY + y + d * (threadx * TX + x) * batch_size])[0] =
          reinterpret_cast<Tx *>(
              &shared_output[thready * TY + y + (threadx * TX + x) * TILEY])[0];
    }
  }
}

#endif
