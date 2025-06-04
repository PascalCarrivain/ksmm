// -*- c++ -*-

#define WARP_SIZE 32

#ifndef KERNELS_HALF
#define KERNELS_HALF

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdio.h>

#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF4_TO_UL(var) *(reinterpret_cast<unsigned long *>(&(var)))
#define __HALF4_TO_CUL(var) *(reinterpret_cast<const unsigned long *>(&(var)))

template <typename T, int sss>
__device__ inline void smem_load_sc(__half *reg, __half *shared_mem, int idx,
                                    int threadx) {
  // The first thread of each group of threads,which must load the same value,
  // starts to load the value. It is easy to identify the first thread of each
  // group when the size of the group divides 32, the warp size, just using
  // threadx = threadIdx.x % xXx. Otherwise we must do additional computations
  // to compute it correctly.
  if constexpr (sss == 4 || sss == 8 || sss == 16 || sss == 32 || sss == 64) {
    if (threadx == 0) {
      reinterpret_cast<T *>(&reg[0])[0] =
          reinterpret_cast<const T *>(&shared_mem[idx])[0];
    }
  } else if constexpr (sss == 6 || sss == 12) {
    // care with cases where xXx does not divide 32:
    // for xXx=12, threadx != 0 for thread 0 of warp 1, however it should load
    // for the next 12 so first compute the id inside our warp, and then decide
    // based on that if we should load, and to which we should shfl_sync
    int id_in_warp = threadIdx.x % WARP_SIZE;
    if (id_in_warp % sss == 0) {
      reinterpret_cast<T *>(&reg[0])[0] =
          reinterpret_cast<const T *>(&shared_mem[idx])[0];
    }
  }

  // then the first thread of each group shares the value among its group via
  // shfl_sync
  // __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
  // If width is set to a power of two less than warpSize then each subsection
  // of the warp behaves as a separate entity with a starting logical lane ID of
  // 0 Therefore, for dividers of 32, we can directly use the width argument
  // with a fixed mask equal to the first xXx threads of each separate entity
  // For non-dividers of 32, we cannot rely on an automatic separation in
  // groups, and the mask has to adapt to each group, as well as the srcLane
  if constexpr (sss == 32 || sss == 64) {
    reinterpret_cast<T *>(&reg[0])[0] =
        __shfl_sync(0xFFFFFFFF, reinterpret_cast<T *>(&reg[0])[0], 0);
  } else if constexpr (sss == 16) {
    reinterpret_cast<T *>(&reg[0])[0] =
        __shfl_sync(0x0000FFFF, reinterpret_cast<T *>(&reg[0])[0], 0, 16);
  } else if constexpr (sss == 12) {
    unsigned xx[3] = {0x00000FFF, 0x00FFF000, 0xFF000000};
    int id_in_warp = threadIdx.x % WARP_SIZE;
    int id_group = id_in_warp / sss;
    int id_group_leader = id_group * sss;
    reinterpret_cast<T *>(&reg[0])[0] = __shfl_sync(
        xx[id_group], reinterpret_cast<T *>(&reg[0])[0], id_group_leader);
  } else if constexpr (sss == 8) {
    reinterpret_cast<T *>(&reg[0])[0] =
        __shfl_sync(0x000000FF, reinterpret_cast<T *>(&reg[0])[0], 0, 8);
  } else if constexpr (sss == 6) {
    unsigned xx[5] = {0x0000003F, 0x00000FC0, 0x00FC0000, 0x3F000000,
                      0xC0000000};
    int id_in_warp = threadIdx.x % WARP_SIZE;
    int id_group = id_in_warp / sss;
    int id_group_leader = id_group * sss;
    reinterpret_cast<T *>(&reg[0])[0] = __shfl_sync(
        xx[id_group], reinterpret_cast<T *>(&reg[0])[0], id_group_leader);
  } else if constexpr (sss == 4) {
    reinterpret_cast<T *>(&reg[0])[0] =
        __shfl_sync(0x0000000F, reinterpret_cast<T *>(&reg[0])[0], 0, 4);
  }
}

template <typename T, bool uss, int sss>
__device__ inline void smem_load(__half *reg, __half *shared_mem, int idx,
                                 int threadx) {
  // avoid shfl sync if size is too small, as not worth it
  if constexpr (uss && sss >= 4) {
    smem_load_sc<T, sss>(reg, shared_mem, idx, threadx);
  } else {
    reinterpret_cast<T *>(&reg[0])[0] =
        reinterpret_cast<T *>(&shared_mem[idx])[0];
  }
}

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_half(
__global__ void kernel_bs_first_half2(half *input, half *valuesT,
                                      int batch_size, half *output, int a,
                                      int b, int c, int d);

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_half(
__global__ void kernel_bs_first_half2(half *input, half *valuesT,
                                      int batch_size, half *output, int a,
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
  // x = TILEX / TX threads per column
  // Get the current thread
  int threadx = threadIdx.x % (TILEX / TX);
  int thready = threadIdx.x / (TILEX / TX);
  // To store input in shared memory
  __shared__ __half shared_input[2][TILEY * TILEK];
  // To store sparse matrix in shared memory
  __shared__ __half shared_values[2][TILEK * TILEX];
  // // To store output in shared memory
  // __shared__ __half shared_output[TILEY * TILEX];
  __half tmp_acc[TY * TX] = {__float2half(0.0f)};
  __half regY[TY] = {__float2half(0.0f)};
  __half regX[TX] = {__float2half(0.0f)};

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

  // Load (half2) the first batch of input from global to shared memory TILEY
  // * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput) {
    shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + 0] =
        input[d * (InputSubCol + s) + (InputSubRow * VSIZE + 0) * input_size];
    shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + 1] =
        input[d * (InputSubCol + s) + (InputSubRow * VSIZE + 1) * input_size];
  }
  // Load (half2) the Butterfly factor in shared memory TILEK x TILEX
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
      for (int x = 0; x < TX; x += VSIZE) {
        // Since threadx = threadIdx.x % (TILEX / TX), consecutive threads in
        // a warp do not access the same values so no shfl_sync can be used in
        // general
        reinterpret_cast<T *>(&regX[x])[0] = reinterpret_cast<T *>(
            &shared_values[load][i * TILEX + threadx * TX + x])[0];
      }
#pragma unroll
      for (int y = 0; y < TY; y += VSIZE) {
        // Since thready = threadIdx.x / (TILEX / TX), consecutive threads may
        // access the same values so only one of them can load it and share it
        // with the others via shfl_sync
        smem_load<__half2, uss, sss>(&regY[y], &shared_input[load][0],
                                     i * TILEY + thready * TY + y, threadx);
        // reinterpret_cast<T*>(&regY[y])[0] =
        // reinterpret_cast<T*>(&shared_input[load][i * TILEY + thready * TY +
        // y])[0];
      }
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++) {
          tmp_acc[y * TX + x] =
              __hadd(tmp_acc[y * TX + x], __hmul(regY[y], regX[x]));
        }
      }
    }

    load = load ^ 1;
    input += d * TILEK;
    valuesT += TILEK * b;

    // Load (half2) next batch of input from global to shared memory TILEY *
    // TILEK Condition on row of valuesT
    if ((k + TILEK) < c) {
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideInput) {
        // input must use the row-major format
        shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * VSIZE +
                            0] = input[d * (InputSubCol + s) +
                                       (InputSubRow * VSIZE + 0) * input_size];
        shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * VSIZE +
                            1] = input[d * (InputSubCol + s) +
                                       (InputSubRow * VSIZE + 1) * input_size];
      }
      // Load (half2) the Butterfly factor in shared memory TILEK x TILEX
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
  // 				     &shared_output[(thready * TY + y) * TILEX +
  // threadx
  // * TX
  // + x])[0] = 	    reinterpret_cast<T*>(&tmp_acc[y * TX + x])[0];
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
  // = shared_output[(thready * TY + y) * TILEX + threadx * TX + x + 0];
  // 	  output[(thready * TY + y) * output_size + d * (threadx * TX + x + 1)]
  // = shared_output[(thready * TY + y) * TILEX + threadx * TX + x + 1];
  // 	}
  //     }

  // Write out the accumulation (from registers to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int l = 0; l < VSIZE; l++)
        output[(thready * TY + y) * output_size + d * (threadx * TX + x + l)] =
            tmp_acc[y * TX + x + l];
    }
  }
}

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_half(
__global__ void kernel_bs_first_half4(half *input, half *valuesT,
                                      int batch_size, half *output, int a,
                                      int b, int c, int d);

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_first_half(
__global__ void kernel_bs_first_half4(half *input, half *valuesT,
                                      int batch_size, half *output, int a,
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
  asm volatile(".shared .align 2 .b16 shared_input[%0];\n\t"
               :
               : "n"(2 * TILEY * TILEK));
  // To store values in shared memory
  asm volatile(".shared .align 2 .b16 shared_values[%0];\n\t"
               :
               : "n"(2 * TILEX * TILEK));
  __half tmp_acc[TY * TX] = {__float2half(0.0f)};
  __half regY[TY] = {__float2half(0.0f)};
  __half regX[TX] = {__float2half(0.0f)};

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

  long tmp64;
  int sharedByteOffset;
  __half tmp16x4[4];

  // Load (4 half) the first batch of input from global to shared memory TILEY
  // * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput) {
    // instead of doing 4 separate global -> reg and then 4 separate reg ->
    // shared, we first do the 4 global -> reg (non-consecutive adresses in
    // global mem), and then 4 reg -> shared at once (consecutive adresses in
    // shared mem)
#pragma unroll
    for (int l = 0; l < VSIZE; l++) {
      // Option 1 (equally as fast as other options, on chain 1.48.48.1 with
      // hyperparams TILEX=8 TILEK=8 TILEY=8 TX=8 TY=4, we will see a slight
      // difference (about 1%) below when comparing the 3 options at a place
      // where the code is executed multiple times):
      // shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + l]
      // =
      // __ldg(&input[d * (InputSubCol + s) + (InputSubRow * VSIZE + l) *
      // input_size]); Option 2:
      asm volatile("ld.global.nc.b16 %0, [%1];\n\t"
                   : "=h"(__HALF_TO_US(tmp16x4[l]))
                   : "l"(&input[d * (InputSubCol + s) +
                                (InputSubRow * VSIZE + l) * input_size]));
    }
    // Part of Option 2:
    sharedByteOffset =
        sizeof(__half) * ((InputSubCol + s) * TILEY + InputSubRow * VSIZE);
    asm volatile("st.shared.b64 [%0], %1;\n\t"
                 :
                 : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp16x4))
                 : "memory");
    // Option 3:
    // shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + 0] =
    // input[d * (InputSubCol + s) + (InputSubRow * VSIZE + 0) * input_size];
    // shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + 1] =
    // input[d * (InputSubCol + s) + (InputSubRow * VSIZE + 1) * input_size];
    // shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + 2] =
    // input[d * (InputSubCol + s) + (InputSubRow * VSIZE + 2) * input_size];
    // shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + 3] =
    // input[d * (InputSubCol + s) + (InputSubRow * VSIZE + 3) * input_size];
  }

  // Load (4 half) the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideValues) {
    // Reconstruct row and column indices
    tmp_row = ValuesSubRow + s;
    tmp_col = ValuesSubCol * VSIZE;

    // Option 1 (seems a bit faster, about 0.5%):
    sharedByteOffset =
        sizeof(__half) * (2 * TILEY * TILEK + (tmp_row * TILEX + tmp_col));
    asm volatile("ld.global.nc.b64 %0, [%1];\n\t"
                 : "=l"(__HALF4_TO_UL(tmp64))
                 : "l"(&valuesT[b * tmp_row + tmp_col % b]));

    asm volatile("st.shared.b64 [%0], %1;\n\t"
                 :
                 : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp64))
                 : "memory");
    // Option 2:
    // #pragma unroll
    //       for (int l = 0; l < VSIZE; l++)
    // 	shared_values[0][tmp_row * TILEX + tmp_col + l] =
    // __ldg(&valuesT[b * tmp_row + tmp_col % b + l]);//, 4 *
    // sizeof(valuesT));
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
        // Option 1 (it runs, but wrong mse for some reason):
        // sharedByteOffset = sizeof(__half) * (load * TILEY * TILEK + i *
        // TILEY
        // + thready * TY + y); asm volatile(
        //     "ld.shared.b64 %0, [%1];\n\t"
        //     : "=l"(__HALF4_TO_UL(regY[y]))
        //     : "r"(sharedByteOffset)
        // //     // : "l"(&shared_input[load][i * TILEY + thready * TY + y])
        // );
        // sharedByteOffset = sizeof(__half) * (load * TILEY * TILEK + i *
        // TILEY
        // + thready * TY + y); #pragma unroll
        sharedByteOffset = sizeof(__half) * (load * TILEY * TILEK + i * TILEY +
                                             thready * TY + y + 0);
        for (int l = 0; l < VSIZE; l++) {
          // Option 2 (seems 0.5% faster than Option 4):
          // regY[y+l] = shared_input[load][i * TILEY + thready * TY + y + l];
          // regY[y+l] = reinterpret_cast<half *>(&shared_input[0][0] + load *
          // TILEY * TILEK + i * TILEY + thready * TY + y + l)[0];
          // Option 3 (it runs but wrong mse for some reason):
          // sharedByteOffset = sizeof(__half) * (load * TILEY * TILEK + i *
          // TILEY + thready * TY + y + l);
          asm volatile("ld.shared::cta.u16 %0, [%1];\n\t"
                       : "=h"(__HALF_TO_US(regY[y + l]))
                       // : "l"(sharedByteOffset+sizeof(__half)*l)
                       : "r"(sharedByteOffset));
          sharedByteOffset += sizeof(__half);
          // asm volatile(
          //     "ld.shared.f16 %0, [%1];\n\t"
          //     : "=h"(__HALF_TO_US(regY[y+l]))
          //     // : "l"(sharedByteOffset+sizeof(__half)*l)
          //     : "r"(sharedByteOffset)
          //     : "memory"
          // );
        }
        // Option 4:
        // reinterpret_cast<half2 *>(&regY[y])[0] =
        //   reinterpret_cast<half2 *>(&shared_input[load][i * TILEY + thready
        //   * TY + y])[0];
        // reinterpret_cast<half2 *>(&regY[y + 2])[0] =
        //   reinterpret_cast<half2 *>(&shared_input[load][i * TILEY + thready
        //   * TY + y + 2])[0];
      }
      sharedByteOffset =
          sizeof(__half) * (2 * TILEK * TILEY + load * TILEX * TILEK +
                            i * TILEX + threadx * TX + 0);
#pragma unroll
      for (int x = 0; x < TX; x += VSIZE) {
        // reinterpret_cast<half2 *>(&regX[x])[0] =
        //   reinterpret_cast<half2 *>(&shared_values[load][i * TILEX +
        //   threadx
        //   * TX + x])[0];
        // reinterpret_cast<half2 *>(&regX[x + 2])[0] =
        //   reinterpret_cast<half2 *>(&shared_values[load][i * TILEX +
        //   threadx
        //   * TX + x + 2])[0];
        // sharedByteOffset = sizeof(__half) * (2 * TILEK * TILEY + load *
        // TILEX
        // * TILEK + i * TILEX + threadx * TX + x);
        asm volatile("ld.shared.b64 %0, [%1];\n\t"
                     : "=l"(__HALF4_TO_UL(regX[x]))
                     : "r"(sharedByteOffset));
        sharedByteOffset += sizeof(__half) * VSIZE;
      }
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++)
          tmp_acc[y * TX + x] =
              __hadd(tmp_acc[y * TX + x], __hmul(regY[y], regX[x]));
      }
    }

    load = load ^ 1;
    input += d * TILEK;
    valuesT += TILEK * b;

    // Load (4 half) next batch of input from global to shared memory TILEY *
    // TILEK Condition on row of valuesT
    if ((k + TILEK) < c) {
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideInput) {
        // input must use the row-major format
#pragma unroll
        for (int l = 0; l < VSIZE; l++) {
          // Option 1 (seems a bit faster about 1%: Option 1 < by 0.5 % of
          // Option 3 < by 0.5% of Option 2 in terms of speed):
          // shared_input[write][(InputSubCol + s) * TILEY + InputSubRow *
          // VSIZE
          // + l] = __ldg(&input[d * (InputSubCol + s) + (InputSubRow * VSIZE
          // + l) * input_size]); Option 2:
          asm volatile("ld.global.nc.b16 %0, [%1];\n\t"
                       : "=h"(__HALF_TO_US(tmp16x4[l]))
                       : "l"(&input[d * (InputSubCol + s) +
                                    (InputSubRow * VSIZE + l) * input_size]));
        }
        // End of option 2:
        sharedByteOffset =
            sizeof(__half) * (write * TILEY * TILEK +
                              (InputSubCol + s) * TILEY + InputSubRow * VSIZE);
        asm volatile("st.shared.b64 [%0], %1;\n\t"
                     :
                     : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp16x4))
                     : "memory");
        // Option 3:
        // shared_input[write][(InputSubCol + s) * TILEY + InputSubRow *
        // VSIZE] = input[d * (InputSubCol + s) + (InputSubRow * VSIZE) *
        // input_size]; shared_input[write][(InputSubCol + s) * TILEY +
        // InputSubRow * VSIZE + 1] = input[d * (InputSubCol + s) +
        // (InputSubRow * VSIZE + 1) * input_size];
        // shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * VSIZE
        // + 2] = input[d * (InputSubCol + s) + (InputSubRow
        // * VSIZE + 2) * input_size]; shared_input[write][(InputSubCol + s) *
        // TILEY + InputSubRow * VSIZE + 3] = input[d * (InputSubCol + s) +
        // (InputSubRow * VSIZE + 3) * input_size];
      }
      // Load (4 half) the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideValues) {
        // Reconstruct row and column indices
        tmp_row = ValuesSubRow + s;
        tmp_col = ValuesSubCol * VSIZE;
        // Option 1 (seems way faster, about 17%):
        sharedByteOffset =
            sizeof(__half) * (write * TILEX * TILEK + 2 * TILEY * TILEK +
                              (tmp_row * TILEX + tmp_col));
        asm volatile("ld.global.nc.b64 %0, [%1];\n\t"
                     : "=l"(__HALF4_TO_UL(tmp64))
                     : "l"(&valuesT[b * tmp_row + tmp_col % b]));

        asm volatile("st.shared.b64 [%0], %1;\n\t"
                     :
                     : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp64))
                     : "memory");
        // Option 2:
        // #pragma unroll
        // 	      for (int l = 0; l < VSIZE; l++)
        // 		shared_values[write][tmp_row * TILEX + tmp_col + l] =
        // __ldg(&valuesT[b * tmp_row + tmp_col % b + l]);//, 4 *
        // sizeof(valuesT));
      }

      write = write ^ 1;
    }
  }

  // Write out the accumulation (from shared to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int l = 0; l < VSIZE; l++)
        output[(thready * TY + y) * output_size + d * (threadx * TX + x + l)] =
            tmp_acc[y * TX + x + l];
    }
  }
}

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_half(
__global__ void kernel_bs_last_half2(half *inputT, half *valuesT,
                                     int batch_size, half *outputT, int a,
                                     int b, int c, int d);

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_half(
__global__ void kernel_bs_last_half2(half *inputT, half *valuesT,
                                     int batch_size, half *outputT, int a,
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
  // x = TILEX / TX threads per column
  // Get the current thread
  int threadx = threadIdx.x % (TILEX / TX);
  int thready = threadIdx.x / (TILEX / TX);
  // To store input in shared memory
  __shared__ __half shared_input[2][TILEY * TILEK];
  // To store sparse matrix in shared memory
  __shared__ __half shared_values[2][TILEK * TILEX];
  // // To store output in shared memory
  // __shared__ __half shared_output[TILEY * TILEX];
  __half tmp_acc[TY * TX] = {__float2half(0.0f)};
  __half regY[TY] = {__float2half(0.0f)};
  __half regX[TX] = {__float2half(0.0f)};

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

  // Load (half2) the first batch of input from global to shared memory TILEY
  // * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput) {
    // Use half2 to load input into shared memory
    // inputT must use the col-major format
    reinterpret_cast<T *>(
        &shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE])[0] =
        reinterpret_cast<T *>(&inputT[d * (InputSubCol + s) * batch_size +
                                      InputSubRow * VSIZE])[0];
  }
  // Load (half2) the first batch of Butterfly factor from global to shared
  // memory TILEK * TILEX
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
      for (int x = 0; x < TX; x += VSIZE) {
        // Since threadx = threadIdx.x % (TILEX / TX), consecutive threads in
        // a warp do not access the same values so no shfl_sync can be used in
        // general.
        reinterpret_cast<T *>(&regX[x])[0] = reinterpret_cast<T *>(
            &shared_values[load][i * TILEX + threadx * TX + x])[0];
      }
#pragma unroll
      for (int y = 0; y < TY; y += VSIZE) {
        // Since thready = threadIdx.x / (TILEX / TX), consecutive threads may
        // access the same values so only one of them can load it and share it
        // with the others via shfl_sync.
        smem_load<__half2, uss, sss>(&regY[y], &shared_input[load][0],
                                     i * TILEY + thready * TY + y, threadx);
        // reinterpret_cast<T*>(&regY[y])[0] =
        // reinterpret_cast<T*>(&shared_input[load][i * TILEY + thready * TY +
        // y])[0];
      }
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++)
          tmp_acc[y * TX + x] =
              __hadd(tmp_acc[y * TX + x], __hmul(regY[y], regX[x]));
      }
    }

    load = load ^ 1;
    inputT += d * TILEK * batch_size;
    valuesT += TILEK * b;

    // Condition on row of valuesT
    if ((k + TILEK) < c) {
      // Load (half2) next batch of input from global to shared memory TILEY x
      // TILEK
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideInput) {
        // Use half2 to load input into shared memory
        // input must use the col-major format
        reinterpret_cast<T *>(&shared_input[write][(InputSubCol + s) * TILEY +
                                                   InputSubRow * VSIZE])[0] =
            reinterpret_cast<T *>(&inputT[d * (InputSubCol + s) * batch_size +
                                          InputSubRow * VSIZE])[0];
      }
      // Load (half2) the Butterfly factor in shared memory TILEK x TILEX
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
  // 	    tmp_acc[y * TX + x + 0];
  // 	  shared_output[thready * TY + y + (threadx * TX + x + 1) * TILEY] =
  // 	    tmp_acc[y * TX + x + 1];
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
  // 			       &outputT[thready * TY + y + d * (threadx * TX +
  // x)
  // *
  // batch_size])[0] = 	    reinterpret_cast<T*>(
  // &shared_output[thready * TY + y + (threadx * TX + x) * TILEY])[0];
  // 	}
  //     }

  // Write out the accumulation (from registers to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int l = 0; l < VSIZE; l++)
        outputT[thready * TY + y + d * (threadx * TX + x + l) * batch_size] =
            tmp_acc[y * TX + x + l];
    }
  }
}

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_half(
__global__ void kernel_bs_last_half4(half *inputT, half *valuesT,
                                     int batch_size, half *outputT, int a,
                                     int b, int c, int d);

template <typename T, const int TILEX, const int TILEK, const int TILEY,
          const int TX, const int TY, const int VSIZE, const bool uss,
          const int sss>
// __global__ __launch_bounds__(xNTHREADSx) void kernel_bs_last_half(
__global__ void kernel_bs_last_half4(half *inputT, half *valuesT,
                                     int batch_size, half *outputT, int a,
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
  asm volatile(".shared .align 2 .b16 shared_input[%0];\n\t"
               :
               : "n"(2 * TILEY * TILEK));
  // To store values in shared memory
  asm volatile(".shared .align 2 .b16 shared_values[%0];\n\t"
               :
               : "n"(2 * TILEX * TILEK));

  __half tmp_acc[TY * TX] = {__float2half(0.0f)};
  __half regY[TY] = {__float2half(0.0f)};
  __half regX[TX] = {__float2half(0.0f)};

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

  long tmp64;
  int sharedByteOffset;

  // Load (half2) the first batch of input from global to shared memory TILEY
  // * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput) {
    // Use half2 to load input into shared memory
    // inputT must use the col-major format
    //     for(int l = 0; l < VSIZE; l++)
    // shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * VSIZE + l] =
    // __ldg(&inputT[d * (InputSubCol + s) * batch_size + InputSubRow * VSIZE
    // + l]);
    //     // reinterpret_cast<half2 *>(
    //     // 				 &shared_input[0][(InputSubCol +
    //     s)
    //     * TILEY
    //     + InputSubRow * VSIZE])[0] =
    //     //   reinterpret_cast<half2 *>(
    //     // 				   &inputT[d * (InputSubCol + s)
    //     * batch_size
    //     + InputSubRow * VSIZE])[0];
    asm volatile("ld.global.nc.b64 %0, [%1];\n\t"
                 : "=l"(__HALF4_TO_UL(tmp64))
                 : "l"(&inputT[d * (InputSubCol + s) * batch_size +
                               InputSubRow * VSIZE]));
    sharedByteOffset =
        sizeof(__half) * ((InputSubCol + s) * TILEY + InputSubRow * VSIZE);
    asm volatile("st.shared.b64 [%0], %1;\n\t"
                 :
                 : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp64))
                 : "memory");
  }
  // Load (half2) the first batch of Butterfly factor from global to shared
  // memory TILEK * TILEX
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideValues) {
    // Reconstruct row and column indices
    tmp_row = ValuesSubRow + s;
    tmp_col = ValuesSubCol * VSIZE;
    //     for(int l = 0; l < VSIZE; l++)
    // shared_values[0][tmp_row * TILEX + tmp_col + l] = __ldg(&valuesT[b *
    // tmp_row + tmp_col % b + l]);
    asm volatile("ld.global.nc.b64 %0, [%1];\n\t"
                 : "=l"(__HALF4_TO_UL(tmp64))
                 : "l"(&valuesT[b * tmp_row + tmp_col % b]));
    sharedByteOffset =
        sizeof(__half) * (2 * TILEY * TILEK + tmp_row * TILEX + tmp_col);
    asm volatile("st.shared.b64 [%0], %1;\n\t"
                 :
                 : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp64))
                 : "memory");
    // reinterpret_cast<half2 *>(
    // 				 &shared_values[0][tmp_row * TILEX +
    // tmp_col])[0] = 	reinterpret_cast<half2 *>(&valuesT[b * tmp_row +
    // tmp_col % b])[0];
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
        // reinterpret_cast<half2 *>(&regY[y])[0] = reinterpret_cast<half2 *>(
        // 						       &shared_input[load][i
        // * TILEY
        // + thready * TY + y])[0]; reinterpret_cast<half2 *>(&regY[y + 2])[0]
        // = reinterpret_cast<half2
        // *>( &shared_input[load][i
        // * TILEY + thready * TY + y + 2])[0];
        sharedByteOffset = sizeof(__half) * (load * TILEY * TILEK + i * TILEY +
                                             thready * TY + y);
        asm volatile("ld.shared.b64 %0, [%1];\n\t"
                     : "=l"(__HALF4_TO_UL(regY[y]))
                     : "r"(sharedByteOffset));
      }
#pragma unroll
      for (int x = 0; x < TX; x += VSIZE) {
        // reinterpret_cast<half2 *>(&regX[x])[0] = reinterpret_cast<half2 *>(
        // 						       &shared_values[load][i
        // * TILEX
        // + threadx * TX + x])[0]; reinterpret_cast<half2 *>(&regX[x + 2])[0]
        // = reinterpret_cast<half2
        // *>( &shared_values[load][i
        // * TILEX + threadx * TX + x + 2])[0];
        sharedByteOffset =
            sizeof(__half) * (2 * TILEK * TILEY + load * TILEX * TILEK +
                              i * TILEX + threadx * TX + x);
        asm volatile("ld.shared.b64 %0, [%1];\n\t"
                     : "=l"(__HALF4_TO_UL(regX[x]))
                     : "r"(sharedByteOffset));
      }
#pragma unroll
      for (int y = 0; y < TY; y++) {
#pragma unroll
        for (int x = 0; x < TX; x++)
          tmp_acc[y * TX + x] =
              __hadd(tmp_acc[y * TX + x], __hmul(regY[y], regX[x]));
      }
    }

    load = load ^ 1;
    inputT += d * TILEK * batch_size;
    valuesT += TILEK * b;

    // Condition on row of valuesT
    if ((k + TILEK) < c) {
      // Load (half2) next batch of input from global to shared memory TILEY x
      // TILEK
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideInput) {
        // Use half2 to load input into shared memory
        // input must use the col-major format
        //     for(int l = 0; l < VSIZE; l++)
        // shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * VSIZE
        // + l] = __ldg(&inputT[d * (InputSubCol + s) * batch_size +
        // InputSubRow * VSIZE + l]);
        asm volatile("ld.global.nc.b64 %0, [%1];\n\t"
                     : "=l"(__HALF4_TO_UL(tmp64))
                     : "l"(&inputT[d * (InputSubCol + s) * batch_size +
                                   InputSubRow * VSIZE]));

        sharedByteOffset =
            sizeof(__half) * (write * TILEY * TILEK +
                              (InputSubCol + s) * TILEY + InputSubRow * VSIZE);
        asm volatile("st.shared.b64 [%0], %1;\n\t"
                     :
                     : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp64))
                     : "memory");
        // reinterpret_cast<doublef2 *>(
        // 				 &shared_input[write][(InputSubCol + s)
        // * TILEY
        // +
        // InputSubRow * VSIZE])[0] = 	reinterpret_cast<double2
        // *>(&inputT[d
        // * (InputSubCol + s) * batch_size + InputSubRow * VSIZE])[0];
      }
      // Load (half2) the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
      for (int s = 0; s < TILEK; s += StrideValues) {
        // Reconstruct row and column indices
        tmp_row = ValuesSubRow + s;
        tmp_col = ValuesSubCol * VSIZE;
        //     for(int l = 0; l < VSIZE; l++)
        // shared_values[write][tmp_row * TILEX + tmp_col + l] =
        // __ldg(&valuesT[b * tmp_row + tmp_col % b + l]);

        asm volatile("ld.global.nc.b64 %0, [%1];\n\t"
                     : "=l"(__HALF4_TO_UL(tmp64))
                     : "l"(&valuesT[b * tmp_row + tmp_col % b]));

        sharedByteOffset =
            sizeof(__half) * (2 * TILEY * TILEK + write * TILEX * TILEK +
                              tmp_row * TILEX + tmp_col);
        asm volatile("st.shared.b64 [%0], %1;\n\t"
                     :
                     : "r"(sharedByteOffset), "l"(__HALF4_TO_CUL(tmp64))
                     : "memory");
        // reinterpret_cast<double2 *>(
        // 				 &shared_values[write]
        // 				 [tmp_row * TILEX + tmp_col])[0] =
        // 	reinterpret_cast<double2 *>(
        // 				   &valuesT[b * tmp_row + tmp_col %
        // b])[0];
      }
      write = write ^ 1;
    }
  }

  // Write out the accumulation (from shared to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++) {
#pragma unroll
    for (int x = 0; x < TX; x += VSIZE) {
#pragma unroll
      for (int l = 0; l < VSIZE; l++)
        outputT[thready * TY + y + d * (threadx * TX + x + l) * batch_size] =
            tmp_acc[y * TX + x + l];
    }
  }
}

#endif
