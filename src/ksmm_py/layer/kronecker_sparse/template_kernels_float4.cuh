// -*- c++ -*-
#ifndef KERNELS_FLOAT4
#define KERNELS_FLOAT4

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdio.h>

#define WARP_SIZE 32
#define xNTHREADS_FLOAT4x 256

// using namespace nvcuda;


template <const int TILEX, const int TILEK, const int TILEY, const int TX,
          const int TY>
__global__ __launch_bounds__(xNTHREADS_FLOAT4x) void kernel_bs_first_float4(
    float *input, float *valuesT, int batch_size,
    float *output, int a, int b, int c, int d);

template <const int TILEX, const int TILEK, const int TILEY, const int TX,
          const int TY>
__global__ __launch_bounds__(xNTHREADS_FLOAT4x) void kernel_bs_first_float4(
    float *input, float *valuesT, int batch_size,
    float *output, int a, int b, int c, int d)
{
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
  // To store output in shared memory
  __shared__ float shared_output[TILEY * TILEX];
  float tmp_acc[TY * TX] = {0.0f};
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
  int InputSubRow = threadIdx.x % (TILEY / 4);
  int InputSubCol = threadIdx.x / (TILEY / 4);

  // Indices to load (Butterfly factor)
  int ValuesSubRow = threadIdx.x / (TILEX / 4);
  int ValuesSubCol = threadIdx.x % (TILEX / 4);

  // Use stride to load from GMEM to SMEM
  const int nthreads = (TILEX / TX) * (TILEY / TY);
  const int StrideInput = 4 * nthreads / TILEY;
  const int StrideValues = 4 * nthreads / TILEX;

  int tmp_col, tmp_row;

  // Load (float4) the first batch of input from global to shared memory TILEY * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput)
    {
      shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * 4 + 0] = input[d * (InputSubCol + s) + (InputSubRow * 4 + 0) * input_size];
      shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * 4 + 1] = input[d * (InputSubCol + s) + (InputSubRow * 4 + 1) * input_size];
      shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * 4 + 2] = input[d * (InputSubCol + s) + (InputSubRow * 4 + 2) * input_size];
      shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * 4 + 3] = input[d * (InputSubCol + s) + (InputSubRow * 4 + 3) * input_size];
    }
  // Load (float4) the first batch of the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideValues)
    {
      // Reconstruct row and column indices
      tmp_row = ValuesSubRow + s;
      tmp_col = ValuesSubCol * 4;
      reinterpret_cast<float4 *>(
				 &shared_values[0][tmp_row * TILEX + tmp_col])[0] =
	reinterpret_cast<float4 *>(&valuesT[b * tmp_row + tmp_col % b])[0];
    }

  int load = 0;
  int write = 1;

  // Loop over non-zero entries by TILEK
  for (int k = 0; k < c; k += TILEK)
    {
      __syncthreads();
      // Load smem in register and compute accumulation
#pragma unroll
      for (int i = 0; i < TILEK; i++)
	{
#pragma unroll
	  for (int y = 0; y < TY; y += 4)
	    reinterpret_cast<float4 *>(&regY[y])[0] = reinterpret_cast<float4 *>(
										 &shared_input[load][i * TILEY + thready * TY + y])[0];
#pragma unroll
	  for (int x = 0; x < TX; x += 4)
	    reinterpret_cast<float4 *>(&regX[x])[0] = reinterpret_cast<float4 *>(
										 &shared_values[load][i * TILEX + threadx * TX + x])[0];
#pragma unroll
	  for (int y = 0; y < TY; y++)
	    {
#pragma unroll
	      for (int x = 0; x < TX; x++)
		tmp_acc[y * TX + x] += regY[y] * regX[x];
	    }
	}

      load = load ^ 1;
      input += d * TILEK;
      valuesT += TILEK * b;

      // Load (float4) next batch of the input from global to shared memory TILEY * TILEK
      // Condition on row of valuesT
      if ((k + TILEK) < c)
	{
#pragma unroll
	  for (int s = 0; s < TILEK; s += StrideInput)
	    {
	      // Use float4 to load input into shared memory
	      // input must use the row-major format
	      shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * 4] = input[d * (InputSubCol + s) + (InputSubRow * 4) * input_size];
	      shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * 4 + 1] = input[d * (InputSubCol + s) + (InputSubRow * 4 + 1) * input_size];
	      shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * 4 + 2] = input[d * (InputSubCol + s) + (InputSubRow * 4 + 2) * input_size];
	      shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * 4 + 3] = input[d * (InputSubCol + s) + (InputSubRow * 4 + 3) * input_size];
	    }
	  // Load (float4) next batch of the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
	  for (int s = 0; s < TILEK; s += StrideValues)
	    {
	      // Reconstruct row and column indices
	      tmp_row = ValuesSubRow + s;
	      tmp_col = ValuesSubCol * 4;
	      reinterpret_cast<float4 *>(
					 &shared_values[write]
					 [tmp_row * TILEX + tmp_col])[0] =
		reinterpret_cast<float4 *>(
					   &valuesT[b * tmp_row + tmp_col % b])[0];
	    }
	  write = write ^ 1;
	}
    }

  // Store accumulation to shared memory
#pragma unroll
  for (int y = 0; y < TY; y++)
    {
      for (int x = 0; x < TX; x += 4)
	{
	  reinterpret_cast<float4 *>(
				     &shared_output[(thready * TY + y) * TILEX + threadx * TX + x])[0] =
	    reinterpret_cast<float4 *>(&tmp_acc[y * TX + x])[0];
	}
    }

  // Write out the accumulation (from shared to global memory)
#pragma unroll
  for (int y = 0; y < TY; y++)
    {
#pragma unroll
      for (int x = 0; x < TX; x += 4)
	{
	  output[(thready * TY + y) * output_size + d * (threadx * TX + x + 0)] =
	    shared_output[(thready * TY + y) * TILEX + threadx * TX + x + 0];
	  output[(thready * TY + y) * output_size + d * (threadx * TX + x + 1)] =
	    shared_output[(thready * TY + y) * TILEX + threadx * TX + x + 1];
	  output[(thready * TY + y) * output_size + d * (threadx * TX + x + 2)] =
	    shared_output[(thready * TY + y) * TILEX + threadx * TX + x + 2];
	  output[(thready * TY + y) * output_size + d * (threadx * TX + x + 3)] =
	    shared_output[(thready * TY + y) * TILEX + threadx * TX + x + 3];
	}
    }
}


template <const int TILEX, const int TILEK, const int TILEY, const int TX,
          const int TY>
__global__ __launch_bounds__(xNTHREADS_FLOAT4x) void kernel_bs_last_float4(
    float *inputT, float *valuesT, int batch_size,
    float *outputT, int a, int b, int c, int d);

template <const int TILEX, const int TILEK, const int TILEY, const int TX,
          const int TY>
__global__ __launch_bounds__(xNTHREADS_FLOAT4x) void kernel_bs_last_float4(
    float *inputT, float *valuesT, int batch_size,
    float *outputT, int a, int b, int c, int d)
{
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
  // To store output in shared memory
  __shared__ float shared_output[TILEY * TILEX];
  float tmp_acc[TY * TX] = {0.0f};
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
  int InputSubRow = threadIdx.x % (TILEY / 4);
  int InputSubCol = threadIdx.x / (TILEY / 4);

  // Indices to load (Butterfly factor)
  int ValuesSubRow = threadIdx.x / (TILEX / 4);
  int ValuesSubCol = threadIdx.x % (TILEX / 4);

  // Use stride to load from GMEM to SMEM
  const int nthreads = (TILEX / TX) * (TILEY / TY);
  const int StrideInput = 4 * nthreads / TILEY;
  const int StrideValues = 4 * nthreads / TILEX;

  int tmp_col, tmp_row;

  // Load (float4) the first batch of input from global to shared memory TILEY * TILEK
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideInput)
    {
      // Use float4 to load input into shared memory
      // inputT must use the col-major format
      reinterpret_cast<float4 *>(
				 &shared_input[0][(InputSubCol + s) * TILEY + InputSubRow * 4])[0] =
        reinterpret_cast<float4 *>(
				   &inputT[d * (InputSubCol + s) * batch_size + InputSubRow * 4])[0];
    }
  // Load (float4) the first batch of Butterfly factor from global to shared memory TILEK * TILEX
#pragma unroll
  for (int s = 0; s < TILEK; s += StrideValues)
    {
      // Reconstruct row and column indices
      tmp_row = ValuesSubRow + s;
      tmp_col = ValuesSubCol * 4;
      reinterpret_cast<float4 *>(
				 &shared_values[0][tmp_row * TILEX + tmp_col])[0] =
	reinterpret_cast<float4 *>(&valuesT[b * tmp_row + tmp_col % b])[0];
    }

  int load = 0;
  int write = 1;

  // Loop over non-zero entries by TILEK
  for (int k = 0; k < c; k += TILEK)
    {
      __syncthreads();
      // Load smem to register and compute accumulation
#pragma unroll
      for (int i = 0; i < TILEK; i++)
	{
#pragma unroll
	  for (int y = 0; y < TY; y += 4)
	    reinterpret_cast<float4 *>(&regY[y])[0] = reinterpret_cast<float4 *>(
										 &shared_input[load][i * TILEY + thready * TY + y])[0];
#pragma unroll
	  for (int x = 0; x < TX; x += 4)
	    reinterpret_cast<float4 *>(&regX[x])[0] = reinterpret_cast<float4 *>(
										 &shared_values[load][i * TILEX + threadx * TX + x])[0];
#pragma unroll
	  for (int y = 0; y < TY; y++)
	    {
#pragma unroll
	      for (int x = 0; x < TX; x++)
		tmp_acc[y * TX + x] += regY[y] * regX[x];
	    }
	}

      load = load ^ 1;
      inputT += d * TILEK * batch_size;
      valuesT += TILEK * b;

      // Condition on row of valuesT
      if ((k + TILEK) < c)
	{
	  // Load (float4) next batch from global to shared memory TILEY x TILEK
#pragma unroll
	  for (int s = 0; s < TILEK; s += StrideInput)
	    {
	      // Use float4 to load input into shared memory
	      // input must use the col-major format
	      reinterpret_cast<float4 *>(
					 &shared_input[write][(InputSubCol + s) * TILEY + InputSubRow * 4])[0] =
		reinterpret_cast<float4 *>(&inputT[d * (InputSubCol + s) * batch_size + InputSubRow * 4])[0];
	    }
	  // Load (float4) the Butterfly factor in shared memory TILEK x TILEX
#pragma unroll
	  for (int s = 0; s < TILEK; s += StrideValues)
	    {
	      // Reconstruct row and column indices
	      tmp_row = ValuesSubRow + s;
	      tmp_col = ValuesSubCol * 4;
	      reinterpret_cast<float4 *>(
					 &shared_values[write]
					 [tmp_row * TILEX + tmp_col])[0] =
		reinterpret_cast<float4 *>(
					   &valuesT[b * tmp_row + tmp_col % b])[0];
	    }
	  write = write ^ 1;
	}
    }

  // Store accumulation to shared memory
#pragma unroll
  for (int y = 0; y < TY; y++)
    {
      for (int x = 0; x < TX; x += 4)
	{
	  shared_output[thready * TY + y + (threadx * TX + x + 0) * TILEY] =
	    tmp_acc[y * TX + x + 0];
	  shared_output[thready * TY + y + (threadx * TX + x + 1) * TILEY] =
	    tmp_acc[y * TX + x + 1];
	  shared_output[thready * TY + y + (threadx * TX + x + 2) * TILEY] =
	    tmp_acc[y * TX + x + 2];
	  shared_output[thready * TY + y + (threadx * TX + x + 3) * TILEY] =
	    tmp_acc[y * TX + x + 3];
	}
    }

  // Write out the accumulation (from shared to global memory)
#pragma unroll
  for (int y = 0; y < TY; y += 4)
    {
#pragma unroll
      for (int x = 0; x < TX; x++)
	{
	  reinterpret_cast<float4 *>(
				     &outputT[thready * TY + y + d * (threadx * TX + x) * batch_size])[0] =
	    reinterpret_cast<float4 *>(
				       &shared_output[thready * TY + y + (threadx * TX + x) * TILEY])[0];
	}
    }
}

#endif
