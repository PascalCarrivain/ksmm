// -*- c++ -*-

#include <torch/extension.h>

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <vector>
// #include <cuda_fp8.h>
// #include <cuda_fp16.h>

// #include "kernel_bs_first_fp8e4m3.cuh"
#include "kernel_bs_first_float.cuh"
#include "kernel_bs_first_half.cuh"

// #include "kernel_bs_last_fp8e4m3.cuh"
#include "kernel_bs_last_float.cuh"
#include "kernel_bs_last_half.cuh"

using namespace nvcuda;

// at::Half is not compatible with cuda native type __half, a workaround taken
// from https://discuss.pytorch.org/t/getting-half-out-of-an-fp16-tensor/85743
// for any scalar_t that is different from at::Half, at::scalar_t is compatible
// with native scalar_t so we can juste use the same type
template <typename U> struct native_type { using T = U; };
// for scalar_t = at::Half, at::scalar_t is not compatible with native scalar_t
// so we use instead __half

template <> struct native_type<c10::Half> { using T = __half; };
// this function takes a tensor as input, recast it to __half if scalar_t =
// at::Half, and returns a pointer
template <typename U> typename native_type<U>::T *ptr(at::Tensor t) {
  return reinterpret_cast<typename native_type<U>::T *>(t.data_ptr<U>());
}

torch::Tensor kernel(torch::Tensor input, torch::Tensor values,
                     const int bs_last, const int a, const int b, const int c,
                     const int d, const char *dtype, int vsize) {
  // Performs the multiplication between input, a tensor of size (batch_size,
  // input_size), and values, a Kronecker-sparse matrix stored in format
  // (a*d*c,b), and returns the result as a tensor of size (batch_size,
  // output_size).
  const int batch_size = bs_last ? input.size(1) : input.size(0);
  const int input_size = bs_last ? input.size(0) : input.size(1);
  const int output_size = a * b * d;
  torch::Tensor output = torch::empty(
      {bs_last ? output_size : batch_size, bs_last ? batch_size : output_size},
      input.options());

  // the dispatching macro can be chosen from
  // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h#L131
  // for fp16, 32 and 64, one can choose
  // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h#L244
  // if fp32 and fp64 are default and one allows another type one can use
  // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h#L260
  // there is also
  // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h#L417
  // for smaller precision format

  dim3 blockGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, 1);
  if (strcmp(dtype, "torch.float8_e4m3fn") == 0) {
    switch (bs_last) {
      //   case 0:
      // best_kernel_bs_first_fp8e4m3(reinterpret_cast<__nv_fp8_e4m3
      // *>(input.data_ptr()), reinterpret_cast<__nv_fp8_e4m3
      // *>(values.data_ptr()), reinterpret_cast<__nv_fp8_e4m3
      // *>(output.data_ptr()), batch_size, a, b, c, d, blockGrid,
      // threadsPerBlock); break;
      //   case 1:
      // best_kernel_bs_last_fp8e4m3(reinterpret_cast<__nv_fp8_e4m3
      // *>(input.data_ptr()), reinterpret_cast<__nv_fp8_e4m3
      // *>(values.data_ptr()), reinterpret_cast<__nv_fp8_e4m3
      // *>(output.data_ptr()), batch_size, a, b, c, d, blockGrid,
      // threadsPerBlock); break;
    default:
      break;
    }
  } else if (strcmp(dtype, "torch.half") == 0 ||
             strcmp(dtype, "torch.float16") == 0) {
    switch (bs_last) {
    case 0:
      best_kernel_bs_first_half(reinterpret_cast<half *>(input.data_ptr()),
                                reinterpret_cast<half *>(values.data_ptr()),
                                reinterpret_cast<half *>(output.data_ptr()),
                                batch_size, a, b, c, d, blockGrid,
                                threadsPerBlock);
      break;
    case 1:
      best_kernel_bs_last_half(reinterpret_cast<half *>(input.data_ptr()),
                               reinterpret_cast<half *>(values.data_ptr()),
                               reinterpret_cast<half *>(output.data_ptr()),
                               batch_size, a, b, c, d, blockGrid,
                               threadsPerBlock);
      break;
    default:
      break;
    }
  } else if (strcmp(dtype, "torch.float") == 0 ||
             strcmp(dtype, "torch.float32") == 0) {
    switch (bs_last) {
    case 0:
      best_kernel_bs_first_float(reinterpret_cast<float *>(input.data_ptr()),
                                 reinterpret_cast<float *>(values.data_ptr()),
                                 reinterpret_cast<float *>(output.data_ptr()),
                                 batch_size, a, b, c, d, blockGrid,
                                 threadsPerBlock);
      break;
    case 1:
      best_kernel_bs_last_float(reinterpret_cast<float *>(input.data_ptr()),
                                reinterpret_cast<float *>(values.data_ptr()),
                                reinterpret_cast<float *>(output.data_ptr()),
                                batch_size, a, b, c, d, blockGrid,
                                threadsPerBlock);
      break;
    default:
      break;
    }
  } else {
    assert(0 && "Unsupported data type");
  }
  return output;
}
