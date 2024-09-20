#include <torch/extension.h>

#include <vector>
// CUDA forward declarations

torch::Tensor kernel(
    torch::Tensor input,
    torch::Tensor values,
    const int bs_last,
    const int a,
    const int b,
    const int c,
    const int d,
    bool fp16);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor factor,
    const int bs_last,
    const int a,
    const int b,
    const int c,
    const int d,
    bool fp16)
{
  CHECK_INPUT(input);
  CHECK_INPUT(factor);
  int input_size = a * c * d;
  int output_size = a * b * d;
  assert(input_size == input.size(bs_last ? 0 : 1));
  assert(factor.size(0) == input.size(bs_last ? 0 : 1));
  assert(b == factor.size(1));

  return kernel(input, factor, bs_last, a, b, c, d, fp16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &forward, "Kernel for the forward pass with a Kronecker-sparse matrix (CUDA)");
}