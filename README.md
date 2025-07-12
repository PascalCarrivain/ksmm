# Kronecker-Sparse Matrix Multiplication (KSMM)

*Efficient linear layers for deep learning via Kronecker‐sparse factorization.*

KSMM provides efficient implementations of linear layers where the weight matrix is decomposed as a product of Kronecker‐sparse factors.
In practice, this means that a large dense matrix

```math
W = K_1 \cdots K_L
```

is replaced by a sequence of factors $K_\ell$, each following a prescribed sparsity pattern to enable fast multiplications and reduced memory consumption.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Available Algorithms](#available-algorithms)
- [Standalone CUDA Kernel (`ksmm_cu`)](#standalone-cuda-kernel-ksmm_cu)
- [Standalone OpenCL Kernel](#standalone-opencl-kernel)
- [Benchmark Reproduction](#benchmark-reproduction)
- [References](#references)
- [Citation](#citation)

## Getting Started

KSMM integrates easily with your PyTorch workflow through the `KSLinear` layer, which acts as a drop-in replacement for `torch.nn.Linear`. Simply specify your desired decomposition via a list of sparsity patterns.
Under the hood, the layer stores each Kronecker-sparse factor $K_\ell$ of the decomposition $W = K_1 \cdots K_L$, and handles batched matrix-vector multiplications with $W$ by sequentially applying the multiplication with each factor $K_\ell$.


**Quick start example (available at scripts/example.py)**
```python
from ksmm_py.layer.kronecker_sparse.interface import KSLinear
import torch

# Two factors W = K_1 K_2
patterns = [(6, 64, 64, 1), (1, 768, 192, 2)] # [(a_2, b_2, c_2, d_2), (a_1, b_1, c_1, d_1)], and in general for W = K_1 \dots K_L, patterns[0] should correspond to K_L (right-most factor), patterns[1] to K_{L-1}, ..., patterns[-1] to K_1 (left-most factor)
dim_in = patterns[0][0] * patterns[0][2] * patterns[0][3] # a_2 * c_2 * d_2
batch_size = 25088 # support for other batch-sizes is in progress; for now we finetuned the hyperparameters of our kernel only for 25'088= 196 (ctx length) * 128 (batch size) = effective ViT batch size
batch_size_last = True # True or False, whether the batch size is the last or the first dimension of the input and output tensors
device='cuda' # 'cuda' or 'cpu', the kernel is only available on GPU

x = torch.randn((dim_in, batch_size) if batch_size_last else (batch_size, dim_in), 
                dtype=torch.float32, # torch.float16 is fine too
                device=device)

# either provide no weights (or set weights = None), and let KSLinear initialize each factor with uniform(-1/sqrt(c), 1/sqrt(c)), or provide a list of tensor of shape (a, b, c, d) for each pattern (a, b, c, d)
weights = [torch.ones(*pattern) for pattern in patterns]

ksl = KSLinear(patterns=patterns, weights=weights, algo='kernel', dtype=x.dtype, bs_last=batch_size_last, device=device)
y = ksl(x)
ksl = KSLinear(patterns=patterns, weights=weights, algo='dense', dtype=x.dtype, bs_last=batch_size_last, device=device)
z = ksl(x)
print("Relative error between kernel and dense implementation: ",
    torch.linalg.norm(y - z) / torch.linalg.norm(z))
```
This defines a linear layer where the weight matrix is given by the product $W=K_1K_2$ of two Kronecker-sparse factors $K_1,K_2$ with sparsity patterns $(a_1, b_1, c_1, d_1)$ and $(a_2, b_2, c_2, d_2)$.
`KSLinear` handles the decomposition and efficient multiplication behind the scenes.

In batch-size-first, the input has format $X\in\mathbb{R}^{B \times a_2 c_2d_2}$ (where $B$ is the batch-size) and the output is $XW^\top=XK_2^\top K_1^\top\in\mathbb{R}^{B \times a_1 b_1 d_1}$.

In batch-size-last, the input has format $X\in\mathbb{R}^{a_2 c_2d_2\times B}$ and the output is $WX=K_1K_2X\in\mathbb{R}^{a_1 b_1 d_1 \times B}$.

#### Kernel: Batch-Sizes and Patterns Supported

Currently, we tuned the hyperparameters of our kernel only for the ViT batch-size **25,088** and around **700** patterns $(a, b, c, d)$ listed in the file *patterns.in*. Support for other batch-sizes and patterns is in progress.

Other `algo` choices in `KSLinear` support arbitrary batch-sizes and patterns.

#### Kernel Device Requirement

When using `algo="kernel"`, the device must be **CUDA**, or an exception is raised:

```python
>>> ksl = KSLinear(patterns=patterns, algo="kernel", dtype=x.dtype, bs_last=False, device='cpu')
Exception: kernel expects device='cuda'.
```

## Installation

1. **Clone the repository.**
    ```bash
    git clone https://github.com/PascalCarrivain/ksmm.git
    cd ksmm
    ```
2. **Optional: create a virtual environment.**
   *Alternative: use your favorite environment managing tool (e.g., conda).
   ```
   python -m venv ksmm_venv/
   source ksmm_venv/bin/activate
   ```
4. **Install the Python package.**
    ```bash
    pip install -e .
    ```
5. **Optional: run your first example.**
    ```bash
    python scripts/example.py
    ```
## Available Algorithms

When creating a `KSLinear`, you can choose among the following algorithms to perform the batched matrix-vector multiplication with each Kronecker-sparse factor $K_\ell$:
1. **`kernel`:** Our KSMM kernel (integrated with PyTorch, tuned for ViT batch-size 25,088).
2. **`bmm`:** Based on the method from this [paper](#references) by Tri Dao et al.
3. **`bsr`:** Uses the PyTorch BSR library.
4. **`einsum`:** Leveraging PyTorch's `einsum` function for tensor operations.
5. **`dense`:** Using `torch.nn.functional.linear` with dense matrices.
6. **`sparse`:** Also uses `torch.nn.functional.linear`, but the Kronecker-sparse matrices are provided in PyTorch's sparse CSR format.

We recommend starting with `kernel` and `bmm` as they are usually the fastest (but this of course depends on your specific settings).

## Standalone CUDA Kernel (`ksmm_cu`)

### Introduction

The standalone CUDA implementation of the Kronecker-sparse matrix-matrix multiplication kernel is located in the [`src/ksmm_cu`](src/ksmm_cu) directory.
Running the kernel independently of PyTorch can significantly reduce compilation times.

### Available Kernels

We provide four kernels for both single- and half-precision, with support for either batch-size-first or batch-size-last layouts:

1. **Floating-Point Precision:**
   - `kernel_bs_first_float4`
   - `kernel_bs_last_float4`
2. **Half-Precision:**
   - `kernel_bs_first_half2`
   - `kernel_bs_last_half2`

### Fine-Tuning Hyperparameters of the Kernel

For a **long fine-tuning** on your own device:

```bash
./long_run.sh <kernel_name> <GPU_device_index>
```

- `<kernel_name>`: One of the available kernel names listed above.
- `<GPU_device_index>`: The index of your GPU device.

#### Fine-Tuning Output

Fine-tuning scripts output results in the following format:

```bash
batch_size a b c d TILEX TILEK TILEY TX TY nwarpsX nwarpsY ms std mse
25088 1 48 48 1 16 16 16 4 4 16 16 0.0402 0.0007 1.8566e-08
...
```

- **ms**: Execution time in milliseconds.
- **std**: Standard deviation of the execution time.
- **mse**: Mean squared error between CPU and GPU matrix multiplication results.

#### Getting Best Hyperparameters

Once fine-tuning completed, you can copy the output file to the `tuning` folder.
Then, run the `generate_best_from_fine_tuning.sh` script to find the best hyperparameters for each sparsity pattern `(a, b, c, d)`.
This will generate or append to a file named `tuning/<kernel_name>.cuh`, which you should copy into the `src/ksmm_py/layer/kronecker_sparse/` folder for use in PyTorch.

**Example:**

```bash
./long_run.sh 'kernel_bs_first_float4' 0
```

- The output file `kernel_bs_first_float4.out` is generated.
- Move this file to the `tuning` folder.
- Run `generate_best_from_fine_tuning.sh` to create `kernel_bs_first_float4.cuh`, containing the best hyperparameters.

#### Change Hyperparameters Explored at Fine-Tuning

You can adjust the hyperparameters tested during fine-tuning by editing the bash scripts. Note that some configurations might cause errors like out-of-memory issues, depending on your GPU's characteristics.

Currently, the scripts support NVIDIA architectures such as V100, RTX 2080, RTX 4090, and A100. To add support for other architectures, specify the appropriate `sm_xx` codes.

### Template Files

In the `src` folder, you will find three template files:

1. `template_kernels_float4.cuh`
2. `template_kernels_half2.cuh`
3. `template_sparse_mm.cu`

- **`template_kernels_float4.cuh` and `template_kernels_half2.cuh`**: Contain the kernel implementations with placeholders for hyperparameters.
- **`template_sparse_mm.cu`**: Contains the code to run a kernel for given hyperparameters.

These templates can be modified to experiment with different kernel implementations. Placeholders like `xTILEX`, `xTILEKx`, `xTILEYx`, `xTXx`, and `xTYx` represent hyperparameters replaced with constant values when running `long_run.sh`.

After running the script, three new files with the placeholders replaced are generated:

1. `kernels_float4.cuh`
2. `kernels_half2.cuh`
3. `sparse_mm.cu`

These files are used to compile and run the code.

### Assertions and Valid Configurations

The `long_run.sh` script includes several assertions to ensure the kernel functions correctly. For a given pattern `(a, b, c, d)`, some hyperparameter combinations may not pass these assertions.
If you encounter issues, adjust the hyperparameters accordingly.

## Standalone OpenCL Kernel

The Python package [lazylinop](https://faustgrp.gitlabpages.inria.fr/lazylinop/index.html) provides OpenCL version of the kernel.
The repository is available [here](https://gitlab.inria.fr/faustgrp/lazylinop).

---

## Benchmark Reproduction

To reproduce the results from our paper [GZCT24](#GZCT24):

```bash
bash scripts/0_benchmark_ksmm_time.sh
```

This script benchmarks:

- All sparsity patterns `(a, b, c, d)` from [GZCT24](#GZCT24).
- Both float-precision and half-precision.
- Batch-size-first and batch-size-last.
- All available algorithms in the `KSLinear` class.

#### Customizing the Benchmark

Edit `scripts/0_benchmark_ksmm_time.sh` to adjust:
- the **result file**: set `saving_csv` to specify where results are saved.
- the **sparsity patterns**: modify `a_list`, `b_list`, `c_list`, and `d_list` to modify patterns `(a, b, c, d)`. By default, the script iterates over the cartesian product of these lists, skipping patterns that don't satisfy certain conditions (see Appendix B.1 of [GZCT24](#GZCT24)).
- the **precision and batch layout:** change the loops for precision (`fp16` and `fp32`) and batch position `bs_last`.
- the **algorithms:** change the loop `for algo in "kernel" "bmm" "bsr" "einsum" "dense" "sparse"` to select algorithms to benchmark.

#### Execution Time (Example)

- **Time:**  ~140 hours (all configurations).
- **Memory:** ~120 MB for storing results.
- **GPU:** NVIDIA A100-PCIE-40GB
- **CPU:** Intel(R) Xeon(R) Silver 4215R CPU @ 3.20GHz
- **RAM:** 377 GB
  



---

## References

- [GZCT24]<a name="GZCT24"></a> Antoine Gonon, Léon Zheng, Pascal Carrivain, Quoc-Tung Le. [*Fast inference with Kronecker-sparse matrices.*](https://arxiv.org/abs/2405.15013) CoRR abs/2405.15013 (2024)
- [Monarch]<a name="monarch"></a> Tri Dao, Beidi Chen, Nimit Sharad Sohoni, Arjun D. Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, Christopher Ré. [*Monarch: Expressive Structured Matrices for Efficient and Accurate Training.*](https://arxiv.org/abs/2204.00595) ICML 2022: 4690-4721

---

## Citation

If you use this codebase or otherwise found our work valuable, please consider citing:

```bibtex
@article{GZCT24,
  author    = {Antoine Gonon and L\'eon Zheng and Pascal Carrivain and Quoc-Tung Le},
  title     = {Fast inference with Kronecker-sparse matrices},
  journal   = {CoRR},
  volume    = {abs/2405.15013},
  year      = {2024},
  url       = {https://arxiv.org/abs/2405.15013},
}
```
