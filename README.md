# Kronecker-Sparse Matrix Multiplication (KSMM)

## Overview

This repository provides an implementation of a Kronecker-sparse matrix-matrix multiplication kernel designed to accelerate matrix multiplication involving Kronecker-sparse matrices. This is particularly useful in neural networks where such operations are common.

We offer:
- **Standalone CUDA Kernel**: a standalone CUDA version of our new Kronecker-sparse Matrix Multiplication (KSMM) kernel, without PyTorch dependencies for faster compilation and experimentation. [(See Standalone CUDA Kernel Section)](#standalone-cuda-kernel-ksmm-cu)
- **PyTorch-integrated version**: a Kronecker-Sparse Linear (KSLinear) class that represents linear layers that correspond to products of Kronecker-sparse matrices, and perform the multiplication by using one of the KSMM algorithm of your choice (e.g., our new kernel) to perform the sequential multiplication with each factor. It can be used as a drop-in replacement of `torch.nn.Linear`, making easy the integration into neural network models. [(See PyTorch Version Section)](#pytorch-version-ksmm-py)
- **Benchmarking Tools**: scripts and utilities to benchmark various algorithms for KSMM on GPU in PyTorch, and compare them with existing methods. [(See Benchmarking Section)](#benchmarking)

---

## Installation

To install the KSMM package and its dependencies, follow these steps:

### Prerequisites

- **Python 3.12 or higher**
- **CUDA Toolkit**: Ensure that you have the CUDA toolkit installed and properly configured for your GPU.
- **PyTorch**: Install a version compatible with your CUDA toolkit.
- **NVCC Compiler**: Ensure `nvcc` is in your system's PATH.
- **Additional Python Packages**: The installation will automatically install the following packages if they are not already installed:
  - `numpy`
  - `pandas`
  - `packaging`
  - `einops`

### Steps

1. **Clone the Repository**

   Clone the KSMM repository to your local machine:

   ```bash
   git clone https://github.com/PascalCarrivain/ksmm.git
   cd ksmm
   ```

2. **Install the Package**

   Install the package in editable mode using pip:

   ```bash
   pip install -e .
   ```

   This command will install the KSMM package along with its dependencies specified in the `pyproject.toml` file.

**Note**: The installation process assumes that you have a compatible CUDA toolkit installed and that `nvcc` is accessible from your command line. If you encounter issues related to CUDA or NVCC during installation, ensure that your CUDA toolkit is correctly installed and that `nvcc` is in your system's PATH.

---


## Standalone CUDA Kernel (`ksmm_cu`)

### Introduction

The standalone CUDA implementation of the Kronecker-sparse matrix-matrix multiplication kernel is located in the [`src/ksmm_cu`](src/ksmm_cu) directory. Running the kernel independently of PyTorch significantly reduces compilation times, as compiling within PyTorch can be time-consuming.

### Available Kernels

We provide four kernels supporting both floating-point and half-precision computations, with the batch size either at the first or last dimension:

1. **Floating-Point Precision:**
   - `kernel_bs_first_float4`
   - `kernel_bs_last_float4`
2. **Half-Precision:**
   - `kernel_bs_first_half2`
   - `kernel_bs_last_half2`

### Running Fine-Tuning

To fine-tune the kernel parameters, use the `long_run.sh` script:

```bash
./long_run.sh <kernel_name> <GPU_device_index>
```

- `<kernel_name>`: One of the available kernel names listed above.
- `<GPU_device_index>`: The index of your GPU device.

#### Light Version

For a quicker fine-tuning (sanity check), run:

```bash
./short_run.sh
```

### Output

The script outputs results in the following format:

```bash
batch_size a b c d TILEX TILEK TILEY TX TY nwarpsX nwarpsY ms std mse
25088 1 48 48 1 16 16 16 4 4 16 16 0.0402 0.0007 1.8566e-08
...
```

- **ms**: Execution time in milliseconds.
- **std**: Standard deviation of the execution time.
- **mse**: Mean squared error between CPU and GPU matrix multiplication results.

### Modifying Hyperparameters

You can adjust the hyperparameters by editing the bash scripts. Note that some configurations might cause errors like out-of-memory issues, depending on your GPU's characteristics.

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

The `long_run.sh` script includes several assertions to ensure the kernel functions correctly. For a given pattern `(a, b, c, d)`, some hyperparameter combinations may not pass these assertions. If you encounter issues, adjust the hyperparameters accordingly.

### Selecting the Best Hyperparameters

After `long_run.sh` completes, you can copy the output file to the `tuning` folder. Then, run the `generate_best_from_fine_tuning.sh` script to find the best hyperparameters for each tuple `(a, b, c, d)`. This will generate or append to a file named `tuning/<kernel_name>.cuh`, which you should copy into the `src/ksmm_py/layer/kronecker_sparse/` folder for use in PyTorch.

**Example:**

```bash
./long_run.sh 'kernel_bs_first_float4' 0
```

- The output file `kernel_bs_first_float4.out` is generated.
- Move this file to the `tuning` folder.
- Run `generate_best_from_fine_tuning.sh` to create `kernel_bs_first_float4.cuh`, containing the best hyperparameters.

---

## PyTorch Version (`ksmm_py`)

### Introduction

The PyTorch integration of the KSMM kernel is located in the [`src/ksmm_py`](src/ksmm_py) directory.

#### Kronecker-Sparse Linear (KSLinear) layer

This class provides a way to represent linear layers corresponding to products of Kronecker-sparse matrices. This can be used in place of standard `torch.nn.Linear` layers. Examples of such replacements in Vision Transformers (ViT) are given in `src/ksmm_py/model/get_vit.py`

- **File:** `src/ksmm_py/layer/kronecker_sparse/interface.py`
- **Description:** Represents linear transformations corresponding to products of Kronecker-sparse matrices.
- **Sparsity Patterns:** Each Kronecker-sparse matrix has a sparsity pattern corresponding to a pattern given in the format `(a, b, c, d)`, with the corresponding support \( I_a \otimes 1_{b, c} \otimes I_d \), where \( I \) is the identity matrix and \( 1_{b, c} \) is a matrix of ones with dimensions \( b \times c \).

### Benchmarking

We provide tools to benchmark various algorithms for Kronecker-sparse Matrix Multiplication (KSMM) on GPU in PyTorch and compare their performance:

1. **Our Kernel (`kernel`):** Our KSMM kernel originating from ksmm_cu integrated in PyTorch.
2. **Block Matrix Multiplication (`bmm`):** Based on the method from the [Monarch paper](#references) by Tri Dao et al.
3. **Block Sparse Row (`bsr`):** Utilizes the PyTorch BSR library.
4. **Einsum (`einsum`):** Leveraging PyTorch's `einsum` function for tensor operations.
5. **Dense (`dense`):** Using `torch.nn.functional.linear` with dense matrices.
6. **Sparse (`sparse`):** Also uses `torch.nn.functional.linear`, but the matrix is provided in PyTorch's sparse format.

Please refer to the paper [GZCT24](#GZCT24), or look directly into the functions `forward_bs_first` and `forward_bs_last` in `ksmm_py/layer/kronecker_sparse/interface.py` to have details about these algorithms.
#### System Specifications

- **GPU:** NVIDIA A100-PCIE-40GB
- **CPU:** Intel(R) Xeon(R) Silver 4215R CPU @ 3.20GHz
- **RAM:** 377 GB

- **Benchmark Duration:** Approximately 140 hours for all configurations.
- **Storage Requirements:** Around 120 MB for results (may vary).

### Running the Benchmark

To run the benchmark and reproduce the results from our paper:

```bash
bash scripts/0_benchmark_ksmm_time.sh
```

This script benchmarks all configurations, i.e.:

- All sparsity patterns `(a, b, c, d)` from [GZCT24](#GZCT24).
- Both float-precision and half-precision.
- Both batch-size-first and batch-size-last layouts.
- All algorithms mentioned above.

#### Customizing the Benchmark

To customize the benchmark configurations, edit `scripts/0_benchmark_ksmm_time.sh`:

1. **Output File:**
   - Set `saving_csv` to specify where to save the results.
   - If unchanged, new configurations will append to the existing benchmark file, and configurations already there will have their time measurements updated.

2. **Sparsity Patterns:**
   - Adjust `a_list`, `b_list`, `c_list`, and `d_list` to modify patterns `(a, b, c, d)`.
   - By default, the script iterates over the cartesian product of these lists, skipping patterns that don't satisfy certain conditions (see Appendix B.1 of [GZCT24](#GZCT24)).
   - Modify these conditions at the beginning of the inner loop if needed.

   **Note:** Some patterns may exceed your GPU's memory capacity.

3. **Precision:**
   - Modify `for p in fp16 fp32` to select between half-precision (`fp16`) and float-precision (`fp32`).

4. **Batch Positions:**
   - Adjust `for bs_last in 0 1` to choose between batch-size-first (`0`) and batch-size-last (`1`) configurations.

5. **Algorithms:**
   - Modify `for algo in "kernel" "bmm" "bsr" "einsum" "dense" "sparse"` to select algorithms to benchmark.
   - These correspond to the methods described in our paper [GZCT24](#GZCT24).

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
