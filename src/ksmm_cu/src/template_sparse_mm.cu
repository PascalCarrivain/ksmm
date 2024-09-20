// -*- c++ -*-

#include <cassert>
#include <ctime>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <getopt.h>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <stdio.h>

#defineFLOAT4orHALF2

#ifdef FLOAT4
#include "kernels_float4.cuh"
#endif

#ifdef HALF2
#include "kernels_half2.cuh"
#endif

// using namespace nvcuda;

char name[200];
FILE *fout;

int MF(int row, int col, int ld, int mf);

int MF(int row, int col, int ld, int mf)
{
  // mf = 0 reads in row-major format
  // mf = 1 reads in column-major format
  return (mf == 0) ? row * ld + col : col * ld + row;
}

int main(int argc, char **argv)
{
  int nargs = 0, arg_i;
  char kernel_name[100];
  int deviceId = 0;
  int nrepeats = 100;
  const option long_opts[] = {{"kernel", required_argument, nullptr, 'n'},
                              {"device", required_argument, nullptr, 'd'},
			      {"nrepeats", required_argument, nullptr, 'r'}};
  while ((arg_i = getopt_long(argc, argv, "n:d:r:h", long_opts, nullptr)) != -1)
  {
    switch (arg_i)
    {
    case 'n':
      sprintf(kernel_name, "%s", optarg);
      nargs++;
      break;
    case 'd':
      deviceId = (int)atof(optarg);
      nargs++;
      break;
    case 'r':
      nrepeats = (int)atof(optarg);
      nargs++;
      break;
    case 'h':
      break;
    default:
      printf("\n");
      break;
    }
  }

  int seed = 1;
  std::mt19937_64 mt(seed);
  std::uniform_real_distribution<float> u01(.0, 1.);

  // Grid
  const int batch_size = xBATCHSIZEx;
  const int input_size = xINPUTSIZEx;
  const int output_size = xOUTPUTSIZEx;
  int dim1, dim2;
  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, 1);
  // Grid for the tensor cores kernel
  const int WMMA_Y = xWMMA_Yx;
  const int WMMA_X = xWMMA_Xx;
  const int WMMA_K = xWMMA_Kx;
  int nwarpsY = xNWARPSYx;
  int nwarpsX = xNWARPSXx;
  // Tile dimensions
  const int TILEX = xTILEXx;
  const int TILEK = xTILEKx;
  const int TILEY = xTILEYx;
  const int TX = xTXx;
  const int TY = xTYx;

  // Allocate matrices in host memory
  float *input;
  float *valuesT;
  float *bfactor;
  // half precision
  half *h_input;
  half *h_valuesT;
  half *h_bfactor;
  // To store the output of matrix multiplication
  float *gpu_output, *true_output;
  half *h_gpu_output;

  // Allocate matrices in device memory
  float *d_input;
  float *d_bfactor;
  float *d_valuesT;
  float *d_output;
  // half precision
  half *d_h_input;
  half *d_h_bfactor;
  half *d_h_valuesT;
  half *d_h_output;

  // alpha and beta for cuBlas routines
  const float alpha = 1.0;
  const float beta = 0.0;
  const half h_alpha = 1.0;
  const half h_beta = 0.0;

  // Leading dimensions
  int lda, ldb, ldc, mf;

  // Check if user asks for cuBlas routines
  bool is_cublas = false;
  if (strcmp(kernel_name, "cublas_factor0_fp16") == 0 ||
      strcmp(kernel_name, "cublas_stride_factor0_fp16") == 0 ||
      strcmp(kernel_name, "cublas_factor0_fp32") == 0 ||
      strcmp(kernel_name, "cublas_stride_factor0_fp32") == 0)
    is_cublas = true;

  // Check if half-precision
  bool hp = false;
  if (strcmp(kernel_name, "cublas_factor0_fp16") == 0 ||
      strcmp(kernel_name, "cublas_stride_factor0_fp16") == 0 ||
      strcmp(kernel_name, "kernel_bs_first_half2") == 0 ||
      strcmp(kernel_name, "kernel_bs_last_half2") == 0)
    hp = true;

  // Check if batch-size last position kernel
  bool bs_last = false;
  if (strcmp(kernel_name, "kernel_bs_last_float4") == 0 ||
      strcmp(kernel_name, "kernel_bs_last_half2") == 0)
    bs_last = true;

  bool check_output = false;
  bool take_too_long = false;
  bool debug = false;

  // CUDA device properties
  int nDevices;
  int maxDynamicSharedMem;
  int maxSharedMem;
  int maxThreadsPerBlock;
  int maxBlockDimX, maxBlockDimY;
  int maxGridDimX, maxGridDimY;
  int maxRegistersPerBlock;
  cudaDataType cuda_data_type = hp ? CUDA_R_16F : CUDA_R_32F;
  cudaEvent_t cstart, cend;
  float ct, ct_cutoff = 1e6, meant, stdt, *ts = new float[nrepeats]();

  // Mean-Square-Error between CPU and GPU computation
  float mse = 0.0;

  // Grid size as a function of Butterfly factor
  // Dimensions (cf overleaf)
  dim1 = input_size;
  dim2 = output_size;
  printf("batch=%i input_size=%i output_size=%i\n", batch_size, input_size,
         output_size);

  threadsPerBlock.x = xNTHREADS_FLOAT4x;
  threadsPerBlock.y = 1;
  blocksPerGrid.x = (dim2 + TILEX - 1) / TILEX;
  blocksPerGrid.y = (batch_size + TILEY - 1) / TILEY;

  if (!is_cublas)
  {
    printf("TILEX=%i TILEK=%i TILEY=%i TX=%i TY=%i\n", TILEX, TILEK, TILEY, TX, TY);
    printf("threadsPerBlock=%i %i\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("blocksPerGrid=%i %i\n", blocksPerGrid.x, blocksPerGrid.y);
  }

  // Shared memory
  int smem = (hp ? sizeof(half) : sizeof(float)) *
    ((TILEY * TILEK + TILEK * TILEX) + TILEY * TILEX);

  if (strcmp(kernel_name, "cublas_stride_factor0_fp16") == 0 ||
      strcmp(kernel_name, "cublas_stride_factor0_fp32") == 0)
  {
    // If CuBlas stride tranpose (row to column major format)
    // Move everything from row to column major format ?
    lda = batch_size;
    ldb = dim1;
    ldc = batch_size;
    mf = 1;
  }
  else
  {
    lda = dim1;
    ldb = dim2;
    ldc = dim2;
    mf = 0;
  }

  // Host device allocations ...
  printf("Host device matrices allocations ...\n");
  // Random matrix
  printf("Random matrix ...\n");
  input = new float[hp ? 1 : batch_size * dim1]();
  h_input = new half[hp ? batch_size * dim1 : 1]();
  int istride = debug ? 1 : 200;
  int nthreads = 16;
  int bpert = (int)std::ceil(batch_size / (double)nthreads);
  bpert = istride * ((int)(bpert / istride) + 1);
  int row, col;
#pragma omp parallel for num_threads(nthreads)
  for (int n = 0; n < nthreads; n++) {
    for (int b = n * bpert; b < min(batch_size, (n + 1) * bpert); b += istride) {
      for (int i = 0; i < dim1; i++) {
	row = (int)(batch_size * u01(mt));
	col = i;
	if (!debug) {
	  if (hp)
	    h_input[(bs_last) ? MF(row, col, batch_size, 1)
		    : MF(row, col, lda, mf)] =
	      __float2half(u01(mt));
	  else
	    input[(bs_last) ? MF(row, col, batch_size, 1)
		  : MF(row, col, lda, mf)] = u01(mt);
	} else {
	  if (hp)
	    h_input[(bs_last) ?
		    MF(row, col, batch_size, 1) : MF(row, col, lda, mf)] = __float2half(1.0f);
	  else
	    input[(bs_last) ?
		  MF(row, col, batch_size, 1) : MF(row, col, lda, mf)] = (row == 0) ? 1.0 : 0.0;
	}
      }
    }
  }

  // Butterfly factor
  // tuple (a, b, c, d)
  // B = kron(Id_{a,a}, kron(1_{b,c}, Id_{d,d}))
  // There is 'a' super-blocks of shape (b * d, c * d).
  // Number of non-zero per super-block is
  // b per column and c per row.
  // We would like to compute X @ B^T.
  // X shape is (batch, a * c * d).
  // B^T shape is (dim1, dim2).
  // dim1 = a * c * d
  // dim2 = a * b * d
  printf("Butterfly factor (%i,%i,%i,%i) ...\n", xax, xbx, xcx, xdx);
  int NNZ = xax * xbx * xcx * xdx;
  valuesT = new float[hp ? 1 : NNZ]();
  h_valuesT = new half[hp ? NNZ : 1]();
  bfactor = new float[is_cublas ? (hp ? 1 : dim1 * dim2) : 1]();
  h_bfactor = new half[is_cublas ? (hp ? dim1 * dim2 : 1) : 1]();
  int count = 0;
  float value;
  half h_value;
  // Loop over the super-block
  int ii, jj;
  for (int aa = 0; aa < xax; aa++)
    {
      // First row and first column of the current super-block
      ii = aa * dim1 / xax;
      jj = aa * dim2 / xax;
      // Loop over the columns inside super-block and
      // store the index of rows of non-zero entries
      for (int m = 0; m < xdx; m++)
	{
	  for (int j = m; j < (dim2 / xax); j += xdx)
	    {
	      for (int i = m; i < (dim1 / xax); i += xdx)
		{
		  // inside super-block
		  if (hp)
		    {
		      // handle half case
		      h_value = debug ? __float2half(1.0f) : __float2half(u01(mt));
		      // There is b * c * d non-zero per super-block
		      // Current super-block is aa
		      h_valuesT[aa * xbx * xcx * xdx + m * xbx * xcx + (i / xdx) * (dim2 / (xax * xdx)) + j / xdx] = h_value;
		      if (is_cublas)
			h_bfactor[MF(ii + i, jj + j, ldb, mf)] = h_value;
		    }
		  else
		    {
		      value = debug ? 1.0 : u01(mt);
		      // There is b * c * d non-zero per super-block
		      // Current super-block is aa
		      valuesT[aa * xbx * xcx * xdx + m * xbx * xcx + (i / xdx) * (dim2 / (xax * xdx)) + j / xdx] = value;
		      if (is_cublas)
			bfactor[MF(ii + i, jj + j, ldb, mf)] = value;
		    }
		}
	    }
	}
    }

  // output
  gpu_output = new float[hp ? 1 : batch_size * dim2]();
  h_gpu_output = new half[hp ? batch_size * dim2 : 1]();
  true_output = new float[batch_size * dim2]();

  // Loop over devices
  printf("From host to device ...\n");
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++)
  {
    if (i != deviceId)
      continue;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cudaDeviceGetAttribute(&maxDynamicSharedMem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, i);
    cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlock,
                           i);
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock,
                           i);
    cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, i);
    cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, i);
    cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, i);
    cudaDeviceGetAttribute(&maxGridDimY, cudaDevAttrMaxGridDimY, i);
    cudaDeviceGetAttribute(&maxRegistersPerBlock,
                           cudaDevAttrMaxRegistersPerBlock, i);
    printf("device %i/%i\n", i, nDevices);
    printf("maxBlockDimX=%i maxBlockDimY=%i maxGridDimX=%i "
           "maxGridDimY=%i MaxThreadsPerBlock=%i\n",
           maxBlockDimX, maxBlockDimY, maxGridDimX, maxGridDimY,
           maxThreadsPerBlock);
    printf("registers per block=%i\n", maxRegistersPerBlock);
    printf("smem=%i/%i\n", smem, maxSharedMem);
  }
  assert(xNTHREADS_FLOAT4x <= maxThreadsPerBlock);
  assert(xNTHREADS_HALF2x <= maxThreadsPerBlock);
  assert(smem < maxSharedMem);
  assert(blocksPerGrid.x <= maxGridDimX && blocksPerGrid.y <= maxGridDimY);
  assert(threadsPerBlock.x <= maxBlockDimX &&
         threadsPerBlock.y <= maxBlockDimY);

  // Set device
  cudaSetDevice(deviceId);

  // Allocate matrices in device memory
  printf("device memory allocation ...\n");
  if (hp)
  {
    cudaMalloc(&d_h_input, sizeof(half) * batch_size * dim1);
    if (is_cublas)
      cudaMalloc(&d_h_bfactor, sizeof(half) * dim1 * dim2);
    cudaMalloc(&d_h_valuesT, sizeof(half) * NNZ);
    cudaMalloc(&d_h_output, sizeof(half) * batch_size * dim2);
  }
  else
  {
    cudaMalloc(&d_input, sizeof(float) * batch_size * dim1);
    if (is_cublas)
      cudaMalloc(&d_bfactor, sizeof(float) * dim1 * dim2);
    cudaMalloc(&d_valuesT, sizeof(float) * NNZ);
    cudaMalloc(&d_output, sizeof(float) * batch_size * dim2);
  }

  // Copy data from host memory to device memory
  printf("host to device memory ...\n");
  if (hp)
  {
    cudaMemcpy(d_h_input, h_input, sizeof(half) * batch_size * dim1,
               cudaMemcpyHostToDevice);
    if (is_cublas)
      cudaMemcpy(d_h_bfactor, h_bfactor, sizeof(half) * dim1 * dim2,
                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_h_valuesT, h_valuesT, sizeof(half) * NNZ,
               cudaMemcpyHostToDevice);
  }
  else
  {
    cudaMemcpy(d_input, input, sizeof(float) * batch_size * dim1,
               cudaMemcpyHostToDevice);
    if (is_cublas)
      cudaMemcpy(d_bfactor, bfactor, sizeof(float) * dim1 * dim2,
                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_valuesT, valuesT, sizeof(float) * NNZ, cudaMemcpyHostToDevice);
  }

  // Multiple runs of the kernel
  // ikernel is 0 if no valid kernels found
  int ikernel =
      1 * (int)(strcmp(kernel_name, "cublas_stride_factor0_fp16") == 0 ||
                strcmp(kernel_name, "cublas_stride_factor0_fp32") == 0) +
      2 * (int)(strcmp(kernel_name, "cublas_factor0_fp16") == 0 ||
                strcmp(kernel_name, "cublas_factor0_fp32") == 0) +
      3 * (int)(strcmp(kernel_name, "kernel_bs_first_float4") == 0 ||
                strcmp(kernel_name, "kernel_bs_first_half2") == 0 ||
                strcmp(kernel_name, "kernel_bs_last_float4") == 0 ||
                strcmp(kernel_name, "kernel_bs_last_half2") == 0);

  // Set CUDA time threshold to the minimum CUDA time ???
  int i0, i1, i2, i3, i4, i6, i7, i8, i9, i10, i11, i12;
  int s0, s1, s2, s3, s4, s6, s7, s8, s9, s10;
  s0 = s1 = s2 = s3 = s4 = s6 = s7 = s8 = s9 = s10 = -1;
  float f0, f1; //, f2;
  float tmin = 1e6;
  int nhp = 0;
  count = 0;
  sprintf(name, "%s.out", kernel_name);
  fout = fopen(name, "r");
  if (fout != NULL)
    {
      while (!feof(fout))
	{
	  if (count > 0)
	    {
	      fscanf(
		     fout,
		     "%i %i %i %i %i %i %i %i %i %i %i %i %f %f %*e\n",
		     &i0, &i1, &i2, &i3, &i4, &i6, &i7, &i8, &i9, &i10, &i11,
		     &i12, &f0, &f1);
	      if (xBATCHSIZEx == i0 && xax == i1 && xbx == i2 && xcx == i3 &&
		  xdx == i4)
		{
		  s0 = i0;
		  s1 = i1;
		  s2 = i2;
		  s3 = i3;
		  s4 = i4;
		  s6 = i6;
		  s7 = i7;
		  s8 = i8;
		  s9 = i9;
		  s10 = i10;
		  tmin = fmin(tmin, f0);
		  nhp++;
		}
	    }
	  else
	    fscanf(fout, "%*[^\n]\n");
	  count++;
	}
      fclose(fout);
    }
  if (count > 1 && s0 != -1)
    {
      ct_cutoff = 1.25 * tmin;
      printf("tmin=%f (%i, %i %i %i %i, %i %i %i %i %i)\n", tmin, s0, s1, s2,
             s3, s4, s6, s7, s8, s9, s10);
    }
  if(0 && nhp > 100)
    return 0;
  // ???

  const std::clock_t c_start = std::clock();
  cublasHandle_t cublasHandle;
  cublasStatus_t cublasStatus;
  float cum_time = 0.0;
  long long int stride_input = xcx * batch_size;
  long long int stride_values = xbx * xcx;
  long long int stride_output = xbx * batch_size;
  int batch_count = dim1 / xbx;
  // GPU warmup: w = 0
  printf("Warmup and then repeat %i runs ...\n", nrepeats);
  take_too_long = false;
  for (int w = 0; w < 2; w++)
  {
    cum_time = 0.0;
    for (int i = 0; i < ((w == 0) ? 100000000 : nrepeats); i++)
    {
      cudaEventCreate(&cstart);
      cudaEventRecord(cstart, 0);
      switch (ikernel)
      {
      case 0:
        throw std::invalid_argument("Did not find the kernel.");
        break;
      case 1:
        // CuBlas uses column-major format
        // However, we create matrices A and B using row-major format
        // A^T and B^T are in column-major format
        // Therefore, cublas computes C^T = (A @ B)^T that is in column-major
        // format
        if (w == 0 && i == 0)
          cublasCreate(&cublasHandle);
        // CUBLAS_GEMM_ALGO0_TENSOR_OP to CUBLAS_GEMM_ALGO15_TENSOR_OP
        // CUBLAS_GEMM_ALGO0 to CUBLAS_GEMM_ALGO23
        if (xdx == 1)
        {
          // Butterfly factor 0
          // |A B| @ |D1  0| = |A @ D1 B @ D2|
          // |C D|   |0  D2|   |C @ D1 D @ D2|
          // We consider the batch as an horizontal stack
          // of rectangular matrices batch_size x (b * d, c * d)
          // Use CSR format
          if (strcmp(kernel_name, "cublas_stride_factor0_fp16") == 0)
          {
            cublasStatus = cublasGemmStridedBatchedEx(
                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, xcx, xbx,
                &h_alpha, d_h_input, cuda_data_type, batch_size, stride_input,
                d_h_valuesT, cuda_data_type, xbx, stride_values, &h_beta,
                d_h_output, cuda_data_type, batch_size, stride_output,
                batch_count, cuda_data_type, xCUBLAS_GEMM_ALGO_TENSOR_OPx);
          }
          else
          {
            cublasStatus = cublasGemmStridedBatchedEx(
                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, xcx, xbx,
                &alpha, d_input, cuda_data_type, batch_size, stride_input,
                d_valuesT, cuda_data_type, xbx, stride_values, &beta, d_output,
                cuda_data_type, batch_size, stride_output, batch_count,
                cuda_data_type, xCUBLAS_GEMM_ALGOx);
          }
        }
        else
	  throw std::runtime_error("Not implemented for d=1.");
        if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        {
          printf("%i %i %i %i %i %i\n", cublasStatus, CUBLAS_STATUS_SUCCESS,
                 CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED,
                 CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_EXECUTION_FAILED);
          throw std::invalid_argument("cublasStatus != CUBLAS_STATUS_SUCCESS");
        }
        if (w == 1 && i == (nrepeats - 1))
          cublasDestroy(cublasHandle);
        break;
      case 2:
        // CuBlas uses column-major format
        // However, we create matrices A and B using row-major format
        // A^T and B^T are in column-major format
        // Therefore, cublas computes C^T = (A @ B)^T that is in column-major
        // format
        if (w == 0 && i == 0)
          cublasCreate(&cublasHandle);
        // CUBLAS_GEMM_ALGO0_TENSOR_OP to CUBLAS_GEMM_ALGO15_TENSOR_OP
        if (strcmp(kernel_name, "cublas_factor0_fp16") == 0)
          cublasStatus = cublasGemmEx(
              cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim2, batch_size, dim1,
              &h_alpha, d_h_bfactor, cuda_data_type, dim2, d_h_input,
              cuda_data_type, dim1, &h_beta, d_h_output, cuda_data_type, dim2,
              cuda_data_type, xCUBLAS_GEMM_ALGO_TENSOR_OPx);
        else
          cublasStatus = cublasGemmEx(
              cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, dim2, batch_size, dim1,
              &alpha, d_bfactor, cuda_data_type, dim2, d_input, cuda_data_type,
              dim1, &beta, d_output, cuda_data_type, dim2, cuda_data_type,
              xCUBLAS_GEMM_ALGOx);
        if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        {
          printf("%i %i %i %i %i %i\n", cublasStatus, CUBLAS_STATUS_SUCCESS,
                 CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED,
                 CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_EXECUTION_FAILED);
          throw std::invalid_argument("cublasStatus != CUBLAS_STATUS_SUCCESS");
        }
        if (w == 1 && i == (nrepeats - 1))
          cublasDestroy(cublasHandle);
        break;
      case 3:
#ifdef FLOAT4
        if (strcmp(kernel_name, "kernel_bs_first_float4") == 0)
        {
          kernel_bs_first_float4<xTILEXx, xTILEKx, xTILEYx, xTXx, xTYx>
              <<<blocksPerGrid, threadsPerBlock>>>(d_input, d_valuesT,
						   batch_size, d_output,
						   xax, xbx, xcx, xdx);
        }
        if (strcmp(kernel_name, "kernel_bs_last_float4") == 0)
        {
          kernel_bs_last_float4<xTILEXx, xTILEKx, xTILEYx, xTXx, xTYx>
              <<<blocksPerGrid, threadsPerBlock>>>(d_input, d_valuesT,
						   batch_size, d_output,
						   xax, xbx, xcx, xdx);
        }
#endif
#ifdef HALF2
        if (strcmp(kernel_name, "kernel_bs_first_half2") == 0)
        {
          kernel_bs_first_half2<xTILEXx, xTILEKx, xTILEYx, xTXx, xTYx>
              <<<blocksPerGrid, threadsPerBlock>>>(d_h_input, d_h_valuesT,
                                                   batch_size, d_h_output,
                                                   xax, xbx, xcx, xdx);
        }
        if (strcmp(kernel_name, "kernel_bs_last_half2") == 0)
        {
          kernel_bs_last_half2<xTILEXx, xTILEKx, xTILEYx, xTXx, xTYx>
              <<<blocksPerGrid, threadsPerBlock>>>(d_h_input, d_h_valuesT,
                                                   batch_size, d_h_output,
                                                   xax, xbx, xcx, xdx);
        }
#endif
        break;
      default:
        throw std::invalid_argument("Did not find the kernel.");
        break;
      }
      cudaEventCreate(&cend);
      cudaEventRecord(cend, 0);
      cudaEventSynchronize(cend);
      cudaEventElapsedTime(&ct, cstart, cend);
      cum_time += ct;
      if (w == 1 && i == (nrepeats - 1))
        printf("End warmup and repeats.\n");
      // Do 100 ms warmup
      if (w == 0 && cum_time > 100.0)
        break;
      // Save time if not warmup
      if (w == 1)
        ts[i] = ct;
      // Does it take too much time ?
      if (w == 1 && ct > ct_cutoff)
      {
        take_too_long = true;
        for (int j = 0; j < nrepeats; j++)
          ts[j] = 3600.0 * 1e3;
        break;
      }
    }
    if (take_too_long)
      break;
  }
  if (take_too_long)
  {
    printf("Take too long (%f > %f), exit.\n", ct, ct_cutoff);
    return 0;
  }

  const std::clock_t c_end = std::clock();
  meant = 0.0f;
  for (int i = 0; i < nrepeats; i++)
    meant += ts[i];
  meant /= nrepeats;
  stdt = 0.0f;
  for (int i = 0; i < nrepeats; i++)
    stdt += pow(ts[i] - meant, 2.);
  stdt = sqrt(stdt / (nrepeats - 1));
  printf("factor cuda=%f +/- %f ms\n", meant, stdt);

  // Copy data from device memory to host memory
  if (hp)
    cudaMemcpy(h_gpu_output, d_h_output, sizeof(half) * batch_size * dim2,
               cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(gpu_output, d_output, sizeof(float) * batch_size * dim2,
               cudaMemcpyDeviceToHost);

  check_output = (bool)(meant <= tmin);
  if (check_output)
  {
    // batch / Butterfly factor multiplication
    printf("use CPU to compute the output\n");
    const std::clock_t cpu_start = std::clock();
    if (hp)
      {
	// Butterfly factor
#pragma omp parallel for num_threads(nthreads)
	for (int n = 0; n < nthreads; n++)
	  {
	    for (int i = n * bpert; i < min(batch_size, (n + 1) * bpert);
		 i += istride)
	      {
		for (int m = 0; m < xdx; m++)
		  {
		    // TODO: comment about the index access
		    // We store all the columns j such that they have the same j % d
		    // in a contiguous way. The size of such group is dim2 // d.
		    for (int j = m; j < dim2; j += xdx)
		      {
			for (int k = m; k < dim1; k += xdx)
			  {
			    // Same super-block (a super-blocks (b * d, c * d)) ?
			    if ((j / (dim2 / xax)) != (k / (dim1 / xax)))
			      continue;
			    // There is b * c * d non-zero per super-block
			    true_output[MF(i, j, ldc, mf)] +=
			      __half2float(h_input[MF(i, k, (bs_last) ? batch_size : lda, (bs_last) ? 1 : mf)]) *
			      __half2float(h_valuesT[(k / (dim1 / xax)) * xbx * xcx * xdx +
						     m * xbx * xcx +
						     ((k - (dim1 / xax) * (k / (dim1 / xax))) / xdx) * (dim2 / (xax * xdx)) +
						     (j - (dim2 / xax) * (j / (dim2 / xax))) / xdx]);
			  }
		      }
		  }
	      }
	  }
      }
    else
      {
	// Butterfly factor
#pragma omp parallel for num_threads(nthreads)
	for (int n = 0; n < nthreads; n++)
	  {
	    for (int i = n * bpert; i < min(batch_size, (n + 1) * bpert);
		 i += istride)
	      {
		for (int m = 0; m < xdx; m++)
		  {
		    // TODO: comment about the index access
		    // We store all the columns j such that they have the same j % d
		    // in a contiguous way. The size of such group is dim2 // d.
		    for (int j = m; j < dim2; j += xdx)
		      {
			for (int k = m; k < dim1; k += xdx)
			  {
			    // Same super-block (a super-blocks (b * d, c * d)) ?
			    if ((j / (dim2 / xax)) != (k / (dim1 / xax)))
			      continue;
			    // There is b * c * d non-zero per super-block
			    true_output[MF(i, j, ldc, mf)] +=
			      input[MF(i, k, (bs_last) ? batch_size : lda, (bs_last) ? 1 : mf)] *
			      valuesT[(k / (dim1 / xax)) * xbx * xcx * xdx +
				      m * xbx * xcx +
				      ((k - (dim1 / xax) * (k / (dim1 / xax))) / xdx) * (dim2 / (xax * xdx)) +
				      (j - (dim2 / xax) * (j / (dim2 / xax))) / xdx];
			  }
		      }
		  }
	      }
	  }
      }
    const std::clock_t cpu_end = std::clock();
    printf("cpu clock_t=%f ms\n",
	   1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC);

    // Compute Mean-Square-Error between CPU and kernel results
    mse = 0.0;
    if (hp)
    {
      for (int n = 0; n < nthreads; n++)
      {
        for (int i = n * bpert; i < min(batch_size, (n + 1) * bpert);
             i += istride)
        {
          for (int j = 0; j < dim2; j++)
            mse += pow(__half2float(h_gpu_output[MF(i, j, bs_last ? batch_size : ldc, bs_last ? 1 : mf)]) -
                           true_output[MF(i, j, ldc, mf)],
                       2.0);
        }
      }
    }
    else
    {
      for (int n = 0; n < nthreads; n++)
      {
        for (int i = n * bpert; i < min(batch_size, (n + 1) * bpert);
             i += istride)
        {
          for (int j = 0; j < dim2; j++)
            mse += pow(gpu_output[MF(i, j, bs_last ? batch_size : ldc, bs_last ? 1 : mf)] -
                           true_output[MF(i, j, ldc, mf)],
                       2.0);
        }
      }
    }
    mse = sqrt(mse / ((batch_size / istride) * dim2));
    printf("mse(CPU, GPU)=%e\n", mse);
  }
  else
    mse = 0.0;

  // Write CUDA time (add a threshold ?)
  if ((!hp && mse < 1e-5) || (hp && mse < 1e-1))
  {
    if (meant <= tmin)
    {
      sprintf(name, "%s.out", kernel_name);
      fout = fopen(name, "a");
      if (strcmp(kernel_name, "cublas_factor0_fp16") == 0 ||
          strcmp(kernel_name, "cublas_stride_factor0_fp16") == 0)
        fprintf(fout, "%i %i %i %i %i %.4f %.4f %.4e\n", batch_size, xax, xbx,
                xcx, xdx, meant, stdt,
                mse); //, xCUBLAS_GEMM_ALGOx, xCUBLAS_GEMM_ALGO_TENSOR_OPx);
      else
      {
        if (strcmp(kernel_name, "cublas_factor0_fp32") == 0 ||
            strcmp(kernel_name, "cublas_stride_factor0_fp32") == 0)
          fprintf(fout, "%i %i %i %i %i %.4f %.4f %.4e\n", batch_size, xax,
                  xbx, xcx, xdx, meant, stdt,
                  mse); //, xCUBLAS_GEMM_ALGOx, xCUBLAS_GEMM_ALGO_TENSOR_OPx);
        else
        {
	  fprintf(fout,
		  "%i %i %i %i %i %i %i %i %i %i %i %i %.4f %.4f %.4e\n",
		  batch_size, xax, xbx, xcx, xdx, TILEX, TILEK, TILEY,
		  TX, TY, TILEX, TILEY, meant, stdt, mse);
        }
      }
      fclose(fout);
    }
    else
      printf("Too slow, do not save result.\n");
  }
  else
    {
      printf("Wrong mse, do not save result.\n");
      sprintf(name, "wrong_mse_%s.out", kernel_name);
      fout = fopen(name, "a");
      fprintf(fout, "%i %i %i %i %i %i %i %i %i %i\n",
	      batch_size, xax, xbx, xcx, xdx, TILEX, TILEK, TILEY, TX, TY);
      fclose(fout);
    }

  // ???
  if (check_output)
  {
    int row0 = 0 * (batch_size - 5),
      row1 = row0 + 5 * istride,
      col0 = min(dim2 - 5, 0 * xcx * xdx - 0 * 5);
    if (debug) {
      for (int i = row0; i < row1; i += istride)
	printf("%f %f %f %f %f ...\n", input[MF(i, col0 + 0, bs_last ? batch_size : lda, bs_last ? 1 : mf)],
	       input[MF(i, col0 + 1, bs_last ? batch_size : lda, bs_last ? 1 : mf)],
	       input[MF(i, col0 + 2, bs_last ? batch_size : lda, bs_last ? 1 : mf)],
	       input[MF(i, col0 + 3, bs_last ? batch_size : lda, bs_last ? 1 : mf)],
	       input[MF(i, col0 + 4, bs_last ? batch_size : lda, bs_last ? 1 : mf)]);
    }
    if (0 && debug)
      for (int i = 0; i < NNZ; i++)
        assert(valuesT[i] == 1.0);
    printf("true output:\n");
    for (int i = row0; i < row1; i += istride)
      printf("%f %f %f %f %f ...\n", true_output[MF(i, col0 + 0, ldc, mf)],
             true_output[MF(i, col0 + 1, ldc, mf)],
             true_output[MF(i, col0 + 2, ldc, mf)],
             true_output[MF(i, col0 + 3, ldc, mf)],
             true_output[MF(i, col0 + 4, ldc, mf)]);
    printf("output:\n");
    if (hp)
    {
      if (bs_last)
      {
        for (int i = row0; i < row1; i += istride)
          printf("%f %f %f %f %f ...\n",
                 __half2float(h_gpu_output[MF(i, col0 + 0, batch_size, 1)]),
                 __half2float(h_gpu_output[MF(i, col0 + 1, batch_size, 1)]),
                 __half2float(h_gpu_output[MF(i, col0 + 2, batch_size, 1)]),
                 __half2float(h_gpu_output[MF(i, col0 + 3, batch_size, 1)]),
                 __half2float(h_gpu_output[MF(i, col0 + 4, batch_size, 1)]));
      }
      else
      {
        for (int i = row0; i < row1; i += istride)
          printf("%f %f %f %f %f ...\n",
                 __half2float(h_gpu_output[MF(i, col0 + 0, ldc, mf)]),
                 __half2float(h_gpu_output[MF(i, col0 + 1, ldc, mf)]),
                 __half2float(h_gpu_output[MF(i, col0 + 2, ldc, mf)]),
                 __half2float(h_gpu_output[MF(i, col0 + 3, ldc, mf)]),
                 __half2float(h_gpu_output[MF(i, col0 + 4, ldc, mf)]));
      }
    }
    else
    {
      if (bs_last)
      {
        for (int i = row0; i < row1; i += istride)
          printf("%f %f %f %f %f ...\n",
                 gpu_output[MF(i, col0 + 0, batch_size, 1)],
                 gpu_output[MF(i, col0 + 1, batch_size, 1)],
                 gpu_output[MF(i, col0 + 2, batch_size, 1)],
                 gpu_output[MF(i, col0 + 3, batch_size, 1)],
                 gpu_output[MF(i, col0 + 4, batch_size, 1)]);
      }
      else
      {
        for (int i = row0; i < row1; i += istride)
          printf("%f %f %f %f %f ...\n", gpu_output[MF(i, col0 + 0, ldc, mf)],
                 gpu_output[MF(i, col0 + 1, ldc, mf)],
                 gpu_output[MF(i, col0 + 2, ldc, mf)],
                 gpu_output[MF(i, col0 + 3, ldc, mf)],
                 gpu_output[MF(i, col0 + 4, ldc, mf)]);
      }
    }
  }
  // ???

  // Free memory
  printf("free device memory ...\n");
  if (hp)
  {
    cudaFree(d_h_output);
    cudaFree(d_h_valuesT);
    cudaFree(d_h_input);
    cudaFree(d_h_bfactor);
  }
  else
  {
    cudaFree(d_output);
    cudaFree(d_valuesT);
    cudaFree(d_input);
    cudaFree(d_bfactor);
  }
  printf("free host memory ...\n\n");
  delete[] input;
  input = NULL;
  delete[] h_input;
  h_input = NULL;
  delete[] valuesT;
  valuesT = NULL;
  delete[] h_valuesT;
  h_valuesT = NULL;
  delete[] bfactor;
  bfactor = NULL;
  delete[] h_bfactor;
  h_bfactor = NULL;
  delete[] gpu_output;
  gpu_output = NULL;
  delete[] h_gpu_output;
  h_gpu_output = NULL;
  delete[] true_output;
  true_output = NULL;

  delete[] ts;
  ts = NULL;

  return 0;
}
