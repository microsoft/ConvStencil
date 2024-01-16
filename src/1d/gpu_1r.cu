#include <mma.h>
#include <iostream>
#include "1d_utils.h"
#include <chrono>

using namespace nvcuda;

#define BLOCK_SIZE_COL 1024//tune
#define HALO 3
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2)
#define PAD 0
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / 8)
#define UNIT_LENGTH 7
#define TENSOR_CORE_M 8
#define WARP_PER_BLOCK 8//tune
#define MMA_NUM 2
#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / 8)
#define SM_SIZE_COL (7+PAD)
#define SM_SIZE_COL2 (7+PAD2)
#define PAD2 2


extern __constant__ double param_matrix_d[2 * 8 * TENSOR_CORE_M];
// extern __constant__ double param_matrix_d[2 * UNIT_LENGTH * UNIT_LENGTH];
__global__ void gpu_1d1r_kernel(const double *__restrict__ in, double *__restrict__ out, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2)
{
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];

    int begin = blockIdx.x * BLOCK_SIZE_COL+1;
    int laneid = threadIdx.x % 32;

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    for (int i = tid; i <  D_BLOCK_SIZE_COL; i += totalThreads) {
        sharedmem[0][lookup_table1[i]] = in[begin + i];
        sharedmem[1][lookup_table2[i]] = in[begin + i];
    }

    __syncthreads();

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++)
    {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 2 * 4 * 8 + i * 32, 8);
    }

    nvcuda::wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;

    nvcuda::wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    int warp_id=threadIdx.x/32;
#pragma unroll
    for (int row = 2*8*warp_id; row < (warp_id+1)*8*2; row += TENSOR_CORE_M)
    {
#pragma unroll

            nvcuda::wmma::fill_fragment(acc_frag, 0.0);
            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(row, compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
            }

            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(row, compute_idx * 4, SM_SIZE_COL),SM_SIZE_COL);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
            }

            nvcuda::wmma::store_matrix_sync(out + begin + row / 8 * 64 + HALO , acc_frag, TENSOR_CORE_M, nvcuda::wmma::mem_row_major); //+1为了对齐
        }
}

__global__ void breakdown4_kernel(const double *__restrict__ in, double *__restrict__ out, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2)
{
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL2];

    int begin = blockIdx.x * BLOCK_SIZE_COL+1;
    int laneid = threadIdx.x % 32;

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    for (int i = tid; i <  D_BLOCK_SIZE_COL; i += totalThreads) {
        if (lookup_table1[i] != -1) {
            sharedmem[0][lookup_table1[i]] = in[begin + i];
        }
        if (lookup_table2[i] != -1) {
            sharedmem[1][lookup_table2[i]] = in[begin + i];
        }
    }

    __syncthreads();

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++)
    {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 2 * 4 * 8 + i * 32, 8);
    }

    nvcuda::wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;

    nvcuda::wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    int warp_id=threadIdx.x/32;

#pragma unroll
    for (int row = 2*8*warp_id; row < (warp_id+1)*8*2; row += TENSOR_CORE_M)
    {
#pragma unroll
            nvcuda::wmma::fill_fragment(acc_frag, 0.0);
            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {     
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(row, compute_idx * 4, SM_SIZE_COL2), SM_SIZE_COL2);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
            }

            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(row, compute_idx * 4, SM_SIZE_COL2),SM_SIZE_COL2);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
            }

            nvcuda::wmma::store_matrix_sync(out + begin + row / 8 * 64 + HALO , acc_frag, TENSOR_CORE_M, nvcuda::wmma::mem_row_major); //+1为了对齐
        }
}

__global__ void breakdown3_kernel(const double *__restrict__ in, double *__restrict__ out, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2)
{
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];

    int begin = blockIdx.x * BLOCK_SIZE_COL+1;
    int laneid = threadIdx.x % 32;

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    for (int i = tid; i <  D_BLOCK_SIZE_COL; i += totalThreads) {
        if (lookup_table1[i] != -1) {
            sharedmem[0][lookup_table1[i]] = in[begin + i];
        }
        if (lookup_table2[i] != -1) {
            sharedmem[1][lookup_table2[i]] = in[begin + i];
        }
    }

    __syncthreads();

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++)
    {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 2 * 4 * 8 + i * 32, 8);
    }

    nvcuda::wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;

    nvcuda::wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    int warp_id=threadIdx.x/32;
#pragma unroll
    for (int row = 2*8*warp_id; row < (warp_id+1)*8*2; row += TENSOR_CORE_M)
    {
#pragma unroll
            nvcuda::wmma::fill_fragment(acc_frag, 0.0);
            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {
          
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(row, compute_idx * 4, 7), 7);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
            }

            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(row, compute_idx * 4, 7), 7);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
            }

            nvcuda::wmma::store_matrix_sync(out + begin + row / 8 * 64 + HALO , acc_frag, TENSOR_CORE_M, nvcuda::wmma::mem_row_major); //+1为了对齐
        }
}

__global__ void breakdown1_kernel(const double *__restrict__ in, double *__restrict__ out, double* __restrict__ la,double* __restrict__ lb)
{
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];

    int begin = blockIdx.x * BLOCK_SIZE_COL+1;
    int gbegin=(blockIdx.x)*SM_SIZE_COL*SM_SIZE_ROW;
    int x=threadIdx.x;

    
    for(int col=x;col<D_BLOCK_SIZE_COL;col+=blockDim.x){
            if ((col + 1) % 8 != 0 && col < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                la[gbegin+IDX(col / (UNIT_LENGTH + 1),  col % (UNIT_LENGTH + 1), SM_SIZE_COL)]=in[begin+col];
            } 
            if ((col + 2) % 8 != 0 && col > 2 * HALO) {
                lb[gbegin+IDX((col - UNIT_LENGTH) / (UNIT_LENGTH + 1),  (col - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL)]=in[begin+col];
            } 
        }


    __syncthreads();

    for(int row=x;row<SM_SIZE_ROW;row+=blockDim.x){
        double result[8]={};
        for(int i=0;i<7;++i)
            for(int k=0;k<7;++k){
                result[i]+=la[gbegin+IDX(row,k,SM_SIZE_COL)]*param_matrix_d[IDX(k,i,7)];
                result[i+1]+=lb[gbegin+IDX(row,k,SM_SIZE_COL)]*param_matrix_d[7*7+IDX(k,i,7)];
            }

        for(int i=0;i<8;++i)out[begin+HALO+row*8+i]=result[i];
    }
}


/**
 * @param in input array pointer
 * @param out output array pointer
 * @param params parameter array pointer
 *
 */
void gpu_1d1r(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times, const int input_n)
{
    double param_matrix_h[2][8 * 8] = {};

    // Initialize parameter matrix

    for (int row = 0; row < 7; row++) // kernel size 7
        for (int col = 0; col <= row; ++col)
            param_matrix_h[0][row * 8 + col] = params[row - col];

    for (int row = 0; row < 7; row++)
        for (int col = row + 1; col < 8; ++col)
            param_matrix_h[1][row * 8 + col] = params[row + 7 - col];

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 8 * sizeof(double)));

    int lookup_table1_h[D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_COL];

    for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
        if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
            lookup_table1_h[j] = IDX(j / (UNIT_LENGTH + 1),  j % (UNIT_LENGTH + 1), SM_SIZE_COL);//9
        } else {
            lookup_table1_h[j] = SM_SIZE_ROW *SM_SIZE_COL - 1;
        }
        if ((j + 2) % 8 != 0 && j > 2 * HALO) {
            lookup_table2_h[j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
        } else {
            lookup_table2_h[j] = SM_SIZE_ROW*SM_SIZE_COL - 1;
        }
    }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d,  D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    const int cols = input_n + 2 * HALO + 1; // 1 for address alighment
    const size_t array_size = cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++)
    {
        CUDAKERNELCHECK((gpu_1d1r_kernel<<<grid_config, block_config>>>(array_d[i % 2] , array_d[(i + 1) % 2], lookup_table1_d, lookup_table2_d))); // 为了对齐空了4个  
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "ConvStencil(1D): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double)input_n * times * 3) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2] + 1, array_size - sizeof(double), cudaMemcpyDeviceToHost));

    return;
}


void gpu_1d1r_breakdown4(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times, const int input_n)
{
    double param_matrix_h[2][8 * 8] = {};

    // Initialize parameter matrix

    for (int row = 0; row < 7; row++) // kernel size 7
        for (int col = 0; col <= row; ++col)
            param_matrix_h[0][row * 8 + col] = params[row - col];

    for (int row = 0; row < 7; row++)
        for (int col = row + 1; col < 8; ++col)
            param_matrix_h[1][row * 8 + col] = params[row + 7 - col];
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 8 * sizeof(double)));

    int lookup_table1_h[D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_COL];

    for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
        if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
            lookup_table1_h[j] = IDX(j / (UNIT_LENGTH + 1),  j % (UNIT_LENGTH + 1), SM_SIZE_COL2);//9
        } else {
            lookup_table1_h[j] = - 1;
        }
        if ((j + 2) % 8 != 0 && j > 2 * HALO) {
            lookup_table2_h[j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL2);
        } else {
            lookup_table2_h[j] =  - 1;
        }
    }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d,  D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    const int cols = input_n + 2 * HALO + 1; // 1 for address alighment
    const size_t array_size = cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    int smem_size=SM_SIZE_ROW * SM_SIZE_COL2 * 2 * sizeof(double);
    // CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++)
    {
        CUDAKERNELCHECK((breakdown4_kernel<<<grid_config, block_config>>>(array_d[i % 2] , array_d[(i + 1) % 2], lookup_table1_d, lookup_table2_d))); 
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(1D) 4: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n\n", ((double)input_n * times * 3) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2] + 1, array_size - sizeof(double), cudaMemcpyDeviceToHost));

    return;
}


void gpu_1d1r_breakdown3(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times, const int input_n)
{
    double param_matrix_h[2][8 * 8] = {};

    // Initialize parameter matrix

    for (int row = 0; row < 7; row++) // kernel size 7
        for (int col = 0; col <= row; ++col)
            param_matrix_h[0][row * 8 + col] = params[row - col];

    for (int row = 0; row < 7; row++)
        for (int col = row + 1; col < 8; ++col)
            param_matrix_h[1][row * 8 + col] = params[row + 7 - col];
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 8 * sizeof(double)));

    int lookup_table1_h[D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_COL];

    for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
        if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
            lookup_table1_h[j] = IDX(j / (UNIT_LENGTH + 1),  j % (UNIT_LENGTH + 1), SM_SIZE_COL);//9
        } else {
            lookup_table1_h[j] = - 1;//去掉*7 差得更少  得填到padding区域
        }
        if ((j + 2) % 8 != 0 && j > 2 * HALO) {
            lookup_table2_h[j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
        } else {
            lookup_table2_h[j] =  - 1;
        }
    }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d,  D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    const int cols = input_n + 2 * HALO + 1; // 1 for address alighment
    const size_t array_size = cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++)
    {
        CUDAKERNELCHECK((breakdown3_kernel<<<grid_config, block_config>>>(array_d[i % 2] , array_d[(i + 1) % 2], lookup_table1_d, lookup_table2_d))); // 为了对齐空了4个  
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(1D) 3: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n\n", ((double)input_n * times * 3) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2] + 1, array_size - sizeof(double), cudaMemcpyDeviceToHost));

    return;
}

__global__ void breakdown2_kernel(const double *__restrict__ in, double *__restrict__ out, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2)
{
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];

    int begin = blockIdx.x * BLOCK_SIZE_COL+1;
    int x=threadIdx.x;

    
    for(int col=x;col<D_BLOCK_SIZE_COL;col+=blockDim.x){
            if ((col + 1) % 8 != 0 && col < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
               sharedmem[0][IDX(col / (UNIT_LENGTH + 1),  col % (UNIT_LENGTH + 1), SM_SIZE_COL)]= in[begin + col];
            } 
            if ((col + 2) % 8 != 0 && col > 2 * HALO) {
                sharedmem[1][IDX((col - UNIT_LENGTH) / (UNIT_LENGTH + 1),  (col - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL)]= in[begin + col];
            } 
        }


    __syncthreads();

    for(int row=x;row<SM_SIZE_ROW;row+=blockDim.x){
            double result[8]={};
            for(int i=0;i<7;++i)
                for(int k=0;k<7;++k){
                    result[i]+=sharedmem[0][IDX(row,k,SM_SIZE_COL)]*param_matrix_d[IDX(k,i,7)];
                    result[i+1]+=sharedmem[1][IDX(row,k,SM_SIZE_COL)]*param_matrix_d[7*7+IDX(k,i,7)];
                }

            for(int i=0;i<8;++i)out[begin+HALO+row*8+i]=result[i];
    }
}

void gpu_1d1r_breakdown2(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times, const int input_n)
{
    double param_matrix_h[2][7 * 7] = {};

    // Initialize parameter matrix

    for (int row = 0; row < UNIT_LENGTH; row++) // kernel size 7
        for (int col = 0; col < UNIT_LENGTH; ++col)
            param_matrix_h[0][row * UNIT_LENGTH + col] = params[row - col];

    for (int row = 0; row < UNIT_LENGTH; row++)
        for (int col = row; col < UNIT_LENGTH; ++col)
            param_matrix_h[1][row * UNIT_LENGTH + col] = params[row + 6 - col];//+7

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 7 * 7 * sizeof(double)));

    int lookup_table1_h[D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_COL];

    for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
        if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
            lookup_table1_h[j] = IDX(j / (UNIT_LENGTH + 1),  j % (UNIT_LENGTH + 1), SM_SIZE_COL);//9
        } else {
            lookup_table1_h[j] = SM_SIZE_ROW *SM_SIZE_COL - 1;
        }
        if ((j + 2) % 8 != 0 && j > 2 * HALO) {
            lookup_table2_h[j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
        } else {
            lookup_table2_h[j] = SM_SIZE_ROW*SM_SIZE_COL - 1;
        }
    }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d,  D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    const int cols = input_n + 2 * HALO + 1; // 1 for address alighment
    const size_t array_size = cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++)
    {
        CUDAKERNELCHECK((breakdown2_kernel<<<grid_config, block_config>>>(array_d[i % 2] , array_d[(i + 1) % 2], lookup_table1_d, lookup_table2_d))); // 为了对齐空了4个  
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(1D) 2: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n\n", ((double)input_n * times * 3) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2] + 1, array_size - sizeof(double), cudaMemcpyDeviceToHost));

    return;
}

void gpu_1d1r_breakdown1(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times, const int input_n)
{
    double param_matrix_h[2][7 * 7] = {};

    // Initialize parameter matrix

    // 最后一列空
    for (int row = 0; row < UNIT_LENGTH; row++) // kernel size 7
        for (int col = 0; col < UNIT_LENGTH; ++col)
            param_matrix_h[0][row * UNIT_LENGTH + col] = params[row - col];

    // 第一列空
    for (int row = 0; row < UNIT_LENGTH; row++)
        for (int col = row; col < UNIT_LENGTH; ++col)
            param_matrix_h[1][row * UNIT_LENGTH + col] = params[row + 6 - col];//+7

    // 常量内存搬运
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 7 * 7 * sizeof(double)));

    int lookup_table1_h[D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_COL];

    //刨去了若干列
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[j] = IDX(j / (UNIT_LENGTH + 1),  j % (UNIT_LENGTH + 1), SM_SIZE_COL);//9
            } else {
                lookup_table1_h[j] = SM_SIZE_ROW *SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[j] = SM_SIZE_ROW*SM_SIZE_COL - 1;
            }
        }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d,  D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h,  D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    const int cols = input_n + 2 * HALO + 1; // 1 for address alighment
    const size_t array_size = cols * sizeof(double);
    double *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    dim3 grid_config(BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    double* stencil2row[2];
    CUDA_CHECK(cudaMalloc(&stencil2row[0],BLOCK_N*SM_SIZE_COL*SM_SIZE_ROW*sizeof(double)));//*0.75?
    CUDA_CHECK(cudaMalloc(&stencil2row[1], BLOCK_N*SM_SIZE_COL*SM_SIZE_ROW*sizeof(double)));

    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++)
    {
        CUDAKERNELCHECK((breakdown1_kernel<<<grid_config, block_config>>>(array_d[i % 2] , array_d[(i + 1) % 2],stencil2row[0],stencil2row[1]))); // 为了对齐空了4个  
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(1D) 1: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n\n", ((double)input_n * times * 3) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2] + 1, array_size - sizeof(double), cudaMemcpyDeviceToHost));

    return;
}