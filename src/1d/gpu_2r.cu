//!支持任意大小
//  corner case 64 2
#include <mma.h>
#include <iostream>
#include "1d_utils.h"
#include <chrono>

using namespace nvcuda;

#define BLOCK_SIZE_COL 1024//tune
#define HALO 4
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2) //2*HALO 8
#define PAD
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / 8)
#define UNIT_LENGTH 7
#define TENSOR_CORE_M 8
#define WARP_PER_BLOCK 8
#define MMA_NUM 2
#define IDX(x, y, ldm) ((x) * (ldm) + (y))

extern __constant__ double param_matrix_d[2 * 8 * TENSOR_CORE_M];

__global__ void gpu_star_1d2r_step2_kernel(const double *__restrict__ in, double *__restrict__ out)
{
    __shared__ double sharedmem[SM_SIZE_ROW * 8];

    int begin = blockIdx.x * BLOCK_SIZE_COL;
    int laneid = threadIdx.x % 32;

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    for (int i = tid; i < D_BLOCK_SIZE_COL; i += totalThreads)
    {
        // if ( i < D_BLOCK_SIZE_COL)
        // if ( i < D_BLOCK_SIZE_COL  - 2 * HALO)
            // sharedmem[lookup_table[i]] = in[begin + i];
            sharedmem[i] = in[begin + i];
            // sharedmem[i / 8 * 8 + i % 8] = in[begin + i];
        // if (i >= 2 * HALO)
        //     sharedmem[1][(i - 8) / 8 * 8 + (i - 8) % 8] = in[begin + i];
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
    // 行数/warp数
#pragma unroll
    for (int row = 2*8*warp_id; row < (warp_id+1)*8*2; row += TENSOR_CORE_M) //得是8的倍数 8*4
    {
            nvcuda::wmma::fill_fragment(acc_frag, 0.0);
            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem + IDX(row, compute_idx * 4, 8), 8);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
            }

            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++)
            {
                nvcuda::wmma::load_matrix_sync(in_frag, sharedmem + IDX(row+1, compute_idx * 4, 8), 8);
                nvcuda::wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
            }

            nvcuda::wmma::store_matrix_sync(out + begin + row / 8 * 64 + HALO , acc_frag, TENSOR_CORE_M, nvcuda::wmma::mem_row_major); 
        }
}

/**
 * @param in input array pointer
 * @param out output array pointer
 * @param params parameter array pointer
 *
 */
void gpu_star_1d2r_step2(const double *__restrict__ in, double *__restrict__ out, const double *__restrict__ params, const int times, const int input_n)
{
    double param_matrix_h[2][8 * 8] = {};

    // Initialize parameter matrix

    for (int row = 0; row < 8; row++) // kernel size 7
        for (int col = 0; col <= row; ++col)
            param_matrix_h[0][row * 8 + col] = params[row - col];

    for (int row = 0; row < 8; row++)
        for (int col = row; col < 8; ++col)
            param_matrix_h[1][row * 8 + col] = params[row + 8 - col];

    // for(int i=0;i<8;++i){
    //     for(int j=0;j<8;++j)
    //         printf("%8.3f ",param_matrix_h[0][i*8+j]);
    //     printf("\n");
    // }

    // printf("\n");
    // for(int i=0;i<8;++i){
    //     for(int j=0;j<8;++j)
    //         printf("%8.3f ",param_matrix_h[1][i*8+j]);
    //     printf("\n");
    // }


    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 8 * sizeof(double)));

    const int cols = input_n + 2 * HALO ;
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

    // int lookup_table_h[D_BLOCK_SIZE_COL];
    // for(int j=0;j<D_BLOCK_SIZE_COL;++j){
    //     lookup_table_h[j]=IDX(j/8,j%8,8);
    // }

    // int *lookup_table_d;
    // CUDA_CHECK(cudaMalloc(&lookup_table_d,D_BLOCK_SIZE_COL*sizeof(int)));
    // CUDA_CHECK(cudaMemcpy(lookup_table_d,lookup_table_h,D_BLOCK_SIZE_COL*sizeof(int),cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (; i < times; i++)
    {
        CUDAKERNELCHECK((gpu_star_1d2r_step2_kernel<<<grid_config, block_config>>>(array_d[i % 2] , array_d[(i + 1) % 2]))); 
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    // std::cout << secs << std::endl;
    printf("GStencil/s = %f\n", ((double)input_n * times * 2) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2] , array_size - sizeof(double), cudaMemcpyDeviceToHost));

    return;
}
