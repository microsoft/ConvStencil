#include <mma.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include "../utils.h"
#include <iostream>
#include "2d_utils.h"
#include <chrono>

using namespace nvcuda;

#define BLOCK_SIZE_ROW 32
#define BLOCK_SIZE_COL 64
#define HALO 3
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2)
#define D_BLOCK_SIZE_ROW (BLOCK_SIZE_ROW + HALO * 2)
#define PAD 2
#define SM_SIZE_COL (7 * D_BLOCK_SIZE_ROW + PAD)
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / 8)
#define UNIT_LENGTH 7
#define TENSOR_CORE_M 8
#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define WARP_PER_BLOCK 8
// #define ACCS_PER_WARP (BLOCK_SIZE_COL * BLOCK_SIZE_ROW / 64 / WARP_PER_BLOCK)
#define MMA_NUM 13
#define ceild(n,d)	(((n)-1)/(d) + 1)

__constant__ double param_matrix_d[2 * 52 * TENSOR_CORE_M];


__global__ void kernel2d (const double * __restrict__ in, double * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
        sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
    }
    __syncthreads();


    int warp_id = threadIdx.x / 32;

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++) {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 52 * 8 + i * 32, 8);
    }

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    for (int col = warp_id * 28; col < warp_id * 28 + 28; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
}

__global__ void breakdown4_kernel (const double * __restrict__ in, double * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;

        if (lookup_table1[i] != -1) {
            sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
        }
        if (lookup_table2[i] != -1) {
            sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
        }
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++) {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 52 * 8 + i * 32, 8);
    }

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    for (int col = warp_id * 4*7; col < warp_id *4*7 + 4*7; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);//1+
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);//1+
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
}

__global__ void breakdown3_kernel(const double * __restrict__ in, double * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * (SM_SIZE_COL - PAD)];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;

        if (lookup_table1[i] != -1) {
            sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
        }
        if (lookup_table2[i] != -1) {
            sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
         }
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++) {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 52 * 8 + i * 32, 8);
    }

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    for (int col = warp_id * 4*7; col < warp_id *4*7 + 4*7; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            // if(threadIdx.x%32==0)printf("%d\n",IDX(0, col + compute_idx * 4, (SM_SIZE_COL - PAD)));
            wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 4, (SM_SIZE_COL - PAD)), (SM_SIZE_COL - PAD));//1+
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 4, (SM_SIZE_COL - PAD)), (SM_SIZE_COL - PAD));//1+
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
}

__global__ void breakdown2_kernel (const double * __restrict__ in, double * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * (SM_SIZE_COL - PAD)];

    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);//分块


    int x=threadIdx.x;//0~7
    int y=threadIdx.y;//0~31

    int tid=threadIdx.x+threadIdx.y*blockDim.x;
    for(int i=tid;i<D_BLOCK_SIZE_ROW*D_BLOCK_SIZE_COL;i+=blockDim.x*blockDim.y){
        int row=i/D_BLOCK_SIZE_COL;
        int col=i%D_BLOCK_SIZE_COL;
            if ((col + 1) % 8 != 0 && col < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
               sharedmem[0][IDX(col / (UNIT_LENGTH + 1), UNIT_LENGTH * row + col % (UNIT_LENGTH + 1), (SM_SIZE_COL - PAD))]= in[begin + IDX(row, col, ldm)];//读了halo区
            } 
            if ((col + 2) % 8 != 0 && col > 2 * HALO) {
                sharedmem[1][IDX((col - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * row + (col - UNIT_LENGTH) % (UNIT_LENGTH + 1), (SM_SIZE_COL - PAD))]= in[begin + IDX(row, col, ldm)];
            } 
        }
    __syncthreads();

    for(int row=x;row<SM_SIZE_ROW;row+=blockDim.x){
        for(int col=7*y;col<(SM_SIZE_COL - PAD)-49+7;col+=7*blockDim.y){
            double result[8]={};
            for(int i=0;i<7;++i)
                for(int k=0;k<7*7;++k){
                    result[i]+=sharedmem[0][IDX(row,col+k,(SM_SIZE_COL - PAD))]*param_matrix_d[IDX(k,i,7)];
                    result[i+1]+=sharedmem[1][IDX(row,col+k,(SM_SIZE_COL - PAD))]*param_matrix_d[49*7+IDX(k,i,7)];
                }
            for(int i=0;i<8;++i)out[begin+IDX(HALO+col/7,row*8+HALO,ldm)+i]=result[i];
        }

    }
}

__global__ void breakdown1_kernel ( double * __restrict__ in, double * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2,double* __restrict__ la,double * __restrict__ lb) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * (SM_SIZE_COL - PAD)];

    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);

    int gbegin=(blockIdx.x*blockDim.y+blockIdx.y)*(SM_SIZE_COL - PAD)*SM_SIZE_ROW;

    int x=threadIdx.x;//0~7
    int y=threadIdx.y;//0~31

    int tid=threadIdx.x+threadIdx.y*blockDim.x;
    for(int i=tid;i<D_BLOCK_SIZE_ROW*D_BLOCK_SIZE_COL;i+=blockDim.x*blockDim.y){
        int row=i/D_BLOCK_SIZE_COL;
        int col=i%D_BLOCK_SIZE_COL;
            if ((col + 1) % 8 != 0 && col < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                la[gbegin+IDX(col / (UNIT_LENGTH + 1), UNIT_LENGTH * row + col % (UNIT_LENGTH + 1), (SM_SIZE_COL - PAD))]=in[begin + IDX(row, col, ldm)];
            } 
            if ((col + 2) % 8 != 0 && col > 2 * HALO) {
            lb[gbegin+IDX((col - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * row + (col - UNIT_LENGTH) % (UNIT_LENGTH + 1), (SM_SIZE_COL - PAD))]=in[begin + IDX(row, col, ldm)];
            } 
        }

    __syncthreads();
    for(int row=x;row<SM_SIZE_ROW;row+=blockDim.x){
        for(int col=7*y;col<(SM_SIZE_COL - PAD)-49+7;col+=7*blockDim.y){
            double result[8]={};
            for(int i=0;i<7;++i)
                for(int k=0;k<7*7;++k){
                    result[i]+=la[gbegin+IDX(row,col+k,(SM_SIZE_COL - PAD))]*param_matrix_d[IDX(k,i,7)];
                    result[i+1]+=lb[gbegin+IDX(row,col+k,(SM_SIZE_COL - PAD))]*param_matrix_d[49*7+IDX(k,i,7)];
                }

            for(int i=0;i<8;++i)out[begin+IDX(HALO+col/7,row*8+HALO,ldm)+i]=result[i];
        }
    }
}


/**
 * @param in input array pointer
 * @param out output array pointer
 * @param params parameter array pointer (length 49)
 * 
*/
void gpu_box_2d1r(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_m, const int input_n) {
    double param_matrix_h[2][52 * 8] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 52 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO + 2;
    const size_t array_size = rows * cols * sizeof(double);
    double *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    
    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    // dim3 grid_config(1, 1);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // Lookup table
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }


    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((kernel2d<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "ConvStencil(2D): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    printf("GStencil/s = %f\n", ((double)input_m * input_n * times * 3) / secs / 1e9);
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}

void gpu_box_2d3r(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_m, const int input_n) {
    double param_matrix_h[2][52 * 8] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 52 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO + 2;
    const size_t array_size = rows * cols * sizeof(double);
    double *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    
    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    // dim3 grid_config(1, 1);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // Lookup table
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }


    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((kernel2d<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "ConvStencil(2D): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    printf("GStencil/s = %f\n", ((double)input_m * input_n * times) / secs / 1e9);
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}


/**
 * @param in input array pointer
 * @param out output array pointer
 * @param params parameter array pointer (length 49)
 * 
*/
void gpu_box_2d1r_breakdown4(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_m, const int input_n) {
    double param_matrix_h[2][52 * 8] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 52 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO + 2;
    const size_t array_size = rows * cols * sizeof(double);
    double *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    
    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    // dim3 grid_config(1, 1);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // Lookup table
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = - 1;
            }
        }
    }


    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((breakdown4_kernel<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(2D) 4: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    printf("GStencil/s = %f\n\n", ((double)input_m * input_n * times * 3) / secs / 1e9);
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}

/**
 * @param in input array pointer
 * @param out output array pointer
 * @param params parameter array pointer (length 49)
 * 
*/
void gpu_box_2d1r_breakdown3(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_m, const int input_n) {
    double param_matrix_h[2][52 * 8] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 52 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO +2;
    const size_t array_size = rows * cols * sizeof(double);
    double *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    
    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    // dim3 grid_config(1, 1);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // Lookup table
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] =  - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] =  - 1;
            }
        }
    }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((breakdown3_kernel<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(2D) 3: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    printf("GStencil/s = %f\n\n", ((double)input_m * input_n * times * 3) / secs / 1e9);
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}

void gpu_box_2d1r_breakdown2(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_m, const int input_n) {
    double param_matrix_h[2][49 * 7] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < UNIT_LENGTH ; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * UNIT_LENGTH  + col] = params[i * UNIT_LENGTH + j - col];//（i*UNIT_LENGTH+j,col）
                }
            }
        }
    }
    for (int col = 0; col < UNIT_LENGTH ; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j <= col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * UNIT_LENGTH  + col] = params[i * UNIT_LENGTH + j - col + 6];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 7 * 49 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO+1 ;
    const size_t array_size = rows * cols * sizeof(double);
    double *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    
    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    dim3 block_config(8,32);

    // Lookup table
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((breakdown2_kernel<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(2D) 2: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    printf("GStencil/s = %f\n\n", ((double)input_m * input_n * times * 3) / secs / 1e9);
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}

void gpu_box_2d1r_breakdown1(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_m, const int input_n) {
    double param_matrix_h[2][49 * 7] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < UNIT_LENGTH ; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * UNIT_LENGTH  + col] = params[i * UNIT_LENGTH + j - col];//（i*UNIT_LENGTH+j,col）
                }
            }
        }
    }
    for (int col = 0; col < UNIT_LENGTH ; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j <= col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * UNIT_LENGTH  + col] = params[i * UNIT_LENGTH + j - col + 6];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 7 * 49 * sizeof(double)));

    const int rows = input_m + 2 * HALO;
    // const int cols = input_n + 2 * HALO + 2;
    const int cols = input_n + 2 * HALO+1 ;
    const size_t array_size = rows * cols * sizeof(double);
    double *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));



    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    dim3 block_config(8,32);

    double* stencil2row[2];
    CUDA_CHECK(cudaMalloc(&stencil2row[0], BLOCK_M*BLOCK_N*SM_SIZE_COL*SM_SIZE_ROW*sizeof(double)));//*0.75?
    CUDA_CHECK(cudaMalloc(&stencil2row[1], BLOCK_M*BLOCK_N*SM_SIZE_COL*SM_SIZE_ROW*sizeof(double)));

    // Lookup table
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }
    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((breakdown1_kernel<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d,stencil2row[0],stencil2row[1])));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Experiment - Breakdown(2D) 1: " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    printf("GStencil/s = %f\n\n", ((double)input_m * input_n * times * 3) / secs / 1e9);
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}
