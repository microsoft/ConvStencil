#include <mma.h>
#include <iostream>
#include "3d_utils.h"
#include <chrono>

using namespace nvcuda;

#define BLOCK_SIZE_ROW 8
#define BLOCK_SIZE_COL 64
#define HALO 3
#define UNIT_LENGTH 7
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2)
#define D_BLOCK_SIZE_ROW (BLOCK_SIZE_ROW + HALO * 2)
#define PAD 2
#define SM_SIZE_COL (UNIT_LENGTH * D_BLOCK_SIZE_ROW + PAD)
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / (UNIT_LENGTH + 1))
#define SM_DIFF (SM_SIZE_ROW * SM_SIZE_COL - D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL)
#define WARP_PER_BLOCK 8
#define COL_PER_WARP (BLOCK_SIZE_ROW / WARP_PER_BLOCK * UNIT_LENGTH)
#define TENSOR_CORE_M 8
#define MMA_NUM 13
#define IDX2D(x, y, ldm) ((x) * (ldm) + (y))
#define IDX3D(x, y, z, rows, cols) ((x) * (rows) * (cols) + (y) * (cols) + (z))

__constant__ double param_star_matrix_d[2 * 52 * TENSOR_CORE_M];
__constant__ double param_one_d[1];
__constant__ double param_five_d[5];
__constant__ double param_thirteen_d[13];

void copy_temp(double * __restrict__ temp_para, const double * __restrict__ params) {
    temp_para[IDX2D(0, 3, UNIT_LENGTH)] = params[19 + 0];
    temp_para[IDX2D(1, 2, UNIT_LENGTH)] = params[19 + 1];
    temp_para[IDX2D(1, 3, UNIT_LENGTH)] = params[19 + 2];
    temp_para[IDX2D(1, 4, UNIT_LENGTH)] = params[19 + 3];
    temp_para[IDX2D(2, 1, UNIT_LENGTH)] = params[19 + 4];
    temp_para[IDX2D(2, 2, UNIT_LENGTH)] = params[19 + 5];
    temp_para[IDX2D(2, 3, UNIT_LENGTH)] = params[19 + 6];
    temp_para[IDX2D(2, 4, UNIT_LENGTH)] = params[19 + 7];
    temp_para[IDX2D(2, 5, UNIT_LENGTH)] = params[19 + 8];
    temp_para[IDX2D(3, 0, UNIT_LENGTH)] = params[19 + 9];
    temp_para[IDX2D(3, 1, UNIT_LENGTH)] = params[19 + 10];
    temp_para[IDX2D(3, 2, UNIT_LENGTH)] = params[19 + 11];
    temp_para[IDX2D(3, 3, UNIT_LENGTH)] = params[19 + 12];
    temp_para[IDX2D(3, 4, UNIT_LENGTH)] = params[19 + 13];
    temp_para[IDX2D(3, 5, UNIT_LENGTH)] = params[19 + 14];
    temp_para[IDX2D(3, 6, UNIT_LENGTH)] = params[19 + 15];
    temp_para[IDX2D(4, 1, UNIT_LENGTH)] = params[19 + 16];
    temp_para[IDX2D(4, 2, UNIT_LENGTH)] = params[19 + 17];
    temp_para[IDX2D(4, 3, UNIT_LENGTH)] = params[19 + 18];
    temp_para[IDX2D(4, 4, UNIT_LENGTH)] = params[19 + 19];
    temp_para[IDX2D(4, 5, UNIT_LENGTH)] = params[19 + 20];
    temp_para[IDX2D(5, 2, UNIT_LENGTH)] = params[19 + 21];
    temp_para[IDX2D(5, 3, UNIT_LENGTH)] = params[19 + 22];
    temp_para[IDX2D(5, 4, UNIT_LENGTH)] = params[19 + 23];
    temp_para[IDX2D(6, 3, UNIT_LENGTH)] = params[19 + 24];
}

__forceinline__ __device__ void load_original_data(double * __restrict__ data, const double * __restrict__ in, const int h, const int rows, const int cols) {
    int begin = IDX3D(h, blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, rows, cols);
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += total_threads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        data[i] = in[begin + IDX2D(row, col, cols)];
    }
    __syncthreads();
}

__forceinline__ __device__ void load_shared_data(double * __restrict__ data, const double * __restrict__ in, const int h, const int rows, const int cols, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += total_threads) {
        data[IDX2D(0, lookup_table1[i], SM_SIZE_ROW * SM_SIZE_COL)] = in[i];
        data[IDX2D(1, lookup_table2[i], SM_SIZE_ROW * SM_SIZE_COL)] = in[i];
    }
    __syncthreads();
}

__forceinline__ __device__ void load_trans_data(double * __restrict__ data, const double * __restrict__ in, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += total_threads) {
        data[IDX2D(0, lookup_table1[i], SM_SIZE_ROW * SM_SIZE_COL)] = in[i];
        data[IDX2D(1, lookup_table2[i], SM_SIZE_ROW * SM_SIZE_COL)] = in[i];
    }
    __syncthreads();
}

__forceinline__ __device__ void compute_one_point(double * __restrict__ data, double * __restrict__ out) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < BLOCK_SIZE_ROW * BLOCK_SIZE_COL; i += total_threads) {
        int row = i / BLOCK_SIZE_COL;
        int col = i % BLOCK_SIZE_COL;
        out[i] = param_one_d[0] * data[IDX2D(row + HALO, col + HALO, D_BLOCK_SIZE_COL)];
    }
}

__forceinline__ __device__ void compute_five_point(double * __restrict__ data, double * __restrict__ out) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < BLOCK_SIZE_ROW * BLOCK_SIZE_COL; i += total_threads) {
        int row = i / BLOCK_SIZE_COL;
        int col = i % BLOCK_SIZE_COL;
        out[i] = 
            param_five_d[0] * data[IDX2D(HALO + row - 1, HALO + col, D_BLOCK_SIZE_COL)] + 
            param_five_d[1] * data[IDX2D(HALO + row, HALO + col - 1, D_BLOCK_SIZE_COL)] + 
            param_five_d[2] * data[IDX2D(HALO + row, HALO + col, D_BLOCK_SIZE_COL)] + 
            param_five_d[3] * data[IDX2D(HALO + row, HALO + col + 1, D_BLOCK_SIZE_COL)] + 
            param_five_d[4] * data[IDX2D(HALO + row + 1, HALO + col, D_BLOCK_SIZE_COL)];
    }
}

__forceinline__ __device__ void compute_thirteen_point(double * __restrict__ data, double * __restrict__ out) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < BLOCK_SIZE_ROW * BLOCK_SIZE_COL; i += total_threads) {
        int row = i / BLOCK_SIZE_COL;
        int col = i % BLOCK_SIZE_COL;
        out[i] = 
            param_thirteen_d[0] * data[IDX2D(HALO + row - 2, HALO + col, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[1] * data[IDX2D(HALO + row - 1, HALO + col - 1, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[2] * data[IDX2D(HALO + row - 1, HALO + col, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[3] * data[IDX2D(HALO + row - 1, HALO + col + 1, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[4] * data[IDX2D(HALO + row, HALO + col - 2, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[5] * data[IDX2D(HALO + row, HALO + col - 1, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[6] * data[IDX2D(HALO + row, HALO + col, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[7] * data[IDX2D(HALO + row, HALO + col + 1, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[8] * data[IDX2D(HALO + row, HALO + col + 2, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[9] * data[IDX2D(HALO + row + 1, HALO + col - 1, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[10] * data[IDX2D(HALO + row + 1, HALO + col, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[11] * data[IDX2D(HALO + row + 1, HALO + col + 1, D_BLOCK_SIZE_COL)] +
            param_thirteen_d[12] * data[IDX2D(HALO + row + 2, HALO + col, D_BLOCK_SIZE_COL)];
    }
}

__forceinline__ __device__ void compute_tensorcore(double * __restrict__ data, double * __restrict__ out, const int ldm, const int warp_id) {
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++) {
        wmma::load_matrix_sync(param_frag[0][i], param_star_matrix_d + i * 32, 8);
        wmma::load_matrix_sync(param_frag[1][i], param_star_matrix_d + 52 * 8 + i * 32, 8);
    }

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;

    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;

    for (int col = warp_id * COL_PER_WARP; col < warp_id * COL_PER_WARP + COL_PER_WARP; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, data + IDX2D(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, data + SM_SIZE_ROW * SM_SIZE_COL + IDX2D(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + IDX2D(col / UNIT_LENGTH, 0, BLOCK_SIZE_COL), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
    __syncthreads();
}

__forceinline__ __device__ void add(double * __restrict__ data1, double * __restrict__ data2, double * __restrict__ data3, double * __restrict__ data4, double * __restrict__ data5, double * __restrict__ data6, double * __restrict__ data7, double * __restrict__ out, const int cols) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < BLOCK_SIZE_ROW * BLOCK_SIZE_COL; i += total_threads) {
        int row = i / BLOCK_SIZE_COL;
        int col = i % BLOCK_SIZE_COL;
        out[IDX2D(row, col, cols)] = data1[i] + data2[i] + data3[i] + data4[i] + data5[i] + data6[i] + data7[i];
    }
}


__global__ void gpu_star_3d1r_step3_kernel (const double * __restrict__ in, double * __restrict__ out, const int heights, const int rows, const int cols, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    // __shared__ double data[D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL];
    // __shared__ double trans[2][SM_SIZE_ROW * SM_SIZE_COL];
    // __shared__ double intermediate[19][BLOCK_SIZE_ROW * BLOCK_SIZE_COL];
    extern __shared__ double data[];
    double * trans = &data[D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL];
    double * intermediate = &data[D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL + 2 * SM_SIZE_ROW * SM_SIZE_COL];

    int begin =  IDX2D(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL, cols);
    int warp_id = threadIdx.x / 32;
    // int tid = threadIdx.x;
    // int total_threads = blockDim.x;

    load_original_data(data, in, 0, rows, cols);
    compute_one_point(data, intermediate);
    load_original_data(data, in, 1, rows, cols);
    compute_one_point(data, intermediate + BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_five_point(data, intermediate + 7 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    load_original_data(data, in, 2, rows, cols);
    compute_one_point(data, intermediate + 2 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_five_point(data, intermediate + 8 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_thirteen_point(data, intermediate + 12 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    load_original_data(data, in, 3, rows, cols);
    compute_one_point(data, intermediate + 3 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_five_point(data, intermediate + 9 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_thirteen_point(data, intermediate + 13 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    load_trans_data(trans, data, lookup_table1, lookup_table2);
    compute_tensorcore(trans, intermediate + 16 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, SM_SIZE_COL, warp_id);
    load_original_data(data, in, 4, rows, cols);
    compute_one_point(data, intermediate + 4 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_five_point(data, intermediate + 10 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_thirteen_point(data, intermediate + 14 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    load_trans_data(trans, data, lookup_table1, lookup_table2);
    compute_tensorcore(trans, intermediate + 17 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, SM_SIZE_COL, warp_id);
    load_original_data(data, in, 5, rows, cols);
    compute_one_point(data, intermediate + 5 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_five_point(data, intermediate + 11 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    compute_thirteen_point(data, intermediate + 15 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
    load_trans_data(trans, data, lookup_table1, lookup_table2);
    compute_tensorcore(trans, intermediate + 18 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, SM_SIZE_COL, warp_id);
    for (int h = 6; h < heights + 6; h++) {
        load_original_data(data, in, h, rows, cols);
        compute_one_point(data, intermediate + (h % 7) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
        add(
            intermediate + ((h - 6) % 7) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, 
            intermediate + ((h - 6) % 5 + 7) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, 
            intermediate + ((h - 6) % 4 + 12) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, 
            intermediate + ((h - 6) % 3 + 16) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, 
            intermediate + ((h - 4) % 4 + 12) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, 
            intermediate + ((h - 2) % 5 + 7) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, 
            intermediate + (h % 7) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, 
            out + (h - 3) * rows * cols + begin + IDX2D(HALO, HALO, cols),
            cols);
        compute_five_point(data, intermediate + ((h - 6) % 5 + 7) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
        compute_thirteen_point(data, intermediate + ((h - 6) % 4 + 12) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL);
        load_trans_data(trans, data, lookup_table1, lookup_table2);
        compute_tensorcore(trans, intermediate + ((h - 6) % 3 + 16) * BLOCK_SIZE_ROW * BLOCK_SIZE_COL, SM_SIZE_COL, warp_id);
    }
}

void gpu_star_3d1r(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_h, const int input_m, const int input_n) {
    double param_matrix_h[2][52 * 8] = {0.0};

    // Initialize parameter matrix
    CUDA_CHECK(cudaMemcpyToSymbol(param_one_d, params, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(param_five_d, params + 1, 5 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(param_thirteen_d, params + 6, 13 * sizeof(double)));

    double temp_para[49] = {0.0};
    copy_temp(temp_para, params);

    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = temp_para[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = temp_para[i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_star_matrix_d, param_matrix_h, 2 * 8 * 52 * sizeof(double)));

    const int heights = input_h + 2 * HALO;
    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO;
    const size_t array_size = heights * rows * cols * sizeof(double);
    double *array_d[2];
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
    int sm_size = (D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL + 2 * SM_SIZE_ROW * SM_SIZE_COL + 19 * BLOCK_SIZE_ROW * BLOCK_SIZE_COL) * sizeof(double);
    CUDA_CHECK(cudaFuncSetAttribute(gpu_star_3d1r_step3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX2D(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX2D((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
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

    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((gpu_star_3d1r_step3_kernel<<<grid_config, block_config, sm_size>>>(array_d[i % 2], array_d[(i + 1) % 2], input_h, rows, cols, lookup_table1_d, lookup_table2_d)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    std::cout << secs << std::endl;
    printf("GStencil/s = %f\n", ((double)input_m * input_n * input_h * times * 3) / secs / 1e9);
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}