

// #pragma once

// #include <cmath>
// #include <functional>
// #include <iostream>
// #include <random>
// #include <stdexcept>
// #include <string>

// #include <cuda_runtime_api.h>
// #include <library_types.h>



// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)

// device memory pitch alignment
static const size_t device_alignment = 32;

// void gpu_box_2d1r_step3(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int time, const int input_m, const int input_n);

// __global__ void gpu_box_2d1r_step3_kernel (const double * __restrict__ in, double * __restrict__ out);

// #include <helper_cuda.h>
// #include <helper_functions.h>

#define DATA_TYPE double

#define TENSOR_CORE_M 8

#pragma once
#define CUDAKERNELCHECK(expr)                                                               \
    do                                                                                        \
    {                                                                                         \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess)                                                             \
        {                                                                                     \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)


#include <stdio.h>

#define CUDA_CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// #pragma once

enum Shape
{
    box_3d1r,
    star_3d1r,
};

void gpu_box_3d1r(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_h, const int input_m, const int input_n);

void gpu_star_3d1r(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_h, const int input_m, const int input_n);

void gpu_box_3d1r_breakdown4(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_h, const int input_m, const int input_n);

void gpu_box_3d1r_breakdown3(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_h, const int input_m, const int input_n);

void gpu_box_3d1r_breakdown2(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_h, const int input_m, const int input_n);

void gpu_box_3d1r_breakdown1(const double * __restrict__ in, double * __restrict__ out, const double * __restrict__ params, const int times, const int input_h, const int input_m, const int input_n);