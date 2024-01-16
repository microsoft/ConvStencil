#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

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

#define CHECK_CUDNN(expression)                             \
  {                                                         \
    cudnnStatus_t status = (expression);                    \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudnnGetErrorString(status) << std::endl;\
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 输入数据（N=1, C=3, H=8, W=8）
    int H = 512;
    int W = 512;
    int L = 512;
    int T = 512;
    double *input_data_h;
    input_data_h = (double*)malloc(1 * 1 * H * W * L * sizeof(double));

    for (int i = 0; i < H * W * L; i++) {
        input_data_h[i] = 1.0f;
    }

    double *data[2];
    double *input_data;
    CUDA_CHECK(cudaMalloc(&input_data, 1 * 1 * H * W * L * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(input_data, input_data_h, 1 * 1 * H * W * L * sizeof(double), cudaMemcpyHostToDevice));
    data[0] = input_data;

    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    int dims[5] = {1, 1, H, W, L};
    int strides[5] = {1*H*W*L, H*W*L, W*L, L, 1};
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_descriptor,
                                          //  /*format=*/CUDNN_TENSOR_NHWC,
                                           /*dataType=*/CUDNN_DATA_DOUBLE,
                                           5,
                                           dims,
                                           strides));

    // 卷积滤波器（K=2, C=3, H=5, W=5）
    double *filter_data_h;
    int kernel_size = 3;
    filter_data_h = (double*)malloc(1 * 1 * kernel_size * kernel_size * kernel_size * sizeof(double));

    for (int i = 0; i < kernel_size * kernel_size * kernel_size; i++) {
        filter_data_h[i] = (double)1/27;
    }

    double *filter_data;
    CUDA_CHECK(cudaMalloc(&filter_data, 1 * 1 * kernel_size * kernel_size * kernel_size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(filter_data, filter_data_h, 1 * 1 * kernel_size * kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice));

    cudnnFilterDescriptor_t filter_descriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    int kernelDims[5] = {1, 1, kernel_size, kernel_size, kernel_size};
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(filter_descriptor,
                                           /*dataType=*/CUDNN_DATA_DOUBLE,
                                           /*format=*/CUDNN_TENSOR_NCHW,
                                           5,
                                           kernelDims));

    // 卷积描述符
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    int pad = 1;
    int padA[3] = {pad, pad, pad};
    int filterStrideA[3] = {1, 1, 1};
    int dilationA[3] = {1, 1, 1};
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(convolution_descriptor,
                                                3,
                                                padA,
                                                filterStrideA,
                                                dilationA,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_DOUBLE));
    CHECK_CUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // 计算输出数据尺寸
    // int batch_size{0}, channels{0}, height{0}, width{0};
    int outputDims[5];
    CHECK_CUDNN(cudnnGetConvolutionNdForwardOutputDim(convolution_descriptor, 
                                      input_descriptor, 
                                      filter_descriptor, 
                                      5,
                                      outputDims));

    // 输出数据
    // double *output_data_h;
    // output_data_h = (double*)malloc(batch_size * channels * height * width * sizeof(double));

    // double *output_data;
    // cudaMalloc(&output_data, batch_size * channels * height * width * sizeof(double));
    // data[1] = output_data;

    int outputStrides[5] = {outputDims[1]*outputDims[2]*outputDims[3]*outputDims[4], outputDims[2]*outputDims[3]*outputDims[4], outputDims[3]*outputDims[4], outputDims[4], 1};

    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensorNdDescriptor(output_descriptor, 
                              CUDNN_DATA_DOUBLE, 
                              5, 
                              outputDims, 
                              outputStrides);

    double *output_data;
    CUDA_CHECK(cudaMalloc(&output_data, outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3]*outputDims[4] * sizeof(double)));
    data[1] = output_data;

    double *output_data_h;
    output_data_h = (double*)malloc(outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3]*outputDims[4] * sizeof(double));
    // 执行卷积前向传播
    double alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // CHECK_CUDNN(
    //     cudnnFindConvolutionForwardAlgorithm(cudnn,
    //                                         input_descriptor,
    //                                         filter_descriptor,
    //                                         convolution_descriptor,
    //                                         output_descriptor,
    //                                         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //                                         /*memoryLimitInBytes=*/0,
    //                                         &convolution_algorithm));

    size_t workspace_bytes{0};
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        input_descriptor,
                                                        filter_descriptor,
                                                        convolution_descriptor,
                                                        output_descriptor,
                                                        convolution_algorithm,
                                                        &workspace_bytes));

    void* d_workspace{nullptr};
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int t = 0; t < T; t++) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                                            &alpha,
                                            input_descriptor,
                                            data[t % 2],
                                            filter_descriptor,
                                            filter_data,
                                            convolution_descriptor,
                                            convolution_algorithm,
                                            d_workspace,
                                            workspace_bytes,
                                            &beta,
                                            output_descriptor,
                                            data[(t + 1) % 2]));
    }
     CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    std::cout << secs << std::endl;
    printf("GStencil/s = %f\n", ((double)H * W * L * T) / secs / 1e9);

    CUDA_CHECK(cudaMemcpy(output_data_h, output_data, outputDims[0]*outputDims[1]*outputDims[2]*outputDims[3]*outputDims[4] * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 500 * 500 * 499; i < 500 * 500 * 500; i++) {
    //     std::cout << output_data_h[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << height << " " << width << std::endl;
    

    // 释放所有资源
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    cudaFree(input_data);
    cudaFree(filter_data);
    cudaFree(output_data);
    cudaFree(d_workspace);

    return 0;
}
