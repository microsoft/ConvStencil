#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

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

    int H = 10000;
    int W = 10000;
    int T = 10000;
    double *input_data_h;
    input_data_h = (double*)malloc(1 * 1 * H * W * sizeof(double));

    for (int i = 0; i < H * W; i++) {
        input_data_h[i] = 1.0f;
    }

    double *data[2];
    double *input_data;
    cudaMalloc(&input_data, 1 * 1 * H * W * sizeof(double));
    cudaMemcpy(input_data, input_data_h, 1 * 1 * H * W * sizeof(double), cudaMemcpyHostToDevice);
    data[0] = input_data;

    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                           /*format=*/CUDNN_TENSOR_NHWC,
                                           /*dataType=*/CUDNN_DATA_DOUBLE,
                                           /*batch_size=*/1,
                                           /*channels=*/1,
                                           /*image_height=*/H,
                                           /*image_width=*/W));

    double *filter_data_h;
    filter_data_h = (double*)malloc(1 * 1 * 3 * 3 * sizeof(double));

    for (int i = 0; i < 3 * 3; i++) {
        filter_data_h[i] = 0.1111f;
    }

    double *filter_data;
    cudaMalloc(&filter_data, 1 * 1 * 3 * 3 * sizeof(double));
    cudaMemcpy(filter_data, filter_data_h, 1 * 1 * 3 * 3 * sizeof(double), cudaMemcpyHostToDevice);

    cudnnFilterDescriptor_t filter_descriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                           /*dataType=*/CUDNN_DATA_DOUBLE,
                                           /*format=*/CUDNN_TENSOR_NCHW,
                                           /*out_channels=*/1,
                                           /*in_channels=*/1,
                                           /*kernel_height=*/3,
                                           /*kernel_width=*/3));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                /*pad_height=*/1,
                                                /*pad_width=*/1,
                                                /*vertical_stride=*/1,
                                                /*horizontal_stride=*/1,
                                                /*dilation_height=*/1,
                                                /*dilation_width=*/1,
                                                /*mode=*/CUDNN_CROSS_CORRELATION,
                                                /*computeType=*/CUDNN_DATA_DOUBLE));
    CHECK_CUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // 计算输出数据尺寸
    int batch_size{0}, channels{0}, height{0}, width{0};
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                      input_descriptor,
                                                      filter_descriptor,
                                                      &batch_size,
                                                      &channels,
                                                      &height,
                                                      &width));

    double *output_data_h;
    output_data_h = (double*)malloc(batch_size * channels * height * width * sizeof(double));

    double *output_data;
    cudaMalloc(&output_data, batch_size * channels * height * width * sizeof(double));
    data[1] = output_data;

    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                           /*format=*/CUDNN_TENSOR_NHWC,
                                           /*dataType=*/CUDNN_DATA_DOUBLE,
                                           /*batch_size=*/batch_size,
                                           /*channels=*/channels,
                                           /*image_height=*/height,
                                           /*image_width=*/width));

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
    cudaMalloc(&d_workspace, workspace_bytes);

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
    cudaDeviceSynchronize() ;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
    printf("GStencil/s = %f\n", ((double)H * W * T) / secs / 1e9);

    cudaMemcpy(output_data_h, output_data, batch_size * channels * height * width * sizeof(double), cudaMemcpyDeviceToHost);




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
