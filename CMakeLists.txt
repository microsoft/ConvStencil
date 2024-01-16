cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(spmm_coo VERSION 0.01 LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
enable_language(CXX CUDA)

# Define the include DIRs
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

# Define the link libraries
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/usr/local/cuda/lib64)
add_executable(convstencil_1d src/1d/main.cu src/1d/gpu_1r.cu src/1d/gpu_2r.cu src/1d/1d_utils.h)
# target_link_libraries( gpu_box2d1r ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )
set_target_properties(
    convstencil_1d
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(convstencil_1d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --gpu-architecture=sm_80>)


add_executable(cudnn_1d3p src/cudnn/conv_1d3p.cu)
target_link_libraries( cudnn_1d3p ${CUDA_cudnn_LIBRARY})
set_target_properties(
    cudnn_1d3p
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(cudnn_1d3p PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -lcudnn --use_fast_math --gpu-architecture=sm_80>)
target_link_libraries(cudnn_1d3p ${CUDA_LIBRARIES} cudnn)

add_executable(cudnn_1d5p src/cudnn/conv_1d5p.cu)
target_link_libraries( cudnn_1d5p ${CUDA_cudnn_LIBRARY})
set_target_properties(
    cudnn_1d5p
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(cudnn_1d5p PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -lcudnn --use_fast_math --gpu-architecture=sm_80>)
target_link_libraries(cudnn_1d5p ${CUDA_LIBRARIES} cudnn)

add_executable(cudnn_box2d49p src/cudnn/conv_box2d49p.cu)
target_link_libraries( cudnn_box2d49p ${CUDA_cudnn_LIBRARY})
set_target_properties(
    cudnn_box2d49p
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(cudnn_box2d49p PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -lcudnn --use_fast_math --gpu-architecture=sm_80>)
target_link_libraries(cudnn_box2d49p ${CUDA_LIBRARIES} cudnn)

add_executable(cudnn_box2d9p src/cudnn/conv_box2d9p.cu)
target_link_libraries( cudnn_box2d9p ${CUDA_cudnn_LIBRARY})
set_target_properties(
    cudnn_box2d9p
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(cudnn_box2d9p PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -lcudnn --use_fast_math --gpu-architecture=sm_80>)
target_link_libraries(cudnn_box2d9p ${CUDA_LIBRARIES} cudnn)

add_executable(cudnn_box3d27p src/cudnn/conv_box3d27p.cu)
target_link_libraries( cudnn_box2d9p ${CUDA_cudnn_LIBRARY})
set_target_properties(
    cudnn_box3d27p
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(cudnn_box3d27p PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -lcudnn --use_fast_math --gpu-architecture=sm_80>)
target_link_libraries(cudnn_box3d27p ${CUDA_LIBRARIES} cudnn)

add_executable(convstencil_2d src/2d/main.cu src/2d/gpu.cu src/2d/2d_utils.h)
# target_link_libraries( gpu_box2d1r ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )
set_target_properties(
    convstencil_2d
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(convstencil_2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --gpu-architecture=sm_80>)

add_executable(convstencil_3d src/3d/main.cu src/3d/gpu_box.cu src/3d/gpu_star.cu  src/3d/3d_utils.h)
# target_link_libraries( gpu_box2d1r ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )
set_target_properties(
    convstencil_3d
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80")
target_compile_options(convstencil_3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --gpu-architecture=sm_80>)

set(CMAKE_CUDA_ARCHITECTURES 80)  
# add_subdirectory(breakdown)
