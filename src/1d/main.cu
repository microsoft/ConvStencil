#include <iostream>
#include <omp.h>
#include "1d_utils.h"
#include <cmath>

const char *ShapeStr[4] = {
    "1d1r",
    "1d2r"
};

#define FILL_RANDOM
// #define FILL_INDEX

// #define CHECK_ERROR
const double tolerance = 1e-7;
__constant__ double param_matrix_d[2 * 8 * TENSOR_CORE_M];
// #define WRITE_OUTPUT

int HALO;

void save_to_txt(double *arr, int cols, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < cols ; i++)
    {
        fprintf(file, "%d %.0f\n", i , arr[i]);
    }

    fclose(file);
}

void naive_1d(double *in, double *out, double *param, int N, int halo)//halo=r
{
#pragma unroll
    for (int j = halo; j < N + halo; j++)
    {
        out[j] = 0;
        for (int k = -halo; k <= halo; ++k)
            out[j] += param[k + halo] * in[j + k];
    }
}

void printHelp()
{
    const char *helpMessage =
        "Program name: convstencil_1d\n"
        "Usage: convstencil_2 shape input_size time_iteration_size [Options]\n"
        "Shape: 1d1r or 1d2r\n"
        "Options:\n"
        "  --help    Display this help message and exit\n"
        "  --custom  If you want to use costum parameters, please use this option and input your parameters like 0.2 0.2 0.2 0.2 0.2 if the shape is star2d1r\n";
    printf("%s\n", helpMessage);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printHelp();
        return 1;
    }

    // configurable settings
    std::string arg1 = argv[1];

    Shape compute_shape;
    if(arg1 == "1d1r"){
        compute_shape=star_1d1r;
    }
    else if(arg1 == "1d2r"){
        compute_shape=star_1d2r;
    } else {
        printHelp();
        return 1;
    }

    int n = 0;
    int times = 0;

    try
    {
        n = std::stoi(argv[2]);
        times = std::stoi(argv[3]);
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Invalid argument: cannot convert the parameter(s) to integer.\n";
        return 1;
    }
    catch (const std::out_of_range &e)
    {
        std::cerr << "Argument out of range: the parameter(s) is(are) too large.\n";
        return 1;
    }

    double param_1d1r[7] = {};
    double param_1d2r[9] = {};

    bool breakdown = false;

    for (int i = 0; i < 7; i++)
    {
        param_1d1r[i] = i + 1;
    }

    
    for (int i = 0; i < 9; i++)
    {
        param_1d2r[i] = i + 1;
    }

    if (argc == 5 && std::string(argv[4]) == "--custom") {
        int num_param = 0;
        if (arg1 == "1d1r") {
            num_param = 3;
        } else if (arg1 == "1d2r") {
            num_param = 5;
        }
        printf("Please enter %d parameters:\n", num_param);
        double values[num_param];
        for (int i = 0; i < num_param; i++)
        {
            int readNum = scanf("%lf", &values[i]);
            if (readNum != 1)
                return 1;
        }
        if (arg1 == "1d1r") {
            param_1d1r[0] = values[0] * values[0] * values[0];
            param_1d1r[1] = 3 * values[0] * values[0] * values[1];
            param_1d1r[2] = 3 * values[0] * values[0] * values[2] + 3 * values[0] * values[1] * values[1];
            param_1d1r[3] = 6 * values[0] * values[1] * values[2] + values[1] * values[1] * values[1];
            param_1d1r[4] = 3 * values[0] * values[2] * values[2] + 3 * values[1] * values[1] * values[2];
            param_1d1r[5] = 3 * values[1] * values[2] * values[2];
            param_1d1r[6] = values[2] * values[2] * values[2];
        } else if (arg1 == "1d2r") {
            param_1d2r[0] = values[0] * values[0];
            param_1d2r[1] = 2 * values[0] * values[1];
            param_1d2r[2] = 2 * values[0] * values[2] + values[1] * values[1];
            param_1d2r[3] = 2 * values[0] * values[3] + 2 * values[1] * values[2];
            param_1d2r[4] = 2 * values[0] * values[4] + 2 * values[1] * values[3] + values[2] * values[2];
            param_1d2r[5] = 2 * values[1] * values[4] + 2 * values[2] * values[3];
            param_1d2r[6] = 2 * values[2] * values[4] + values[3] * values[3];
            param_1d2r[7] = 2 * values[3] * values[4];
            param_1d2r[8] = values[4] * values[4];
        }
    }

    if (argc == 5 && std::string(argv[4]) == "--breakdown") {
        breakdown = true;
    }

    double *param;

    switch (compute_shape)
    {
    case star_1d1r:
        param = param_1d1r;
        HALO = 3;
        break;
    case star_1d2r:
        param=param_1d2r;
        HALO= 4;
        break;
    }

    // print brief info

    printf("INFO: shape = %s,  n = %d, times = %d\n", ShapeStr[compute_shape], n, times);

    int cols = n + 2 * HALO; //+1

    size_t input_size = (unsigned long)cols * sizeof(double);

    // allocate space

    double *input = (double *)malloc(input_size + sizeof(double)); // alignment for tensor core
    double *output = (double *)malloc(input_size + sizeof(double));

    // fill input matrix

#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < cols + 1; i++)
    {
        input[i] = (double)(rand() % 10000);
    }
#elif defined(FILL_INDEX)
    if(compute_shape==star_1d1r_step3){
        for (int i = 0; i < cols + 1; i++)
        {
            if (i < HALO + 1 || i > cols - HALO)//+1为了对齐
                input[i] = 0;
            else
            {
                input[i] = i + 1 - (HALO+1);
                // printf("%d %lf\n",i,input[i]);
            }
        }
    }
    else{
        for (int i = 0; i < cols ; i++)
        {
            if (i < HALO  || i > cols - HALO -1)
                input[i] = 0;
            else
            {
                input[i] = i + 1 - HALO;
            }
            // printf("%d %lf\n",i,input[i]);
        }

    }
#endif

    switch (compute_shape)
    {
    case star_1d1r:
        if (breakdown) {
            gpu_1d1r_breakdown1(input, output, param, times, n);
            gpu_1d1r_breakdown2(input, output, param, times, n);
            gpu_1d1r_breakdown3(input, output, param, times, n);
            gpu_1d1r_breakdown4(input, output, param, times, n);
        }
        gpu_1d1r(input, output, param, times, n);
        break;
    case star_1d2r:
        gpu_star_1d2r_step2(input, output, param, times, n);
        break;
    }

    // check result correctness
 
#if defined(CHECK_ERROR)
    printf("\nChecking ... \n");
    double *naive[2];
    naive[0] = (double *)malloc(input_size);
    naive[1] = (double *)malloc(input_size);

    for (int i = 0; i < cols; i++)
    {
        if(compute_shape==star_1d2r_step2)naive[0][i] = input[i];
        else naive[0][i]=input[i+1];
        naive[1][i] = 0;
        // printf("%lf ",naive[0][i]);
    }

    int t = 0;

        for (; t < times; t++)
        {
            naive_1d(naive[t % 2], naive[(t + 1) % 2], param, n, HALO);
        }


    printf("Comparing naive and output\n");
    for (int col = HALO; col < cols - HALO; col++)
    {
        if (std::fabs(naive[t % 2][col] - output[col]) > 1e-7)
        {
            printf("col = %d, naive = %lf, output = %lf\n", col, naive[t % 2][col], output[col]);
        }
    }
#endif

    // write to file

#ifdef WRITE_OUTPUT
    printf("Writing output_gpu.txt\n");
    save_to_txt(output, cols, "output_gpu.txt");
    #if defined(CHECK_ERROR)
        save_to_txt(naive[t % 2], cols, "output_naive.txt");
    #endif
#endif

    // free space
    free(output);
    free(input);

    return 0;
}