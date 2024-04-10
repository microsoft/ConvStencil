#include <iostream>
#include <omp.h>
// #include "type.h"
// #include "../utils.h"
#include "2d_utils.h"
// #include "mix/mix.h"
// #include "cpu/cpu.h"
// #include "gpu/gpu.h"
// #include "heat/heat.h"

const char *ShapeStr[5] = {
    "star_2d1r",
    "box_2d1r",
    "star_2d3r",
    "box_2d3r",
};

// Fill the matrix with random numbers or indices
#define FILL_RANDOM
// #define FILL_INDEX

// Check the correctness of the result or not
// #define CHECK_ERROR
const double tolerance = 1e-7;

#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define ABS(x, y) (((x) > (y)) ? ((x) - (y)) : ((y) - (x)))

// Write the output to file or not
// #define WRITE_OUTPUT

/* Global variable */
int NY;
int XSLOPE, YSLOPE;

void save_to_txt(double *arr, int rows, int cols, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fprintf(file, "%.0f\t", arr[IDX(i, j, cols)]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void naive_box2d1r(double *in, double *out, double *param, const int input_m, const int input_n)
{
    for (int row = 3; row < input_m - 3; row++)
    {
        for (int col = 4; col < input_n - 4; col++)
        {
            out[IDX(row, col, input_n)] =
                param[0] * in[IDX(row - 3, col - 3, input_n)] +
                param[1] * in[IDX(row - 3, col - 2, input_n)] +
                param[2] * in[IDX(row - 3, col - 1, input_n)] +
                param[3] * in[IDX(row - 3, col, input_n)] +
                param[4] * in[IDX(row - 3, col + 1, input_n)] +
                param[5] * in[IDX(row - 3, col + 2, input_n)] +
                param[6] * in[IDX(row - 3, col + 3, input_n)] +
                param[7] * in[IDX(row - 2, col - 3, input_n)] +
                param[8] * in[IDX(row - 2, col - 2, input_n)] +
                param[9] * in[IDX(row - 2, col - 1, input_n)] +
                param[10] * in[IDX(row - 2, col, input_n)] +
                param[11] * in[IDX(row - 2, col + 1, input_n)] +
                param[12] * in[IDX(row - 2, col + 2, input_n)] +
                param[13] * in[IDX(row - 2, col + 3, input_n)] +
                param[14] * in[IDX(row - 1, col - 3, input_n)] +
                param[15] * in[IDX(row - 1, col - 2, input_n)] +
                param[16] * in[IDX(row - 1, col - 1, input_n)] +
                param[17] * in[IDX(row - 1, col, input_n)] +
                param[18] * in[IDX(row - 1, col + 1, input_n)] +
                param[19] * in[IDX(row - 1, col + 2, input_n)] +
                param[20] * in[IDX(row - 1, col + 3, input_n)] +
                param[21] * in[IDX(row, col - 3, input_n)] +
                param[22] * in[IDX(row, col - 2, input_n)] +
                param[23] * in[IDX(row, col - 1, input_n)] +
                param[24] * in[IDX(row, col, input_n)] +
                param[25] * in[IDX(row, col + 1, input_n)] +
                param[26] * in[IDX(row, col + 2, input_n)] +
                param[27] * in[IDX(row, col + 3, input_n)] +
                param[28] * in[IDX(row + 1, col - 3, input_n)] +
                param[29] * in[IDX(row + 1, col - 2, input_n)] +
                param[30] * in[IDX(row + 1, col - 1, input_n)] +
                param[31] * in[IDX(row + 1, col, input_n)] +
                param[32] * in[IDX(row + 1, col + 1, input_n)] +
                param[33] * in[IDX(row + 1, col + 2, input_n)] +
                param[34] * in[IDX(row + 1, col + 3, input_n)] +
                param[35] * in[IDX(row + 2, col - 3, input_n)] +
                param[36] * in[IDX(row + 2, col - 2, input_n)] +
                param[37] * in[IDX(row + 2, col - 1, input_n)] +
                param[38] * in[IDX(row + 2, col, input_n)] +
                param[39] * in[IDX(row + 2, col + 1, input_n)] +
                param[40] * in[IDX(row + 2, col + 2, input_n)] +
                param[41] * in[IDX(row + 2, col + 3, input_n)] +
                param[42] * in[IDX(row + 3, col - 3, input_n)] +
                param[43] * in[IDX(row + 3, col - 2, input_n)] +
                param[44] * in[IDX(row + 3, col - 1, input_n)] +
                param[45] * in[IDX(row + 3, col, input_n)] +
                param[46] * in[IDX(row + 3, col + 1, input_n)] +
                param[47] * in[IDX(row + 3, col + 2, input_n)] +
                param[48] * in[IDX(row + 3, col + 3, input_n)];
        }
    }
}

void printHelp()
{
    const char *helpMessage =
        "Program name: convstencil_2d\n"
        "Usage: convstencil_2d shape input_size_of_first_dimension input_size_of_second_dimension time_iteration_size [Options]\n"
        "Shape: box2d1r or star2d1r or box2d3r or star2d3r\n"
        "Options:\n"
        "  --help    Display this help message and exit\n"
        "  --custom  If you want to use costum parameters, please use this option and input your parameters like 0.2 0.2 0.2 0.2 0.2 if the shape is star2d1r\n";
    printf("%s\n", helpMessage);
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        printHelp();
        return 1;
    }

    // configurable settings
    Shape compute_shape;
    std::string arg1 = argv[1];
    if (arg1 == "box2d1r")
    {
        compute_shape = box_2d1r;
    }
    else if (arg1 == "star2d1r")
    {
        compute_shape = star_2d1r;
    }
    else if (arg1 == "star2d3r")
    {
        compute_shape = star_2d3r;
    }
    else if (arg1 == "box2d3r")
    {
        compute_shape = box_2d3r;
    }
    else
    {
        printHelp();
        return 1;
    }

    int m = 0;
    int n = 0;
    int times = 0;

    try
    {
        m = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        times = std::stoi(argv[4]);
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

    double param_1r[9] = {0.0};
    bool breakdown = false;
    if (argc == 6 && std::string(argv[5]) == "--custom")
    {
        int num_param = 9;
        if (arg1 == "box2d1r")
        {
            num_param = 9;
        }
        else if (arg1 == "star2d1r")
        {
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
        if (num_param == 9)
        {
            for (int i = 0; i < 9; i++)
            {
                param_1r[i] = values[i];
            }
        }
        else
        {
            param_1r[1] = values[0];
            param_1r[3] = values[1];
            param_1r[4] = values[2];
            param_1r[5] = values[3];
            param_1r[7] = values[4];
        }
    }

    if (argc == 6 && std::string(argv[5]) == "--breakdown") {
        breakdown = true;
    }

    double param_box_2d1r[49] = {0.0};
    double param_star_2d1r[49] = {0.0};

    for (int i = 0; i < 49; i++)
    {
        param_box_2d1r[i] = 0.021;
    }

    param_box_2d1r[16] = (3 * param_1r[0] * param_1r[0] * param_1r[8] + 6 * param_1r[0] * param_1r[1] * param_1r[7] + 6 * param_1r[0] * param_1r[2] * param_1r[6] + 6 * param_1r[0] * param_1r[3] * param_1r[5] + 3 * param_1r[0] * param_1r[4] * param_1r[4] + 3 * param_1r[1] * param_1r[1] * param_1r[6] + 6 * param_1r[1] * param_1r[3] * param_1r[4] + 3 * param_1r[2] * param_1r[3] * param_1r[3]);
    param_box_2d1r[15] = (3 * param_1r[0] * param_1r[0] * param_1r[7] + 6 * param_1r[0] * param_1r[1] * param_1r[6] + 6 * param_1r[0] * param_1r[3] * param_1r[4] + 3 * param_1r[1] * param_1r[3] * param_1r[3]);
    param_box_2d1r[14] = (3 * param_1r[0] * param_1r[0] * param_1r[6] + 3 * param_1r[0] * param_1r[3] * param_1r[3]);
    param_box_2d1r[17] = (6 * param_1r[0] * param_1r[1] * param_1r[8] + 6 * param_1r[0] * param_1r[2] * param_1r[7] + 6 * param_1r[0] * param_1r[4] * param_1r[5] + 3 * param_1r[1] * param_1r[1] * param_1r[7] + 6 * param_1r[1] * param_1r[2] * param_1r[6] + 6 * param_1r[1] * param_1r[3] * param_1r[5] + 3 * param_1r[1] * param_1r[4] * param_1r[4] + 6 * param_1r[2] * param_1r[3] * param_1r[4]);
    param_box_2d1r[18] = (6 * param_1r[0] * param_1r[2] * param_1r[8] + 3 * param_1r[0] * param_1r[5] * param_1r[5] + 3 * param_1r[1] * param_1r[1] * param_1r[8] + 6 * param_1r[1] * param_1r[2] * param_1r[7] + 6 * param_1r[1] * param_1r[4] * param_1r[5] + 3 * param_1r[2] * param_1r[2] * param_1r[6] + 6 * param_1r[2] * param_1r[3] * param_1r[5] + 3 * param_1r[2] * param_1r[4] * param_1r[4]);
    param_box_2d1r[19] = (6 * param_1r[1] * param_1r[2] * param_1r[8] + 3 * param_1r[1] * param_1r[5] * param_1r[5] + 3 * param_1r[2] * param_1r[2] * param_1r[7] + 6 * param_1r[2] * param_1r[4] * param_1r[5]);
    param_box_2d1r[20] = (3 * param_1r[2] * param_1r[2] * param_1r[8] + 3 * param_1r[2] * param_1r[5] * param_1r[5]);
    param_box_2d1r[9] = (3 * param_1r[0] * param_1r[0] * param_1r[5] + 6 * param_1r[0] * param_1r[1] * param_1r[4] + 6 * param_1r[0] * param_1r[2] * param_1r[3] + 3 * param_1r[1] * param_1r[1] * param_1r[3]);
    param_box_2d1r[8] = (3 * param_1r[0] * param_1r[0] * param_1r[4] + 6 * param_1r[0] * param_1r[1] * param_1r[3]);
    param_box_2d1r[7] = 3 * param_1r[0] * param_1r[0] * param_1r[3];
    param_box_2d1r[10] = (6 * param_1r[0] * param_1r[1] * param_1r[5] + 6 * param_1r[0] * param_1r[2] * param_1r[4] + 3 * param_1r[1] * param_1r[1] * param_1r[4] + 6 * param_1r[1] * param_1r[2] * param_1r[3]);
    param_box_2d1r[11] = (6 * param_1r[0] * param_1r[2] * param_1r[5] + 3 * param_1r[1] * param_1r[1] * param_1r[5] + 6 * param_1r[1] * param_1r[2] * param_1r[4] + 3 * param_1r[2] * param_1r[2] * param_1r[3]);
    param_box_2d1r[12] = (6 * param_1r[1] * param_1r[2] * param_1r[5] + 3 * param_1r[2] * param_1r[2] * param_1r[4]);
    param_box_2d1r[13] = 3 * param_1r[2] * param_1r[2] * param_1r[5];
    param_box_2d1r[2] = (3 * param_1r[0] * param_1r[0] * param_1r[2] + 3 * param_1r[0] * param_1r[1] * param_1r[1]);
    param_box_2d1r[1] = 3 * param_1r[0] * param_1r[0] * param_1r[1];
    param_box_2d1r[0] = param_1r[0] * param_1r[0] * param_1r[0];
    param_box_2d1r[3] = (6 * param_1r[0] * param_1r[1] * param_1r[2] + param_1r[1] * param_1r[1] * param_1r[1]);
    param_box_2d1r[4] = (3 * param_1r[0] * param_1r[2] * param_1r[2] + 3 * param_1r[1] * param_1r[1] * param_1r[2]);
    param_box_2d1r[5] = 3 * param_1r[1] * param_1r[2] * param_1r[2];
    param_box_2d1r[6] = param_1r[2] * param_1r[2] * param_1r[2];
    param_box_2d1r[23] = (6 * param_1r[0] * param_1r[3] * param_1r[8] + 6 * param_1r[0] * param_1r[4] * param_1r[7] + 6 * param_1r[0] * param_1r[5] * param_1r[6] + 6 * param_1r[1] * param_1r[3] * param_1r[7] + 6 * param_1r[1] * param_1r[4] * param_1r[6] + 6 * param_1r[2] * param_1r[3] * param_1r[6] + 3 * param_1r[3] * param_1r[3] * param_1r[5] + 3 * param_1r[3] * param_1r[4] * param_1r[4]);
    param_box_2d1r[22] = (6 * param_1r[0] * param_1r[3] * param_1r[7] + 6 * param_1r[0] * param_1r[4] * param_1r[6] + 6 * param_1r[1] * param_1r[3] * param_1r[6] + 3 * param_1r[3] * param_1r[3] * param_1r[4]);
    param_box_2d1r[21] = (6 * param_1r[0] * param_1r[3] * param_1r[6] + param_1r[3] * param_1r[3] * param_1r[3]);
    param_box_2d1r[24] = (6 * param_1r[0] * param_1r[4] * param_1r[8] + 6 * param_1r[0] * param_1r[5] * param_1r[7] + 6 * param_1r[1] * param_1r[3] * param_1r[8] + 6 * param_1r[1] * param_1r[4] * param_1r[7] + 6 * param_1r[1] * param_1r[5] * param_1r[6] + 6 * param_1r[2] * param_1r[3] * param_1r[7] + 6 * param_1r[2] * param_1r[4] * param_1r[6] + 6 * param_1r[3] * param_1r[4] * param_1r[5] + pow(param_1r[4], 3));
    param_box_2d1r[25] = (6 * param_1r[0] * param_1r[5] * param_1r[8] + 6 * param_1r[1] * param_1r[4] * param_1r[8] + 6 * param_1r[1] * param_1r[5] * param_1r[7] + 6 * param_1r[2] * param_1r[3] * param_1r[8] + 6 * param_1r[2] * param_1r[4] * param_1r[7] + 6 * param_1r[2] * param_1r[5] * param_1r[6] + 3 * param_1r[3] * param_1r[5] * param_1r[5] + 3 * param_1r[4] * param_1r[4] * param_1r[5]);
    param_box_2d1r[26] = (6 * param_1r[1] * param_1r[5] * param_1r[8] + 6 * param_1r[2] * param_1r[4] * param_1r[8] + 6 * param_1r[2] * param_1r[5] * param_1r[7] + 3 * param_1r[4] * param_1r[5] * param_1r[5]);
    param_box_2d1r[27] = (6 * param_1r[2] * param_1r[5] * param_1r[8] + param_1r[5] * param_1r[5] * param_1r[5]);
    param_box_2d1r[30] = (6 * param_1r[0] * param_1r[6] * param_1r[8] + 3 * param_1r[0] * param_1r[7] * param_1r[7] + 6 * param_1r[1] * param_1r[6] * param_1r[7] + 3 * param_1r[2] * param_1r[6] * param_1r[6] + 3 * param_1r[3] * param_1r[3] * param_1r[8] + 6 * param_1r[3] * param_1r[4] * param_1r[7] + 6 * param_1r[3] * param_1r[5] * param_1r[6] + 3 * param_1r[4] * param_1r[4] * param_1r[6]);
    param_box_2d1r[29] = (6 * param_1r[0] * param_1r[6] * param_1r[7] + 3 * param_1r[1] * param_1r[6] * param_1r[6] + 3 * param_1r[3] * param_1r[3] * param_1r[7] + 6 * param_1r[3] * param_1r[4] * param_1r[6]);
    param_box_2d1r[28] = (3 * param_1r[0] * param_1r[6] * param_1r[6] + 3 * param_1r[3] * param_1r[3] * param_1r[6]);
    param_box_2d1r[31] = (6 * param_1r[0] * param_1r[7] * param_1r[8] + 6 * param_1r[1] * param_1r[6] * param_1r[8] + 3 * param_1r[1] * param_1r[7] * param_1r[7] + 6 * param_1r[2] * param_1r[6] * param_1r[7] + 6 * param_1r[3] * param_1r[4] * param_1r[8] + 6 * param_1r[3] * param_1r[5] * param_1r[7] + 3 * param_1r[4] * param_1r[4] * param_1r[7] + 6 * param_1r[4] * param_1r[5] * param_1r[6]);
    param_box_2d1r[32] = (3 * param_1r[0] * param_1r[8] * param_1r[8] + 6 * param_1r[1] * param_1r[7] * param_1r[8] + 6 * param_1r[2] * param_1r[6] * param_1r[8] + 3 * param_1r[2] * param_1r[7] * param_1r[7] + 6 * param_1r[3] * param_1r[5] * param_1r[8] + 3 * param_1r[4] * param_1r[4] * param_1r[8] + 6 * param_1r[4] * param_1r[5] * param_1r[7] + 3 * param_1r[5] * param_1r[5] * param_1r[6]);
    param_box_2d1r[33] = (3 * param_1r[1] * param_1r[8] * param_1r[8] + 6 * param_1r[2] * param_1r[7] * param_1r[8] + 6 * param_1r[4] * param_1r[5] * param_1r[8] + 3 * param_1r[5] * param_1r[5] * param_1r[7]);
    param_box_2d1r[34] = (3 * param_1r[2] * param_1r[8] * param_1r[8] + 3 * param_1r[5] * param_1r[5] * param_1r[8]);
    param_box_2d1r[37] = (6 * param_1r[3] * param_1r[6] * param_1r[8] + 3 * param_1r[3] * param_1r[7] * param_1r[7] + 6 * param_1r[4] * param_1r[6] * param_1r[7] + 3 * param_1r[5] * param_1r[6] * param_1r[6]);
    param_box_2d1r[36] = (6 * param_1r[3] * param_1r[6] * param_1r[7] + 3 * param_1r[4] * param_1r[6] * param_1r[6]);
    param_box_2d1r[35] = 3 * param_1r[3] * param_1r[6] * param_1r[6];
    param_box_2d1r[38] = (6 * param_1r[3] * param_1r[7] * param_1r[8] + 6 * param_1r[4] * param_1r[6] * param_1r[8] + 3 * param_1r[4] * param_1r[7] * param_1r[7] + 6 * param_1r[5] * param_1r[6] * param_1r[7]);
    param_box_2d1r[39] = (3 * param_1r[3] * param_1r[8] * param_1r[8] + 6 * param_1r[4] * param_1r[7] * param_1r[8] + 6 * param_1r[5] * param_1r[6] * param_1r[8] + 3 * param_1r[5] * param_1r[7] * param_1r[7]);
    param_box_2d1r[40] = (3 * param_1r[4] * param_1r[8] * param_1r[8] + 6 * param_1r[5] * param_1r[7] * param_1r[8]);
    param_box_2d1r[41] = 3 * param_1r[5] * param_1r[8] * param_1r[8];
    param_box_2d1r[44] = (3 * param_1r[6] * param_1r[6] * param_1r[8] + 3 * param_1r[6] * param_1r[7] * param_1r[7]);
    param_box_2d1r[43] = 3 * param_1r[6] * param_1r[6] * param_1r[7];
    param_box_2d1r[42] = param_1r[6] * param_1r[6] * param_1r[6];
    param_box_2d1r[45] = (6 * param_1r[6] * param_1r[7] * param_1r[8] + param_1r[7] * param_1r[7] * param_1r[7]);
    param_box_2d1r[46] = (3 * param_1r[6] * param_1r[8] * param_1r[8] + 3 * param_1r[7] * param_1r[7] * param_1r[8]);
    param_box_2d1r[47] = 3 * param_1r[7] * param_1r[8] * param_1r[8];
    param_box_2d1r[48] = param_1r[8] * param_1r[8] * param_1r[8];

    double *param;
    int halo;
    switch (compute_shape)
    {
    case box_2d1r:
        param = param_box_2d1r;
        halo = 3;
        break;
    case star_2d1r:
        param = param_star_2d1r;
        halo = 3;
        break;
    case star_2d3r:
        param = param_star_2d1r;
        halo = 3;
        break;
    case box_2d3r:
        param = param_box_2d1r;
        halo = 3;
        break;
    }

    // print brief info
    printf("INFO: shape = %s, m = %d, n = %d, times = %d\n", ShapeStr[compute_shape], m, n, times);

    int rows = m + 2 * halo;
    int cols = n + 2 * halo + 2;
    NY = n;
    size_t matrix_size = (unsigned long)rows * cols * sizeof(double);

    // allocate space

    double *matrix = (double *)malloc(matrix_size);
    double *output = (double *)malloc(matrix_size);

    // fill input matrix

#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = (double)(rand() % 100);
    }
#elif defined(FILL_INDEX)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 1; j < cols - 1; j++)
        {
            matrix[i * cols + j] = (double)(i * (cols - 2) + j);
        }
    }
#else
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            matrix[i * cols + j] = 1.0;
        }
    }
    // std::fill_n(matrix, rows * cols, 1.0);
#endif

    switch (compute_shape)
    {
    case box_2d1r:
    case star_2d1r:
        if (breakdown)
        {
            gpu_box_2d1r_breakdown1(matrix, output, param, times, m, n);
            gpu_box_2d1r_breakdown2(matrix, output, param, times, m, n);
            gpu_box_2d1r_breakdown3(matrix, output, param, times, m, n);
            gpu_box_2d1r_breakdown4(matrix, output, param, times, m, n);
            gpu_box_2d1r(matrix, output, param, times, m, n);

        }
        else
        {
            gpu_box_2d1r(matrix, output,
                     param, times,
                     m, n);
        }
        break;
    case star_2d3r:
    case box_2d3r:
        gpu_box_2d3r(matrix, output,
                      param, times,
                      m, n);
        break;
    }

    // check result correctness

#if defined(CHECK_ERROR)
    printf("\nChecking ... \n");
    double *naive[2];
    naive[0] = (double *)malloc(matrix_size);
    naive[1] = (double *)malloc(matrix_size);

    for (int i = 0; i < rows * cols; i++)
    {
        naive[0][i] = matrix[i];
        naive[1][i] = 0;
    }

    int t = 0;
    if (compute_shape == box_2d1r_step3)
    {
        for (; t < times; t++)
        {
            naive_box2d1r(naive[t % 2], naive[(t + 1) % 2], param, rows, cols);
        }
    }
    printf("Comparing naive and output\n");
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            if (ABS(naive[t % 2][IDX(row, col, cols)], output[IDX(row, col, cols)]) > 1e-7)
            {
                printf("row = %d, col = %d, naive = %lf, output = %lf\n", row, col, naive[t % 2][IDX(row, col, cols)], output[IDX(row, col, cols)]);
            }
        }
    }
#endif

    // write to file

#ifdef WRITE_OUTPUT
#ifdef RUN_GPU
    printf("Writing output_gpu.txt\n");
    save_to_txt(output, rows, cols, "output_gpu.txt");
    save_to_txt(naive[t % 2], rows, cols, "output_naive.txt");
#endif
#endif

    // free space
    free(output);
    free(matrix);

    return 0;
}
