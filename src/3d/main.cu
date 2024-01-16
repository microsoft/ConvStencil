#include <iostream>
#include <omp.h>
#include "3d_utils.h"

const char *ShapeStr[5] = {
    "box_3d1r",
    "star_3d1r",
    };

#define FILL_RANDOM
// #define FILL_INDEX


// #define CHECK_ERROR
const double tolerance = 1e-7;

#define IDX2D(x, y, ldm) ((x) * (ldm) + (y))
#define IDX3D(x, y, z, rows, cols) ((x) * (rows) * (cols) + (y) * (cols) + (z))
#define ABS(x,y)  (((x) > (y))? ((x)-(y)) : ((y)-(x)))
// #define WRITE_OUTPUT

void save_to_txt(double* arr, int x, int y, int z, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
                fprintf(file, "%.0f\t", arr[IDX3D(i, j, k, y , z)]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\nNext\n");
    }

    fclose(file);
}

void naive_star3d1r_step3(double *in, double *out, double*param, const int heights, const int rows, const int cols) {
    for (int height = 3; height < heights - 3; height++) {
        for (int row = 3; row < rows - 3; row++) {
            for (int col = 3; col < cols - 3; col++) {
                out[IDX3D(height, row, col, rows, cols)] = 
                    param[0] * in[IDX3D(height - 3, row, col, rows, cols)] + 
                    param[1] * in[IDX3D(height - 2, row - 1, col, rows, cols)] +
                    param[2] * in[IDX3D(height - 2, row, col - 1, rows, cols)] +
                    param[3] * in[IDX3D(height - 2, row, col, rows, cols)] +
                    param[4] * in[IDX3D(height - 2, row, col + 1, rows, cols)] +
                    param[5] * in[IDX3D(height - 2, row + 1, col, rows, cols)] +
                    param[6] * in[IDX3D(height - 1, row - 2, col, rows, cols)] +
                    param[7] * in[IDX3D(height - 1, row - 1, col - 1, rows, cols)] +
                    param[8] * in[IDX3D(height - 1, row - 1, col, rows, cols)] +
                    param[9] * in[IDX3D(height - 1, row - 1, col + 1, rows, cols)] +
                    param[10] * in[IDX3D(height - 1, row, col - 2, rows, cols)] +
                    param[11] * in[IDX3D(height - 1, row, col - 1, rows, cols)] +
                    param[12] * in[IDX3D(height - 1, row, col, rows, cols)] +
                    param[13] * in[IDX3D(height - 1, row, col + 1, rows, cols)] +
                    param[14] * in[IDX3D(height - 1, row, col + 2, rows, cols)] +
                    param[15] * in[IDX3D(height - 1, row + 1, col - 1, rows, cols)] +
                    param[16] * in[IDX3D(height - 1, row + 1, col, rows, cols)] +
                    param[17] * in[IDX3D(height - 1, row + 1, col + 1, rows, cols)] +
                    param[18] * in[IDX3D(height - 1, row + 2, col, rows, cols)] +
                    param[19] * in[IDX3D(height, row - 3, col, rows, cols)] +
                    param[20] * in[IDX3D(height, row - 2, col - 1, rows, cols)] +
                    param[21] * in[IDX3D(height, row - 2, col, rows, cols)] +
                    param[22] * in[IDX3D(height, row - 2, col + 1, rows, cols)] +
                    param[23] * in[IDX3D(height, row - 1, col - 2, rows, cols)] +
                    param[24] * in[IDX3D(height, row - 1, col - 1, rows, cols)] +
                    param[25] * in[IDX3D(height, row - 1, col, rows, cols)] +
                    param[26] * in[IDX3D(height, row - 1, col + 1, rows, cols)] +
                    param[27] * in[IDX3D(height, row - 1, col + 2, rows, cols)] +
                    param[28] * in[IDX3D(height, row, col - 3, rows, cols)] +
                    param[29] * in[IDX3D(height, row, col - 2, rows, cols)] +
                    param[30] * in[IDX3D(height, row, col - 1, rows, cols)] +
                    param[31] * in[IDX3D(height, row, col, rows, cols)] +
                    param[32] * in[IDX3D(height, row, col + 1, rows, cols)] +
                    param[33] * in[IDX3D(height, row, col + 2, rows, cols)] +
                    param[34] * in[IDX3D(height, row, col + 3, rows, cols)] +
                    param[35] * in[IDX3D(height, row + 1, col - 2, rows, cols)] +
                    param[36] * in[IDX3D(height, row + 1, col - 1, rows, cols)] +
                    param[37] * in[IDX3D(height, row + 1, col, rows, cols)] +
                    param[38] * in[IDX3D(height, row + 1, col + 1, rows, cols)] +
                    param[39] * in[IDX3D(height, row + 1, col + 2, rows, cols)] +
                    param[40] * in[IDX3D(height, row + 2, col - 1, rows, cols)] +
                    param[41] * in[IDX3D(height, row + 2, col, rows, cols)] +
                    param[42] * in[IDX3D(height, row + 2, col + 1, rows, cols)] +
                    param[43] * in[IDX3D(height, row + 3, col, rows, cols)] +
                    param[6] * in[IDX3D(height + 1, row - 2, col, rows, cols)] +
                    param[7] * in[IDX3D(height + 1, row - 1, col - 1, rows, cols)] +
                    param[8] * in[IDX3D(height + 1, row - 1, col, rows, cols)] +
                    param[9] * in[IDX3D(height + 1, row - 1, col + 1, rows, cols)] +
                    param[10] * in[IDX3D(height + 1, row, col - 2, rows, cols)] +
                    param[11] * in[IDX3D(height + 1, row, col - 1, rows, cols)] +
                    param[12] * in[IDX3D(height + 1, row, col, rows, cols)] +
                    param[13] * in[IDX3D(height + 1, row, col + 1, rows, cols)] +
                    param[14] * in[IDX3D(height + 1, row, col + 2, rows, cols)] +
                    param[15] * in[IDX3D(height + 1, row + 1, col - 1, rows, cols)] +
                    param[16] * in[IDX3D(height + 1, row + 1, col, rows, cols)] +
                    param[17] * in[IDX3D(height + 1, row + 1, col + 1, rows, cols)] +
                    param[18] * in[IDX3D(height + 1, row + 2, col, rows, cols)] +
                    param[1] * in[IDX3D(height + 2, row - 1, col, rows, cols)] +
                    param[2] * in[IDX3D(height + 2, row, col - 1, rows, cols)] +
                    param[3] * in[IDX3D(height + 2, row, col, rows, cols)] +
                    param[4] * in[IDX3D(height + 2, row, col + 1, rows, cols)] +
                    param[5] * in[IDX3D(height + 2, row + 1, col, rows, cols)] +
                    param[0] * in[IDX3D(height + 3, row, col, rows, cols)];
            }
        }
    }
}

void naive_box3d1r_step3(double *in, double *out, double *param, const int heights, const int rows, const int cols) {
    for(int height = 3; height < heights - 3; height++) {
        for (int row = 3; row < rows - 3; row++) {
            for (int col = 3; col < cols - 3; col++) {
                // printf("%lf\n", in[IDX3D(height, row, col, rows, cols)]);
                out[IDX3D(height, row, col, rows, cols)] = 
                    param[0] * in[IDX3D(height - 3, row - 3, col - 3, rows, cols)] +
                    param[1] * in[IDX3D(height - 3, row - 3, col - 2, rows, cols)] +
                    param[2] * in[IDX3D(height - 3, row - 3, col - 1, rows, cols)] +
                    param[3] * in[IDX3D(height - 3, row - 3, col, rows, cols)] +
                    param[4] * in[IDX3D(height - 3, row - 3, col + 1, rows, cols)] +
                    param[5] * in[IDX3D(height - 3, row - 3, col + 2, rows, cols)] +
                    param[6] * in[IDX3D(height - 3, row - 3, col + 3, rows, cols)] +
                    param[7] * in[IDX3D(height - 3, row - 2, col - 3, rows, cols)] +
                    param[8] * in[IDX3D(height - 3, row - 2, col - 2, rows, cols)] +
                    param[9] * in[IDX3D(height - 3, row - 2, col - 1, rows, cols)] +
                    param[10] * in[IDX3D(height - 3, row - 2, col, rows, cols)] +
                    param[11] * in[IDX3D(height - 3, row - 2, col + 1, rows, cols)] +
                    param[12] * in[IDX3D(height - 3, row - 2, col + 2, rows, cols)] +
                    param[13] * in[IDX3D(height - 3, row - 2, col + 3, rows, cols)] +
                    param[14] * in[IDX3D(height - 3, row - 1, col - 3, rows, cols)] +
                    param[15] * in[IDX3D(height - 3, row - 1, col - 2, rows, cols)] +
                    param[16] * in[IDX3D(height - 3, row - 1, col - 1, rows, cols)] +
                    param[17] * in[IDX3D(height - 3, row - 1, col, rows, cols)] +
                    param[18] * in[IDX3D(height - 3, row - 1, col + 1, rows, cols)] +
                    param[19] * in[IDX3D(height - 3, row - 1, col + 2, rows, cols)] +
                    param[20] * in[IDX3D(height - 3, row - 1, col + 3, rows, cols)] +
                    param[21] * in[IDX3D(height - 3, row, col - 3, rows, cols)] +
                    param[22] * in[IDX3D(height - 3, row, col - 2, rows, cols)] +
                    param[23] * in[IDX3D(height - 3, row, col - 1, rows, cols)] +
                    param[24] * in[IDX3D(height - 3, row, col, rows, cols)] +
                    param[25] * in[IDX3D(height - 3, row, col + 1, rows, cols)] +
                    param[26] * in[IDX3D(height - 3, row, col + 2, rows, cols)] +
                    param[27] * in[IDX3D(height - 3, row, col + 3, rows, cols)] +
                    param[28] * in[IDX3D(height - 3, row + 1, col - 3, rows, cols)] +
                    param[29] * in[IDX3D(height - 3, row + 1, col - 2, rows, cols)] +
                    param[30] * in[IDX3D(height - 3, row + 1, col - 1, rows, cols)] +
                    param[31] * in[IDX3D(height - 3, row + 1, col, rows, cols)] +
                    param[32] * in[IDX3D(height - 3, row + 1, col + 1, rows, cols)] +
                    param[33] * in[IDX3D(height - 3, row + 1, col + 2, rows, cols)] +
                    param[34] * in[IDX3D(height - 3, row + 1, col + 3, rows, cols)] +
                    param[35] * in[IDX3D(height - 3, row + 2, col - 3, rows, cols)] +
                    param[36] * in[IDX3D(height - 3, row + 2, col - 2, rows, cols)] +
                    param[37] * in[IDX3D(height - 3, row + 2, col - 1, rows, cols)] +
                    param[38] * in[IDX3D(height - 3, row + 2, col, rows, cols)] +
                    param[39] * in[IDX3D(height - 3, row + 2, col + 1, rows, cols)] +
                    param[40] * in[IDX3D(height - 3, row + 2, col + 2, rows, cols)] +
                    param[41] * in[IDX3D(height - 3, row + 2, col + 3, rows, cols)] +
                    param[42] * in[IDX3D(height - 3, row + 3, col - 3, rows, cols)] +
                    param[43] * in[IDX3D(height - 3, row + 3, col - 2, rows, cols)] +
                    param[44] * in[IDX3D(height - 3, row + 3, col - 1, rows, cols)] +
                    param[45] * in[IDX3D(height - 3, row + 3, col, rows, cols)] +
                    param[46] * in[IDX3D(height - 3, row + 3, col + 1, rows, cols)] +
                    param[47] * in[IDX3D(height - 3, row + 3, col + 2, rows, cols)] +
                    param[48] * in[IDX3D(height - 3, row + 3, col + 3, rows, cols)] +
                    param[49] * in[IDX3D(height - 2, row - 3, col - 3, rows, cols)] +
                    param[50] * in[IDX3D(height - 2, row - 3, col - 2, rows, cols)] +
                    param[51] * in[IDX3D(height - 2, row - 3, col - 1, rows, cols)] +
                    param[52] * in[IDX3D(height - 2, row - 3, col, rows, cols)] +
                    param[53] * in[IDX3D(height - 2, row - 3, col + 1, rows, cols)] +
                    param[54] * in[IDX3D(height - 2, row - 3, col + 2, rows, cols)] +
                    param[55] * in[IDX3D(height - 2, row - 3, col + 3, rows, cols)] +
                    param[56] * in[IDX3D(height - 2, row - 2, col - 3, rows, cols)] +
                    param[57] * in[IDX3D(height - 2, row - 2, col - 2, rows, cols)] +
                    param[58] * in[IDX3D(height - 2, row - 2, col - 1, rows, cols)] +
                    param[59] * in[IDX3D(height - 2, row - 2, col, rows, cols)] +
                    param[60] * in[IDX3D(height - 2, row - 2, col + 1, rows, cols)] +
                    param[61] * in[IDX3D(height - 2, row - 2, col + 2, rows, cols)] +
                    param[62] * in[IDX3D(height - 2, row - 2, col + 3, rows, cols)] +
                    param[63] * in[IDX3D(height - 2, row - 1, col - 3, rows, cols)] +
                    param[64] * in[IDX3D(height - 2, row - 1, col - 2, rows, cols)] +
                    param[65] * in[IDX3D(height - 2, row - 1, col - 1, rows, cols)] +
                    param[66] * in[IDX3D(height - 2, row - 1, col, rows, cols)] +
                    param[67] * in[IDX3D(height - 2, row - 1, col + 1, rows, cols)] +
                    param[68] * in[IDX3D(height - 2, row - 1, col + 2, rows, cols)] +
                    param[69] * in[IDX3D(height - 2, row - 1, col + 3, rows, cols)] +
                    param[70] * in[IDX3D(height - 2, row, col - 3, rows, cols)] +
                    param[71] * in[IDX3D(height - 2, row, col - 2, rows, cols)] +
                    param[72] * in[IDX3D(height - 2, row, col - 1, rows, cols)] +
                    param[73] * in[IDX3D(height - 2, row, col, rows, cols)] +
                    param[74] * in[IDX3D(height - 2, row, col + 1, rows, cols)] +
                    param[75] * in[IDX3D(height - 2, row, col + 2, rows, cols)] +
                    param[76] * in[IDX3D(height - 2, row, col + 3, rows, cols)] +
                    param[77] * in[IDX3D(height - 2, row + 1, col - 3, rows, cols)] +
                    param[78] * in[IDX3D(height - 2, row + 1, col - 2, rows, cols)] +
                    param[79] * in[IDX3D(height - 2, row + 1, col - 1, rows, cols)] +
                    param[80] * in[IDX3D(height - 2, row + 1, col, rows, cols)] +
                    param[81] * in[IDX3D(height - 2, row + 1, col + 1, rows, cols)] +
                    param[82] * in[IDX3D(height - 2, row + 1, col + 2, rows, cols)] +
                    param[83] * in[IDX3D(height - 2, row + 1, col + 3, rows, cols)] +
                    param[84] * in[IDX3D(height - 2, row + 2, col - 3, rows, cols)] +
                    param[85] * in[IDX3D(height - 2, row + 2, col - 2, rows, cols)] +
                    param[86] * in[IDX3D(height - 2, row + 2, col - 1, rows, cols)] +
                    param[87] * in[IDX3D(height - 2, row + 2, col, rows, cols)] +
                    param[88] * in[IDX3D(height - 2, row + 2, col + 1, rows, cols)] +
                    param[89] * in[IDX3D(height - 2, row + 2, col + 2, rows, cols)] +
                    param[90] * in[IDX3D(height - 2, row + 2, col + 3, rows, cols)] +
                    param[91] * in[IDX3D(height - 2, row + 3, col - 3, rows, cols)] +
                    param[92] * in[IDX3D(height - 2, row + 3, col - 2, rows, cols)] +
                    param[93] * in[IDX3D(height - 2, row + 3, col - 1, rows, cols)] +
                    param[94] * in[IDX3D(height - 2, row + 3, col, rows, cols)] +
                    param[95] * in[IDX3D(height - 2, row + 3, col + 1, rows, cols)] +
                    param[96] * in[IDX3D(height - 2, row + 3, col + 2, rows, cols)] +
                    param[97] * in[IDX3D(height - 2, row + 3, col + 3, rows, cols)] +
                    param[98] * in[IDX3D(height - 1, row - 3, col - 3, rows, cols)] +
                    param[99] * in[IDX3D(height - 1, row - 3, col - 2, rows, cols)] +
                    param[100] * in[IDX3D(height - 1, row - 3, col - 1, rows, cols)] +
                    param[101] * in[IDX3D(height - 1, row - 3, col, rows, cols)] +
                    param[102] * in[IDX3D(height - 1, row - 3, col + 1, rows, cols)] +
                    param[103] * in[IDX3D(height - 1, row - 3, col + 2, rows, cols)] +
                    param[104] * in[IDX3D(height - 1, row - 3, col + 3, rows, cols)] +
                    param[105] * in[IDX3D(height - 1, row - 2, col - 3, rows, cols)] +
                    param[106] * in[IDX3D(height - 1, row - 2, col - 2, rows, cols)] +
                    param[107] * in[IDX3D(height - 1, row - 2, col - 1, rows, cols)] +
                    param[108] * in[IDX3D(height - 1, row - 2, col, rows, cols)] +
                    param[109] * in[IDX3D(height - 1, row - 2, col + 1, rows, cols)] +
                    param[110] * in[IDX3D(height - 1, row - 2, col + 2, rows, cols)] +
                    param[111] * in[IDX3D(height - 1, row - 2, col + 3, rows, cols)] +
                    param[112] * in[IDX3D(height - 1, row - 1, col - 3, rows, cols)] +
                    param[113] * in[IDX3D(height - 1, row - 1, col - 2, rows, cols)] +
                    param[114] * in[IDX3D(height - 1, row - 1, col - 1, rows, cols)] +
                    param[115] * in[IDX3D(height - 1, row - 1, col, rows, cols)] +
                    param[116] * in[IDX3D(height - 1, row - 1, col + 1, rows, cols)] +
                    param[117] * in[IDX3D(height - 1, row - 1, col + 2, rows, cols)] +
                    param[118] * in[IDX3D(height - 1, row - 1, col + 3, rows, cols)] +
                    param[119] * in[IDX3D(height - 1, row, col - 3, rows, cols)] +
                    param[120] * in[IDX3D(height - 1, row, col - 2, rows, cols)] +
                    param[121] * in[IDX3D(height - 1, row, col - 1, rows, cols)] +
                    param[122] * in[IDX3D(height - 1, row, col, rows, cols)] +
                    param[123] * in[IDX3D(height - 1, row, col + 1, rows, cols)] +
                    param[124] * in[IDX3D(height - 1, row, col + 2, rows, cols)] +
                    param[125] * in[IDX3D(height - 1, row, col + 3, rows, cols)] +
                    param[126] * in[IDX3D(height - 1, row + 1, col - 3, rows, cols)] +
                    param[127] * in[IDX3D(height - 1, row + 1, col - 2, rows, cols)] +
                    param[128] * in[IDX3D(height - 1, row + 1, col - 1, rows, cols)] +
                    param[129] * in[IDX3D(height - 1, row + 1, col, rows, cols)] +
                    param[130] * in[IDX3D(height - 1, row + 1, col + 1, rows, cols)] +
                    param[131] * in[IDX3D(height - 1, row + 1, col + 2, rows, cols)] +
                    param[132] * in[IDX3D(height - 1, row + 1, col + 3, rows, cols)] +
                    param[133] * in[IDX3D(height - 1, row + 2, col - 3, rows, cols)] +
                    param[134] * in[IDX3D(height - 1, row + 2, col - 2, rows, cols)] +
                    param[135] * in[IDX3D(height - 1, row + 2, col - 1, rows, cols)] +
                    param[136] * in[IDX3D(height - 1, row + 2, col, rows, cols)] +
                    param[137] * in[IDX3D(height - 1, row + 2, col + 1, rows, cols)] +
                    param[138] * in[IDX3D(height - 1, row + 2, col + 2, rows, cols)] +
                    param[139] * in[IDX3D(height - 1, row + 2, col + 3, rows, cols)] +
                    param[140] * in[IDX3D(height - 1, row + 3, col - 3, rows, cols)] +
                    param[141] * in[IDX3D(height - 1, row + 3, col - 2, rows, cols)] +
                    param[142] * in[IDX3D(height - 1, row + 3, col - 1, rows, cols)] +
                    param[143] * in[IDX3D(height - 1, row + 3, col, rows, cols)] +
                    param[144] * in[IDX3D(height - 1, row + 3, col + 1, rows, cols)] +
                    param[145] * in[IDX3D(height - 1, row + 3, col + 2, rows, cols)] +
                    param[146] * in[IDX3D(height - 1, row + 3, col + 3, rows, cols)] +
                    param[147] * in[IDX3D(height, row - 3, col - 3, rows, cols)] +
                    param[148] * in[IDX3D(height, row - 3, col - 2, rows, cols)] +
                    param[149] * in[IDX3D(height, row - 3, col - 1, rows, cols)] +
                    param[150] * in[IDX3D(height, row - 3, col, rows, cols)] +
                    param[151] * in[IDX3D(height, row - 3, col + 1, rows, cols)] +
                    param[152] * in[IDX3D(height, row - 3, col + 2, rows, cols)] +
                    param[153] * in[IDX3D(height, row - 3, col + 3, rows, cols)] +
                    param[154] * in[IDX3D(height, row - 2, col - 3, rows, cols)] +
                    param[155] * in[IDX3D(height, row - 2, col - 2, rows, cols)] +
                    param[156] * in[IDX3D(height, row - 2, col - 1, rows, cols)] +
                    param[157] * in[IDX3D(height, row - 2, col, rows, cols)] +
                    param[158] * in[IDX3D(height, row - 2, col + 1, rows, cols)] +
                    param[159] * in[IDX3D(height, row - 2, col + 2, rows, cols)] +
                    param[160] * in[IDX3D(height, row - 2, col + 3, rows, cols)] +
                    param[161] * in[IDX3D(height, row - 1, col - 3, rows, cols)] +
                    param[162] * in[IDX3D(height, row - 1, col - 2, rows, cols)] +
                    param[163] * in[IDX3D(height, row - 1, col - 1, rows, cols)] +
                    param[164] * in[IDX3D(height, row - 1, col, rows, cols)] +
                    param[165] * in[IDX3D(height, row - 1, col + 1, rows, cols)] +
                    param[166] * in[IDX3D(height, row - 1, col + 2, rows, cols)] +
                    param[167] * in[IDX3D(height, row - 1, col + 3, rows, cols)] +
                    param[168] * in[IDX3D(height, row, col - 3, rows, cols)] +
                    param[169] * in[IDX3D(height, row, col - 2, rows, cols)] +
                    param[170] * in[IDX3D(height, row, col - 1, rows, cols)] +
                    param[171] * in[IDX3D(height, row, col, rows, cols)] +
                    param[172] * in[IDX3D(height, row, col + 1, rows, cols)] +
                    param[173] * in[IDX3D(height, row, col + 2, rows, cols)] +
                    param[174] * in[IDX3D(height, row, col + 3, rows, cols)] +
                    param[175] * in[IDX3D(height, row + 1, col - 3, rows, cols)] +
                    param[176] * in[IDX3D(height, row + 1, col - 2, rows, cols)] +
                    param[177] * in[IDX3D(height, row + 1, col - 1, rows, cols)] +
                    param[178] * in[IDX3D(height, row + 1, col, rows, cols)] +
                    param[179] * in[IDX3D(height, row + 1, col + 1, rows, cols)] +
                    param[180] * in[IDX3D(height, row + 1, col + 2, rows, cols)] +
                    param[181] * in[IDX3D(height, row + 1, col + 3, rows, cols)] +
                    param[182] * in[IDX3D(height, row + 2, col - 3, rows, cols)] +
                    param[183] * in[IDX3D(height, row + 2, col - 2, rows, cols)] +
                    param[184] * in[IDX3D(height, row + 2, col - 1, rows, cols)] +
                    param[185] * in[IDX3D(height, row + 2, col, rows, cols)] +
                    param[186] * in[IDX3D(height, row + 2, col + 1, rows, cols)] +
                    param[187] * in[IDX3D(height, row + 2, col + 2, rows, cols)] +
                    param[188] * in[IDX3D(height, row + 2, col + 3, rows, cols)] +
                    param[189] * in[IDX3D(height, row + 3, col - 3, rows, cols)] +
                    param[190] * in[IDX3D(height, row + 3, col - 2, rows, cols)] +
                    param[191] * in[IDX3D(height, row + 3, col - 1, rows, cols)] +
                    param[192] * in[IDX3D(height, row + 3, col, rows, cols)] +
                    param[193] * in[IDX3D(height, row + 3, col + 1, rows, cols)] +
                    param[194] * in[IDX3D(height, row + 3, col + 2, rows, cols)] +
                    param[195] * in[IDX3D(height, row + 3, col + 3, rows, cols)] +
                    param[98] * in[IDX3D(height + 1, row - 3, col - 3, rows, cols)] +
                    param[99] * in[IDX3D(height + 1, row - 3, col - 2, rows, cols)] +
                    param[100] * in[IDX3D(height + 1, row - 3, col - 1, rows, cols)] +
                    param[101] * in[IDX3D(height + 1, row - 3, col, rows, cols)] +
                    param[102] * in[IDX3D(height + 1, row - 3, col + 1, rows, cols)] +
                    param[103] * in[IDX3D(height + 1, row - 3, col + 2, rows, cols)] +
                    param[104] * in[IDX3D(height + 1, row - 3, col + 3, rows, cols)] +
                    param[105] * in[IDX3D(height + 1, row - 2, col - 3, rows, cols)] +
                    param[106] * in[IDX3D(height + 1, row - 2, col - 2, rows, cols)] +
                    param[107] * in[IDX3D(height + 1, row - 2, col - 1, rows, cols)] +
                    param[108] * in[IDX3D(height + 1, row - 2, col, rows, cols)] +
                    param[109] * in[IDX3D(height + 1, row - 2, col + 1, rows, cols)] +
                    param[110] * in[IDX3D(height + 1, row - 2, col + 2, rows, cols)] +
                    param[111] * in[IDX3D(height + 1, row - 2, col + 3, rows, cols)] +
                    param[112] * in[IDX3D(height + 1, row - 1, col - 3, rows, cols)] +
                    param[113] * in[IDX3D(height + 1, row - 1, col - 2, rows, cols)] +
                    param[114] * in[IDX3D(height + 1, row - 1, col - 1, rows, cols)] +
                    param[115] * in[IDX3D(height + 1, row - 1, col, rows, cols)] +
                    param[116] * in[IDX3D(height + 1, row - 1, col + 1, rows, cols)] +
                    param[117] * in[IDX3D(height + 1, row - 1, col + 2, rows, cols)] +
                    param[118] * in[IDX3D(height + 1, row - 1, col + 3, rows, cols)] +
                    param[119] * in[IDX3D(height + 1, row, col - 3, rows, cols)] +
                    param[120] * in[IDX3D(height + 1, row, col - 2, rows, cols)] +
                    param[121] * in[IDX3D(height + 1, row, col - 1, rows, cols)] +
                    param[122] * in[IDX3D(height + 1, row, col, rows, cols)] +
                    param[123] * in[IDX3D(height + 1, row, col + 1, rows, cols)] +
                    param[124] * in[IDX3D(height + 1, row, col + 2, rows, cols)] +
                    param[125] * in[IDX3D(height + 1, row, col + 3, rows, cols)] +
                    param[126] * in[IDX3D(height + 1, row + 1, col - 3, rows, cols)] +
                    param[127] * in[IDX3D(height + 1, row + 1, col - 2, rows, cols)] +
                    param[128] * in[IDX3D(height + 1, row + 1, col - 1, rows, cols)] +
                    param[129] * in[IDX3D(height + 1, row + 1, col, rows, cols)] +
                    param[130] * in[IDX3D(height + 1, row + 1, col + 1, rows, cols)] +
                    param[131] * in[IDX3D(height + 1, row + 1, col + 2, rows, cols)] +
                    param[132] * in[IDX3D(height + 1, row + 1, col + 3, rows, cols)] +
                    param[133] * in[IDX3D(height + 1, row + 2, col - 3, rows, cols)] +
                    param[134] * in[IDX3D(height + 1, row + 2, col - 2, rows, cols)] +
                    param[135] * in[IDX3D(height + 1, row + 2, col - 1, rows, cols)] +
                    param[136] * in[IDX3D(height + 1, row + 2, col, rows, cols)] +
                    param[137] * in[IDX3D(height + 1, row + 2, col + 1, rows, cols)] +
                    param[138] * in[IDX3D(height + 1, row + 2, col + 2, rows, cols)] +
                    param[139] * in[IDX3D(height + 1, row + 2, col + 3, rows, cols)] +
                    param[140] * in[IDX3D(height + 1, row + 3, col - 3, rows, cols)] +
                    param[141] * in[IDX3D(height + 1, row + 3, col - 2, rows, cols)] +
                    param[142] * in[IDX3D(height + 1, row + 3, col - 1, rows, cols)] +
                    param[143] * in[IDX3D(height + 1, row + 3, col, rows, cols)] +
                    param[144] * in[IDX3D(height + 1, row + 3, col + 1, rows, cols)] +
                    param[145] * in[IDX3D(height + 1, row + 3, col + 2, rows, cols)] +
                    param[146] * in[IDX3D(height + 1, row + 3, col + 3, rows, cols)] +
                    param[49] * in[IDX3D(height + 2, row - 3, col - 3, rows, cols)] +
                    param[50] * in[IDX3D(height + 2, row - 3, col - 2, rows, cols)] +
                    param[51] * in[IDX3D(height + 2, row - 3, col - 1, rows, cols)] +
                    param[52] * in[IDX3D(height + 2, row - 3, col, rows, cols)] +
                    param[53] * in[IDX3D(height + 2, row - 3, col + 1, rows, cols)] +
                    param[54] * in[IDX3D(height + 2, row - 3, col + 2, rows, cols)] +
                    param[55] * in[IDX3D(height + 2, row - 3, col + 3, rows, cols)] +
                    param[56] * in[IDX3D(height + 2, row - 2, col - 3, rows, cols)] +
                    param[57] * in[IDX3D(height + 2, row - 2, col - 2, rows, cols)] +
                    param[58] * in[IDX3D(height + 2, row - 2, col - 1, rows, cols)] +
                    param[59] * in[IDX3D(height + 2, row - 2, col, rows, cols)] +
                    param[60] * in[IDX3D(height + 2, row - 2, col + 1, rows, cols)] +
                    param[61] * in[IDX3D(height + 2, row - 2, col + 2, rows, cols)] +
                    param[62] * in[IDX3D(height + 2, row - 2, col + 3, rows, cols)] +
                    param[63] * in[IDX3D(height + 2, row - 1, col - 3, rows, cols)] +
                    param[64] * in[IDX3D(height + 2, row - 1, col - 2, rows, cols)] +
                    param[65] * in[IDX3D(height + 2, row - 1, col - 1, rows, cols)] +
                    param[66] * in[IDX3D(height + 2, row - 1, col, rows, cols)] +
                    param[67] * in[IDX3D(height + 2, row - 1, col + 1, rows, cols)] +
                    param[68] * in[IDX3D(height + 2, row - 1, col + 2, rows, cols)] +
                    param[69] * in[IDX3D(height + 2, row - 1, col + 3, rows, cols)] +
                    param[70] * in[IDX3D(height + 2, row, col - 3, rows, cols)] +
                    param[71] * in[IDX3D(height + 2, row, col - 2, rows, cols)] +
                    param[72] * in[IDX3D(height + 2, row, col - 1, rows, cols)] +
                    param[73] * in[IDX3D(height + 2, row, col, rows, cols)] +
                    param[74] * in[IDX3D(height + 2, row, col + 1, rows, cols)] +
                    param[75] * in[IDX3D(height + 2, row, col + 2, rows, cols)] +
                    param[76] * in[IDX3D(height + 2, row, col + 3, rows, cols)] +
                    param[77] * in[IDX3D(height + 2, row + 1, col - 3, rows, cols)] +
                    param[78] * in[IDX3D(height + 2, row + 1, col - 2, rows, cols)] +
                    param[79] * in[IDX3D(height + 2, row + 1, col - 1, rows, cols)] +
                    param[80] * in[IDX3D(height + 2, row + 1, col, rows, cols)] +
                    param[81] * in[IDX3D(height + 2, row + 1, col + 1, rows, cols)] +
                    param[82] * in[IDX3D(height + 2, row + 1, col + 2, rows, cols)] +
                    param[83] * in[IDX3D(height + 2, row + 1, col + 3, rows, cols)] +
                    param[84] * in[IDX3D(height + 2, row + 2, col - 3, rows, cols)] +
                    param[85] * in[IDX3D(height + 2, row + 2, col - 2, rows, cols)] +
                    param[86] * in[IDX3D(height + 2, row + 2, col - 1, rows, cols)] +
                    param[87] * in[IDX3D(height + 2, row + 2, col, rows, cols)] +
                    param[88] * in[IDX3D(height + 2, row + 2, col + 1, rows, cols)] +
                    param[89] * in[IDX3D(height + 2, row + 2, col + 2, rows, cols)] +
                    param[90] * in[IDX3D(height + 2, row + 2, col + 3, rows, cols)] +
                    param[91] * in[IDX3D(height + 2, row + 3, col - 3, rows, cols)] +
                    param[92] * in[IDX3D(height + 2, row + 3, col - 2, rows, cols)] +
                    param[93] * in[IDX3D(height + 2, row + 3, col - 1, rows, cols)] +
                    param[94] * in[IDX3D(height + 2, row + 3, col, rows, cols)] +
                    param[95] * in[IDX3D(height + 2, row + 3, col + 1, rows, cols)] +
                    param[96] * in[IDX3D(height + 2, row + 3, col + 2, rows, cols)] +
                    param[97] * in[IDX3D(height + 2, row + 3, col + 3, rows, cols)] +
                    param[0] * in[IDX3D(height + 3, row - 3, col - 3, rows, cols)] +
                    param[1] * in[IDX3D(height + 3, row - 3, col - 2, rows, cols)] +
                    param[2] * in[IDX3D(height + 3, row - 3, col - 1, rows, cols)] +
                    param[3] * in[IDX3D(height + 3, row - 3, col, rows, cols)] +
                    param[4] * in[IDX3D(height + 3, row - 3, col + 1, rows, cols)] +
                    param[5] * in[IDX3D(height + 3, row - 3, col + 2, rows, cols)] +
                    param[6] * in[IDX3D(height + 3, row - 3, col + 3, rows, cols)] +
                    param[7] * in[IDX3D(height + 3, row - 2, col - 3, rows, cols)] +
                    param[8] * in[IDX3D(height + 3, row - 2, col - 2, rows, cols)] +
                    param[9] * in[IDX3D(height + 3, row - 2, col - 1, rows, cols)] +
                    param[10] * in[IDX3D(height + 3, row - 2, col, rows, cols)] +
                    param[11] * in[IDX3D(height + 3, row - 2, col + 1, rows, cols)] +
                    param[12] * in[IDX3D(height + 3, row - 2, col + 2, rows, cols)] +
                    param[13] * in[IDX3D(height + 3, row - 2, col + 3, rows, cols)] +
                    param[14] * in[IDX3D(height + 3, row - 1, col - 3, rows, cols)] +
                    param[15] * in[IDX3D(height + 3, row - 1, col - 2, rows, cols)] +
                    param[16] * in[IDX3D(height + 3, row - 1, col - 1, rows, cols)] +
                    param[17] * in[IDX3D(height + 3, row - 1, col, rows, cols)] +
                    param[18] * in[IDX3D(height + 3, row - 1, col + 1, rows, cols)] +
                    param[19] * in[IDX3D(height + 3, row - 1, col + 2, rows, cols)] +
                    param[20] * in[IDX3D(height + 3, row - 1, col + 3, rows, cols)] +
                    param[21] * in[IDX3D(height + 3, row, col - 3, rows, cols)] +
                    param[22] * in[IDX3D(height + 3, row, col - 2, rows, cols)] +
                    param[23] * in[IDX3D(height + 3, row, col - 1, rows, cols)] +
                    param[24] * in[IDX3D(height + 3, row, col, rows, cols)] +
                    param[25] * in[IDX3D(height + 3, row, col + 1, rows, cols)] +
                    param[26] * in[IDX3D(height + 3, row, col + 2, rows, cols)] +
                    param[27] * in[IDX3D(height + 3, row, col + 3, rows, cols)] +
                    param[28] * in[IDX3D(height + 3, row + 1, col - 3, rows, cols)] +
                    param[29] * in[IDX3D(height + 3, row + 1, col - 2, rows, cols)] +
                    param[30] * in[IDX3D(height + 3, row + 1, col - 1, rows, cols)] +
                    param[31] * in[IDX3D(height + 3, row + 1, col, rows, cols)] +
                    param[32] * in[IDX3D(height + 3, row + 1, col + 1, rows, cols)] +
                    param[33] * in[IDX3D(height + 3, row + 1, col + 2, rows, cols)] +
                    param[34] * in[IDX3D(height + 3, row + 1, col + 3, rows, cols)] +
                    param[35] * in[IDX3D(height + 3, row + 2, col - 3, rows, cols)] +
                    param[36] * in[IDX3D(height + 3, row + 2, col - 2, rows, cols)] +
                    param[37] * in[IDX3D(height + 3, row + 2, col - 1, rows, cols)] +
                    param[38] * in[IDX3D(height + 3, row + 2, col, rows, cols)] +
                    param[39] * in[IDX3D(height + 3, row + 2, col + 1, rows, cols)] +
                    param[40] * in[IDX3D(height + 3, row + 2, col + 2, rows, cols)] +
                    param[41] * in[IDX3D(height + 3, row + 2, col + 3, rows, cols)] +
                    param[42] * in[IDX3D(height + 3, row + 3, col - 3, rows, cols)] +
                    param[43] * in[IDX3D(height + 3, row + 3, col - 2, rows, cols)] +
                    param[44] * in[IDX3D(height + 3, row + 3, col - 1, rows, cols)] +
                    param[45] * in[IDX3D(height + 3, row + 3, col, rows, cols)] +
                    param[46] * in[IDX3D(height + 3, row + 3, col + 1, rows, cols)] +
                    param[47] * in[IDX3D(height + 3, row + 3, col + 2, rows, cols)] +
                    param[48] * in[IDX3D(height + 3, row + 3, col + 3, rows, cols)];
            }
        }
    }
}

void printHelp()
{
    const char *helpMessage =
        "Program name: convstencil_2d\n"
        "Usage: convstencil_3d shape input_size_of_first_dimension input_size_of_second_dimension input_size_of_third_dimension time_iteration_size [Options]\n"
        "Shape: box3d1r or star3d1r\n"
        "Options:\n"
        "  --help    Display this help message and exit\n"
        "  --custom  If you want to use costum parameters, please use this option and input your parameters like 0.2 0.2 0.2 0.2 0.2 if the shape is star2d1r\n";
    printf("%s\n", helpMessage);
}

int main(int argc, char *argv[]) {

    if (argc < 6)
    {
        printHelp();
        return 1;
    }

    // configurable settings
    Shape compute_shape;
    std::string arg1 = argv[1];
    if (arg1 == "box3d1r") {
        compute_shape = box_3d1r;
    }
    else if(arg1 == "star3d1r"){
        compute_shape=star_3d1r;
    } else {
        printHelp();
        return 1;
    }

    int h = 0;
    int m = 0;
    int n = 0;
    int times = 0;

    try
    {
        h = std::stoi(argv[2]);
        m = std::stoi(argv[3]);
        n = std::stoi(argv[4]);
        times = std::stoi(argv[5]);
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
    

    double param_box_3d1r_step3[49 * 4] = {0.0};
    for (int i = 0; i < 49 * 4; i++) {
        param_box_3d1r_step3[i] = (double)i/100;
    }
    
    double param_star_3d1r_step3[44] = {0.0};
    for (int i = 0; i < 44; i++) {
        param_star_3d1r_step3[i] = 1;
    }

    double param_1r[27];
    if (argc == 7 && std::string(argv[6]) == "--custom") {
        int num_param = 0;
        if (arg1 == "box3d1r") {
            num_param = 27;
        } else if (arg1 == "star3d1r") {
            num_param = 7;
        }
        printf("Please enter %d parameters:\n", num_param);
        double values[num_param];
        for (int i = 0; i < num_param; i++)
        {
            int readNum = scanf("%lf", &values[i]);
            if (readNum != 1)
                return 1;
        }
        if(num_param == 27) {
            for (int i = 0; i < 27; i++) {
                param_1r[i] = values[i];
            }
        } else if (num_param == 7) {
            for (int i = 0; i < 7; i++) {
                param_1r[4] = values[0];
                param_1r[10] = values[1];
                param_1r[12] = values[2];
                param_1r[13] = values[3];
                param_1r[14] = values[4];
                param_1r[16] = values[5];
                param_1r[22] = values[6];
            }
        }
    }

    bool breakdown = false;
    if (argc == 7 && std::string(argv[6]) == "--breakdown") {
        breakdown = true;
    }

    double *param;
    int halo;
    switch (compute_shape)
    {
    case box_3d1r:
        param = param_box_3d1r_step3;
        halo = 3;
        break;
    case star_3d1r:
        param = param_star_3d1r_step3;
        halo = 3;
        break;
    }

    // print brief info

    printf("INFO: shape = %s, h = %d, m = %d, n = %d, times = %d\n", ShapeStr[compute_shape], h, m, n, times);

    int heights = h + 2 * halo;
    int rows = m + 2 * halo;
    int cols = n + 2 * halo;
    size_t matrix_size = (unsigned long)heights * rows * cols * sizeof(double);

    // allocate space

    double *matrix = (double *)malloc(matrix_size);
    double *output = (double *)malloc(matrix_size);


    // fill input matrix

#if defined(FILL_RANDOM)
#pragma unroll
    for (int i = 0; i < heights * rows * cols; i++)
    {
        matrix[i] = (double)(rand() % 100);
    }
#elif defined(FILL_INDEX)
    for (int i = 0; i < heights * rows * cols; i++)
    {
        matrix[i] = (double)i;
    }
#else
    std::fill_n(matrix, heights * rows * cols, 1.0);
#endif

    switch (compute_shape)
    {
    case box_3d1r:
        if (breakdown) {
            gpu_box_3d1r_breakdown1(matrix, output, param, times, h, m, n);
            gpu_box_3d1r_breakdown2(matrix, output, param, times, h, m, n);
            gpu_box_3d1r_breakdown3(matrix, output, param, times, h, m, n);
            gpu_box_3d1r_breakdown4(matrix, output, param, times, h, m, n);
        }
        gpu_box_3d1r(matrix, output,
                           param, times,
                           h, m, n);
        break;
    case star_3d1r:
        gpu_star_3d1r(matrix, output,
                            param, times,
                            h, m, n);
        break;
    }

    // check result correctness

#if defined(CHECK_ERROR)
    printf("\nChecking ... \n");
    double *naive[2];
    naive[0] = (double *)malloc(matrix_size);
    naive[1] = (double *)malloc(matrix_size);

    for (int i = 0; i < heights * rows * cols; i++) {
        naive[0][i] = matrix[i];
        naive[1][i] = 0;
    }

    int t = 0;
    if (compute_shape == box_3d1r_step3) {
        for (; t < times; t++) {
            naive_box3d1r_step3(naive[t % 2], naive[(t + 1) % 2], param, heights, rows, cols);
        }
    } else if (compute_shape == star_3d1r_step3) {
        for (; t < times; t++) {
            naive_star3d1r_step3(naive[t % 2], naive[(t + 1) % 2], param, heights, rows, cols);
        }
    }
    printf("Comparing naive and output\n");
    int error = 0;
    for (int height = 0; height < heights; height++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (ABS(naive[t % 2][IDX3D(height, row, col, rows, cols)], output[IDX3D(height, row, col, rows, cols)]) > tolerance) {
                    // printf("height = %d, row = %d, col = %d, naive = %lf, output = %lf\n", height, row, col, naive[t % 2][IDX3D(height, row, col, rows, cols)], output[IDX3D(height, row, col, rows, cols)]);
                    error++;
                }
            }
        }
    }
    printf("error = %d\n", error);
#endif

    // write to file

#ifdef WRITE_OUTPUT
#ifdef RUN_GPU
    printf("Writing output_gpu.txt\n");
    save_to_txt(output, heights, rows, cols, "output_gpu.txt");
    // save_to_txt(naive[t % 2], rows, cols, "output_naive.txt");
#endif
#endif

    // free space
    free(output);
    free(matrix);

    return 0;
}