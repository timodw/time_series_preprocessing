#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stddef.h>

double dtw(int len_s1, double* s1, int len_s2, double* s2) {
    double *mat_d = malloc((len_s1 + 1) * (len_s2 + 1) * sizeof(double));
    if (!mat_d) return -1;  // Memory allocation failed

    for (int i = 0; i <= len_s1; i++) {
        for (int j = 0; j <= len_s2; j++) {
            mat_d[i * (len_s2 + 1) + j] = DBL_MAX;
        }
    }
    mat_d[0] = 0.0;

    for (int i = 1; i <= len_s1; i++) {
        for (int j = 1; j <= len_s2; j++) {
            double d = pow(s1[i - 1] - s2[j - 1], 2);
            double min_val = fmin(mat_d[(i - 1) * (len_s2 + 1) + j], mat_d[i * (len_s2 + 1) + (j - 1)]);
            min_val = fmin(min_val, mat_d[(i - 1) * (len_s2 + 1) + (j - 1)]);
            mat_d[i * (len_s2 + 1) + j] = d + min_val;
        }
    }

    double result = sqrt(mat_d[len_s1 * (len_s2 + 1) + len_s2]);
    free(mat_d);
    return result;
}
