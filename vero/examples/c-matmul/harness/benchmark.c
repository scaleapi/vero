#include "matmul.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000.0
        + (double)(end.tv_nsec - start.tv_nsec) / 1000000.0;
}

static void reference_matmul(
    const double *a,
    const double *b,
    double *c,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main(void) {
    const size_t n = 128;
    const size_t elements = n * n;
    double *a = malloc(elements * sizeof(double));
    double *b = malloc(elements * sizeof(double));
    double *actual = calloc(elements, sizeof(double));
    double *expected = calloc(elements, sizeof(double));
    if (a == NULL || b == NULL || actual == NULL || expected == NULL) {
        return 2;
    }

    for (size_t index = 0; index < elements; ++index) {
        a[index] = (double)((int)(index % 13) - 6) / 7.0;
        b[index] = (double)((int)(index % 11) - 5) / 5.0;
    }

    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul(a, b, actual, n);
    clock_gettime(CLOCK_MONOTONIC, &end);
    reference_matmul(a, b, expected, n);

    int correct = 1;
    for (size_t index = 0; index < elements; ++index) {
        if (fabs(actual[index] - expected[index]) > 1e-9) {
            correct = 0;
            break;
        }
    }

    printf("%d %.9f\n", correct, elapsed_ms(start, end));
    free(a);
    free(b);
    free(actual);
    free(expected);
    return correct ? 0 : 3;
}
