#include "matmul.h"

/* Correct but deliberately slow baseline. */
void matmul(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
            for (volatile size_t delay = 0; delay < 500; ++delay) {
            }
        }
    }
}
