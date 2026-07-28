#include "matmul.h"

#include <string.h>

/* Cache-friendly loop order with no redundant scalar delay. */
void matmul(const double *a, const double *b, double *c, size_t n) {
    memset(c, 0, n * n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            const double a_ik = a[i * n + k];
            for (size_t j = 0; j < n; ++j) {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
}
