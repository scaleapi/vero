"""Naive matrix multiply kernel. Optimize this program."""


def multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices using the naive O(n^3) algorithm.

    Args:
        a: Matrix of shape (n, k)
        b: Matrix of shape (k, m)

    Returns:
        Result matrix of shape (n, m)
    """
    n = len(a)
    m = len(b[0])
    k = len(b)
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                result[i][j] += a[i][p] * b[p][j]
    return result
