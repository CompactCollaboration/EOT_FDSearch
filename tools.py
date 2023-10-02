import numpy as np
import numba as nb

@nb.njit(parallel=True, cache=True)
def product(arrays):
    n = 1
    for x in arrays: n *= x.size
    m = len(arrays)
    out = np.zeros((n, m), dtype=arrays[0].dtype)

    for i in range(m):
        p = arrays[i].size
        m = int(n / p)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= p

    n = arrays[-1].size
    for k in range(m - 2, -1, -1):
        p = arrays[k].size
        n *= p
        m = int(n / p)
        for j in range(1, p):
            out[j*m:(j+1)*m, k+1:] = out[0:m, k+1:]
    return out
