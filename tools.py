import numpy as np
import numba as nb

from numba.types import float64


@nb.njit(parallel=True, cache=True)
def product(arrays):
    """
    Generate the Cartesian product of input arrays.

    The function computes the Cartesian product of a list of arrays, 
    where each array in the input list is one dimension in the output 
    array. The resulting array is a 2D array where each row is a 
    combination of elements from each of the input arrays.

    Args:
        arrays (List[np.ndarray]): A list of 1D NumPy arrays of possibly 
                                   different lengths.

    Returns:
        np.ndarray: A 2D NumPy array of shape (N, M) where N is the 
                    product of the sizes of the arrays in `arrays` and M 
                    is the number of arrays. Each row represents a unique 
                    combination of elements from the input arrays.

    Examples:
        >>> product([np.array([1, 2]), np.array([3, 4])])
        array([[1, 3],
               [1, 4],
               [2, 3],
               [2, 4]])

    Notes:
        - The function utilizes Numba's JIT compilation with parallelism 
          and caching for improved performance on large inputs.
        - The input arrays are assumed to be non-empty and have compatible 
          dtypes.
    """
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

@nb.njit(cache=True)
def equal(a, b):
    """
    Check if two arrays are element-wise equal.

    This function compares two arrays and determines if they are equal, 
    i.e., all their corresponding elements are the same. The comparison 
    is done element-wise.

    Args:
        a (np.ndarray): The first array to compare.
        b (np.ndarray): The second array to compare.

    Returns:
        bool: True if the arrays are equal, and False otherwise.

    Examples:
        >>> equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        True
        >>> equal(np.array([1, 2, 3]), np.array([4, 5, 6]))
        False

    Notes:
        - The function is JIT-compiled with Numba for performance.
        - The input arrays should be of the same shape and data type.
        - If arrays are of different shapes, a broadcasting error will 
          be raised by NumPy.
    """
    return np.all(a == b)

@nb.njit(
    float64[:, :, :](float64[:, :, :, :]),
    fastmath=True, parallel=True, cache=True,
)
def compute_distances(x):
    x_sh = x.shape
    distances = np.zeros((x_sh[0], x_sh[1], x_sh[2]), dtype=np.float64)
    for i in nb.prange(x_sh[0]):
        for j in range(x_sh[1]):
            for k in range(x_sh[2]):
                x0 = x[i, j, k, 0]
                x1 = x[i, j, k, 1]
                x2 = x[i, j, k, 2]
                distances[i, j, k] = np.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    return distances

@nb.njit(
    float64[:, :](float64[:, :], float64[:, :]),
    fastmath=True, parallel=True, cache=True,
)
def compute_dot(M, x):
    y = np.empty_like(x)
    for i in nb.prange(x.shape[0]):
        y[i, 0] = M[0, 0] * x[i, 0] + M[0, 1] * x[i, 1] + M[0, 2] * x[i, 2]
        y[i, 1] = M[1, 0] * x[i, 0] + M[1, 1] * x[i, 1] + M[1, 2] * x[i, 2]
        y[i, 2] = M[2, 0] * x[i, 0] + M[2, 1] * x[i, 1] + M[2, 2] * x[i, 2]
    return y
