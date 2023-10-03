import numpy as np
import numba as nb

from numbers import Integral, Real
from typing import Type, List, Tuple, Literal
from numpy.typing import NDArray

from time import process_time

from manifold import Manifold


# @nb.njit(
#     nb.float64[:, :](nb.types.unicode_type, nb.int32),
#     parallel=True,
# )
def sample_associated_E1_topology(
    manifold_name: str,
    size: Integral,
) -> NDArray:
    eq_tops = ["E3", "E4", "E5"]
    L = np.zeros((size**3, 3), dtype=np.float64)
    L1 = np.linspace(1, 2, size)
    L2 = np.linspace(1, 2, size) if manifold_name not in eq_tops else L1.copy()
    L3 = np.linspace(0.5, 1.1, size)

    size2 = size**2
    for i in range(size):
        for j in range(size):
            for k in range(size):
                idx = i * size2 + j * size + k
                L[idx, 0] = L1[k]
                L[idx, 1] = L2[j]
                L[idx, 2] = L3[i]

    return L

# @nb.njit
def find_circles(
    manifold: Type[Manifold],
    precision: Integral,
):
    num_gens = manifold.num_gens
    pure_translations = manifold.pure_translations

    scatter = sample_topology_box(manifold, precision)
    
    match num_gens:
        case 2:
            positions = (
                scatter
                - 0.5 * (pure_translations[0] + pure_translations[1])
            )
        case 3:
            positions = (
                scatter
                - 0.5 * (
                    pure_translations[0] + pure_translations[1]
                    - pure_translations[2]
                )
            )
    # ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=0.5, c='r')
    allowed_pts_idx = []
    excluded_pts_idx = []
    for i in range(precision):
        # ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], s=20, c='k')
        distance = compute_topology_distance(manifold, positions[i])
        # break
        if distance < 1:
            excluded_pts_idx.append(i)
        else:
            allowed_pts_idx.append(i)

    allowed_frac = len(allowed_pts_idx) / precision
    excluded_frac = len(excluded_pts_idx) / precision

    if len(allowed_pts_idx) > 0:
        allowed_pts = positions[np.array(allowed_pts_idx)]
    else:
        allowed_pts = []

    if len(excluded_pts_idx) > 0:
        excluded_pts = positions[np.array(excluded_pts_idx)]
    else:
        excluded_pts = []

    return allowed_frac, excluded_frac, excluded_pts, allowed_pts

# @nb.njit(parallel=True)
def sample_topology_box(
    manifold: Type[Manifold],
    size: Integral,
) -> NDArray:
    pure_translations = manifold.pure_translations
    num_gens = manifold.num_gens
    L3 = manifold.L3
    points = np.random.rand(size, 3)

    match num_gens:
        case 2:
            scatter = (
                np.outer(points[:, 0], pure_translations[0])
                + np.outer(points[:, 1], pure_translations[1])
                + np.outer((points[:, 2] - 0.5), np.array([0, 0, 2 * L3]))
            )
        case 3:
            scatter = (
                np.outer(points[:, 0], pure_translations[0])
                + np.outer(points[:, 1], pure_translations[1])
                - np.outer(points[:, 2], pure_translations[2])
            )

    # ax.scatter(scatter[:, 0], scatter[:, 1], scatter[:, 2], s=0.5, c='b')
    return scatter

# @nb.njit
def compute_topology_distance(
    manifold: Type[Manifold],
    point: NDArray,
):
    all_translations = manifold.all_translations
    clones = find_point_clones(manifold, point)
    # ax.scatter(np.array(clones)[:, 0], np.array(clones)[:, 1], np.array(clones)[:, 2], s=20, c='g')
    translated_clone_positions = find_translated_clones(clones, all_translations)
    # print(translated_clone_positions.shape)
    # ax.scatter(translated_clone_positions[:, :, 0], translated_clone_positions[:, :, 1], translated_clone_positions[:, :, 2], s=20, c='r')
    nearest_pt_from_layer = np.zeros((translated_clone_positions.shape[0], 3), dtype=np.float64)
    nearest_distances = np.zeros(translated_clone_positions.shape[0], dtype=np.float64)
    for i in range(translated_clone_positions.shape[0]):
        nearest_pt_from_layer[i], nearest_distances[i] = find_distances(
            translated_clone_positions[i],
            point,
        )
    closest_clone, distance = find_closest_clone(
        manifold,
        nearest_pt_from_layer,
        nearest_distances,
        point,
    )
    return distance

# @nb.njit
def find_point_clones(
    manifold: Type[Manifold],
    point: NDArray,
) -> List[NDArray]:
    g = manifold.g
    num_gens = manifold.num_gens
    x0 = manifold.x0
    M = manifold.M
    translations = manifold.translations
    full_clone_list = []
    
    for comb in manifold.nontriv_g_seqs:
        x = point
        for i in range(num_gens):
            for _ in range(1, comb[i]):
                x = apply_gen(x, x0, M[i], translations[i])
        full_clone_list.append(x)

    if len(full_clone_list) == 0: full_clone_list = [point]
    return full_clone_list

# @nb.njit
def apply_gen(
    x: NDArray,
    x0: NDArray,
    M: NDArray,
    translation: NDArray,
) -> NDArray:
    """
    Apply a generator to a vector
    """
    return M.dot(x - x0) + translation + x0

# @nb.njit
def find_translated_clones(
    clones: List[NDArray],
    translations: List[NDArray],
) -> List[List[NDArray]]:
    """
    Apply the set of all translations of a manifold to the set of clones
    """
    clones_arr = np.empty((len(clones), 3))
    for i in range(len(clones)): clones_arr[i] = clones[i]
    translations_arr = np.empty((len(translations), 3))
    for i in range(len(translations)): translations_arr[i] = translations[i]
    translated_clone_positions = clones_arr[:, None] + translations_arr[None, :]
    return translated_clone_positions

# @nb.njit
def find_distances(
    clone_positions: NDArray,
    point: NDArray,
) -> Tuple[NDArray, Real]:
    distances = dist(clone_positions, point)
    idx = distances.argmin()
    return clone_positions[idx], distances[idx]

# @nb.njit
def dist(x, y):
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    for i in range(N):
        result[i] = np.linalg.norm(x[i] - y)
    return result

# @nb.njit
def find_closest_clone(
    manifold: Type[Manifold],
    generated_clones: NDArray,
    generated_distances: NDArray,
    point: NDArray,
) -> Tuple[NDArray, Real]:
    pure_translations = manifold.pure_translations
    x0 = manifold.x0

    translated_clones = np.empty((3, 3), dtype=np.float64)
    translated_clone_distances = np.empty(3, dtype=np.float64)

    for i in range(3):
        translated_clones[i] = pure_translations[i] + point
    translated_clone_distances = dist(pure_translations, x0)

    all_clone_pts = np.vstack((generated_clones, translated_clones))
    all_clones_dists = np.hstack((generated_distances, translated_clone_distances))

    min_idx = all_clones_dists.argmin()
    return all_clone_pts[min_idx], all_clones_dists[min_idx]


if __name__ == "__main__":
    np.random.seed(1234)

    manifold_name = "E5"
    manifold = Manifold(manifold_name)
    α = β = γ = np.pi / 2
    angles = np.array([α, β, γ])
    
    precision = 10000
    param_precision = 2
    
    L_samples = sample_associated_E1_topology(manifold_name, param_precision)
    L_samples = np.array([[2.1, 2.1, 0.32564806]])
    # L_samples = np.array([[1.62210877, 1.94390309, 0.92723427]])

    L_accept = []
    L_reject = []

    import matplotlib.pyplot as plt
    from plot import Arrow3D

    for i in range(L_samples.shape[0]):
        manifold.construct_generators(L_samples[i], angles)
        
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')

        # colors = ["r", "g", "b"]
        
        # for i, tr in enumerate(manifold.translations):
        #     arw = Arrow3D([0, tr[0]], [0, tr[1]], [0, tr[2]], arrowstyle="->", color=colors[i], lw = 2, mutation_scale=25)
        #     ax.add_artist(arw)

        # for i, ptr in enumerate(manifold.pure_translations):
        #     arw = Arrow3D([0, ptr[0]], [0, ptr[1]], [0, ptr[2]], ls="--", arrowstyle="->", color=colors[i], lw = 2, mutation_scale=25)
        #     ax.add_artist(arw)

        # ax.set_xlim(-8, 8)
        # ax.set_ylim(-8, 8)
        # ax.set_zlim(-8, 8)

        

        allowed_frac, excluded_frac, allowed_pts, excluded_pts = find_circles(
            manifold, precision,
        )

        # plt.show()

        print(allowed_frac, excluded_frac)
        exit()

        if excluded_frac > 0.05:
            L_accept.append(L_samples[i])
        else:
            L_reject.append(L_samples[i])
        # if (i%10 == 0):
        #     print(i)

        # exit()

    # print(L_accept)
    # print(L_reject)
