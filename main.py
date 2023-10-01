import numpy as np
import numba as nb
import itertools

from numbers import Integral, Real
from typing import Type, List, Tuple, Literal
from numpy.typing import NDArray

from manifold import Manifold


def scatter_points(
    manifold: Type[Manifold],
    N: int,
) -> NDArray:
    translations = manifold.translations
    num_gens = manifold.num_gens
    L3 = manifold.L3

    points = np.random.rand(N, 3)

    match num_gens:
        case 2:
            scatter = (
                np.outer(points[:, 0], translations[0])
                + np.outer(points[:, 1], translations[1])
                + np.outer((points[:, 2] - 0.5), np.array([0, 0, 2 * L3]))
            )
        case 3:
            scatter = (
                np.outer(points[:, 0], translations[0])
                + np.outer(points[:, 1], translations[1])
                - np.outer(points[:, 2], translations[2])
            )
    
    return scatter

def find_all_translations_corner(pure_translations):
    trans1 = [list(itertools.combinations_with_replacement(pure_translations, i)) for i in range(len(pure_translations) + 2)]
    trans2 = [[(np.add.reduce(trans1[i][j])) for j in range(len(trans1[i]))] for i in range(len(trans1))]

    trans2.append([pure_translations[0] - pure_translations[1]])
    trans2.append([pure_translations[1] - pure_translations[0]])

    trans2[0] = [[0, 0, 0]]

    trans_upplane = list(itertools.chain.from_iterable(trans2))
    all_new_trans = np.array((np.unique(trans_upplane, axis = 0)))

    return all_new_trans

def apply_generator(
    x: NDArray,
    x0: NDArray,
    M: NDArray,
    translation: NDArray,
) -> NDArray:
    """
    Apply a generator to a vector
    """
    return M.dot(x - x0) + translation + x0

def find_clones(manifold: Type[Manifold], positions: NDArray) -> List[NDArray]:
    g = manifold.g
    num_gens = manifold.num_gens
    x0 = manifold.x0
    M = manifold.M
    translations = manifold.translations

    g_ranges = [[x for x in range(1, gi + 1)] for gi in g]
    clone_combs = list(itertools.product(*g_ranges))
    
    full_clone_list = []

    for comb in clone_combs:
        if comb != (1, 1, 1):
            x = positions
            for i in range(num_gens):
                for _ in range(comb[i]):
                    x = apply_generator(x, x0, M[i], translations[i])
            full_clone_list.append(x)

    if full_clone_list == []:
        full_clone_list = positions.copy()
    return full_clone_list

def translate_clones(
    clones: List[NDArray],
    translations: List[NDArray],
) -> List[List[NDArray]]:
    """
    Apply the set of all translations of a manifold to the set of clones
    """
    clones = np.array(clones)
    translations = np.array(translations)
    translated_clone_positions = clones[:, None] + translations[None, :]
    return translated_clone_positions

# @nb.njit
def dist(x, y):
    return np.linalg.norm(x - y, axis=-1)
    # if len(x.shape) == 1:
    #     N = 1
    # else:
    #     N = x.shape[0]

    # res = np.empty(N, dtype=x.dtype)
    # for i in nb.prange(x.shape[0]):
    #     acc = 0
    #     for j in range(x.shape[-1]):
    #         acc += (x[i, j] - y[j])**2
    #     res[i] = np.sqrt(acc)
    # return res

def distances(
    clone_positions: NDArray,
    position: NDArray,
) -> Tuple[NDArray, Real]:
    distances = dist(clone_positions, position)
    closest_clone_position = clone_positions[idx := distances.argmin()]
    return closest_clone_position, distances[idx]

def find_closest_clone(
    manifold: Type[Manifold],
    generated_clones: NDArray,
    positions: NDArray,
) -> Tuple[NDArray, Real]:
    generated_clone_points = np.array([gc[0] for gc in generated_clones])
    generated_clone_distances = np.array([gc[1] for gc in generated_clones])

    pure_translations = manifold.pure_translations
    x0 = manifold.x0
    
    # translate_clone = [
    #     [(pt + positions), dist(pt, x0)] for pt in pure_translations
    # ]

    translated_clones = np.empty((3, 3), dtype=np.float64)
    translated_clone_distances = np.empty(3, dtype=np.float64)

    for i in range(3):
        pure_translation = pure_translations[i]
        translated_clones[i] = pure_translation + positions
        translated_clone_distances[i] = dist(x0, pure_translation)

    all_clone_pts = np.vstack((generated_clone_points, translated_clones))
    all_clones_dists = np.hstack((generated_clone_distances, translated_clone_distances))

    # closest_translated_clone = min(translate_clone, key = lambda x: x[1] if (x[1] > 10e-12) else np.nan)
    # closest_generated_clone = min(generated_clones, key = lambda x: x[1] if (x[1]> 10e-12) else np.nan, default = closest_translated_clone)

    min_idx = all_clones_dists.argmin()

    return (all_clone_pts[min_idx], all_clones_dists[min_idx])
    # return min((closest_generated_clone, closest_translated_clone), key = lambda x: x[1])

def E_general_topology(
    manifold: Type[Manifold],
    positions: NDArray,
):
    translations = manifold.translations
    # x0 = manifold.x0

    clones = find_clones(manifold, positions)
    translated_clone_positions = translate_clones(clones, translations)
    
    """Rewrite in a better way, split points and distances"""
    nearest_from_layer = [
        distances(translated_clone_positions[i], positions)
        for i in range(len(translated_clone_positions))
    ]

    closest_clone = find_closest_clone(manifold, nearest_from_layer, positions)

    return closest_clone[1]

def sample_topology(
    manifold: Type[Manifold],
    positions: NDArray,
):
    manifold.find_all_translations()
    distance = E_general_topology(manifold, positions)
    print(distance)
    return distance

def sample_points(
    manifold: Type[Manifold],
    precision: int,
):
    num_gens = manifold.num_gens
    pure_translations = manifold.pure_translations
    L_scale = manifold.L
    L_scatter = scatter_points(manifold, precision)
    
    match num_gens:
        case 2:
            positions = (
                L_scatter
                - 0.5 * (pure_translations[0] + pure_translations[1])
            )
        case 3:
            positions = (
                L_scatter
                - 0.5 * (
                    pure_translations[0] + pure_translations[1]
                    - pure_translations[2]
                )
            )

    count = 0
    allowed_points = []
    excluded_points = []

    allowed_pts_idx = []
    excluded_pts_idx = []
    
    for k in range(precision):
        distance = sample_topology(manifold, positions[k])
        if distance < 1:
            count +=1
            excluded_pts_idx.append(k)
        else:
            allowed_pts_idx.append(k)

    percents = 1 - count / precision

    allowed_pts = np.array(positions[allowed_pts_idx])
    excluded_pts = np.array(positions[excluded_pts_idx])

    return percents, excluded_pts, allowed_pts

    # L_x = [allowed_points[i][0] for i in range(len(allowed_points))]
    # L_y = [allowed_points[i][1] for i in range(len(allowed_points))]
    # L_z = [allowed_points[i][2] for i in range(len(allowed_points))]

    # excluded_points_x = [excluded_points[i][0] for i in range(len(excluded_points))]
    # excluded_points_y = [excluded_points[i][1] for i in range(len(excluded_points))]
    # excluded_points_z = [excluded_points[i][2] for i in range(len(excluded_points))]

    # return percents, [excluded_points_x, excluded_points_y, excluded_points_z], [L_x, L_y, L_z]

def sample_assoc_E1(N: int):
    """
    Rewrite with a grid in each direction instead of random sampling
    """
    L1 = np.random.uniform(low = 1, high = 2, size = N)
    L2 = np.random.uniform(low = L1[0], high = 2, size = N)
    L3 = np.random.uniform(low = 0.5, high = 1.1, size = N)
    L_samples = np.stack((L1, L2, L3)).T
    return L_samples

"""
===============================================================================
"""

@nb.njit(
    nb.float64[:, :](nb.types.unicode_type, nb.int32),
    parallel=True,
)
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


if __name__ == "__main__":
    np.random.seed(1234)

    manifold_name = "E2"

    α = β = γ = np.pi / 2
    angles = np.array([α, β, γ])
    
    precision = 20
    param_precision = 10
    
    L_samples = sample_associated_E1_topology(manifold_name, param_precision)


    # L_samples = sample_assoc_E1(param_precision)

    L_accept = []
    L_reject = []

    for i in range(param_precision):
        manifold = Manifold(manifold_name)
        manifold.construct_generators(L_samples[i], angles)
        exit()
        percents, excludedPoints, allowedPoints = sample_points(
            manifold, precision,
        )
        print(percents, excludedPoints, allowedPoints)
        
        exit()

        if percents > 0.05:
            L_accept.append(L_samples[i])
        else:
            L_reject.append(L_samples[i])
        if (i%10 == 0):
            print(i)

        # exit()

    print(L_accept)
    print(L_reject)
