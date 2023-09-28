import numpy as np
import numba as nb
import itertools

from numbers import Integral, Real
from typing import Type, List, Tuple
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

def dist(x, y):
    return np.linalg.norm(x - y, axis=-1)

def distances(
    clone_positions: NDArray,
    position: NDArray,
) -> Tuple[NDArray, Real]:
    distances = dist(position, clone_positions)
    closest_clone_position = clone_positions[idx := distances.argmin()]
    return closest_clone_position, distances[idx]

def find_closest_clone(
    generated_clones,
    pure_translations,
    x0,
    pos,
):
    translate_clone = [[(pure_translations[x] + pos), distance.euclidean(x0, pure_translations[x])] for x in range(len(pure_translations))]
    closest_translated_clone = min(translate_clone, key = lambda x: x[1] if (x[1] > 10e-12) else np.nan)
    closest_generated_clone = min(generated_clones, key = lambda x: x[1] if (x[1]> 10e-12) else np.nan, default = closest_translated_clone)
    return min((closest_generated_clone, closest_translated_clone), key = lambda x: x[1])

def E_general_topology(
    manifold: Type[Manifold],
    positions: NDArray,
):
    translations = manifold.all_translations
    x0 = manifold.x0

    clones = find_clones(manifold, positions)
    translated_clone_positions = translate_clones(clones, translations)
    
    """Rewrite in a better way, split points and distances"""
    nearest_from_layer = [distances(translated_clone_positions[i], positions) for i in range(len(translated_clone_positions))]

    closest_clone = find_closest_clone(nearest_from_layer, translations, x0, positions)
    return closest_clone[1]

def sample_topology(
    manifold: Type[Manifold],
    positions: NDArray,
):
    manifold.find_all_translations()
    distance = E_general_topology(manifold, positions)
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
    
    for k in range(precision):
        distance = sample_topology(manifold, positions[k])
        if distance < 1:
            count +=1
            excluded_points.append(positions[k])
        else:
            allowed_points.append(positions[k])

    percents = 1 - count / precision

    L_x = [allowed_points[i][0] for i in range(len(allowed_points))]
    L_y = [allowed_points[i][1] for i in range(len(allowed_points))]
    L_z = [allowed_points[i][2] for i in range(len(allowed_points))]

    excluded_points_x = [excluded_points[i][0] for i in range(len(excluded_points))]
    excluded_points_y = [excluded_points[i][1] for i in range(len(excluded_points))]
    excluded_points_z = [excluded_points[i][2] for i in range(len(excluded_points))]

    return percents, [excluded_points_x, excluded_points_y, excluded_points_z], [L_x, L_y, L_z]

def sample_assoc_E1(N: int):
    """
    Rewrite with a grid in each direction instead of random sampling
    """
    L1 = np.random.uniform(low = 1, high = 2, size = N)
    L2 = np.random.uniform(low = L1[0], high = 2, size = N)
    L3 = np.random.uniform(low = 0.5, high = 1.1, size = N)
    L_samples = np.stack((L1, L2, L3)).T
    return L_samples

if __name__ == "__main__":
    np.random.seed(1234)

    manifold_name = "E2"

    α = β = γ = np.pi / 2
    angles = np.array([α, β, γ])
    
    precision = 20
    param_precision = 10
    
    L_samples = sample_assoc_E1(param_precision)

    L_accept = []
    L_reject = []

    for i in range(param_precision):
        manifold = Manifold(manifold_name) #
        manifold.construct_generators(L_samples[i], angles) #
        percents, excludedPoints, allowedPoints = sample_points(
            manifold, precision,
        )
        
        if percents > 0.05:
            L_accept.append(L_samples[i])
        else:
            L_reject.append(L_samples[i])
        if (i%10 == 0):
            print(i)

        exit()

    print(L_accept)
    print(L_reject)
