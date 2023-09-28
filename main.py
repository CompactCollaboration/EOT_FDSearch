import numpy as np
import itertools
from scipy.spatial import distance

from numbers import Integral
from typing import Type

from manifold import Manifold


def scatter_points(
    manifold,
    points,
):
    translations = manifold.translations
    num_gens = manifold.num_gens
    L3 = manifold.L3

    if num_gens == 2:
        scatter = (
            np.outer(points[:, 0], translations[0])
            + np.outer(points[:, 1], translations[1])
            + np.outer((points[:, 2] - 0.5), np.array([0, 0, 2 * L3]))
        )
    elif num_gens == 3:
        scatter = (
            np.outer(points[:, 0], translations[0])
            + np.outer(points[:, 1], translations[1])
            - np.outer(points[:, 2], translations[2])
        )
    
    return scatter

# def find_all_translations_center(
#     pure_translations,
#     num_gens,
# ):
#     layer_trans = [
#         pure_translations[0],
#         pure_translations[1],
#         -pure_translations[0],
#         -pure_translations[1],
#         -2 * pure_translations[0],
#         -2 * pure_translations[1],
#         2 * pure_translations[0],
#         2 * pure_translations[1],
#         pure_translations[0] + pure_translations[1],
#         pure_translations[0] - pure_translations[1], 
#         -pure_translations[0] + pure_translations[1],
#         -pure_translations[0] - pure_translations[1],
#     ]

#     if num_gens == 2:
#         return layer_trans
#     elif num_gens == 3:
#         all_new_trans = np.concatenate([
#             layer_trans,
#             layer_trans + pure_translations[2],
#             [pure_translations[2]]
#         ])
#         return all_new_trans
#     else:
#         raise Exception("num_gens can be 2 or 3 only")

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
    x,
    x0,
    M,
    translation,
):  
    return M.dot(x - x0) + translation + x0

def find_clones(
    manifold,
    positions,
):
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
    clones,
    translations,
):
    translated_clone_positions = [
        [clone + translation for translation in translations]
        for clone in clones
    ]
    return translated_clone_positions

def translate_clones_old(clone_pos, translations):
    translated_clone_pos = [clone_pos + translations[i] for i in range(len(translations))]
    return translated_clone_pos

def distances(clone_pos, pos):
    trans_dist = [distance.euclidean(pos, clone_pos[i]) for i in range(len(clone_pos))]
    min_trans_dist = min(trans_dist)
    closest_clone_pos = clone_pos[trans_dist.index(min_trans_dist)]
    return closest_clone_pos, min_trans_dist

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
    positions,
):
    translations = manifold.all_translations
    x0 = manifold.x0
    print(x0)


    clones = find_clones(manifold, positions)
    translated_clone_positions = translate_clones(clones, translations)
    
    nearest_from_layer = [distances(translated_clone_positions[i], positions) for i in range(len(translated_clone_positions))]
    closest_clone = find_closest_clone(nearest_from_layer, translations, x0, positions)
    return closest_clone[1]

def sample_topology(
    manifold,
    positions,
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

    points = np.random.rand(precision, 3)

    L_scatter = scatter_points(manifold, points)
    
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
    
    # L_sample = np.dstack((L1[0], L1[0], L3[0]))[0]
    # manifold.construct_generators(L_sample[0], angles)

    L_accept = []
    L_reject = []

    for i in range(param_precision):
        manifold = Manifold(manifold_name)
        manifold.construct_generators(L_samples[i], angles)
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
