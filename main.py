from abc import ABC

import numpy as np
import itertools
from scipy.spatial import distance

from numbers import Integral


def get_num_of_generators(topology_name: str) -> Integral:
    if topology_name in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        num_gens = 3
    elif topology_name in ["E11", "E12"]:
        num_gens = 2
    return num_gens

def scatter_points(
    points,
    translations,
    # precision, # UNUSED
    num_gens,
    L3,
):
    if num_gens == 2:
        scatter = (
            points[:][0] * translations[0]
            + points[:][1] * translations[1]
            + (points[:][2] - 0.5) * np.array([0, 0, 2 * L3])
        )
    elif num_gens == 3:
        scatter = (
            points[:][0] * translations[0]
            + points[:][1] * translations[1]
            - points[:][2] * translations[2]
        )
    else:
        raise Exception("num_gens can be 2 or 3 only")
    
    return scatter

def find_all_translations_center(
    pure_translations,
    num_gens,
):
    layer_trans = [
        pure_translations[0],
        pure_translations[1],
        -pure_translations[0],
        -pure_translations[1],
        -2 * pure_translations[0],
        -2 * pure_translations[1],
        2 * pure_translations[0],
        2 * pure_translations[1],
        pure_translations[0] + pure_translations[1],
        pure_translations[0] - pure_translations[1], 
        -pure_translations[0] + pure_translations[1],
        -pure_translations[0] - pure_translations[1],
    ]

    if num_gens == 2:
        return layer_trans
    elif num_gens == 3:
        all_new_trans = np.concatenate([
            layer_trans,
            layer_trans + pure_translations[2],
            [pure_translations[2]]
        ])
        return all_new_trans
    else:
        raise Exception("num_gens can be 2 or 3 only")

def find_all_translations_corner(pure_translations):
    trans1 = [list(itertools.combinations_with_replacement(pure_translations, i)) for i in range(len(pure_translations) + 2)]
    trans2 = [[(np.add.reduce(trans1[i][j])) for j in range(len(trans1[i]))] for i in range(len(trans1))]

    trans2.append([pure_translations[0] - pure_translations[1]])
    trans2.append([pure_translations[1] - pure_translations[0]])

    trans2[0] = [[0, 0, 0]]

    trans_upplane = list(itertools.chain.from_iterable(trans2))
    all_new_trans = np.array((np.unique(trans_upplane, axis = 0)))

    return all_new_trans

def constructions(
    manifold,
    L_scale,
    angles,
):
    num_gens = get_num_of_generators(manifold)

    if num_gens == 2:
        M, translations, pure_translations, E1_dict, center, x0 = Manifold.construct_2_generators(Manifold, L_scale, angles)
        translation_list = find_all_translations_center(pure_translations, num_gens)
    elif num_gens == 3:
        M, translations, pure_translations, E1_dict, center, x0 = Manifold.construct_3_generators(Manifold, L_scale, angles)
        if center == True:
            translation_list = find_all_translations_center(pure_translations, num_gens)
        else:
            translation_list = find_all_translations_corner(pure_translations)
    else:
        raise Exception("num_gens can be 2 or 3 only")
    
    return M, translations, pure_translations, E1_dict, translation_list, num_gens, x0

def generator_pos(
    x,
    x0,
    M,
    translations,
):  
    x_out = M.dot(x - x0) + translations + x0
    return x_out

def find_clones(
    pos,
    x0,
    M,
    translations,
    E1_dict,
    num_gens,
):
    clone_pos = []

    for i in range(E1_dict[0]):
        for j in range(E1_dict[1]):
            if num_gens == 3:
                for k in range(E1_dict[2]):
                    clone_pos.append([i, j, k])
            elif num_gens == 2:
                clone_pos.append([i, j])
    
    full_clone_list = []

    for i in range(len(clone_pos)):
        if not (all(x == 0 for x in clone_pos[i])):
            _x = pos
            for j in range(len(clone_pos[i])):
                for k in range(clone_pos[i][j]):
                    _x = generator_pos(_x, x0, M[j], translations[j])
            full_clone_list.append(_x)

    if not full_clone_list:
        return pos, clone_pos
    else:
        return full_clone_list, clone_pos

def translate_clones(clone_pos, translations):
    translated_clone_pos = [clone_pos + translations[i] for i in range(len(translations))]
    return translated_clone_pos

def distances(clone_pos, pos):
    trans_dist = [distance.euclidean(pos, clone_pos[i]) for i in range(len(clone_pos))]
    min_trans_dist = min(trans_dist)
    closest_clone_pos = clone_pos[trans_dist.index(min_trans_dist)]
    return closest_clone_pos, min_trans_dist

def E_general_topol(
    pos,
    x0,
    M,
    translations,
    pure_translations,
    E1_dict,
    num_gens,
    translation_list,
):
    clone_positions, _ = find_clones(pos, x0, M, translations, E1_dict, num_gens)
    translated_clone_pos = [translate_clones(clone_positions[i], translation_list) for i in range(len(clone_positions))]
    nearest_from_layer = [distances(translated_clone_pos[i], pos, x0) for i in range(len(translated_clone_pos))]

def sample_topology(
    manifold,
    L_scale,
    pos,
    angles,
):
    M, translations, pure_translations, E1_dict, translation_list, num_gens, x0 = constructions(manifold, L_scale, angles)
    distance = E_general_topol(pos, x0, M, translations, pure_translations, E1_dict, num_gens, translation_list)

def sample_points(
    manifold,
    angles,
    precision,
    L_scale,
):
    num_gens = get_num_of_generators(manifold)
    points = np.random.rand(precision, 3)
    
    match num_gens:
        case 2:
            _, _, pure_translations, _, _, _ = Manifold.construct_2_generators(manifold, L_scale, angles)
            L_scatter = scatter_points(points, pure_translations, num_gens, L_scale[2])
            pos = L_scatter - 0.5 * ((pure_translations[0] + pure_translations[1]))
        case 3:
            _, _, pure_translations, _, _, _ = Manifold.construct_3_generators(manifold, L_scale, angles)
            L_scatter = scatter_points(points, pure_translations, num_gens, L_scale[2])
            pos = L_scatter - 0.5 * ((pure_translations[0] + pure_translations[1] - pure_translations[2]))

    count = 0
    allowed_points = []
    excluded_points = []
    
    for k in range(precision):
        distance = sample_topology(manifold, L_scale, pos[k], angles)


class Manifold(ABC):
    def __init__(self):
        pass

    def construct_3_generators(self, name, L_scale, angles):
        L1, L2, L3 = L_scale

        match name:
            case "E1":
                M1 = M2 = M3 = np.eye(3)
                g1 = g2 = g3 = 1
                T1 = TA1 = L1 * np.array([1, 0, 0])
                T2 = TA2 = L2 * np.array([np.cos(angles[0]), np.sin(angles[0]), 0])
                T3 = TB = L3 * np.array([
                    np.cos(angles[1]) * np.cos(angles[2]),
                    np.cos(angles[1]) * np.sin(angles[2]),
                    np.sin(angles[1]),
                ])
                center = False

            case "E2":
                M1 = M2 = np.eye(3)
                M3 = np.diag([-1, -1, 1])
                g1 = g2 = 1
                g3 = 2
                TA1 = L1 * np.array([1, 0, 0])
                TA2 = L2 * np.array([np.cos(angles[0]), np.sin(angles[0]), 0])
                TB  = np.array([0, 0, L3])
                
                T1 = L1 * np.array([1, 0, 0])
                T2 = L2 * np.array([np.cos(angles[0]), np.sin(angles[0]), 0])
                T3 = np.array([0, 0, 2 * L3])
                center = True

            case "E3":
                M1 = M2 = MA = np.eye(3)
                M3 = MB = np.array([
                    [0,  1,  0],
                    [-1, 0,  0],
                    [0,  0,  1],
                ])
                TA1 = L1 * np.array([1, 0, 0])
                TA2 = L2 * np.array([np.cos(angles[0]), np.sin(angles[0]), 0])
                TB = np.array([0, 0, L3])
                T1 = L1 * np.array([1, 0, 0])
                T2 = L2 * np.array([np.cos(angles[0]), np.sin(angles[0]), 0])
                T3 = np.array([0, 0, 4 * L3])
                center = True

                if L1 != L2 or angles[0] != np.pi/2:
                    raise ValueError("Restrictions on E3: L1 = L2 and alpha = pi/2")
                
            case "E4":
                g1 = 1
                g2 = 1
                g3 = 3
                
                M1 = M2 = MA = np.eye(3)
                M3 = MB = np.array([
                    [-1/2, np.sqrt(3)/2, 0],
                    [-np.sqrt(3)/2, -1/2, 0],
                    [0, 0, 1],
                ])
                
                TA1 = L1 * np.array([1, 0, 0])
                TA2 = L2 * np.array([-1/2, np.sqrt(3) / 2, 0])
                TB = np.array([0, 0, L3])
                
                T1 = L1 * np.array([1, 0, 0])
                T2 = L2 * np.array([-1/2, np.sqrt(3) / 2, 0])
                # T3 = L3 * np.array([0, 0, 3 * np.sin(angles[1])])
                T3 = np.array([0, 0, 3 * L3])
                
                center = True
                
                if (L1 != L2):
                    raise ValueError("Restrictions on E4: L1 = L2")
                
            case "E5":
                g1 = 1
                g2 = 1
                g3 = 6
                
                M1 = M2 = MA = np.eye(3)
                M3  = MB = np.array([
                    [1/2, np.sqrt(3)/2, 0],
                    [-np.sqrt(3)/2, 1/2, 0],
                    [0, 0, 1],
                ])
                
                TA1 = L1 * np.array([1, 0, 0])
                TA2 = L2 * np.array([-1/2, np.sqrt(3)/2, 0])
                TB = np.array([0, 0, L3])
                
                T1 = L1 * np.array([1, 0, 0])
                T2 = L2 * np.array([-1/2, np.sqrt(3)/2, 0])
                T3 = np.array([0, 0, 6 * L3])
                
                center = True
                
                if (L1 != L2):
                    raise ValueError("Restrictions on E5: L1 = L2")
                
            case "E6":
                LCx, LAy, LBz = L_scale
            
                g1 = 2
                g2 = 2
                g3 = 2
                
                M1 = np.diag(([1, -1, -1]))
                M2 = np.diag(([-1, 1, -1]))
                M3 = np.diag(([-1, -1, 1]))
                
                LAx = LCx
                LBy = LAy
                LCz = LBz
                
                TA1 = np.array([LAx, LAy, 0])
                TA2 = np.array([0, LBy, LBz])
                TB  = np.array([LCx, 0, LCz])
                
                T1 = 2 * LAx * np.array([1, 0, 0])
                T2 = 2 * LBy * np.array([0, 1, 0])
                T3 = 2 * LCz * np.array([0, 0, 1])
                
                center = True
            
        translations = np.around(np.array([TA1, TA2, TB]), decimals = 5)
        # Probably an iffy way to solve this problem, needs to figure out if 
        # theres a better way to do this
        pureTranslations = np.around(np.array([T1, T2, -T3]), decimals = 5)
        associatedE1Dict = np.array([g1, g2, g3])
        M = [M1, M2, M3]
        x0 = np.array([0, 0, 0])
        
        return M, translations, pureTranslations, associatedE1Dict, center, x0

    def construct_2_generators(self, name, L_scale, angles):
        L1, L2 = L_scale

        match name:
            case "E11":
                g1 = 1
                g2 = 1
                
                M1 = M2 = np.eye(3)
                TA1 = T1 = L1 * np.array([1, 0, 0])
                TA2 = T2 = L2 * np.array([np.cos(angles[0]), np.sin(angles[0]), 0])
                
                center = False

            case "E12":
                g1 = 1
                g2 = 2
                
                M1 = np.eye(3)
                M2 = np.diag([-1, 1, -1])
                
                TA1 = T1 = L1 * np.array([np.cos(angles[0]), 0, np.sin(angles[0])])
                TA2 = np.array([0, L2, 0])
                
                T2 = np.array([0, 2 * L2, 0])
                
                center = True
        
        translations = np.around(np.array([TA1, TA2]), decimals = 5)
        pureTranslations = np.around(np.array([T1, T2]), decimals = 5)
        associatedE1Dict = np.array([g1, g2])
        M = [M1, M2]
        x0 = np.array([0, 0, 0])

        return M, translations, pureTranslations, associatedE1Dict, center, x0
