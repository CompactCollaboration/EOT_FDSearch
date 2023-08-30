from abc import ABC

import numpy as np
import itertools
from scipy.spatial import distance

from numbers import Integral


def scatter_points(
    points,
    translations,
    num_gens,
    L3,
):
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
        M, translations, pure_translations, E1_dict, center, x0 = Manifold.construct_2_generators(manifold, L_scale, angles)
        translation_list = find_all_translations_center(pure_translations, num_gens)
    elif num_gens == 3:
        M, translations, pure_translations, E1_dict, center, x0 = Manifold.construct_3_generators(manifold, L_scale, angles)
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
    nearest_from_layer = [distances(translated_clone_pos[i], pos) for i in range(len(translated_clone_pos))]
    closest_clone = find_closest_clone(nearest_from_layer, pure_translations, x0, pos)
    return closest_clone[1]

def sample_topology(
    manifold,
    L_scale,
    pos,
    angles,
):
    M, translations, pure_translations, E1_dict, translation_list, num_gens, x0 = constructions(manifold, L_scale, angles)
    distance = E_general_topol(pos, x0, M, translations, pure_translations, E1_dict, num_gens, translation_list)
    return distance

def sample_points(
    manifold,
    precision: int,
):
    num_gens = manifold.num_gens
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
        if distance < 1:
            count +=1
            excluded_points.append(pos[k])
        else:
            allowed_points.append(pos[k])

    percents = 1 - count / precision

    L_x = [allowed_points[i][0] for i in range(len(allowed_points))]
    L_y = [allowed_points[i][1] for i in range(len(allowed_points))]
    L_z = [allowed_points[i][2] for i in range(len(allowed_points))]

    excluded_points_x = [excluded_points[i][0] for i in range(len(excluded_points))]
    excluded_points_y = [excluded_points[i][1] for i in range(len(excluded_points))]
    excluded_points_z = [excluded_points[i][2] for i in range(len(excluded_points))]

    return percents, [excluded_points_x, excluded_points_y, excluded_points_z], [L_x, L_y, L_z]


class Manifold(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.num_gens = None
        self.L = None
        self.L1 = self.L2 = self.L3 = None
        self.angles = None
        self.α = self.β = self.γ = None
        self.M1 = self.M2 = self.M3 = None
        self.M = None
        self.g1 = self.g2 = self.g3 = None
        self.T1 = self.T2 = self.T3 = None
        self.TA1 = self.TA2 = None
        self.TB = None
        self.center = None
        self.x0 = np.array([0, 0, 0])
        self.pure_translations = None
        self.translations = None

        topologies = ["E1", "E2", "E3", "E4", "E5", "E6", "E11", "E12"]
        assert self.name in topologies, f"Topology {self.name} is not supported."

    def construct_generators(self, L_scale, angles):
        self._get_num_generators()
        match self.num_gens:
            case 2:
                self.L1, self.L2 = self.L = L_scale
                self.α, self.β = self.angles = angles
            case 3:
                self.L1, self.L2, self.L3 = self.L = L_scale
                self.α, self.β, self.γ = self.angles = angles
        
        self._construct_generators()

    def _get_num_generators(self):
        match self.name:
            case "E1" | "E2" | "E3" | "E4" | "E5" | "E6":
                self.num_gens = 3
            case "E11" | "E12":
                self.num_gens = 2

    def _construct_generators(self) -> None:
        match self.name:
            case "E1":
                self.M1 = self.M2 = self.M3 = np.eye(3)
                self.g1 = self.g2 = self.g3 = 1
                self.T1 = self.TA1 = self.L1 * np.array([1, 0, 0])
                self.T2 = self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0,
                ])
                self.T3 = self.TB = self.L3 * np.array([
                    np.cos(self.β) * np.cos(self.γ),
                    np.cos(self.β) * np.sin(self.γ),
                    np.sin(self.β),
                ])
                self.center = False

            case "E2":
                self.M1 = self.M2 = np.eye(3)
                self.M3 = np.diag([-1, -1, 1])
                self.g1 = self.g2 = 1
                self.g3 = 2
                self.T1 = self.L1 * np.array([1, 0, 0])
                self.T2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0,
                ])
                self.T3 = np.array([0, 0, 2 * self.L3])
                self.TA1 = self.L1 * np.array([1, 0, 0])
                self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0,
                ])
                self.TB  = np.array([0, 0, self.L3])
                self.center = True

            case "E3":
                assert self.L1 == self.L2, "Restriction on E3: L1 = L2"
                assert self.α == np.pi/2, "Restriction on E3: α = π/2"

                self.M1 = self.M2 = self.MA = np.eye(3)
                self.M3 = self.MB = np.array([
                    [0,  1,  0],
                    [-1, 0,  0],
                    [0,  0,  1],
                ])
                self.g1 = self.g2 = 1
                self.g3 = 4
                self.T1 = self.L1 * np.array([1, 0, 0])
                self.T2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0,
                ])
                self.T3 = np.array([0, 0, 4 * self.L3])
                self.TA1 = self.L1 * np.array([1, 0, 0])
                self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0,
                ])
                self.TB = np.array([0, 0, self.L3])
                self.center = True

            case "E4":
                assert self.L1 == self.L2, "Restriction on E4: L1 = L2"

                self.M1 = self.M2 = self.MA = np.eye(3)
                self.M3 = self.MB = np.array([
                    [-1/2, np.sqrt(3)/2, 0],
                    [-np.sqrt(3)/2, -1/2, 0],
                    [0, 0, 1],
                ])
                self.g1 = self.g2 = 1
                self.g3 = 3
                self.T1 = self.L1 * np.array([1, 0, 0])
                self.T2 = self.L2 * np.array([-1/2, np.sqrt(3) / 2, 0])
                self.T3 = np.array([0, 0, 3 * self.L3])
                self.TA1 = self.L1 * np.array([1, 0, 0])
                self.TA2 = self.L2 * np.array([-1/2, np.sqrt(3) / 2, 0])
                self.TB = np.array([0, 0, self.L3])
                self.center = True

            case "E5":
                assert self.L1 == self.L2, "Restriction on E5: L1 = L2"

                self.M1 = self.M2 = self.MA = np.eye(3)
                self.M3 = self.MB = np.array([
                    [1/2, np.sqrt(3)/2, 0],
                    [-np.sqrt(3)/2, 1/2, 0],
                    [0, 0, 1],
                ])
                self.g1 = self.g2 = 1
                self.g3 = 6
                self.T1 = self.L1 * np.array([1, 0, 0])
                self.T2 = self.L2 * np.array([-1/2, np.sqrt(3)/2, 0])
                self.T3 = np.array([0, 0, 6 * self.L3])
                self.TA1 = self.L1 * np.array([1, 0, 0])
                self.TA2 = self.L2 * np.array([-1/2, np.sqrt(3)/2, 0])
                self.TB = np.array([0, 0, self.L3])
                self.center = True

            case "E6":
                LAx = LCx = self.L1
                LBy = LAy = self.L2
                LCz = LBz = self.L3
                
                self.M1 = np.diag(([1, -1, -1]))
                self.M2 = np.diag(([-1, 1, -1]))
                self.M3 = np.diag(([-1, -1, 1]))
                self.g1 = self.g2 = self.g3 = 2
                self.T1 = 2 * LAx * np.array([1, 0, 0])
                self.T2 = 2 * LBy * np.array([0, 1, 0])
                self.T3 = 2 * LCz * np.array([0, 0, 1])
                self.TA1 = np.array([LAx, LAy, 0])
                self.TA2 = np.array([0, LBy, LBz])
                self.TB  = np.array([LCx, 0, LCz])
                self.center = True

            case "E11":
                self.M1 = self.M2 = np.eye(3)
                self.g1 = self.g2 = 1
                self.T1 = self.TA1 = self.L1 * np.array([1, 0, 0])
                self.T2 = self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0,
                ])
                self.center = False

            case "E12":
                self.M1 = np.eye(3)
                self.M2 = np.diag([-1, 1, -1])
                self.g1 = 1
                self.g2 = 2
                self.T1 = self.TA1 = self.L1 * np.array([
                    np.cos(self.α), 0, np.sin(self.α),
                ])
                self.T2 = np.array([0, 2 * L2, 0])
                self.TA2 = np.array([0, L2, 0])
                self.center = True

        if self.num_gens == 2:
            self.M = [self.M1, self.M2]
            self.g = [self.g1, self.g2]
            self.pure_translations = np.array([self.T1, self.T2])
            self.translations = np.array([self.TA1, self.TA2])
        elif self.num_gens == 3:
            self.M = [self.M1, self.M2, self.M3]
            self.g = [self.g1, self.g2, self.g3]
            self.pure_translations = np.array([self.T1, self.T2, -self.T3])
            self.translations = np.array([self.TA1, self.TA2, self.TB])


if __name__ == "__main__":
    np.random.seed(1234)

    manifold_name = "E2"
    manifold = Manifold(manifold_name)

    α = β = γ = np.pi / 2
    angles = np.array([α, β, γ])
    
    precision = 20
    param_precision = 10
    
    L1 = np.array([np.random.uniform(low = 1, high = 2, size = param_precision)])
    L2 = np.array([np.random.uniform(low = L1[0], high = 2, size = param_precision)])
    L3 = np.array([np.random.uniform(low = 0.5, high = 1.1, size = param_precision)])
    
    random_L_sample = np.dstack((L1[0], L1[0], L3[0]))[0]
    manifold.construct_generators(random_L_sample[0], angles)

    print(manifold.translations)
    exit()

    L_accept = []
    L_reject = []

    for i in range(param_precision):
        percents, excludedPoints, allowedPoints = sample_points(manifold, angles, precision, random_L_sample[i])
        
        if percents > 0.05:
            L_accept.append(random_L_sample[i])
        else:
            L_reject.append(random_L_sample[i])
        if (i%10 == 0):
            print(i)

    print(L_accept)
    print(L_reject)
