from abc import ABC

import numpy as np

from numbers import Integral


def get_num_of_generators(topology_name: str) -> Integral:
    if topology_name in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        num_gens = 3
    elif topology_name in ["E11", "E12"]:
        num_gens = 2
    return num_gens

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
                M3  = MB = np.array([
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
