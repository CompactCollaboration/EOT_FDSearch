from abc import ABC

import numpy as np
import numba as nb
from numba.experimental import jitclass

from typing import Literal, List
from numpy.typing import NDArray


@jitclass([('name', nb.types.unicode_type)])
class Manifold():
    def __init__(self, name: Literal) -> None:
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
                self.T2 = np.array([0, 2 * self.L2, 0])
                self.TA2 = np.array([0, self.L2, 0])
                self.center = True

        if self.num_gens == 2:
            self.M = [self.M1, self.M2]
            self.g = [self.g1, self.g2]
            self.pure_translations = [self.T1, self.T2]
            self.translations = [self.TA1, self.TA2]
        elif self.num_gens == 3:
            self.M = [self.M1, self.M2, self.M3]
            self.g = [self.g1, self.g2, self.g3]
            self.pure_translations = [self.T1, self.T2, -self.T3]
            self.translations = [self.TA1, self.TA2, self.TB]
        
        self.pure_translations = list(np.round(self.pure_translations, 5))
        self.translations = list(np.round(self.translations, 5))

    def apply_generator(self, x):
        return [
            self.M[i].dot(x - self.x0) + self.translations[i] + self.x0
            for i in range(self.num_gens)
        ]

    def get_generator(self):
        return lambda x: self.apply_generator(x)

    def find_all_translations(self):
        if self.center is True:
            translations = self._find_all_translations_center()
        else:
            translations = self._find_all_translations_corner()
        self.all_translations = translations

    def _find_all_translations_center(self):
        layer_translations = [
            self.pure_translations[0],
            self.pure_translations[1],
            -self.pure_translations[0],
            -self.pure_translations[1],
            -2 * self.pure_translations[0],
            -2 * self.pure_translations[1],
            2 * self.pure_translations[0],
            2 * self.pure_translations[1],
            self.pure_translations[0] + self.pure_translations[1],
            self.pure_translations[0] - self.pure_translations[1], 
            -self.pure_translations[0] + self.pure_translations[1],
            -self.pure_translations[0] - self.pure_translations[1],
        ]

        if self.num_gens == 3:
            layer_translations.extend(layer_translations + self.pure_translations[2])
            layer_translations.append(self.pure_translations[2])

        return layer_translations

    def _find_all_translations_corner(self):
        """
        To be implemented
        """
        return None
    