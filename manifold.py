import numpy as np
from numba.experimental import jitclass
from numba.typed import List as TypedList
from numba.types import (
    string, uint8, float64, boolean, ListType, List
)

from typing import Annotated, Literal, TypeVar, List as ListType
from numbers import Real, Integral
from numpy.typing import NDArray

from tools import product, equal

DType = TypeVar("DType", bound=np.generic)
Array3 = Annotated[NDArray[DType], Literal[3]]
Array3x3 = Annotated[NDArray[DType], Literal[3, 3]]
List3 = Annotated[ListType[DType], Literal[3]]
List3x3 = Annotated[ListType[Array3], Literal[3]]
List3x3x3 = Annotated[ListType[Array3x3], Literal[3]]


@jitclass({
    "name": string,
    "num_gens": uint8,
    "L": float64[:],
    "L1": float64, "L2": float64, "L3": float64,
    "angles": float64[:],
    "α": float64, "β": float64, "γ": float64,
    "M1": float64[:, :], "M2": float64[:, :], "M3": float64[:, :],
    "MA": float64[:, :], "MB": float64[:, :],
    "M": List(float64[:, :]),
    "g1": uint8, "g2": uint8, "g3": uint8,
    "g": List(uint8),
    "T1": float64[:], "T2": float64[:], "T3": float64[:],
    "TA1": float64[:], "TA2": float64[:],
    "TB": float64[:],
    "center": boolean,
    "x0": float64[:],
    "g_seqs": List(uint8[:]),
    "nontriv_g_seqs": List(uint8[:]),
    "pure_translations": List(float64[:]),
    "translations": List(float64[:]),
    "all_translations": List(float64[:]),
})
class Manifold(object):
    def __init__(self, name: Literal) -> None:
        self.name = name
        self.num_gens = self._get_num_generators()

        self.L: Array3; self.L1: Real; self.L2: Real; self.L3: Real
        self.angles: Array3; self.α: Real; self.β: Real; self.γ: Real
        self.M: List3x3x3; self.M1: Array3; self.M2: Array3; self.M3: Array3
        self.MA: Array3; self.MB: Array3
        self.g: List3; self.g1: Integral; self.g2: Integral; self.g3: Integral
        self.pure_translations: List3x3
        self.T1: Array3; self.T2: Array3; self.T3: Array3
        self.translations: List3x3
        self.TA1: Array3; self.TA2: Array3; self.TB: Array3
        self.all_translations: List3x3
        self.center: boolean
        self.g_seqs: List3
        self.nontriv_g_seqs: List3

        self.x0 = np.array([0., 0., 0.])

        topologies = ["E1", "E2", "E3", "E4", "E5", "E6", "E11", "E12"]
        assert self.name in topologies, f"Topology {self.name} is not supported."

    def _get_num_generators(self):
        match self.name:
            case "E1" | "E2" | "E3" | "E4" | "E5" | "E6":
                return 3
            case "E11" | "E12":
                return 2

    def construct_generators(self, L_scale: Array3, angles: Array3) -> None:
        match self.num_gens:
            case 2:
                self.L1, self.L2 = self.L = L_scale
                self.α, self.β = self.angles = angles
            case 3:
                self.L1, self.L2, self.L3 = self.L = L_scale
                self.α, self.β, self.γ = self.angles = angles
        self._construct_generators()

    def _construct_generators(self) -> None:
        match self.name:
            case "E1":
                self.M1 = self.M2 = self.M3 = np.eye(3)
                self.g1 = self.g2 = self.g3 = 1
                self.T1 = self.TA1 = self.L1 * np.array([1., 0., 0.])
                self.T2 = self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0.,
                ])
                self.T3 = self.TB = self.L3 * np.array([
                    np.cos(self.β) * np.cos(self.γ),
                    np.cos(self.β) * np.sin(self.γ),
                    np.sin(self.β),
                ])
                self.center = False

            case "E2":
                self.M1 = self.M2 = np.eye(3)
                self.M3 = np.diag(np.array([-1., -1., 1.]))
                self.g1 = self.g2 = 1
                self.g3 = 2
                self.T1 = self.L1 * np.array([1., 0., 0.])
                self.T2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0.,
                ])
                self.T3 = np.array([0., 0., 2 * self.L3])
                self.TA1 = self.L1 * np.array([1., 0., 0.])
                self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0.,
                ])
                self.TB  = np.array([0., 0., self.L3])
                self.center = True

            case "E3":
                assert self.L1 == self.L2, "Restriction on E3: L1 = L2"
                assert self.α == np.pi/2, "Restriction on E3: α = π/2"

                self.M1 = self.M2 = self.MA = np.eye(3)
                self.M3 = self.MB = np.array([
                    [0.,  1.,  0.],
                    [-1., 0.,  0.],
                    [0.,  0.,  1.],
                ])
                self.g1 = self.g2 = 1
                self.g3 = 4
                self.T1 = self.L1 * np.array([1., 0., 0.])
                self.T2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0.,
                ])
                self.T3 = np.array([0., 0., 4 * self.L3])
                self.TA1 = self.L1 * np.array([1., 0., 0.])
                self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0.,
                ])
                self.TB = np.array([0., 0., self.L3])
                self.center = True

            case "E4":
                assert self.L1 == self.L2, "Restriction on E4: L1 = L2"

                self.M1 = self.M2 = self.MA = np.eye(3)
                self.M3 = self.MB = np.array([
                    [-1/2,          np.sqrt(3)/2, 0.],
                    [-np.sqrt(3)/2, -1/2,         0.],
                    [0.,            0.,           1.],
                ])
                self.g1 = self.g2 = 1
                self.g3 = 3
                self.T1 = self.L1 * np.array([1., 0., 0.])
                self.T2 = self.L2 * np.array([-1/2, np.sqrt(3) / 2, 0.])
                self.T3 = np.array([0., 0., 3 * self.L3])
                self.TA1 = self.L1 * np.array([1., 0., 0.])
                self.TA2 = self.L2 * np.array([-1/2, np.sqrt(3) / 2, 0.])
                self.TB = np.array([0., 0., self.L3])
                self.center = True

            case "E5":
                assert self.L1 == self.L2, "Restriction on E5: L1 = L2"

                self.M1 = self.M2 = self.MA = np.eye(3)
                self.M3 = self.MB = np.array([
                    [1/2,           np.sqrt(3)/2, 0.],
                    [-np.sqrt(3)/2, 1/2,          0.],
                    [0.,            0.,           1.],
                ])
                self.g1 = self.g2 = 1
                self.g3 = 6
                self.T1 = self.L1 * np.array([1., 0., 0.])
                self.T2 = self.L2 * np.array([-1/2, np.sqrt(3)/2, 0.])
                self.T3 = np.array([0., 0., 6 * self.L3])
                self.TA1 = self.L1 * np.array([1., 0., 0.])
                self.TA2 = self.L2 * np.array([-1/2, np.sqrt(3)/2, 0.])
                self.TB = np.array([0., 0., self.L3])
                self.center = True

            case "E6":
                LAx = LCx = self.L1
                LBy = LAy = self.L2
                LCz = LBz = self.L3
                
                self.M1 = np.diag(np.array([1., -1., -1.]))
                self.M2 = np.diag(np.array([-1., 1., -1.]))
                self.M3 = np.diag(np.array([-1., -1., 1.]))
                self.g1 = self.g2 = self.g3 = 2
                self.T1 = 2 * LAx * np.array([1., 0., 0.])
                self.T2 = 2 * LBy * np.array([0., 1., 0.])
                self.T3 = 2 * LCz * np.array([0., 0., 1.])
                self.TA1 = np.array([LAx, LAy, 0.])
                self.TA2 = np.array([0., LBy, LBz])
                self.TB  = np.array([LCx, 0., LCz])
                self.center = True

            case "E11":
                self.M1 = self.M2 = np.eye(3)
                self.g1 = self.g2 = 1
                self.T1 = self.TA1 = self.L1 * np.array([1., 0., 0.])
                self.T2 = self.TA2 = self.L2 * np.array([
                    np.cos(self.α), np.sin(self.α), 0.,
                ])
                self.center = False

            case "E12":
                self.M1 = np.eye(3)
                self.M2 = np.diag(np.array([-1., 1., -1.]))
                self.g1 = 1
                self.g2 = 2
                self.T1 = self.TA1 = self.L1 * np.array([
                    np.cos(self.α), 0., np.sin(self.α),
                ])
                self.T2 = np.array([0., 2 * self.L2, 0.])
                self.TA2 = np.array([0., self.L2, 0.])
                self.center = True

        self.T1 = self._round(self.T1, 5)
        self.T2 = self._round(self.T2, 5)
        if self.num_gens == 3: self.T3 = self._round(self.T3, 5)

        self.TA1 = self._round(self.TA1, 5)
        self.TA2 = self._round(self.TA2, 5)
        if self.num_gens == 3: self.TB = self._round(self.TB, 5)

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

        self._find_generator_seqs()
        self._find_all_translations()
        
    @staticmethod
    def _round(x: Array3, n: Integral) -> Array3:
        return np.round(x, n, np.zeros_like(x))
    
    def _find_generator_seqs(self) -> None:
        g_ranges = [
            np.array([x for x in range(1, gi + 1)], dtype=np.uint8)
            for gi in self.g
        ]
        self.g_seqs = list(product(g_ranges))
        self.nontriv_g_seqs = [
            seq for seq in self.g_seqs if not equal(seq, np.ones(3))
        ]

    def apply_generator(self, x: Array3) -> Array3:
        return [
            self.M[i].dot(x - self.x0) + self.translations[i] + self.x0
            for i in range(self.num_gens)
        ]

    def get_generator(self):
        return lambda x: self.apply_generator(x)

    def _find_all_translations(self):
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
            layer_translations.extend([
                tr + self.pure_translations[2] for tr in layer_translations
            ])
            layer_translations.append(self.pure_translations[2])

        return layer_translations

    def _find_all_translations_corner(self):
        """
        To be implemented
        """
        return None
    