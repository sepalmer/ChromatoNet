"""ChromatoNet with 24 muscles per chromatophore.

Extends chromatonet.ChromatoNet so that each chromatophore connects to its
first (L1), second (L2) and fourth (L4) neighbor shells, 24 muscles in total
(S1 Fig). The model equations are the same as in chromatonet.py; only the
muscle layout, the gap junction neighbors and some default parameters differ.

Each chromatophore has 24 muscle slots, numbered counterclockwise from the
right:
    L1 (distance dx): slots 0, 4, 8, 12, 16, 20.
    L2 (distance sqrt(3) dx): slots 2, 6, 10, 14, 18, 22.
    L4 (distance sqrt(7) dx): the odd slots 1, 3, ..., 23.
A muscle's gap junction neighbors are the muscles in the adjacent slots
(slot - 1 and slot + 1) at each of its two ends.

Typical usage example:

    net = chromatonet_24.ChromatoNet24(N=20, M=20, integration_method='rk4')
    net.set_vi(stims.vi_edge)
    vals = net.run(666, name='edge_24', dt=0.15, save_freq=6)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

import chromatonet

# Parameters used for the 24-connect simulations. They replace the values in
# defaults.py; anything passed in params replaces them in turn.
PARAMS_24: dict[str, float] = {
    'kc': 0.5,
    'kp': 0.2,
    'ks': 0.5,
    'ggap': 0.005,
    'gl': 0.42,
    'phi': 0.05,
}

# Muscle offsets (di, dj, slot_a, slot_b) for even and odd rows. A muscle
# connects chromatophore (i, j) to (i + di, j + dj).
_L1_OFFSETS = (
    ((0, 1, 0, 12), (1, 0, 4, 16), (1, -1, 8, 20)),  # Even rows.
    ((0, 1, 0, 12), (1, 0, 8, 20), (1, 1, 4, 16)),  # Odd rows.
)
_L2_OFFSETS = (
    ((1, 1, 2, 14), (2, 0, 6, 18), (1, -2, 10, 22)),  # Even rows.
    ((1, 2, 2, 14), (2, 0, 6, 18), (1, -1, 10, 22)),  # Odd rows.
)
_L4_OFFSETS = (
    ((1, 2, 1, 13), (2, 2, 3, 15), (3, 0, 5, 17),
     (3, -1, 7, 19), (2, -2, 9, 21), (1, -3, 11, 23)),  # Even rows.
    ((1, 3, 1, 13), (2, 2, 3, 15), (3, 1, 5, 17),
     (3, 0, 7, 19), (2, -2, 9, 21), (1, -2, 11, 23)),  # Odd rows.
)


class Muscle24(chromatonet.Muscle):
    """A radial muscle with a slot index at each end.

    Attributes:
        slot_a: Slot of this muscle on chroma (-1 if none).
        slot_b: Slot of this muscle on chromb (-1 if none).
    """

    def __init__(
        self,
        chroma: chromatonet.Chromatophore,
        chromb: chromatonet.Chromatophore,
        parameters: Mapping[str, chromatonet.ParamFunction],
        active: bool = True,
        slot_a: int = -1,
        slot_b: int = -1,
    ):
        """Creates the muscle and records its slot at each end.

        Args:
            chroma: Chromatophore at one end.
            chromb: Chromatophore at the other end.
            parameters: Functions of position that give each parameter.
            active: Whether the muscle is active.
            slot_a: Slot of this muscle on chroma.
            slot_b: Slot of this muscle on chromb.
        """
        super().__init__(chroma, chromb, parameters, active=active)
        self.slot_a = slot_a
        self.slot_b = slot_b


class ChromatoNet24(chromatonet.ChromatoNet):
    """ChromatoNet with 24 muscles per chromatophore (hex layout only).

    Takes the same arguments as chromatonet.ChromatoNet. The differences are:
    muscles connect the L1, L2 and L4 neighbor shells, gap junction neighbors
    come from adjacent slots, PARAMS_24 replaces some defaults, and removed
    muscles also lose their gap junction conductance.

    Attributes:
        chromat_slots: For each chromatophore, a dict mapping each slot to
            the global index of the muscle endpoint in that slot.
    """

    def __init__(
        self,
        N: int,
        M: int | None = None,
        dx: float = 1.,
        layout: str = 'hex',
        noise: float = 0,
        iapp: chromatonet.CurrentFunction | None = None,
        params: Mapping[str, chromatonet.ParamFunction] | None = None,
        integration_method: str = 'euler',
        connection_removal_fraction: float = 0.0,
        removal_seed: int | None = None,
    ):
        """Builds and compiles the network with the 24-connect parameters.

        Args are the same as for chromatonet.ChromatoNet. Values in params
        replace those in PARAMS_24.
        """
        merged = {k: (lambda p, v=v: v) for k, v in PARAMS_24.items()}
        if params is not None:
            merged.update(params)
        super().__init__(
            N, M=M, dx=dx, layout=layout, noise=noise, iapp=iapp,
            params=merged, integration_method=integration_method,
            connection_removal_fraction=connection_removal_fraction,
            removal_seed=removal_seed)

    def make_hex_net(self, noise: float) -> None:
        """Builds a hexagonal lattice with 24 muscles per chromatophore.

        Chromatophores on the perimeter are marked as edge and held fixed.

        Args:
            noise: Standard deviation of the noise on rest positions.
        """
        dx = np.array([self.dx, 0])
        dy = np.array([0., np.sqrt(3)/2])*self.dx
        self.chromats = []

        for i in range(self.M):
            self.chromats.append([])
            bx = np.array([((i%2)*self.dx)/2, 0])
            for j in range(self.N - (i % 2)):
                is_edge = (
                    i == 0  # Bottom row.
                    or i == self.M - 1  # Top row.
                    or j == 0  # Left edge.
                    or j == self.N - (i % 2) - 1  # Right edge.
                )
                self.chromats[i].append(
                    chromatonet.Chromatophore(
                        zi=i*dy + bx + j*dx, noise=noise,
                        parameters=self.init_func, is_edge=is_edge))

        def has(i, j):
            return 0 <= i < self.M and 0 <= j < len(self.chromats[i])

        # Add the layers in order: L1, then L2, then L4.
        for offsets in (_L1_OFFSETS, _L2_OFFSETS, _L4_OFFSETS):
            for i in range(self.M):
                for j in range(self.N - (i % 2)):
                    for di, dj, slot_a, slot_b in offsets[i % 2]:
                        if has(i + di, j + dj):
                            self.muscles.append(Muscle24(
                                self.chromats[i][j],
                                self.chromats[i + di][j + dj],
                                parameters=self.init_func,
                                slot_a=slot_a, slot_b=slot_b))

    def make_chain_net(self, noise: float) -> None:
        """Not supported: the 24-muscle layout is defined for hex only.

        Raises:
            ValueError: Always.
        """
        raise ValueError("ChromatoNet24 supports only layout='hex'.")

    def precompute_tempgv_neighbors(self) -> None:
        """Finds the gap junction neighbors of each muscle from its slots.

        A muscle's neighbors are the muscles in the adjacent slots (slot - 1
        and slot + 1) at each of its two ends, so up to 4 in total.
        """
        # Endpoint in each slot: chromat_slots[chromatophore][slot].
        self.chromat_slots = [{} for _ in range(len(self.x))]
        for m in self.muscles:
            if not m.active:
                continue
            if m.slot_a >= 0:
                self.chromat_slots[m.chroma.index][m.slot_a] = (
                    m.chroma.y[m.a_index][2])
            if m.slot_b >= 0:
                self.chromat_slots[m.chromb.index][m.slot_b] = (
                    m.chromb.y[m.b_index][2])

        max_slot = 0
        for m in self.muscles:
            max_slot = max(max_slot, m.slot_a, m.slot_b)
        if max_slot < 6:
            n_slots = 6
        elif max_slot < 12:
            n_slots = 12
        else:
            n_slots = 24

        self.muscle_tempgv_neighbors = []
        for m in self.muscles:
            if not m.active:
                self.muscle_tempgv_neighbors.append([])
                continue
            neighbor_y_indices = []
            for x_idx, slot in ((m.chroma.index, m.slot_a),
                                (m.chromb.index, m.slot_b)):
                if slot < 0:
                    continue
                for adj_slot in ((slot - 1) % n_slots, (slot + 1) % n_slots):
                    if adj_slot in self.chromat_slots[x_idx]:
                        neighbor_y_indices.append(
                            self.chromat_slots[x_idx][adj_slot])
            self.muscle_tempgv_neighbors.append(neighbor_y_indices)
            # Store the lists by compiled muscle index (m.index), which is how
            # the simulation looks them up.
            by_index = [[] for _ in range(len(self.v))]
            for m, neighbors in zip(self.muscles, self.muscle_tempgv_neighbors):
                by_index[m.index] = neighbors
            self.muscle_tempgv_neighbors = by_index

    def sum_voltages_per_chromatophore(
        self,
        tempv: torch.Tensor,
    ) -> torch.Tensor:
        """Returns zeros: the 24-muscle gap junctions do not use this sum.

        Args:
            tempv: Voltage at each muscle endpoint (not used).
        """
        return torch.zeros(len(self.x), device=self.device)

    def sum_voltages_per_muscle(
        self,
        Vt: torch.Tensor,
        tempyv: torch.Tensor | None,
    ) -> torch.Tensor:
        """Returns the summed voltage of each muscle's gap junction neighbors.

        Args:
            Vt: Muscle voltages.
            tempyv: Not used.
        """
        tempgv = torch.zeros_like(Vt)
        for m_idx in range(len(self.v)):
            neighbor_y_indices = self.muscle_tempgv_neighbors[m_idx]
            if len(neighbor_y_indices) > 0:
                neighbor_m_indices = self.m_to_y[neighbor_y_indices]
                tempgv[m_idx] = torch.sum(Vt[neighbor_m_indices])
        return tempgv

    def _apply_connection_removal(self) -> None:
        """Removes a random fraction of muscles, including gap junctions."""
        super()._apply_connection_removal()
        for m_idx in self.severed_muscles:
            self.cached_params['ggap'][m_idx] = 0.0