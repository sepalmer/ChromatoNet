"""ChromatoNet: a network model of chromatophores in cephalopod skin.

Each chromatophore is a point tethered to its rest position by a skin spring
(ks). Radial muscles connect neighboring chromatophores. Each muscle has one
endpoint on each chromatophore. An endpoint is pulled toward its own
chromatophore by the chromatophore-muscle coupling (kc), and along the muscle
by a passive spring (kp) and a voltage-gated active spring (ka). Muscle
voltage follows Morris-Lecar dynamics with a stretch-sensitive calcium
conductance and gap junction coupling between muscles. Boundary
chromatophores are held fixed.

State variables:
    x: Chromatophore positions, shape (n_chromatophores, 2).
    z: Chromatophore rest positions, shape (n_chromatophores, 2).
    y: Muscle endpoints (two per muscle), shape (n_endpoints, 2).
    v: Muscle voltages, shape (n_muscles,).
    w: Potassium gating variables, shape (n_muscles,).

Short names such as N, M, x, y, z, V, W, L0, KP and KA follow the notation of
Eqs 2-6 in the ChromatoNet paper.

Typical usage example:

    net = chromatonet.ChromatoNet(N=20, M=20, layout='hex')
    net.set_vi(stims.vi_gaussian)
    vals = net.run(1500, name='example', dt=0.1, save_freq=10)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import random
import sys
import time
from typing import Any, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F

import defaults

PYTORCH = True

sys.setrecursionlimit(50000)

# Enable optimization flags.
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = True

# Maps a position, shape (2,), to a parameter value.
ParamFunction: TypeAlias = Callable[[np.ndarray], Any]

# Maps muscle positions, shape (n_muscles, 2), and the time to currents.
CurrentFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor],
                                      torch.Tensor]


def sigmoid(x: np.ndarray, th: float, sh: float) -> np.ndarray:
    """Returns the logistic sigmoid 1 / (1 + exp(-(x - th) / sh))."""
    return 1. / (1. + np.exp(-(x - th)/sh))


def sigmoid_torch(
    x: torch.Tensor,
    th: torch.Tensor | float,
    sh: torch.Tensor | float,
) -> torch.Tensor:
    """Returns the logistic sigmoid of (x - th) / sh for tensors."""
    return torch.sigmoid((x - th)/sh)


class Muscle:
    """A radial muscle connecting two chromatophores.

    The muscle has one endpoint on each chromatophore. Its parameters are
    evaluated at the midpoint of the two rest positions.

    Attributes:
        index: Position in the compiled network (-1 until compiled).
        active: Whether the muscle is active.
        chroma: Chromatophore at one end.
        chromb: Chromatophore at the other end.
        par: Parameter values of this muscle.
        a_index: Position of this muscle in chroma's endpoint list.
        b_index: Position of this muscle in chromb's endpoint list.
        V: Initial voltage.
        W: Initial potassium gating variable.
        L0: Rest length, the distance between the two rest positions.
    """

    def __init__(
        self,
        chroma: Chromatophore,
        chromb: Chromatophore,
        parameters: Mapping[str, ParamFunction],
        active: bool = True,
    ):
        """Creates the muscle and adds an endpoint to each chromatophore.

        Args:
            chroma: Chromatophore at one end.
            chromb: Chromatophore at the other end.
            parameters: Functions of position that give each parameter.
            active: Whether the muscle is active.
        """
        self.index = -1  # Set in compile_net.
        self.active = active
        self.chroma = chroma
        self.chromb = chromb

        self.par = {}
        for k in parameters:
            self.par[k] = parameters[k]((self.chroma.z + self.chromb.z)/2)

        # Position of this muscle in each chromatophore's endpoint list.
        self.a_index = len(self.chroma.y)
        self.b_index = len(self.chromb.y)

        # Endpoint entry: [position, muscle, global endpoint index]. The
        # global index is set in compile_net.
        self.chroma.y.append([np.copy(self.chroma.x), self, -1])
        self.chromb.y.append([np.copy(self.chromb.x), self, -1])

        self.V = self.par['vinit']
        self.W = self.par['winit']
        self.L0 = np.linalg.norm(chroma.z - chromb.z)

    def other_att(self, ch: Chromatophore) -> list[Any]:
        """Returns the endpoint entry at the other end of this muscle.

        Args:
            ch: The chromatophore at one end of this muscle.

        Returns:
            The endpoint entry [position, muscle, global endpoint index] on
            the other chromatophore.

        Raises:
            ValueError: If ch is not attached to this muscle.
        """
        if ch == self.chroma:
            return self.chromb.y[self.b_index]
        if ch == self.chromb:
            return self.chroma.y[self.a_index]
        raise ValueError('Chromatophore is not attached to this muscle.')


class Chromatophore:
    """A single chromatophore.

    Attributes:
        index: Position in the compiled network (-1 until compiled).
        is_edge: Whether this is a boundary chromatophore, held fixed.
        z: Rest position (lattice position plus spatial noise).
        par: Parameter values of this chromatophore.
        x: Current position.
        x_clean: Noise-free lattice position, used to find gap junction
            neighbors.
        y: Muscle endpoint entries, filled in by Muscle.
    """

    def __init__(
        self,
        zi: np.ndarray,
        parameters: Mapping[str, ParamFunction],
        noise: float = 0,
        is_edge: bool = False,
    ):
        """Creates the chromatophore at a lattice position.

        Args:
            zi: Lattice position, shape (2,).
            parameters: Functions of position that give each parameter.
            noise: Standard deviation of the Gaussian noise added to the rest
                position.
            is_edge: Whether this is a boundary chromatophore.
        """
        self.index = -1  # Set in compile_net.
        self.is_edge = is_edge
        self.z = np.copy(parameters['z'](zi) + np.random.randn(2)*noise)

        self.par = {}
        for k in parameters:
            self.par[k] = np.copy(parameters[k](self.z))

        self.x = np.copy(self.z)
        self.x_clean = np.copy(parameters['z'](zi))
        self.y = []


class ChromatoNet:  # ChromatophoreNetwork
    """A network of chromatophores connected by radial muscles.

    The network is built and compiled to PyTorch tensors on creation. Call
    run() to simulate it.

    Attributes:
        chromats: Chromatophore objects, one list per row.
        muscles: Muscle objects.
        layout: 'hex' or 'chain'.
        N: Number of columns (hex) or chromatophores (chain).
        M: Number of rows (hex only).
        dx: Lattice spacing.
        device: Torch device used for the simulation.
        iapp: Applied current function.
        x: Chromatophore positions (see the module docstring).
        y: Muscle endpoints.
        z: Chromatophore rest positions.
        v: Muscle voltages.
        w: Potassium gating variables.
        cached_params: Parameter tensors used during the simulation.
        edge_mask: True for boundary chromatophores, which are held fixed.
        severed_muscles: Indices of removed muscles, if any were removed.
    """

    def __init__(
        self,
        N: int,
        M: int | None = None,
        dx: float = 1.,
        layout: str = 'hex',
        noise: float = 0,
        iapp: CurrentFunction | None = None,
        params: Mapping[str, ParamFunction] | None = None,
        integration_method: str = 'euler',
        connection_removal_fraction: float = 0.0,
        removal_seed: int | None = None,
    ):
        """Builds and compiles the network.

        Args:
            N: Number of columns (hex) or chromatophores (chain).
            M: Number of rows (hex only). Defaults to N.
            dx: Lattice spacing.
            layout: 'hex' or 'chain'.
            noise: Standard deviation of the frozen Gaussian noise added to
                the rest positions.
            iapp: Applied current, a function iapp(muscle_positions, t) that
                returns one current per muscle. None uses the constant
                defaults.DEFAULT2D['Iapp'] on every muscle.
            params: Functions of position that override defaults.DEFAULT2D,
                keyed by parameter name. 'z' maps lattice positions to rest
                positions.
            integration_method: 'euler' or 'rk4'.
            connection_removal_fraction: Fraction of muscles to remove, from
                0 to 1.
            removal_seed: Seed for choosing which muscles to remove. None
                gives a random choice.

        Raises:
            ValueError: If layout or integration_method is not recognized.
        """
        if params is None:
            params = {}

        self.chromats = []
        self.muscles = []

        # Every parameter is a function of position; defaults are constants.
        self.init_func = {}
        for k in defaults.DEFAULT2D:
            def constructor(k):
                def f(x):
                    return defaults.DEFAULT2D[k]
                return f
            self.init_func[k] = constructor(k)

        for k in params:
            self.init_func[k] = params[k]

        if 'z' not in params:
            self.init_func['z'] = lambda x: x

        self.layout = layout
        self.N = N
        self.M = M if M is not None else N
        self.dx = dx

        if layout == 'hex':
            self.make_hex_net(noise=noise)
        elif layout == 'chain':
            self.make_chain_net(noise=noise)
        else:
            raise ValueError(
                f"Unknown layout: {layout!r}. Use 'hex' or 'chain'.")

        # Placeholders, filled in by compile_net.
        self.x = []
        self.x_clean = []
        self.y = []
        self.z = []
        self.w = []
        self.v = []
        self.ac = []
        self.par = {}
        self.compiled = False

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')

        # Force deterministic behavior.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if iapp is None:
            self.iapp = self.iapp_default
        else:
            self.iapp = iapp

        self.integration_method = integration_method
        self.connection_removal_fraction = connection_removal_fraction
        self.removal_seed = removal_seed

        self.compile_net()
        if self.connection_removal_fraction > 0:
            self._apply_connection_removal()

    def iapp_default(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns the constant DEFAULT2D['Iapp'] current on every muscle."""
        return (torch.ones(len(self.v), device=self.device)
                * defaults.DEFAULT2D['Iapp'])

    def get_m_pos(
        self,
        yv: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Returns the muscle positions, the midpoints of their endpoints.

        Args:
            yv: Endpoint positions to use. Defaults to the current endpoints.
        """
        if yv is None or len(yv) == 0:
            y = self.y
        else:
            y = yv

        out = []
        for k in range(len(self.v)):
            out.append((y[self.y_to_m[k, 0]] + y[self.y_to_m[k, 1]])/2)

        return torch.stack(out) if isinstance(y, torch.Tensor) else np.array(out)

    # Network layouts.

    def make_hex_net(self, noise: float) -> None:
        """Builds a hexagonal lattice.

        Odd rows are shifted by dx/2 and hold one fewer chromatophore.
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
                    Chromatophore(zi=i*dy + bx + j*dx, noise=noise,
                                  parameters=self.init_func,
                                  is_edge=is_edge))

        # Muscles to the right neighbor, the neighbor above, and the
        # diagonal neighbor above.
        for i in range(self.M):
            fac = ((i%2)*2 - 1)
            for j in range(self.N - (i % 2)):
                if j+1 < len(self.chromats[i]):
                    self.muscles.append(
                        Muscle(self.chromats[i][j],
                               self.chromats[i][j+1],
                               parameters=self.init_func))
                if i+1 < len(self.chromats) and j < len(self.chromats[i+1]):
                    self.muscles.append(
                        Muscle(self.chromats[i][j],
                               self.chromats[i+1][j],
                               parameters=self.init_func))
                if (i+1 < len(self.chromats) and j + fac >= 0 and
                    j + fac < len(self.chromats[i+1])):
                    self.muscles.append(
                        Muscle(self.chromats[i][j],
                               self.chromats[i+1][j + fac],
                               parameters=self.init_func))

    def make_chain_net(self, noise: float) -> None:
        """Builds a 1D chain of N chromatophores with both ends held fixed.

        Args:
            noise: Standard deviation of the noise on rest positions.
        """
        self.chromats = []

        for i in range(self.N):
            anchor_pos = np.array([i * self.dx, 0.0])
            chrom = Chromatophore(zi=anchor_pos, noise=noise,
                                  parameters=self.init_func)
            self.chromats.append([chrom])

        for i in range(self.N - 1):
            self.muscles.append(
                Muscle(self.chromats[i][0], self.chromats[i+1][0],
                       parameters=self.init_func))

    def remove_actives(self, scale: float = 1) -> None:
        """Deletes a random fraction of the active muscles from self.muscles.

        Args:
            scale: Fraction of active muscles to delete.
        """
        startlen = len(self.muscles)
        np_actives = np.array(
            [spring for spring in self.muscles if spring.active])
        rng = np.random.rand(len(np_actives))
        todel = np_actives[rng < scale]
        self.muscles = [
            spring for spring in self.muscles if spring not in todel]
        ndel = startlen - len(self.muscles)
        print(f'Deleted {ndel/len(np_actives)*100}% ({ndel}) of active '
              'springs.')

    # Model functions. use_torch=True is the PyTorch path used during the
    # simulation. use_torch=False evaluates on NumPy arrays and is not used
    # by the simulator.

    def lamn(
        self,
        use_torch: bool = False,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Returns the rate of w: phi * cosh((v - vc) / (2 vd)).

        Args:
            use_torch: Whether to compute with PyTorch from v.
            v: Muscle voltages, used when use_torch is True.
        """
        if use_torch:
            phi_vals = self.cached_params['phi']
            vc_vals = self.cached_params['vc']
            vd_vals = self.cached_params['vd']
            return phi_vals * torch.cosh((v - vc_vals)/(2*vd_vals))
        return self.par['phi'] * np.cosh(
            (self.v - self.par['vc'])/(2*self.par['vd']))

    def ninf(
        self,
        use_torch: bool = False,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Returns the potassium activation (1 + tanh((v - vc) / vd)) / 2.

        Args:
            use_torch: Whether to compute with PyTorch from v.
            v: Muscle voltages, used when use_torch is True.
        """
        if use_torch:
            vc_vals = self.cached_params['vc']
            vd_vals = self.cached_params['vd']
            return (1. + torch.tanh((v - vc_vals)/vd_vals))/2
        return (1. + np.tanh((self.v - self.par['vc'])/self.par['vd']))/2

    def minf(
        self,
        use_torch: bool = False,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Returns the calcium activation (1 + tanh((v - va) / vb)) / 2.

        Args:
            use_torch: Whether to compute with PyTorch from v.
            v: Muscle voltages, used when use_torch is True.
        """
        if use_torch:
            va_vals = self.cached_params['va']
            vb_vals = self.cached_params['vb']
            return (1. + torch.tanh((v - va_vals)/vb_vals))/2
        return (1. + np.tanh((self.v - self.par['va'])/self.par['vb']))/2

    def gc(
        self,
        use_torch: bool = False,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Returns the stretch-sensitive Ca conductance per muscle.

        The conductance is gcmax * sigmoid(l; lth*L0, sig), where l is the
        current muscle length (the distance between its two endpoints).

        Args:
            use_torch: Whether to compute with PyTorch from y.
            y: Muscle endpoints, used when use_torch is True.
        """
        if use_torch:
            L = torch.sqrt(
                ((y[self.y_to_m[:, 0]] - y[self.y_to_m[:, 1]])**2).sum(dim=1))
            gcmax_vals = self.cached_params['gcmax']
            lth_vals = self.cached_params['lth']
            sig_vals = self.cached_params['sig']
            L0_vals = self.cached_params['L0']
            return gcmax_vals * sigmoid_torch(L, lth_vals * L0_vals, sig_vals)
        L = np.sqrt(
            ((self.y[self.y_to_m[:, 0]] - self.y[self.y_to_m[:, 1]])**2)
            .sum(axis=1))
        return self.par['gcmax'] * sigmoid(
            np.reshape(L, [-1, 1]), (self.par['lth']*self.L0),
            self.par['sig'])

    def ka(
        self,
        use_torch: bool = False,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Returns the active spring constant kmax * sigmoid(v; vth, vshp).

        Args:
            use_torch: Whether to compute with PyTorch from v.
            v: Muscle voltages, used when use_torch is True.
        """
        if use_torch:
            kmax_vals = self.cached_params['kmax']
            vth_vals = self.cached_params['vth']
            vshp_vals = self.cached_params['vshp']
            return kmax_vals * sigmoid_torch(v, vth_vals, vshp_vals)
        return self.par['kmax'] * sigmoid(
            np.reshape(self.v, [-1, 1]), self.par['vth'], self.par['vshp'])

    # Gap junctions.

    def sum_voltages_per_chromatophore(
        self,
        tempv: torch.Tensor,
    ) -> torch.Tensor:
        """Returns, per chromatophore, the summed voltage of its muscles.

        Args:
            tempv: Voltage at each muscle endpoint.
        """
        tempxv = torch.zeros(len(self.x), device=self.device)
        for x in range(len(self.x)):
            tempxv[x] = torch.sum(tempv[self.x_to_y[x]])
        return tempxv

    def sum_voltages_per_muscle(
        self,
        Vt: torch.Tensor,
        tempyv: torch.Tensor,
    ) -> torch.Tensor:
        """Returns, per muscle, tempyv summed over its two endpoints.

        Args:
            Vt: Muscle voltages (sets the output shape and type).
            tempyv: Value at each muscle endpoint.
        """
        tempgv = torch.zeros_like(Vt)
        for x in range(len(self.v)):
            y_indices = self.y_to_m[x]
            tempgv[x] = torch.sum(tempyv[y_indices])
        return tempgv

    def _apply_connection_removal(self) -> None:
        """Removes a random fraction of muscles by setting kp and kmax to 0."""
        if self.removal_seed is not None:
            random.seed(self.removal_seed)

        n_muscles = len(self.v)
        n_remove = int(n_muscles * self.connection_removal_fraction)
        muscles_to_sever = random.sample(range(n_muscles), n_remove)

        for m_idx in muscles_to_sever:
            self.cached_params['kp'][m_idx] = 0.0
            self.cached_params['kmax'][m_idx] = 0.0

        self.severed_muscles = muscles_to_sever

    def precompute_tempgv_neighbors(self) -> None:
        """Finds the gap junction neighbors of each muscle, to count nc_m.

        A neighbor shares a chromatophore with this muscle, and its far end
        is one lattice step (distance ~1) from this muscle's far end, so the
        two muscles form a triangle (the bow-tie pattern in Fig 1D). Uses
        noise-free lattice positions.
        """
        self.muscle_tempgv_neighbors = []

        for m_idx in range(len(self.v)):
            y1_idx, y2_idx = self.y_to_m[m_idx].cpu().numpy()
            chromat1_idx = self.y_to_x[y1_idx].item()
            chromat2_idx = self.y_to_x[y2_idx].item()

            neighbor_y_indices = []

            # Look from each end of the muscle.
            ends = [(y1_idx, chromat1_idx, chromat2_idx),
                    (y2_idx, chromat2_idx, chromat1_idx)]
            for y_idx, chromat_idx, other_chromat_idx in ends:
                target_chromat_pos = (
                    self.x_clean[other_chromat_idx].cpu().numpy())

                # Other muscles attached to this chromatophore.
                for other_y_idx in self.x_to_y[chromat_idx]:
                    if other_y_idx == y_idx:
                        continue

                    other_muscle_idx = self.m_to_y[other_y_idx].item()
                    other_y1, other_y2 = (
                        self.y_to_m[other_muscle_idx].cpu().numpy())

                    # Chromatophore at the far end of the other muscle.
                    if other_y1 == other_y_idx:
                        far_chromat_idx = self.y_to_x[other_y2].item()
                    else:
                        far_chromat_idx = self.y_to_x[other_y1].item()

                    far_chromat_pos = (
                        self.x_clean[far_chromat_idx].cpu().numpy())

                    dist = np.linalg.norm(target_chromat_pos - far_chromat_pos)
                    if abs(dist - 1.0) < 0.1:
                        neighbor_y_indices.append(other_y_idx)

            self.muscle_tempgv_neighbors.append(neighbor_y_indices)

    def precompute_nc_m_values(self) -> None:
        """Counts the gap junction neighbors of each muscle (nc_m)."""
        self.nc_m_precomputed = []
        for m_idx in range(len(self.v)):
            self.nc_m_precomputed.append(
                len(self.muscle_tempgv_neighbors[m_idx]))
        self.nc_m_precomputed = torch.tensor(
            self.nc_m_precomputed, dtype=torch.float32, device=self.device)

    # Compilation.

    def compile_net(self) -> int | None:
        """Flattens the network into tensors and builds the update function.

        Index maps:
            x_to_y[i]: Endpoints on chromatophore i.
            y_to_x[j]: Chromatophore that endpoint j sits on.
            m_to_y[j]: Muscle that endpoint j belongs to.
            y_to_m[m]: The two endpoints of muscle m.
            y_to_y[j]: Endpoint at the other end of the same muscle.

        Returns:
            0 if PyTorch is not available, otherwise None.
        """
        # Names below follow the paper's notation (Eqs 2-6).
        # pylint: disable=invalid-name
        if not PYTORCH:
            print('No PyTorch available!')
            return 0

        # Per-muscle and per-chromatophore parameters.
        mparam = {
            'phi', 'vc', 'vd', 'va', 'vb', 'Iapp', 'gl', 'gk', 'vl', 'vk',
            'gca', 'vca', 'ggap', 'gcmax', 'lth', 'sig', 'kmax', 'vshp',
            'kp', 'vth', 'ls',
        }
        cparam = {'kc', 'ks'}

        self.x = []
        self.x_clean = []
        self.y = []
        self.z = []
        self.w = []
        self.v = []
        self.L0 = []
        self.x_to_y = []
        self.y_to_x = []
        self.m_to_y = []
        self.y_to_y = []
        self.y_to_m = []

        self.par = {}
        for k in defaults.DEFAULT2D:
            self.par[k] = []

        for c in range(len(self.chromats)):
            for k in range(len(self.chromats[c])):
                self.chromats[c][k].index = len(self.x)

                self.x.append(self.chromats[c][k].x)
                self.x_clean.append(self.chromats[c][k].x_clean)
                self.z.append(self.chromats[c][k].z)
                self.x_to_y.append([])

                for j in cparam:
                    self.par[j].append(self.chromats[c][k].par[j])

                for y in self.chromats[c][k].y:
                    self.y_to_x.append(len(self.x)-1)
                    self.x_to_y[-1].append(len(self.y))
                    y[2] = len(self.y)
                    self.y.append(y[0])

                    # Register the muscle the first time we see it.
                    if y[1].index == -1:
                        y[1].index = len(self.w)
                        self.w.append(y[1].W)
                        self.v.append(y[1].V)
                        self.L0.append(y[1].L0)
                        self.ac.append(y[1].active)

                        for j in mparam:
                            self.par[j].append(y[1].par[j])
                        self.y_to_m.append([-1, -1])

                    self.m_to_y.append(y[1].index)
                    self.y_to_y.append(-1)

                    # Link the two endpoints once both are registered.
                    if y[1].other_att(self.chromats[c][k])[2] == -1:
                        continue
                    else:
                        my_index = y[2]
                        other_index = y[1].other_att(self.chromats[c][k])[2]
                        self.y_to_y[my_index] = other_index
                        self.y_to_y[other_index] = my_index
                        self.y_to_m[y[1].index] = [my_index, other_index]

        # Convert to tensors and move to the device.
        self.y_to_y = torch.tensor(
            self.y_to_y, dtype=torch.long, device=self.device)
        self.y_to_x = torch.tensor(
            self.y_to_x, dtype=torch.long, device=self.device)
        self.y_to_m = torch.tensor(
            self.y_to_m, dtype=torch.long, device=self.device)
        self.m_to_y = torch.tensor(
            self.m_to_y, dtype=torch.long, device=self.device)

        self.x = torch.tensor(
            np.array(self.x), dtype=torch.float32, device=self.device)
        self.x_clean = torch.tensor(
            np.array(self.x_clean), dtype=torch.float32, device=self.device)
        self.z = torch.tensor(
            np.array(self.z), dtype=torch.float32, device=self.device)
        self.y = torch.tensor(
            np.array(self.y), dtype=torch.float32, device=self.device)
        self.w = torch.tensor(self.w, dtype=torch.float32, device=self.device)
        self.v = torch.tensor(self.v, dtype=torch.float32, device=self.device)
        self.L0 = torch.tensor(
            self.L0, dtype=torch.float32, device=self.device)

        # Parameters to tensors. ks and kc are per chromatophore with shape
        # (n, 1); all others are per muscle with shape (n_muscles,).
        self.cached_params = {}
        for k in self.par:
            param_data = self.par[k]
            try:
                if isinstance(param_data, list):
                    values = []
                    for p in param_data:
                        if hasattr(p, 'item'):
                            values.append(p.item())
                        elif isinstance(p, (int, float)):
                            values.append(float(p))
                        else:
                            values.append(float(p))
                    param_tensor = torch.tensor(
                        values, dtype=torch.float32, device=self.device)
                elif isinstance(param_data, (int, float)):
                    param_tensor = torch.tensor(
                        [param_data], dtype=torch.float32, device=self.device)
                elif hasattr(param_data, 'shape'):
                    param_tensor = torch.tensor(
                        param_data, dtype=torch.float32, device=self.device)
                else:
                    param_tensor = torch.tensor(
                        [float(param_data)], dtype=torch.float32,
                        device=self.device)

                if k in ['ks', 'kc']:
                    if param_tensor.shape[0] == 1:
                        self.cached_params[k] = (
                            param_tensor.expand(len(self.x)).unsqueeze(1))
                    else:
                        self.cached_params[k] = param_tensor.unsqueeze(1)
                else:
                    if param_tensor.shape[0] == 1:
                        self.cached_params[k] = (
                            param_tensor.expand(len(self.v)))
                    else:
                        self.cached_params[k] = param_tensor

                self.par[k] = param_tensor

            except (TypeError, ValueError, IndexError):
                print(f'Error converting parameter {k}: {param_data}, '
                      f'type: {type(param_data)}')
                fallback_val = (2.0 if k == 'kc'
                                else defaults.DEFAULT2D.get(k, 1.0))
                if k in ['ks', 'kc']:
                    self.cached_params[k] = torch.full(
                        (len(self.x), 1), fallback_val, device=self.device)
                else:
                    self.cached_params[k] = torch.full(
                        (len(self.v),), fallback_val, device=self.device)
                self.par[k] = torch.tensor([fallback_val], device=self.device)

        self.cached_params['L0'] = self.L0

        # Endpoint lists as tensors (for indexing) and as NumPy arrays.
        self.x_to_y_tensors = []
        for k in range(len(self.x_to_y)):
            self.x_to_y_tensors.append(torch.tensor(
                self.x_to_y[k], dtype=torch.long, device=self.device))

        for k in range(len(self.x_to_y)):
            self.x_to_y[k] = np.array(self.x_to_y[k])

        self.precompute_tempgv_neighbors()
        self.precompute_nc_m_values()

        def get_updates_base(Xt, Yt, Wt, Vt, Tt, DT):
            """Takes one Euler step of the full model, edges not fixed."""

            # Chromatophore positions (Eq 2):
            # dx/dt = ks (z - x) + kc sum_mu (y_mu - x).
            dx_zterm = -(Xt - self.z)

            dx_yterm = torch.zeros_like(Xt)
            for x in range(len(self.x)):
                y_connections = self.x_to_y_tensors[x]
                if len(y_connections) > 0:
                    dx_yterm[x] = -torch.sum(
                        Xt[x:x+1] - Yt[y_connections], dim=0)

            ks_vals = self.cached_params['ks']
            kc_vals = self.cached_params['kc']

            if ks_vals.shape[1] == 1:
                ks_vals = ks_vals.expand(-1, 2)
            if kc_vals.shape[1] == 1:
                kc_vals = kc_vals.expand(-1, 2)

            tempdx = ks_vals * dx_zterm + kc_vals * dx_yterm

            # Recovery variable (Eq 5): dw/dt = lambda_n(v) (n_inf(v) - w).
            ninf_vals = self.ninf(use_torch=True, v=Vt)
            lamn_vals = self.lamn(use_torch=True, v=Vt)
            tempdw = (ninf_vals - Wt) * lamn_vals

            # Muscle endpoints (Eq 3): pulled toward their own chromatophore
            # by kc...
            kc_indexed = self.cached_params['kc'][self.y_to_x]
            if kc_indexed.shape[1] == 1:
                kc_indexed = kc_indexed.expand(-1, 2)
            dy_xterm = -kc_indexed * (Yt - Xt[self.y_to_x])

            # ...and along the muscle by the passive (kp) and active (ka)
            # springs.
            ka_values = self.ka(use_torch=True, v=Vt)
            KA = ka_values[self.m_to_y].unsqueeze(1)

            kp_vals = self.cached_params['kp']
            if kp_vals.dim() == 1:
                kp_vals = kp_vals.unsqueeze(1)
            KP = kp_vals[self.m_to_y]

            # Vector to the other endpoint, and the muscle length.
            ydiff = Yt - Yt[self.y_to_y]
            ynorm = torch.sqrt((ydiff*ydiff).sum(dim=1, keepdim=True))
            ynorm_safe = torch.maximum(
                ynorm, torch.tensor(1e-10, device=self.device))
            L0 = self.L0[self.m_to_y].unsqueeze(1)

            ls_vals = self.cached_params['ls']
            if ls_vals.dim() == 1:
                ls_vals = ls_vals.unsqueeze(1)
            LS = ls_vals[self.m_to_y]

            # Expansion is faster than retraction: muscles lengthening past
            # their rest length have reduced passive and active stiffness.
            retracting = ynorm > L0
            KP_mod = torch.where(retracting, KP * 0.0005, KP)
            KA_mod = torch.where(retracting, KA * 0.5, KA)

            dy_mterm = (-(KP_mod*(ynorm - L0) + KA_mod * (ynorm - L0*LS))
                        * ydiff / ynorm_safe)

            tempdy = dy_xterm + dy_mterm

            # Membrane voltage (Eq 4): applied current, leak, potassium,
            # calcium and gap junction currents.
            m_pos = (Yt[self.y_to_m[:, 0]] + Yt[self.y_to_m[:, 1]])/2
            iapp = self.iapp(m_pos, Tt)

            gl_vals = self.cached_params['gl']
            vl_vals = self.cached_params['vl']
            leak = gl_vals * (vl_vals - Vt)

            gk_vals = self.cached_params['gk']
            vk_vals = self.cached_params['vk']
            pota = gk_vals * Wt * (vk_vals - Vt)

            # Voltage-gated plus stretch-sensitive calcium.
            gca_vals = self.cached_params['gca']
            vca_vals = self.cached_params['vca']
            minf_vals = self.minf(use_torch=True, v=Vt)
            gc_vals = self.gc(use_torch=True, y=Yt)
            calc = (gca_vals * minf_vals + gc_vals) * (vca_vals - Vt)

            # Gap junctions (Eq 6): I_gap = -ggap (nc_m V - coupled voltages).
            # tempv is the voltage at each endpoint, tempxv sums it per
            # chromatophore, tempyv leaves out the muscle itself, and tempgv
            # sums tempyv over both ends of each muscle.
            tempv = Vt[self.m_to_y]
            tempxv = self.sum_voltages_per_chromatophore(tempv)
            tempyv = tempxv[self.y_to_x] - tempv
            tempgv = self.sum_voltages_per_muscle(Vt, tempyv)
            nc_m = self.nc_m_precomputed

            ggap_vals = self.cached_params['ggap']
            gap = -ggap_vals * (nc_m*Vt - tempgv)

            tempdv = iapp + leak + pota + calc + gap

            outx = Xt + DT*tempdx
            outy = Yt + DT*tempdy
            outw = Wt + DT*tempdw
            outv = Vt + DT*tempdv

            return outx, outy, outw, outv, Tt + DT

        def get_updates_euler(Xt, Yt, Wt, Vt, Tt, DT):
            """Takes one Euler step with boundary chromatophores fixed."""
            outX, outY, outW, outV, Tt_DT = get_updates_base(
                Xt, Yt, Wt, Vt, Tt, DT)
            outX = torch.where(self.edge_mask.unsqueeze(1), Xt, outX)
            return outX, outY, outW, outV, Tt_DT

        def get_updates_rk4(Xt, Yt, Wt, Vt, Tt, DT):
            """Takes one RK4 step with boundary chromatophores fixed.

            Each slope is recovered from one Euler step.
            """
            # k1 at the current state.
            X1, Y1, W1, V1, _ = get_updates_base(Xt, Yt, Wt, Vt, Tt, DT)
            dX1 = (X1 - Xt) / DT
            dY1 = (Y1 - Yt) / DT
            dW1 = (W1 - Wt) / DT
            dV1 = (V1 - Vt) / DT

            # k2 at the midpoint, using k1.
            X_mid2 = Xt + 0.5*DT*dX1
            Y_mid2 = Yt + 0.5*DT*dY1
            W_mid2 = Wt + 0.5*DT*dW1
            V_mid2 = Vt + 0.5*DT*dV1
            X2, Y2, W2, V2, _ = get_updates_base(
                X_mid2, Y_mid2, W_mid2, V_mid2, Tt + 0.5*DT, DT)
            dX2 = (X2 - X_mid2) / DT
            dY2 = (Y2 - Y_mid2) / DT
            dW2 = (W2 - W_mid2) / DT
            dV2 = (V2 - V_mid2) / DT

            # k3 at the midpoint, using k2.
            X_mid3 = Xt + 0.5*DT*dX2
            Y_mid3 = Yt + 0.5*DT*dY2
            W_mid3 = Wt + 0.5*DT*dW2
            V_mid3 = Vt + 0.5*DT*dV2
            X3, Y3, W3, V3, _ = get_updates_base(
                X_mid3, Y_mid3, W_mid3, V_mid3, Tt + 0.5*DT, DT)
            dX3 = (X3 - X_mid3) / DT
            dY3 = (Y3 - Y_mid3) / DT
            dW3 = (W3 - W_mid3) / DT
            dV3 = (V3 - V_mid3) / DT

            # k4 at the endpoint, using k3.
            X_end = Xt + DT*dX3
            Y_end = Yt + DT*dY3
            W_end = Wt + DT*dW3
            V_end = Vt + DT*dV3
            X4, Y4, W4, V4, _ = get_updates_base(
                X_end, Y_end, W_end, V_end, Tt + DT, DT)
            dX4 = (X4 - X_end) / DT
            dY4 = (Y4 - Y_end) / DT
            dW4 = (W4 - W_end) / DT
            dV4 = (V4 - V_end) / DT

            outX = Xt + (DT/6) * (dX1 + 2*dX2 + 2*dX3 + dX4)
            outY = Yt + (DT/6) * (dY1 + 2*dY2 + 2*dY3 + dY4)
            outW = Wt + (DT/6) * (dW1 + 2*dW2 + 2*dW3 + dW4)
            outV = Vt + (DT/6) * (dV1 + 2*dV2 + 2*dV3 + dV4)

            # Hold boundary chromatophores fixed.
            outX = torch.where(self.edge_mask.unsqueeze(1), Xt, outX)

            return outX, outY, outW, outV, Tt + DT

        if self.integration_method.lower() == 'rk4':
            self.get_updates = get_updates_rk4
        elif self.integration_method.lower() == 'euler':
            self.get_updates = get_updates_euler
        else:
            raise ValueError(
                f'Unknown integration_method: {self.integration_method!r}. '
                "Use 'euler' or 'rk4'.")

        # Fixed chromatophores: both ends of a chain, or the perimeter of a
        # hex lattice.
        self.edge_mask = torch.zeros(
            len(self.x), dtype=torch.bool, device=self.device)
        if self.layout == 'chain':
            self.edge_mask[0] = True
            self.edge_mask[-1] = True
        else:
            idx = 0
            for c in range(len(self.chromats)):
                for k in range(len(self.chromats[c])):
                    if self.chromats[c][k].is_edge:
                        self.edge_mask[idx] = True
                    idx += 1

        self.compiled = True
        return None

    # Running and saving.

    def get_all_radii(
        self,
        xi: torch.Tensor | np.ndarray | None = None,
        yi: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Returns the radius of every chromatophore.

        The radius is the norm of the mean absolute offset between a
        chromatophore's position and its muscle endpoints.

        Args:
            xi: Chromatophore positions. Defaults to the current positions.
            yi: Muscle endpoints. Defaults to the current endpoints.
        """
        if xi is None or yi is None or len(xi) == 0 or len(yi) == 0:
            if isinstance(self.x, torch.Tensor):
                x = self.x.clone()
            else:
                x = self.x.copy()
            if isinstance(self.y, torch.Tensor):
                y = self.y.clone()
            else:
                y = self.y.copy()
        else:
            x = xi
            y = yi

        if isinstance(x, torch.Tensor):
            x = x.reshape(-1, 1, 2)
            out = []
            for z in range(len(x)):
                out.append(torch.norm(torch.mean(
                    torch.abs(x[z].reshape(1, 2) - y[self.x_to_y[z]]),
                    dim=0)))
            return torch.stack(out)
        x = np.reshape(x, [-1, 1, 2])
        out = []
        for z in range(len(x)):
            out.append(np.linalg.norm(np.mean(
                np.abs(np.reshape(x[z], [1, 2]) - y[self.x_to_y[z]]),
                axis=0)))
        return np.array(out)

    def run(
        self,
        steps: int,
        name: str = 'test',
        dt: float = .01,
        save_freq: int = 10,
    ) -> list[torch.Tensor] | None:
        """Simulates the network and saves the result to files.

        Writes the x_pos_, v_pos_, z_pos_ and time_ files for `name` (see
        save_chr_files).

        Args:
            steps: Number of time steps.
            name: Name used in the output file names.
            dt: Time step.
            save_freq: Save the state every save_freq steps.

        Returns:
            [X, Y, W, V] at the saved times, or None if the network is not
            compiled.
        """
        if not PYTORCH or not self.compiled:
            print("Either PyTorch is not available or you haven't compiled "
                  'yet.')
            return None

        current_x = self.x.clone()
        current_y = self.y.clone()
        current_w = self.w.clone()
        current_v = self.v.clone()
        current_t = torch.tensor(0.0, device=self.device)
        dt_tensor = torch.tensor(dt, device=self.device)

        # Saved states are kept on the CPU.
        num_saves = 1 + steps // save_freq
        x_states = torch.zeros(num_saves, *current_x.shape, device='cpu')
        y_states = torch.zeros(num_saves, *current_y.shape, device='cpu')
        w_states = torch.zeros(num_saves, *current_w.shape, device='cpu')
        v_states = torch.zeros(num_saves, *current_v.shape, device='cpu')

        x_states[0] = current_x.cpu()
        y_states[0] = current_y.cpu()
        w_states[0] = current_w.cpu()
        v_states[0] = current_v.cpu()

        save_idx = 1

        # Use mixed precision if on GPU.
        use_amp = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        start_time = time.time()
        for step in range(steps):
            if use_amp:
                with torch.cuda.amp.autocast():
                    (current_x, current_y, current_w, current_v,
                     current_t) = self.get_updates(
                         current_x, current_y, current_w, current_v,
                         current_t, dt_tensor)
            else:
                (current_x, current_y, current_w, current_v,
                 current_t) = self.get_updates(
                     current_x, current_y, current_w, current_v,
                     current_t, dt_tensor)

            if (step + 1) % save_freq == 0:
                x_states[save_idx] = current_x.cpu()
                y_states[save_idx] = current_y.cpu()
                w_states[save_idx] = current_w.cpu()
                v_states[save_idx] = current_v.cpu()
                save_idx += 1

            if (step + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                eta = (steps - step - 1) / steps_per_sec
                print(f'Step {step+1}/{steps}, {steps_per_sec:.1f} '
                      f'steps/sec, ETA: {eta:.1f}s')

        vals = [x_states, y_states, w_states, v_states]

        self.save_chr_files(name, vals[0], vals[1], vals[2], vals[3], skip=1)

        time_points = [0] + [
            i * dt for i in range(save_freq, steps + 1, save_freq)]
        tvec = np.array(time_points)
        np.savetxt(f'./time_{name}.csv', tvec)

        return vals

    def save_chr_files(
        self,
        name: str,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        w: torch.Tensor | np.ndarray,
        v: torch.Tensor | np.ndarray,
        skip: int = 1,
    ) -> None:
        """Writes a simulation to text files.

        x_pos_{name}.csv: one row per saved time, [x, y, radius] per
            chromatophore.
        v_pos_{name}.csv: one row per saved time, [x, y, v] per muscle, at
            the muscle midpoint.
        z_pos_{name}.csv: chromatophore rest positions.

        Args:
            name: Name used in the output file names.
            x: Chromatophore positions at the saved times.
            y: Muscle endpoints at the saved times.
            w: Gating variables at the saved times (not saved).
            v: Muscle voltages at the saved times.
            skip: Not used.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            v = v.detach().cpu().numpy()

        x_save = x
        y_save = y
        v_save = v

        # Radius of every chromatophore at every saved time.
        r_list = []
        for t_idx in range(len(x_save)):
            r_t = self.get_all_radii(
                torch.tensor(x_save[t_idx], device=self.device),
                torch.tensor(y_save[t_idx], device=self.device))
            if isinstance(r_t, torch.Tensor):
                r_t = r_t.detach().cpu().numpy()
            r_list.append(r_t)

        r = np.array(r_list)
        if r.ndim == 2:
            r = r.reshape(r.shape[0], -1, 1)

        x_out = np.concatenate([x_save, r], axis=2)
        x_out = x_out.reshape(x_out.shape[0], -1)

        # Muscle midpoints at every saved time.
        mpos_list = []
        for t_idx in range(len(y_save)):
            y_tensor = torch.tensor(y_save[t_idx], device=self.device)
            mpos_t = (y_tensor[self.y_to_m[:, 0]]
                      + y_tensor[self.y_to_m[:, 1]])/2
            if isinstance(mpos_t, torch.Tensor):
                mpos_t = mpos_t.detach().cpu().numpy()
            mpos_list.append(mpos_t)

        mpos = np.array(mpos_list)

        v_reshaped = v_save.reshape(len(v_save), -1, 1)
        v_out = np.concatenate([mpos, v_reshaped], axis=2)
        v_out = v_out.reshape(v_out.shape[0], -1)

        if x_out.ndim == 1:
            x_out = x_out.reshape(1, -1)
        if v_out.ndim == 1:
            v_out = v_out.reshape(1, -1)

        np.savetxt(f'./x_pos_{name}.csv', x_out, delimiter=' ', fmt='%.6e')
        np.savetxt(f'./v_pos_{name}.csv', v_out, delimiter=' ', fmt='%.6e')

        if isinstance(self.z, torch.Tensor):
            z_save = self.z.detach().cpu().numpy()
        else:
            z_save = self.z
        np.savetxt(f'./z_pos_{name}.csv', z_save, delimiter=' ', fmt='%.6e')

    # Initial conditions.

    def set_vi(self, vi_func: Callable[[Any], Any]) -> None:
        """Sets the initial muscle voltages from vi_func(muscle positions).

        vi_func is first called with all muscle positions at once, as a
        tensor of shape (n_muscles, 2). If that fails, it is called with one
        position at a time, as a NumPy array of shape (2,).

        Args:
            vi_func: Function of muscle positions that returns voltages.

        Raises:
            TypeError: If vi_func is not callable.
        """
        if not callable(vi_func):
            raise TypeError('vi_func must be callable.')

        positions = self.get_m_pos()
        if isinstance(positions, torch.Tensor):
            positions_tensor = positions
        else:
            positions_tensor = torch.tensor(positions, device=self.device)

        try:
            new_voltages = vi_func(positions_tensor)
        except (TypeError, IndexError, ValueError, RuntimeError):
            # vi_func does not take all positions at once.
            if isinstance(positions, torch.Tensor):
                positions_np = positions.cpu().numpy()
            else:
                positions_np = positions
            new_voltages = torch.tensor(
                [vi_func(pos) for pos in positions_np],
                dtype=torch.float32, device=self.device)
        else:
            if not isinstance(new_voltages, torch.Tensor):
                new_voltages = torch.tensor(
                    new_voltages, dtype=torch.float32, device=self.device)
            elif new_voltages.device != self.device:
                new_voltages = new_voltages.to(self.device)

        self.v = new_voltages.flatten()

    def set_xi(self, xi: Callable[[np.ndarray], np.ndarray]) -> None:
        """Moves each chromatophore and its endpoints to xi(rest position).

        Args:
            xi: Function of one rest position, shape (2,), that returns the
                new position.
        """
        for x in range(len(self.x)):
            if isinstance(self.z, torch.Tensor):
                z_val = self.z[x].cpu().numpy()
            else:
                z_val = self.z[x]
            new_pos = xi(z_val)
            if isinstance(self.x, torch.Tensor):
                self.x[x] = torch.tensor(new_pos, device=self.device)
            else:
                self.x[x] = new_pos.copy()
            if isinstance(self.y, torch.Tensor):
                self.y[self.x_to_y[x]] = torch.tensor(
                    new_pos, device=self.device).reshape(1, 2)
            else:
                self.y[self.x_to_y[x]] = np.reshape(new_pos, [1, 2]).copy()

    def reset_net(
        self,
        xi: Callable[[np.ndarray], np.ndarray] | None = None,
        vi: Callable[[np.ndarray], float] | None = None,
    ) -> int:
        """Resets the network state.

        Muscle endpoints start on their chromatophore, and w is reset to
        DEFAULT2D['winit'].

        Args:
            xi: Function of one rest position that returns the starting
                position. None starts every chromatophore at rest.
            vi: Function of one muscle position that returns its voltage.
                None uses DEFAULT2D['vinit'].

        Returns:
            1 on success, 0 if the network is not compiled.
        """
        if not self.compiled:
            print("must compile network before using 'reset_net'")
            return 0

        if xi is None:
            if isinstance(self.z, torch.Tensor):
                self.x = self.z.clone()
            else:
                self.x = np.copy(self.z)
        else:
            if isinstance(self.z, torch.Tensor):
                self.x = torch.stack(
                    [torch.tensor(xi(z.cpu().numpy()), device=self.device)
                     for z in self.z])
            else:
                self.x = torch.tensor(
                    [xi(z) for z in self.z], device=self.device)

        if isinstance(self.x, torch.Tensor):
            self.y = self.x[self.y_to_x].clone()
        else:
            self.y = torch.tensor(self.x[self.y_to_x], device=self.device)

        if vi is None:
            self.v = torch.ones_like(self.v) * defaults.DEFAULT2D['vinit']
        else:
            loc = (self.y[self.y_to_m[:, 0]] + self.y[self.y_to_m[:, 1]])/2
            if isinstance(loc, torch.Tensor):
                self.v = torch.stack(
                    [torch.tensor(vi(x.cpu().numpy()), device=self.device)
                     for x in loc])
            else:
                self.v = torch.tensor(
                    [vi(x) for x in loc], device=self.device)

        self.w = torch.ones_like(self.w) * defaults.DEFAULT2D['winit']

        return 1