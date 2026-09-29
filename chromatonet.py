"""ChromatoNet: a biophysical model of chromatophore networks in cephalopod skin.

Each chromatophore is a point tethered to its rest position by a skin spring (ks).
Radial muscles connect neighboring chromatophores. Each muscle has one endpoint
on each chromatophore. An endpoint is pulled toward its own chromatophore by the
chromatophore-muscle coupling (kc), and along the muscle by a passive spring (kp)
and a voltage-gated active spring (ka). Muscle voltage follows Morris-Lecar
dynamics with a stretch-sensitive calcium conductance and gap junction coupling
between muscles. Boundary chromatophores are held fixed.

State variables:
    x: chromatophore positions, shape (n_chromatophores, 2)
    z: chromatophore rest positions, shape (n_chromatophores, 2)
    y: muscle endpoints (two per muscle), shape (n_endpoints, 2)
    v: muscle voltages, shape (n_muscles,)
    w: potassium gating variables, shape (n_muscles,)
"""

import torch
import torch.nn.functional as F
PYTORCH = True

import numpy as np
import time
import random
import sys
sys.setrecursionlimit(50000)

from defaults import DEFAULT2D

# Enable optimization flags
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = True


def sigmoid(x, th, sh):
    return 1. / (1. + np.exp(-(x - th)/sh))


def sigmoid_torch(x, th, sh):
    return torch.sigmoid((x - th)/sh)


class muscle:
    """A radial muscle connecting two chromatophores.

    The muscle has one endpoint on each chromatophore. Its parameters are
    evaluated at the midpoint of the two rest positions.
    """

    def __init__(self, chroma, chromb, parameters, active=True):
        self.index = -1  # Set in compile_net
        self.active = active
        self.chroma = chroma
        self.chromb = chromb

        self.par = {}
        for k in parameters.keys():
            self.par[k] = parameters[k]((self.chroma.z + self.chromb.z)/2)

        # Position of this muscle in each chromatophore's endpoint list
        self.a_index = len(self.chroma.y)
        self.b_index = len(self.chromb.y)

        # Endpoint entry: [position, muscle, global endpoint index (set in compile_net)]
        self.chroma.y.append([np.copy(self.chroma.x), self, -1])
        self.chromb.y.append([np.copy(self.chromb.x), self, -1])

        self.V = self.par["vinit"]
        self.W = self.par["winit"]
        self.L0 = np.linalg.norm(chroma.z - chromb.z)  # Rest length: distance between rest positions

    def other_att(self, ch):
        """Return the endpoint entry at the other end of this muscle."""
        if ch == self.chroma:
            return self.chromb.y[self.b_index]
        elif ch == self.chromb:
            return self.chroma.y[self.a_index]
        else:
            print("I am not your muscle!")
            exit()


class chromatophore:
    """A single chromatophore.

    zi is the lattice position, z is the rest position (lattice position plus
    spatial noise), x is the current position, and x_clean is the noise-free
    lattice position used to find gap junction neighbors.
    """

    def __init__(self, zi, parameters, noise=0, is_edge=False):
        self.index = -1  # Set in compile_net
        self.is_edge = is_edge  # Boundary chromatophores are held fixed
        self.z = np.copy(parameters["z"](zi) + np.random.randn(2)*noise)

        self.par = {}
        for k in parameters.keys():
            self.par[k] = np.copy(parameters[k](self.z))

        self.x = np.copy(self.z)
        self.x_clean = np.copy(parameters["z"](zi))
        self.y = []  # Muscle endpoints, filled in by muscle()


class chromatophore_network:
    def __init__(self, N, M=None, dx=1., layout="hex", noise=0, iapp=-1, PINITS={},
                 integration_method='euler', connection_removal_fraction=0.0, removal_seed=None):
        """Build and compile a network of chromatophores and muscles.

        Args:
            N: number of columns (hex) or chromatophores (chain)
            M: number of rows (hex only), defaults to N
            dx: lattice spacing
            layout: 'hex' or 'chain'
            noise: standard deviation of the frozen Gaussian noise added to rest positions
            iapp: applied current, a function iapp(muscle_positions, t).
                  -1 uses the constant DEFAULT2D['Iapp'] everywhere
            PINITS: dict of functions f(position) that override DEFAULT2D values.
                    'z' maps lattice positions to rest positions
            integration_method: 'euler' or 'rk4'
            connection_removal_fraction: fraction of muscles to remove (0 to 1)
            removal_seed: seed for choosing which muscles to remove (None for random)
        """
        self.chromats = []
        self.muscles = []

        # Every parameter is a function of position; defaults are constants
        self.init_func = {}
        for k in DEFAULT2D.keys():
            def constructor(k):
                def f(x):
                    return DEFAULT2D[k]
                return f
            self.init_func[k] = constructor(k)

        for k in PINITS.keys():
            self.init_func[k] = PINITS[k]

        if "z" not in PINITS.keys():
            self.init_func["z"] = lambda x: x

        self.layout = layout
        self.N = N
        self.M = M if M is not None else N
        self.dx = dx

        if layout == "hex":
            self.make_hex_net(noise=noise)
        elif layout == "chain":
            self.make_chain_net(noise=noise)
        else:
            raise ValueError(f"Unknown layout: {layout}. Use 'hex' or 'chain'")

        # Placeholders, filled in by compile_net
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
        print(f"Using device: {self.device}")

        # Force deterministic behavior
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if iapp == -1:
            self.IAPP = self.iapp_default
        else:
            self.IAPP = iapp

        self.integration_method = integration_method
        self.connection_removal_fraction = connection_removal_fraction
        self.removal_seed = removal_seed

        self.compile_net()
        if self.connection_removal_fraction > 0:
            self._apply_connection_removal()

    def iapp_default(self, x, t):
        """Default applied current: the constant DEFAULT2D['Iapp'] on every muscle."""
        return torch.ones(len(self.v), device=self.device) * DEFAULT2D["Iapp"]

    def get_m_pos(self, YV=[]):
        """Muscle positions, taken as the midpoint of the two endpoints."""
        if len(YV) == 0:
            Y = self.y
        else:
            Y = YV

        out = []
        for k in range(len(self.v)):
            out.append((Y[self.y_to_m[k, 0]] + Y[self.y_to_m[k, 1]])/2)

        return torch.stack(out) if isinstance(Y, torch.Tensor) else np.array(out)

    # Network layouts

    def make_hex_net(self, noise):
        """Hexagonal lattice.

        Odd rows are shifted by dx/2 and hold one fewer chromatophore.
        Chromatophores on the perimeter are marked as edge and held fixed.
        """
        dx = np.array([self.dx, 0])
        dy = np.array([0., np.sqrt(3)/2])*self.dx
        self.chromats = []

        for i in range(self.M):
            self.chromats.append([])
            bx = np.array([((i%2)*self.dx)/2, 0])
            for j in range(self.N - (i % 2)):
                is_edge = (
                    i == 0 or                           # Bottom row
                    i == self.M - 1 or                  # Top row
                    j == 0 or                           # Left edge
                    j == self.N - (i % 2) - 1           # Right edge (odd rows are shorter)
                )
                self.chromats[i].append(chromatophore(zi=i*dy + bx + j*dx, noise=noise, parameters=self.init_func, is_edge=is_edge))

        # Muscles to the right neighbor, the neighbor above, and the diagonal neighbor above
        for i in range(self.M):
            fac = ((i%2)*2 - 1)
            for j in range(self.N - (i % 2)):
                if j+1 < len(self.chromats[i]):
                    self.muscles.append(muscle(self.chromats[i][j], self.chromats[i][j+1], parameters=self.init_func))
                if i+1 < len(self.chromats) and j < len(self.chromats[i+1]):
                    self.muscles.append(muscle(self.chromats[i][j], self.chromats[i+1][j], parameters=self.init_func))
                if i+1 < len(self.chromats) and j + fac >= 0 and j + fac < len(self.chromats[i+1]):
                    self.muscles.append(muscle(self.chromats[i][j], self.chromats[i+1][j + fac], parameters=self.init_func))

    def make_chain_net(self, noise):
        """1D chain of N chromatophores. Both end chromatophores are held fixed."""
        self.chromats = []

        for i in range(self.N):
            anchor_pos = np.array([i * self.dx, 0.0])
            chrom = chromatophore(zi=anchor_pos, noise=noise, parameters=self.init_func)
            self.chromats.append([chrom])

        for i in range(self.N - 1):
            self.muscles.append(
                muscle(self.chromats[i][0], self.chromats[i+1][0], parameters=self.init_func)
            )

    def remove_actives(self, scale=1):
        """Delete a random fraction (scale) of the active muscles from self.muscles."""
        startlen = len(self.muscles)
        np_actives = np.array([spring for spring in self.muscles if spring.active])
        rng = np.random.rand(len(np_actives))
        todel = np_actives[rng < scale]
        self.muscles = [spring for spring in self.muscles if spring not in todel]
        ndel = startlen - len(self.muscles)
        print('Deleted {}% ({}) of active springs.'.format(ndel/len(np_actives)*100, ndel))

    # Model functions.
    # COMPILE=True is the PyTorch path used during simulation.
    # COMPILE=False evaluates on NumPy arrays and is not used by the simulator.

    def lamn(self, COMPILE=False, v=-1):
        """Rate of w: lambda_n(v) = phi * cosh((v - vc) / (2 vd))."""
        if COMPILE:
            phi_vals = self.cached_params['phi']
            vc_vals = self.cached_params['vc']
            vd_vals = self.cached_params['vd']
            return phi_vals * torch.cosh((v - vc_vals)/(2*vd_vals))
        else:
            return self.par['phi'] * np.cosh((self.v - self.par['vc'])/(2*self.par['vd']))

    def ninf(self, COMPILE=False, v=-1):
        """Potassium activation: n_inf(v) = (1 + tanh((v - vc) / vd)) / 2."""
        if COMPILE:
            vc_vals = self.cached_params['vc']
            vd_vals = self.cached_params['vd']
            return (1. + torch.tanh((v - vc_vals)/vd_vals))/2
        else:
            return (1. + np.tanh((self.v - self.par['vc'])/self.par['vd']))/2

    def minf(self, COMPILE=False, v=-1):
        """Calcium activation: m_inf(v) = (1 + tanh((v - va) / vb)) / 2."""
        if COMPILE:
            va_vals = self.cached_params['va']
            vb_vals = self.cached_params['vb']
            return (1. + torch.tanh((v - va_vals)/vb_vals))/2
        else:
            return (1. + np.tanh((self.v - self.par['va'])/self.par['vb']))/2

    def gc(self, COMPILE=False, y=-1):
        """Stretch-sensitive calcium conductance: gcmax * sigmoid(l; lth*L0, sig).

        l is the current muscle length (distance between its two endpoints).
        """
        if COMPILE:
            L = torch.sqrt(((y[self.y_to_m[:, 0]] - y[self.y_to_m[:, 1]])**2).sum(dim=1))
            gcmax_vals = self.cached_params['gcmax']
            lth_vals = self.cached_params['lth']
            sig_vals = self.cached_params['sig']
            L0_vals = self.cached_params['L0']
            return gcmax_vals * sigmoid_torch(L, lth_vals * L0_vals, sig_vals)
        else:
            L = np.sqrt(((self.y[self.y_to_m[:, 0]] - self.y[self.y_to_m[:, 1]])**2).sum(axis=1))
            return self.par['gcmax'] * sigmoid(np.reshape(L, [-1, 1]), (self.par['lth']*self.L0), self.par['sig'])

    def ka(self, COMPILE=False, v=-1):
        """Active spring constant: kmax * sigmoid(v; vth, vshp)."""
        if COMPILE:
            kmax_vals = self.cached_params['kmax']
            vth_vals = self.cached_params['vth']
            vshp_vals = self.cached_params['vshp']
            return kmax_vals * sigmoid_torch(v, vth_vals, vshp_vals)
        else:
            return self.par['kmax'] * sigmoid(np.reshape(self.v, [-1, 1]), self.par['vth'], self.par['vshp'])

    # Gap junctions

    def sum_voltages_per_chromatophore(self, tempv):
        """For each chromatophore, sum the voltages of all muscles attached to it."""
        tempxv = torch.zeros(len(self.x), device=self.device)
        for x in range(len(self.x)):
            tempxv[x] = torch.sum(tempv[self.x_to_y[x]])
        return tempxv

    def sum_voltages_per_muscle(self, Vt, tempyv):
        """For each muscle, sum tempyv over its two endpoints."""
        tempgv = torch.zeros_like(Vt)
        for x in range(len(self.v)):
            y_indices = self.y_to_m[x]
            tempgv[x] = torch.sum(tempyv[y_indices])
        return tempgv

    def _apply_connection_removal(self):
        """Remove a random fraction of muscles by setting kp and kmax to 0."""
        if self.removal_seed is not None:
            random.seed(self.removal_seed)

        n_muscles = len(self.v)
        n_remove = int(n_muscles * self.connection_removal_fraction)
        muscles_to_sever = random.sample(range(n_muscles), n_remove)

        for m_idx in muscles_to_sever:
            self.cached_params['kp'][m_idx] = 0.0
            self.cached_params['kmax'][m_idx] = 0.0

        self.severed_muscles = muscles_to_sever

    def precompute_tempgv_neighbors(self):
        """Find the gap junction neighbors of each muscle, used to count nc_m.

        A neighbor shares a chromatophore with this muscle, and its far end is
        one lattice step (distance ~1) from this muscle's far end, so the two
        muscles form a triangle (the bow-tie pattern in Fig 1D).
        Uses noise-free lattice positions.
        """
        self.muscle_tempgv_neighbors = []

        for m_idx in range(len(self.v)):
            y1_idx, y2_idx = self.y_to_m[m_idx].cpu().numpy()
            chromat1_idx = self.y_to_x[y1_idx].item()
            chromat2_idx = self.y_to_x[y2_idx].item()

            neighbor_y_indices = []

            # Look from each end of the muscle
            for y_idx, chromat_idx, other_chromat_idx in [(y1_idx, chromat1_idx, chromat2_idx),
                                                          (y2_idx, chromat2_idx, chromat1_idx)]:

                target_chromat_pos = self.x_clean[other_chromat_idx].cpu().numpy()

                # Other muscles attached to this chromatophore
                for other_y_idx in self.x_to_y[chromat_idx]:
                    if other_y_idx == y_idx:
                        continue

                    other_muscle_idx = self.m_to_y[other_y_idx].item()
                    other_y1, other_y2 = self.y_to_m[other_muscle_idx].cpu().numpy()

                    # Chromatophore at the far end of the other muscle
                    if other_y1 == other_y_idx:
                        far_chromat_idx = self.y_to_x[other_y2].item()
                    else:
                        far_chromat_idx = self.y_to_x[other_y1].item()

                    far_chromat_pos = self.x_clean[far_chromat_idx].cpu().numpy()

                    dist = np.linalg.norm(target_chromat_pos - far_chromat_pos)
                    if abs(dist - 1.0) < 0.1:
                        neighbor_y_indices.append(other_y_idx)

            self.muscle_tempgv_neighbors.append(neighbor_y_indices)

    def precompute_nc_m_values(self):
        """Number of gap junction neighbors of each muscle (nc_m)."""
        self.nc_m_precomputed = []
        for m_idx in range(len(self.v)):
            self.nc_m_precomputed.append(len(self.muscle_tempgv_neighbors[m_idx]))
        self.nc_m_precomputed = torch.tensor(self.nc_m_precomputed, dtype=torch.float32, device=self.device)

    # Compilation

    def compile_net(self):
        """Flatten the chromatophore and muscle objects into tensors and build
        the update function.

        Index maps:
            x_to_y[i]: endpoints on chromatophore i
            y_to_x[j]: chromatophore that endpoint j sits on
            m_to_y[j]: muscle that endpoint j belongs to
            y_to_m[m]: the two endpoints of muscle m
            y_to_y[j]: endpoint at the other end of the same muscle
        """
        if PYTORCH == False:
            print("No PyTorch available!")
            return 0

        # Per-muscle and per-chromatophore parameters
        mparam = {'phi', 'vc', 'vd', 'va', 'vb', 'Iapp', 'gl', 'gk', 'vl', 'vk', 'gca', 'vca', 'ggap', 'gcmax', 'lth', 'sig', 'kmax', 'vshp', 'kp', 'vth', 'ls'}
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
        for k in DEFAULT2D.keys():
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

                    # Register the muscle the first time we see it
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

                    # Link the two endpoints once both are registered
                    if y[1].other_att(self.chromats[c][k])[2] == -1:
                        continue
                    else:
                        my_index = y[2]
                        other_index = y[1].other_att(self.chromats[c][k])[2]
                        self.y_to_y[my_index] = other_index
                        self.y_to_y[other_index] = my_index
                        self.y_to_m[y[1].index] = [my_index, other_index]

        # Convert to tensors and move to device
        self.y_to_y = torch.tensor(self.y_to_y, dtype=torch.long, device=self.device)
        self.y_to_x = torch.tensor(self.y_to_x, dtype=torch.long, device=self.device)
        self.y_to_m = torch.tensor(self.y_to_m, dtype=torch.long, device=self.device)
        self.m_to_y = torch.tensor(self.m_to_y, dtype=torch.long, device=self.device)

        self.x = torch.tensor(np.array(self.x), dtype=torch.float32, device=self.device)
        self.x_clean = torch.tensor(np.array(self.x_clean), dtype=torch.float32, device=self.device)
        self.z = torch.tensor(np.array(self.z), dtype=torch.float32, device=self.device)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32, device=self.device)
        self.w = torch.tensor(self.w, dtype=torch.float32, device=self.device)
        self.v = torch.tensor(self.v, dtype=torch.float32, device=self.device)
        self.L0 = torch.tensor(self.L0, dtype=torch.float32, device=self.device)

        # Parameters to tensors. ks and kc are per chromatophore with shape (n, 1);
        # all others are per muscle with shape (n_muscles,)
        self.cached_params = {}
        for k in self.par.keys():
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
                    param_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
                elif isinstance(param_data, (int, float)):
                    param_tensor = torch.tensor([param_data], dtype=torch.float32, device=self.device)
                elif hasattr(param_data, 'shape'):
                    param_tensor = torch.tensor(param_data, dtype=torch.float32, device=self.device)
                else:
                    param_tensor = torch.tensor([float(param_data)], dtype=torch.float32, device=self.device)

                if k in ['ks', 'kc']:
                    if param_tensor.shape[0] == 1:
                        self.cached_params[k] = param_tensor.expand(len(self.x)).unsqueeze(1)
                    else:
                        self.cached_params[k] = param_tensor.unsqueeze(1)
                else:
                    if param_tensor.shape[0] == 1:
                        self.cached_params[k] = param_tensor.expand(len(self.v))
                    else:
                        self.cached_params[k] = param_tensor

                self.par[k] = param_tensor

            except Exception as e:
                print(f"Error converting parameter {k}: {param_data}, type: {type(param_data)}")
                fallback_val = 2.0 if k == 'kc' else DEFAULT2D.get(k, 1.0)
                if k in ['ks', 'kc']:
                    self.cached_params[k] = torch.full((len(self.x), 1), fallback_val, device=self.device)
                else:
                    self.cached_params[k] = torch.full((len(self.v),), fallback_val, device=self.device)
                self.par[k] = torch.tensor([fallback_val], device=self.device)

        self.cached_params['L0'] = self.L0

        # Endpoint lists as tensors (for indexing) and as NumPy arrays
        self.x_to_y_tensors = []
        for k in range(len(self.x_to_y)):
            self.x_to_y_tensors.append(torch.tensor(self.x_to_y[k], dtype=torch.long, device=self.device))

        for k in range(len(self.x_to_y)):
            self.x_to_y[k] = np.array(self.x_to_y[k])

        self.precompute_tempgv_neighbors()
        self.precompute_nc_m_values()

        def get_updates_base(Xt, Yt, Wt, Vt, Tt, DT):
            """One Euler step of the full model, without holding boundary chromatophores fixed."""

            # Chromatophore positions (Eq 2): dx/dt = ks (z - x) + kc sum_mu (y_mu - x)
            dx_zterm = -(Xt - self.z)

            dx_yterm = torch.zeros_like(Xt)
            for x in range(len(self.x)):
                y_connections = self.x_to_y_tensors[x]
                if len(y_connections) > 0:
                    dx_yterm[x] = -torch.sum(Xt[x:x+1] - Yt[y_connections], dim=0)

            ks_vals = self.cached_params['ks']
            kc_vals = self.cached_params['kc']

            if ks_vals.shape[1] == 1:
                ks_vals = ks_vals.expand(-1, 2)
            if kc_vals.shape[1] == 1:
                kc_vals = kc_vals.expand(-1, 2)

            tempdx = ks_vals * dx_zterm + kc_vals * dx_yterm

            # Recovery variable (Eq 5): dw/dt = lambda_n(v) (n_inf(v) - w)
            ninf_vals = self.ninf(COMPILE=True, v=Vt)
            lamn_vals = self.lamn(COMPILE=True, v=Vt)
            tempdw = (ninf_vals - Wt) * lamn_vals

            # Muscle endpoints (Eq 3): pulled toward their chromatophore by kc...
            kc_indexed = self.cached_params['kc'][self.y_to_x]
            if kc_indexed.shape[1] == 1:
                kc_indexed = kc_indexed.expand(-1, 2)
            dy_xterm = -kc_indexed * (Yt - Xt[self.y_to_x])

            # ...and along the muscle by the passive (kp) and active (ka) springs
            ka_values = self.ka(COMPILE=True, v=Vt)
            KA = ka_values[self.m_to_y].unsqueeze(1)

            kp_vals = self.cached_params['kp']
            if kp_vals.dim() == 1:
                kp_vals = kp_vals.unsqueeze(1)
            KP = kp_vals[self.m_to_y]

            ydiff = Yt - Yt[self.y_to_y]  # Vector to the other endpoint
            ynorm = torch.sqrt((ydiff*ydiff).sum(dim=1, keepdim=True))  # Muscle length
            ynorm_safe = torch.maximum(ynorm, torch.tensor(1e-10, device=self.device))
            L0 = self.L0[self.m_to_y].unsqueeze(1)

            ls_vals = self.cached_params['ls']
            if ls_vals.dim() == 1:
                ls_vals = ls_vals.unsqueeze(1)
            LS = ls_vals[self.m_to_y]

            # Expansion is faster than retraction: muscles lengthening past
            # their rest length have reduced passive and active stiffness
            retracting = ynorm > L0
            KP_mod = torch.where(retracting, KP * 0.0005, KP)
            KA_mod = torch.where(retracting, KA * 0.5, KA)

            dy_mterm = - (KP_mod*(ynorm - L0) + KA_mod * (ynorm - L0*LS))*ydiff/ynorm_safe

            tempdy = dy_xterm + dy_mterm

            # Membrane voltage (Eq 4): Iapp + leak + potassium + calcium + gap
            m_pos = (Yt[self.y_to_m[:,0]] + Yt[self.y_to_m[:,1]])/2
            iapp = self.IAPP(m_pos, Tt)

            gl_vals = self.cached_params['gl']
            vl_vals = self.cached_params['vl']
            leak = gl_vals * (vl_vals - Vt)

            gk_vals = self.cached_params['gk']
            vk_vals = self.cached_params['vk']
            pota = gk_vals * Wt * (vk_vals - Vt)

            # Voltage-gated plus stretch-sensitive calcium
            gca_vals = self.cached_params['gca']
            vca_vals = self.cached_params['vca']
            minf_vals = self.minf(COMPILE=True, v=Vt)
            gc_vals = self.gc(COMPILE=True, y=Yt)
            calc = (gca_vals * minf_vals + gc_vals) * (vca_vals - Vt)

            # Gap junctions (Eq 6): I_gap = -ggap (nc_m V - sum of coupled voltages)
            tempv = Vt[self.m_to_y]                              # Voltage at each endpoint
            tempxv = self.sum_voltages_per_chromatophore(tempv)  # Summed per chromatophore
            tempyv = tempxv[self.y_to_x] - tempv                 # Excluding the muscle itself
            tempgv = self.sum_voltages_per_muscle(Vt, tempyv)    # Summed over both ends
            nc_m = self.nc_m_precomputed

            ggap_vals = self.cached_params['ggap']
            gap = - ggap_vals * (nc_m*Vt - tempgv)

            tempdv = iapp + leak + pota + calc + gap

            outx = Xt + DT*tempdx
            outy = Yt + DT*tempdy
            outw = Wt + DT*tempdw
            outv = Vt + DT*tempdv

            return outx, outy, outw, outv, Tt + DT

        def get_updates_euler(Xt, Yt, Wt, Vt, Tt, DT):
            outX, outY, outW, outV, Tt_DT = get_updates_base(Xt, Yt, Wt, Vt, Tt, DT)
            # Hold boundary chromatophores fixed
            outX = torch.where(self.edge_mask.unsqueeze(1), Xt, outX)
            return outX, outY, outW, outV, Tt_DT

        def get_updates_rk4(Xt, Yt, Wt, Vt, Tt, DT):
            """Fourth-order Runge-Kutta. Each slope is recovered from one Euler step."""
            # k1 at the current state
            X1, Y1, W1, V1, _ = get_updates_base(Xt, Yt, Wt, Vt, Tt, DT)
            dX1 = (X1 - Xt) / DT
            dY1 = (Y1 - Yt) / DT
            dW1 = (W1 - Wt) / DT
            dV1 = (V1 - Vt) / DT

            # k2 at the midpoint using k1
            X_mid2 = Xt + 0.5*DT*dX1
            Y_mid2 = Yt + 0.5*DT*dY1
            W_mid2 = Wt + 0.5*DT*dW1
            V_mid2 = Vt + 0.5*DT*dV1
            X2, Y2, W2, V2, _ = get_updates_base(X_mid2, Y_mid2, W_mid2, V_mid2, Tt + 0.5*DT, DT)
            dX2 = (X2 - X_mid2) / DT
            dY2 = (Y2 - Y_mid2) / DT
            dW2 = (W2 - W_mid2) / DT
            dV2 = (V2 - V_mid2) / DT

            # k3 at the midpoint using k2
            X_mid3 = Xt + 0.5*DT*dX2
            Y_mid3 = Yt + 0.5*DT*dY2
            W_mid3 = Wt + 0.5*DT*dW2
            V_mid3 = Vt + 0.5*DT*dV2
            X3, Y3, W3, V3, _ = get_updates_base(X_mid3, Y_mid3, W_mid3, V_mid3, Tt + 0.5*DT, DT)
            dX3 = (X3 - X_mid3) / DT
            dY3 = (Y3 - Y_mid3) / DT
            dW3 = (W3 - W_mid3) / DT
            dV3 = (V3 - V_mid3) / DT

            # k4 at the endpoint using k3
            X_end = Xt + DT*dX3
            Y_end = Yt + DT*dY3
            W_end = Wt + DT*dW3
            V_end = Vt + DT*dV3
            X4, Y4, W4, V4, _ = get_updates_base(X_end, Y_end, W_end, V_end, Tt + DT, DT)
            dX4 = (X4 - X_end) / DT
            dY4 = (Y4 - Y_end) / DT
            dW4 = (W4 - W_end) / DT
            dV4 = (V4 - V_end) / DT

            outX = Xt + (DT/6) * (dX1 + 2*dX2 + 2*dX3 + dX4)
            outY = Yt + (DT/6) * (dY1 + 2*dY2 + 2*dY3 + dY4)
            outW = Wt + (DT/6) * (dW1 + 2*dW2 + 2*dW3 + dW4)
            outV = Vt + (DT/6) * (dV1 + 2*dV2 + 2*dV3 + dV4)

            # Hold boundary chromatophores fixed
            outX = torch.where(self.edge_mask.unsqueeze(1), Xt, outX)

            return outX, outY, outW, outV, Tt + DT

        if self.integration_method.lower() == 'rk4':
            self.get_updates = get_updates_rk4
        elif self.integration_method.lower() == 'euler':
            self.get_updates = get_updates_euler
        else:
            raise ValueError(f"Unknown integration_method: {self.integration_method}. Use 'euler' or 'rk4'")

        # Boundary chromatophores: both ends of a chain, or the perimeter of a hex lattice
        self.edge_mask = torch.zeros(len(self.x), dtype=torch.bool, device=self.device)
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

    # Running and saving

    def get_all_radii(self, xi=[], yi=[]):
        """Chromatophore radius: norm of the mean absolute offset between the
        chromatophore position and its muscle endpoints."""
        if len(xi) == 0 or len(yi) == 0:
            x = self.x.clone() if isinstance(self.x, torch.Tensor) else self.x.copy()
            y = self.y.clone() if isinstance(self.y, torch.Tensor) else self.y.copy()
        else:
            x = xi
            y = yi

        if isinstance(x, torch.Tensor):
            x = x.reshape(-1, 1, 2)
            out = []
            for z in range(len(x)):
                out.append(torch.norm(torch.mean(torch.abs(x[z].reshape(1, 2) - y[self.x_to_y[z]]), dim=0)))
            return torch.stack(out)
        else:
            x = np.reshape(x, [-1, 1, 2])
            out = []
            for z in range(len(x)):
                out.append(np.linalg.norm(np.mean(np.abs(np.reshape(x[z], [1, 2]) - y[self.x_to_y[z]]), axis=0)))
            return np.array(out)

    def run(self, steps, name="test", dt=.01, save_freq=10):
        """Simulate `steps` steps of size dt, saving every `save_freq` steps.

        Writes the x_pos_, v_pos_, z_pos_ and time_ files for `name`
        (see save_CHR_files) and returns [X, Y, W, V] at the saved times.
        """
        if PYTORCH == False or self.compiled == False:
            print("Either PyTorch is not available or you haven't compiled yet.")
            return

        current_x = self.x.clone()
        current_y = self.y.clone()
        current_w = self.w.clone()
        current_v = self.v.clone()
        current_t = torch.tensor(0.0, device=self.device)
        dt_tensor = torch.tensor(dt, device=self.device)

        # Saved states are kept on the CPU
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

        # Use mixed precision if on GPU
        use_amp = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        start_time = time.time()
        for step in range(steps):
            if use_amp:
                with torch.cuda.amp.autocast():
                    current_x, current_y, current_w, current_v, current_t = self.get_updates(
                        current_x, current_y, current_w, current_v, current_t, dt_tensor
                    )
            else:
                current_x, current_y, current_w, current_v, current_t = self.get_updates(
                    current_x, current_y, current_w, current_v, current_t, dt_tensor
                )

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
                print(f"Step {step+1}/{steps}, {steps_per_sec:.1f} steps/sec, ETA: {eta:.1f}s")

        vals = [x_states, y_states, w_states, v_states]

        self.save_CHR_files(name, vals[0], vals[1], vals[2], vals[3], skip=1)

        time_points = [0] + [i * dt for i in range(save_freq, steps + 1, save_freq)]
        tvec = np.array(time_points)
        np.savetxt("./time_"+name+".csv", tvec)

        return vals

    def save_CHR_files(self, name, X, Y, W, V, skip=1):
        """Write the simulation to text files.

        x_pos_{name}.csv: one row per saved time, [x, y, radius] per chromatophore
        v_pos_{name}.csv: one row per saved time, [x, y, v] per muscle (at the muscle midpoint)
        z_pos_{name}.csv: chromatophore rest positions
        W is not saved and skip is not used.
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            W = W.detach().cpu().numpy()
            V = V.detach().cpu().numpy()

        X_save = X
        Y_save = Y
        V_save = V

        # Radius of every chromatophore at every saved time
        R_list = []
        for t_idx in range(len(X_save)):
            R_t = self.get_all_radii(torch.tensor(X_save[t_idx], device=self.device),
                                     torch.tensor(Y_save[t_idx], device=self.device))
            if isinstance(R_t, torch.Tensor):
                R_t = R_t.detach().cpu().numpy()
            R_list.append(R_t)

        R = np.array(R_list)
        if R.ndim == 2:
            R = R.reshape(R.shape[0], -1, 1)

        xout = np.concatenate([X_save, R], axis=2)
        xout = xout.reshape(xout.shape[0], -1)

        # Muscle midpoints at every saved time
        mpos_list = []
        for t_idx in range(len(Y_save)):
            y_tensor = torch.tensor(Y_save[t_idx], device=self.device)
            mpos_t = (y_tensor[self.y_to_m[:,0]] + y_tensor[self.y_to_m[:,1]])/2
            if isinstance(mpos_t, torch.Tensor):
                mpos_t = mpos_t.detach().cpu().numpy()
            mpos_list.append(mpos_t)

        mpos = np.array(mpos_list)

        V_reshaped = V_save.reshape(len(V_save), -1, 1)
        Vout = np.concatenate([mpos, V_reshaped], axis=2)
        Vout = Vout.reshape(Vout.shape[0], -1)

        if xout.ndim == 1:
            xout = xout.reshape(1, -1)
        if Vout.ndim == 1:
            Vout = Vout.reshape(1, -1)

        np.savetxt(f"./x_pos_{name}.csv", xout, delimiter=' ', fmt='%.6e')
        np.savetxt(f"./v_pos_{name}.csv", Vout, delimiter=' ', fmt='%.6e')

        z_save = self.z.detach().cpu().numpy() if isinstance(self.z, torch.Tensor) else self.z
        np.savetxt(f"./z_pos_{name}.csv", z_save, delimiter=' ', fmt='%.6e')

    # Initial conditions

    def set_VI(self, vi_func):
        """Set the initial muscle voltages from vi_func(muscle positions).

        vi_func can take all positions at once as a tensor, or one position
        at a time as a NumPy array.
        """
        P = self.get_m_pos()

        if callable(vi_func):
            try:
                # Try all positions at once
                if isinstance(P, torch.Tensor):
                    new_voltages = vi_func(P)
                else:
                    new_voltages = vi_func(torch.tensor(P, device=self.device))

                if not isinstance(new_voltages, torch.Tensor):
                    new_voltages = torch.tensor(new_voltages, dtype=torch.float32, device=self.device)
                elif new_voltages.device != self.device:
                    new_voltages = new_voltages.to(self.device)

                self.v = new_voltages.flatten()

            except:
                # Fall back to one position at a time
                if isinstance(P, torch.Tensor):
                    P_np = P.cpu().numpy()
                else:
                    P_np = P

                new_voltages_list = []
                for pos in P_np:
                    new_voltages_list.append(vi_func(pos))

                new_voltages = torch.tensor(new_voltages_list, dtype=torch.float32, device=self.device)
                self.v = new_voltages.flatten()
        else:
            raise ValueError("vi_func must be callable")

    def set_XI(self, xi):
        """Move each chromatophore and its muscle endpoints to xi(rest position)."""
        for x in range(len(self.x)):
            if isinstance(self.z, torch.Tensor):
                z_val = self.z[x].cpu().numpy()
            else:
                z_val = self.z[x]
            new_pos = xi(z_val)
            self.x[x] = torch.tensor(new_pos, device=self.device) if isinstance(self.x, torch.Tensor) else new_pos.copy()
            if isinstance(self.y, torch.Tensor):
                self.y[self.x_to_y[x]] = torch.tensor(new_pos, device=self.device).reshape(1, 2)
            else:
                self.y[self.x_to_y[x]] = np.reshape(new_pos, [1, 2]).copy()

    def reset_net(self, xi=-1, vi=-1):
        """Reset the network state.

        xi(rest position) sets the chromatophore positions (default: at rest).
        vi(muscle position) sets the voltages (default: DEFAULT2D['vinit']).
        Muscle endpoints start on their chromatophore, and w is reset to DEFAULT2D['winit'].
        """
        if self.compiled == False:
            print("must compile network before using 'reset_net'")
            return 0

        if xi == -1:
            self.x = self.z.clone() if isinstance(self.z, torch.Tensor) else np.copy(self.z)
        else:
            if isinstance(self.z, torch.Tensor):
                self.x = torch.stack([torch.tensor(xi(z.cpu().numpy()), device=self.device) for z in self.z])
            else:
                self.x = torch.tensor([xi(z) for z in self.z], device=self.device)

        if isinstance(self.x, torch.Tensor):
            self.y = self.x[self.y_to_x].clone()
        else:
            self.y = torch.tensor(self.x[self.y_to_x], device=self.device)

        if vi == -1:
            self.v = torch.ones_like(self.v) * DEFAULT2D['vinit']
        else:
            loc = (self.y[self.y_to_m[:,0]] + self.y[self.y_to_m[:,1]])/2
            if isinstance(loc, torch.Tensor):
                self.v = torch.stack([torch.tensor(vi(x.cpu().numpy()), device=self.device) for x in loc])
            else:
                self.v = torch.tensor([vi(x) for x in loc], device=self.device)

        self.w = torch.ones_like(self.w) * DEFAULT2D["winit"]

        return 1