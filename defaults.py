"""Default parameters for ChromatoNet.

Voltage quantities are dimensionless (normalized to vca) and time is in seconds.
Any parameter can be made position dependent by passing a function f(position)
through PINITS in chromatophore_network.
"""

DEFAULT2D = {
    'dt': .01,       # Integration time step (not used by the simulator; run() takes dt)
    'Iapp': .15,     # Baseline applied current
    'phi': .05,      # Rate scaling of the gating variable w
    'va': -.01,      # Ca activation midpoint (m_inf)
    'vb': .15,       # Ca activation slope (m_inf)
    'vd': .3,        # K activation slope (n_inf)
    'vc': 0.,        # K activation midpoint (n_inf)
    'gca': 1.1,      # Ca conductance
    'gcmax': .1,     # Maximum stretch-sensitive Ca conductance
    'gk': 2.,        # K conductance
    'vk': -.7,       # K reversal potential
    'gl': 0.6,       # Leak conductance
    'vl': -.5,       # Leak reversal potential
    'lth': 1.1,      # Stretch activation threshold (in units of L0)
    'sig': .1,       # Stretch activation sharpness
    'kmax': 2.,      # Maximum active spring constant
    'vth': 0.,       # Active spring activation midpoint
    'vshp': .005,    # Active spring activation sharpness
    'ls': .33,       # Contracted length as a fraction of L0
    'l0': 1.,        # Not used by the simulator (L0 is the distance between rest positions)
    'kp': .1,        # Passive muscle spring constant
    'ggap': .005,    # Gap junction conductance
    'x0': 1.,        # Not used by the simulator
    'vca': 1.0,      # Ca reversal potential
    'kc': 2.,        # Chromatophore-muscle coupling
    'ks': 1.,        # Skin spring constant
    'vinit': -.266,  # Initial membrane voltage
    'winit': .144    # Initial gating variable
}