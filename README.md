# ChromatoNet

**A biophysical model of chromatophore networks in cephalopod skin**

Simulate cephalopod skin! 

Cephalopods produce elaborate, fast-changing skin patterns for camouflage and communication. A long-standing question is whether these patterns are controlled entirely by the brain, or whether the skin itself helps generate them. ChromatoNet is a model built to test the second possibility.

ChromatoNet models a network of chromatophores (the tiny color organs in squid and cuttlefish skin), connected by excitable muscles on a 1D chain or a 2D hexagonal lattice. Poke it, and watch waves, spirals and flickers spread across the skin.

With only sparse input, the model reproduces many patterns seen in real and denervated skin:

- traveling waves started by brief, local pulses,
- an intrinsic wave frequency set by the skin's parameters,
- spiral waves, and waves that travel around stiff barriers,
- noisy, asynchronous flickering and whole-skin flashes.

ChromatoNet also provides a testbed for inferring biophysical parameters from recorded skin patterns, and for designing experiments that distinguish central from peripheral control.

This repository contains the simulation code and examples for:

> Ersoy Y, Peter R, Barello G, Meyer E, Mackevicius EL, Senft SL, Hanlon RT, Ermentrout GB, Palmer SE. *ChromatoNet: Bio-inspired chromatophore network model generates dynamic skin patterns akin to those observed in cephalopods.* bioRxiv. 2026. doi:[10.64898/2026.09.30.755305](https://www.biorxiv.org/content/10.64898/2026.09.30.755305v1)


## Files

| File | Contents |
|---|---|
| `chromatonet_demo.ipynb` | Notebook of examples, start here |
| `chromatonet.py` | The model: `ChromatoNet()` |
| `defaults.py` | Default parameter values |
| `stims.py` | Stimulation/setup inputs (initial voltages, continuous applied currents, initial positions and parameter maps) |
| `animate_module.py` | Animation tool for simulations |
| `plot_helper.py` | Plotting helpers |
| `calc_helper.py` | Analysis helpers  |
| `requirements.txt` | Required Python packages |
| `figures/` | Paper figures |
| `videos/` | Example simulation videos and reference footage of cephalopod skin |
| `media_plos_format/` | Copies of the figures and videos, formatted for PLOS Comp Bio |

## Installation

ChromatoNet was tested with **Python 3.11.13**. We recommend a separate virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

ChromatoNet runs on the CPU and uses a CUDA GPU automatically if one is available. On Linux with an NVIDIA GPU, you may need the PyTorch install command for your CUDA version from [pytorch.org](https://pytorch.org).

## Quick start

```python
import chromatonet as cn
import stims
import animate_module as an

# 20 x 20 hexagonal lattice, integrated with fourth-order Runge-Kutta
net = cn.ChromatoNet(N=20, M=20, layout='hex', integration_method='rk4')

# Activate muscles near the lower-left corner
net.set_vi(stims.vi_gaussian)

# Simulate 150 s with dt = 0.1, saving once per second
vals = net.run(1500, name='quickstart', dt=0.1, save_freq=10)

anim = an.animate('quickstart')
```

## Changing parameters

Pass `params`, a dictionary of functions of position. A function that returns a constant sets the parameter everywhere; the helpers in `stims.py` can make it take a different value inside a region:

```python
params = {
    'gl': lambda p: 0.42,  # Leak conductance everywhere
    'kp': lambda p: stims.param_disk(p, center=(14., 12.), radius=3.,
                                     inside=10., outside=0.1),  # Stiff disk
}
net = cn.ChromatoNet(N=20, M=20, params=params)
```

All parameters and their defaults are listed in `defaults.py`.

## Output files

`run(steps, name=...)` writes these files to the current working directory:

| File | Contents |
|---|---|
| `x_pos_{name}.csv` | One row per saved time: `[x, y, radius]` for each chromatophore |
| `v_pos_{name}.csv` | One row per saved time: `[x, y, voltage]` for each muscle, at the muscle midpoint |
| `z_pos_{name}.csv` | Chromatophore rest positions |
| `time_{name}.csv` | Saved time points |

## Authors

Yasemin Ersoy, Robin Peter, Gabriel Barello, Emily Meyer, Emily L. Mackevicius, Stephen L. Senft, Roger T. Hanlon, G. Bard Ermentrout and Stephanie E. Palmer.

ChromatoNet builds on several years of work by students in summer programs at the Marine Biological Laboratory, Woods Hole, and uses data from the Hanlon Laboratory.

## Questions and issues

Found a bug or have a question about the code? Please [open an issue](https://github.com/sepalmer/ChromatoNet/issues). For questions about the code, paper or collaborations, contact Yasemin Ersoy (yasersoy@gmail.com) or Stephanie E. Palmer (sepalmer@uchicago.edu).

## Citation

If you use ChromatoNet, please cite our preprint:

```bibtex
@article{ersoy2026chromatonet,
  title   = {ChromatoNet: Bio-inspired chromatophore network model generates dynamic skin patterns akin to those observed in cephalopods},
  author  = {Ersoy, Yasemin and Peter, Robin and Barello, Gabriel and Meyer, Emily and Mackevicius, Emily L. and Senft, Stephen L. and Hanlon, Roger T. and Ermentrout, G. Bard and Palmer, Stephanie E.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.30.755305}
}
```

## License

ChromatoNet is released under the [MIT License](LICENSE).