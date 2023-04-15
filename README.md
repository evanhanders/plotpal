# plotpal
Post-processing tools for plotting results from Dedalus simulations.
The code in this repository is backed up on Zenodo, and is citeable at: [![DOI](https://zenodo.org/badge/265006663.svg)](https://zenodo.org/badge/latestdoi/265006663)

# Installation
To install plotpal on your local machine, navigate to the root plotpal/ directory (where setup.py is located), and type:

> pip3 install -e .

Note that plotpal is built on [matplotlib](https://matplotlib.org/), and uses some parallel functionality of [Dedalus](https://dedalus-project.org/). 

## Dependencies
Plotpal relies on the full [Dedalus](https://dedalus-project.org/) stack (e.g., numpy, matplotlib, and dedalus itself) for full functionality.

Some additional features also rely on:
* Volumetric plotting requires [PyVista](https://docs.pyvista.org/version/stable/index.html) >= 0.38.5.
* Orthographic projections require [CartoPy](https://scitools.org.uk/cartopy/docs/latest/).

# Usage

1. Copy one of the python scripts from the example/ directory somewhere closer to where you're running dedalus simulations (or just modify one of your local files there).
2. Put in the fields you care about plotting.
3. Make some plots!

By default, plots will be located within the parent directory of your Dedalus simulation, in a new folder.
