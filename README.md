# plotpal
Post-processing tools for plotting results from Dedalus simulations.

# Installation
To install plotpal on your local machine, navigate to the root plotpal/ directory (where setup.py is located), and type:

> pip3 install -e .

# Usage

1. Copy one of the python scripts from the example/ directory somewhere closer to where you're running dedalus simulations (or just modify one of your local files there).
2. Put in the fields you care about plotting.
3. Make some plots!

By default, plots will be located within the parent directory of your Dedalus simulation, close to where the handlers output the data.
