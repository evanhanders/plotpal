"""
Script for plotting traces of evaluated scalar quantities vs. time.

Usage:
    plot_traces.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: scalar]
    --fig_name=<fig_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
from docopt import docopt
args = docopt(__doc__)
from logic.scalars import ScalarFigure, ScalarPlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dirs is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

fig_name    = args['--fig_name']
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

figs = []

# Nu vs time
fig1 = ScalarFigure(1, 1, col_in=6, fig_name='nu_trace')
fig1.add_field(0, 'Nu')
figs.append(fig1)

# Re vs. time
fig2 = ScalarFigure(1, 1, col_in=6, fig_name='pe_trace')
fig2.add_field(0, 'Pe')
figs.append(fig2)

# dT 
fig3 = ScalarFigure(1, 1, col_in=6, fig_name='delta_T')
fig3.add_field(0, 'delta_T')
figs.append(fig3)

# Energies
fig4 = ScalarFigure(4, 1, col_in=6, row_in=2.5, fig_name='energies')
fig4.add_field(0, 'KE')
fig4.add_field(1, 'KE')
fig4.add_field(0, 'IE')
fig4.add_field(2, 'IE')
fig4.add_field(0, 'TE')
fig4.add_field(3, 'TE')
figs.append(fig4)

# Energies
fig5 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_temps')
fig5.add_field(0, 'left_T')
fig5.add_field(1, 'right_T')
figs.append(fig5)

# Fluxes
fig6 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_fluxes')
fig6.add_field(0, 'left_flux')
fig6.add_field(1, 'right_flux')
figs.append(fig6)

# KE sources
fig7 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='ke_sources')
fig7.add_field(0, 'KE')
fig7.add_field(1, 'enstrophy')
fig7.add_field(2, 'wT')
figs.append(fig7)

# KE sources
fig8 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='ke_sources2')
fig8.add_field(0, 'KE')
fig8.add_field(1, 'visc_KE_source')
fig8.add_field(2, 'buoy_KE_source')
figs.append(fig8)

# Re vs. time
if 'rotating' in root_dir:
    figRo = ScalarFigure(1, 1, col_in=6, fig_name='ro_trace')
    figRo.add_field(0, 'Ro')
    figRo.add_field(0, 'true_Ro')
    figs.append(figRo)


# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir='scalar', fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))
