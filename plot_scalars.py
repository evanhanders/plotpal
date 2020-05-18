"""
Script for plotting traces of evaluated scalar quantities vs. time.

Usage:
    plot_scalars.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from plot_logic.scalars import ScalarFigure, ScalarPlotter

root_dir = args['<root_dir>']
fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)


# Nu vs time
fig1 = ScalarFigure(1, 1, col_in=6, fig_name='nu_trace')
fig1.add_field(0, 'Nu')

# Re vs. time
fig2 = ScalarFigure(1, 1, col_in=6, fig_name='pe_trace')
fig2.add_field(0, 'Pe')

# dT 
fig3 = ScalarFigure(1, 1, col_in=6, fig_name='delta_T')
fig3.add_field(0, 'delta_T')

# Energies
fig4 = ScalarFigure(4, 1, col_in=6, row_in=2.5, fig_name='energies')
fig4.add_field(0, 'KE')
fig4.add_field(1, 'KE')
fig4.add_field(0, 'IE')
fig4.add_field(2, 'IE')
fig4.add_field(0, 'TE')
fig4.add_field(3, 'TE')

# Energies
fig5 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_temps')
fig5.add_field(0, 'left_T')
fig5.add_field(1, 'right_T')

# Fluxes
fig6 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_fluxes')
fig6.add_field(0, 'left_flux')
fig6.add_field(1, 'right_flux')

# KE sources
fig7 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='ke_sources')
fig7.add_field(0, 'KE')
fig7.add_field(1, 'enstrophy')
fig7.add_field(2, 'wT')

# KE sources
fig8 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='ke_sources2')
fig8.add_field(0, 'KE')
fig8.add_field(1, 'visc_KE_source')
fig8.add_field(2, 'buoy_KE_source')

# Re vs. time
if 'rotating' in root_dir:
    figRo = ScalarFigure(1, 1, col_in=6, fig_name='ro_trace')
    figRo.add_field(0, 'Ro')
    figRo.add_field(0, 'true_Ro')




# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir='scalar', fig_name=fig_name, start_file=start_file, n_files=n_files)
figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]
if 'rotating' in root_dir: figs.append(figRo)
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))
