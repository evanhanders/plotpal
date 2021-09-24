"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_rolled_vs_instantaneous_profiles.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: profiles]
    --subdir_name=<subdir_name>               Name of figure output directory & base name of saved figures [default: rolled_v_instant_profiles]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                    Figure width (inches) [default: 6]
    --row_inch=<in>                   Figure height (inches) [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
import matplotlib.pyplot as plt
from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularPlotGrid

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_file  = int(args['--start_file'])
subdir_name    = args['--subdir_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)


# Create Plotter object, tell it which fields to plot
plotter_instant = SingleTypeReader(root_dir, file_dir=data_dir, out_name=subdir_name, start_file=start_file, n_files=n_files)
plotter_rolled = SingleTypeReader(root_dir, file_dir=data_dir, out_name=subdir_name, roll_writes=50, start_file=start_file, n_files=n_files)
grid = RegularPlotGrid(num_rows=2, num_cols=1, col_inch=float(args['--col_inch']), row_inch=float(args['--row_inch']))
ax1, ax2 = grid.axes['ax_0-0'], grid.axes['ax_1-0']

instant_kwargs = {'lw' : 1, 'ls' : '--'}
rolled_kwargs  = {'lw' : 2, 'ls' : '-'}

fields = ['b', 'cond_flux', 'conv_flux']
while plotter_instant.writes_remain() and plotter_rolled.writes_remain():
    instant_dsets, ii = plotter_instant.get_dsets(fields)
    rolled_dsets, ri  = plotter_rolled.get_dsets(fields)

    for dset, kwargs, ind in zip([instant_dsets, rolled_dsets], [instant_kwargs, rolled_kwargs], [ii, ri]):
        z = match_basis(dset['b'], 'z')
        ax1.plot(z, dset['b'][ind,0,:], c='k', **kwargs)
        if dset == rolled_dsets:
            label1 = 'cond flux'
            label2 = 'conv flux'
        else:
            label1 = label2 = None
        ax2.plot(z, dset['cond_flux'][ind,0,:], label=label1, c='b', **kwargs)
        ax2.plot(z, dset['conv_flux'][ind,0,:], label=label2, c='r', **kwargs)
    ax2.set_xlabel('z')
    ax2.legend(loc='upper right')

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)

    time_data = instant_dsets['b'].dims[0]
    plt.suptitle('t = {:.4e}'.format(time_data['sim_time'][ii]))
    grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(plotter_instant.out_dir, subdir_name, time_data['write_number'][ii]), dpi=int(args['--dpi']), bbox_inches='tight')
    for ax in [ax1, ax2]: ax.clear()
