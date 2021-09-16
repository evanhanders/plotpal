"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_avg_structure.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: profiles]
    --subdir_name=<subdir_name>               Name of figure output directory & base name of saved figures [default: avg_structure]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --writes_per_avg=<int>              Number of writes over which to take average [default: 15]
    --dpi=<dpi>                         Image pixel density [default: 200]

    --fig_width=<in>                    Figure width (inches) [default: 6]
    --fig_height=<in>                   Figure height (inches) [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.profiles import AveragedProfilePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_file  = int(args['--start_file'])
writes_per_avg = int(args['--writes_per_avg'])
subdir_name    = args['--subdir_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)


# Create Plotter object, tell it which fields to plot
plotter = AveragedProfilePlotter(root_dir, writes_per_avg=writes_per_avg, file_dir=data_dir, fig_name=subdir_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'fig_width' : int(args['--fig_width']), 'fig_height' : int(args['--fig_height']) }
plotter.add_average_plot(x_basis='r', y_tasks='T profile', name='T_vs_r', **plotter_kwargs)
plotter.add_average_plot(x_basis='r', y_tasks=('conv luminosity', 'cond luminosity'), name='lum_vs_r', **plotter_kwargs)

plotter.plot_average_profiles(dpi=int(args['--dpi']), save_data=False)
