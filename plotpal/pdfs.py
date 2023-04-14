import os
import logging
from collections import OrderedDict
from sys import stdout
from sys import path

import numpy as np
import h5py
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

from dedalus.tools.parallel import Sync

from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularPlotGrid

logger = logging.getLogger(__name__.split('.')[-1])


class PdfPlotter(SingleTypeReader):
    """
    A class for plotting probability distributions of a dedalus output.

    PDF plots are currently implemented for 2D slices and 3D volumes. 
    When one axis is represented by polynomials that exist on an uneven basis (e.g., Chebyshev),
    that basis is evenly interpolated to avoid skewing of the distribution by uneven grid sampling.

    # Public Methods
    - __init__()
    - calculate_pdfs()
    - plot_pdfs()

    # Additional Attributes
        pdfs (OrderedDict) :
            Contains PDF data (x, y, dx)
        pdf_stats (OrderedDict) :
            Contains scalar stats for the PDFS (mean, stdev, skew, kurtosis)
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PDF plotter.

        # Arguments
            *args, **kwargs : Additional keyword arguments for super().__init__()  (see file_reader.py)
        """
        super(PdfPlotter, self).__init__(*args, distribution='even-write', **kwargs)
        self.pdfs = OrderedDict()
        self.pdf_stats = OrderedDict()


    def _calculate_pdf_statistics(self):
        """ Calculate statistics of the PDFs stored in self.pdfs. Store results in self.pdf_stats. """
        for k, data in self.pdfs.items():
            pdf, x_vals, dx = data

            mean = np.sum(x_vals*pdf*dx)
            stdev = np.sqrt(np.sum((x_vals-mean)**2*pdf*dx))
            skew = np.sum((x_vals-mean)**3*pdf*dx)/stdev**3
            kurt = np.sum((x_vals-mean)**4*pdf*dx)/stdev**4
            self.pdf_stats[k] = (mean, stdev, skew, kurt)

    
    def _get_interpolated_slices(self, dsets, ni, uneven_basis=None):
        """
        For 2D data on an uneven grid, interpolates that data on to an evenly spaced grid.

        # Arguments
            uneven_basis (string, optional) :
                The basis on which the grid has uneven spacing.
        """
        #Read data
        bases = self.current_bases

        # Put data on an even grid
        x, y = [match_basis(dsets[next(iter(dsets))], bn) for bn in bases]
        if bases[0] == uneven_basis:
            even_x = np.linspace(x.min(), x.max(), len(x))
            even_y = y
        elif bases[1] == uneven_basis:
            even_x = x
            even_y = np.linspace(y.min(), y.max(), len(y))
        else:
            even_x, even_y = x, y
        eyy, exx = np.meshgrid(even_y, even_x)

        file_data = OrderedDict()
        for k in dsets.keys(): 
            file_data[k] = np.zeros(dsets[k][ni].shape)
            interp = RegularGridInterpolator((x.flatten(), y.flatten()), dsets[k][ni], method='linear')
            file_data[k][:,:] = interp((exx, eyy))

        return file_data

    def _get_interpolated_volumes(self, dsets, ni, uneven_basis=None):
        """
        For 3D data on an uneven grid, interpolates that data on to an evenly spaced grid.

        # Arguments
            uneven_basis (string, optional) :
                The basis on which the grid has uneven spacing.
        """
        #Read data
        bases = self.current_bases

        # Put data on an even grid
        x, y, z = [match_basis(dsets[next(iter(dsets))], bn) for bn in bases]
        uneven_index  = None
        if bases[0] == uneven_basis:
            even_x = np.linspace(x.min(), x.max(), len(x))
            even_y = y
            even_z = z
            uneven_index = 0
        elif bases[1] == uneven_basis:
            even_x = x
            even_y = np.linspace(y.min(), y.max(), len(y))
            even_z = z
            uneven_index = 1
        elif bases[2] == uneven_basis:
            even_x = x
            even_y = y
            even_z = np.linspace(z.min(), z.max(), len(z))
            uneven_index = 2
        else:
            even_x, even_y, even_z = x, y, z

        exx, eyy, ezz = None, None, None
        if uneven_index == 0:
            eyy, exx = np.meshgrid(even_y, even_x)
        elif uneven_index == 1:
            eyy, exx = np.meshgrid(even_y, even_x)
        elif uneven_index == 2:
            ezz, exx = np.meshgrid(even_z, even_x)

        file_data = OrderedDict()

        #TODO: Double-check logic here -- this is years old.
        for k in dsets.keys(): 
            file_data[k] = np.zeros(dsets[k][ni].shape)
            for i in range(file_data[k].shape[0]):
                if self.comm.rank == 0:
                    print('interpolating {} ({}/{})...'.format(k, i+1, file_data[k].shape[0]))
                    stdout.flush()
                if uneven_index is None:
                    file_data[k][i,:] = tdsets[k][ni][i,:]
                elif uneven_index == 2:
                    for j in range(file_data[k].shape[-2]): # loop over y
                        interp = RegularGridInterpolator((x.flatten(), z.flatten()), tsk[k][i,:,j,:], method='linear')
                        file_data[k][i,:,j,:] = interp((exx, ezz))
                else:
                    for j in range(file_data[k].shape[-1]): # loop over z
                        interp = RegularGridInterpolator((x.flatten(), y.flatten()), tsk[k][i,:,:,j], method='linear')
                        file_data[k][i,:,:,j] = interp((exx, eyy))

        return file_data



    def _get_bounds(self, pdf_list):
        """
        Finds the global minimum and maximum value of histogram boundaries
        """
        
        with self.my_sync:
            if self.idle : return

            bounds = OrderedDict()
            for field in pdf_list:
                bounds[field] = np.zeros(2)
                bounds[field][:] = np.nan

            while self.writes_remain():
                dsets, ni = self.get_dsets(pdf_list)
                for field in pdf_list:
                    if np.isnan(bounds[field][0]):
                        bounds[field][0], bounds[field][1] = dsets[field][ni].min(), dsets[field][ni].max()

            for field in pdf_list:
                buff     = np.zeros(1)
                self.comm.Allreduce(bounds[field][0], buff, op=MPI.MIN)
                bounds[field][0] = buff

                self.comm.Allreduce(bounds[field][1], buff, op=MPI.MAX)
                bounds[field][1] = buff

            return bounds


    def calculate_pdfs(self, pdf_list, bins=100, threeD=False, bases=['x', 'z'], **kwargs):
        """
        Calculate probability distribution functions of the specified tasks.

        # Arguments
            pdf_list (list) :
                The names of the tasks to create PDFs of
            bins (int, optional) :
                The number of bins the PDF should have
            threeD (bool, optional) :
                If True, find PDF of a 3D volume
            bases (list, optional) :
                A list of strings of the bases over which the simulation information spans. 
                Should have 2 elements if threeD is false, 3 elements if threeD is true.
            **kwargs : additional keyword arguments for the self._get_interpolated_slices() function.
        """
        self.current_bases = bases
        bounds = self._get_bounds(pdf_list)

        histograms = OrderedDict()
        bin_edges  = OrderedDict()
        for field in pdf_list:
            histograms[field] = np.zeros(bins)
            bin_edges[field] = np.zeros(bins+1)

        with self.my_sync:
            if self.idle : return

            while self.writes_remain():
                dsets, ni = self.get_dsets(pdf_list)
                if threeD:
                    file_data = self._get_interpolated_volumes(dsets, ni, **kwargs)
                else:
                    file_data = self._get_interpolated_slices(dsets, ni, **kwargs)

                # Create histograms of data
                for field in pdf_list:
                    hist, bin_vals = np.histogram(file_data[field], bins=bins, range=bounds[field])
                    histograms[field] += hist
                    bin_edges[field] = bin_vals


            for field in pdf_list:
                loc_hist    = np.array(histograms[field], dtype=np.float64)
                global_hist = np.zeros_like(loc_hist, dtype=np.float64)
                self.comm.Allreduce(loc_hist, global_hist, op=MPI.SUM)

                dx = bin_edges[field][1]-bin_edges[field][0]
                x_vals  = bin_edges[field][:-1] + dx/2
                pdf     = global_hist/np.sum(global_hist)/dx
                self.pdfs[field] = (pdf, x_vals, dx)

            self._calculate_pdf_statistics()
        

    def plot_pdfs(self, dpi=150, **kwargs):
        """
        Plot the probability distribution functions and save them to file.

        # Arguments
            dpi (int, optional) :
                Pixel density of output image.
            **kwargs : additional keyword arguments for PlotGrid()
        """
        with self.my_sync:
            if self.comm.rank != 0: return

            grid = RegularPlotGrid(num_rows=1,num_cols=1, **kwargs)
            ax = grid.axes['ax_0-0']
            
            for k, data in self.pdfs.items():
                pdf, xs, dx = data
                mean, stdev, skew, kurt = self.pdf_stats[k]
                title = r'$\mu$ = {:.2g}, $\sigma$ = {:.2g}, skew = {:.2g}, kurt = {:.2g}'.format(mean, stdev, skew, kurt)
                ax.set_title(title)
                ax.axvline(mean, c='orange')

                ax.plot(xs, pdf, lw=2, c='k')
                ax.fill_between((mean-stdev, mean+stdev), pdf.min(), pdf.max(), color='orange', alpha=0.5)
                ax.fill_between(xs, 1e-16, pdf, color='k', alpha=0.5)
                ax.set_xlim(xs.min(), xs.max())
                ax.set_ylim(pdf[pdf > 0].min(), pdf.max())
                ax.set_yscale('log')
                ax.set_xlabel(k)
                ax.set_ylabel('P({:s})'.format(k))

                grid.fig.savefig('{:s}/{:s}_pdf.png'.format(self.out_dir, k), dpi=dpi, bbox_inches='tight')
                ax.clear()

            self._save_pdfs()

    def _save_pdfs(self):
        """ 
        Save PDFs to file. For each PDF, e.g., 'entropy' and 'w', the file will have a dataset with:
            xs  - the x-values of the PDF
            pdf - the (normalized) y-values of the PDF
            dx  - the spacing between x values, for use in integrals.
        """
        if self.comm.rank == 0:
            with h5py.File('{:s}/pdf_data.h5'.format(self.out_dir), 'w') as f:
                for k, data in self.pdfs.items():
                    pdf, xs, dx = data
                    this_group = f.create_group(k)
                    for d, n in ((pdf, 'pdf'), (xs, 'xs')):
                        dset = this_group.create_dataset(name=n, shape=d.shape, dtype=np.float64)
                        f['{:s}/{:s}'.format(k, n)][:] = d
                    dset = this_group.create_dataset(name='dx', shape=(1,), dtype=np.float64)
                    f['{:s}/dx'.format(k)][0] = dx

