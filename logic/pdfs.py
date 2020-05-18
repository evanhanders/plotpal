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

path.insert(0, './plot_logic')
from plot_logic.file_reader import SingleFiletypePlotter
from plot_logic.plot_grid import PlotGrid

logger = logging.getLogger(__name__.split('.')[-1])


class PdfPlotter(SingleFiletypePlotter):
    """
    A class for plotting probability distributions of a dedalus output.

    PDF plots are currently implemented for 2D slices and 3D volumes. 
    When one axis is represented by polynomials that exist on an uneven basis (e.g., Chebyshev),
    that basis is evenly interpolated to avoid skewing of the distribution by uneven grid sampling.

    Public Methods:
    ---------------
    calculate_pdfs
    plot_pdfs


    Additional Attributes:
    ----------------------
    pdfs : OrderedDict
        Contains PDF data (x, y, dx)
    pdf_stats : OrderedDict
        Contains scalar stats for the PDFS (mean, stdev, skew, kurtosis)
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PDF plotter.

        Arguments:
        -----------
        *args, **kwargs : Additional keyword arguments for super().__init__() 
        """
        super(PdfPlotter, self).__init__(*args, distribution='even', **kwargs)
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

    
    def _get_interpolated_slices(self, file_name, pdf_list, bases=['x', 'z'], uneven_basis=None):
        """
        For 2D data on an uneven grid, interpolates that data on to an evenly spaced grid.

        Arguments:
        ----------
        pdf_list : list
            list of strings of the Dedalus tasks to make PDFs of.
        bases : list, optional
            The names of the Dedalus bases on which the data exists
        uneven_basis : string, optional
            The basis on which the grid has uneven spacing.
        """
        #Read data
        bs, tsk, writenum, times = self.reader.read_file(file_name, bases=bases, tasks=pdf_list)

        # Put data on an even grid
        x, y = bs[bases[0]], bs[bases[1]]
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
        for k in pdf_list: 
            file_data[k] = np.zeros(tsk[k].shape)
            for i in range(file_data[k].shape[0]):
                if self.reader.comm.rank == 0:
                    print('interpolating {} ({}/{})...'.format(k, i+1, file_data[k].shape[0]))
                    stdout.flush()
                interp = RegularGridInterpolator((x.flatten(), y.flatten()), tsk[k][i,:], method='linear')
                file_data[k][i,:] = interp((exx, eyy))

        return file_data

    def _get_interpolated_volumes(self, file_name, pdf_list, bases=['x', 'y', 'z'], uneven_basis=None):
        """
        For 3D data on an uneven grid, interpolates that data on to an evenly spaced grid.

        Arguments:
        ----------
        pdf_list : list
            list of strings of the Dedalus tasks to make PDFs of.
        bases : list, optional
            The names of the Dedalus bases on which the data exists
        uneven_basis : string, optional
            The basis on which the grid has uneven spacing.
        """
        #Read data
        bs, tsk, writenum, times = self.reader.read_file(file_name, bases=bases, tasks=pdf_list)

        # Put data on an even grid
        x, y, z = bs[bases[0]], bs[bases[1]], bs[bases[2]]
        dedalus_bases = (x, y, z)
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
        for k in pdf_list: 
            file_data[k] = np.zeros(tsk[k].shape)
            for i in range(file_data[k].shape[0]):
                if self.reader.comm.rank == 0:
                    print('interpolating {} ({}/{})...'.format(k, i+1, file_data[k].shape[0]))
                    stdout.flush()
                if uneven_index is None:
                    file_data[k][i,:] = tsk[k][i,:]
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

            for i, f in enumerate(self.files):
                if self.reader.comm.rank == 0:
                    print('getting bounds from file {}/{}...'.format(i+1, len(self.files)))
                    stdout.flush()
                bs, tsk, writenum, times = self.reader.read_file(f, bases=[], tasks=pdf_list)
                for field in pdf_list:
                    if np.isnan(bounds[field][0]):
                        bounds[field][0], bounds[field][1] = tsk[field].min(), tsk[field].max()

            for field in pdf_list:
                buff     = np.zeros(1)
                self.dist_comm.Allreduce(bounds[field][0], buff, op=MPI.MIN)
                bounds[field][0] = buff

                self.dist_comm.Allreduce(bounds[field][1], buff, op=MPI.MAX)
                bounds[field][1] = buff

            return bounds


    def calculate_pdfs(self, pdf_list, bins=100, threeD=False, **kwargs):
        """
        Calculate probability distribution functions of the specified tasks.

        Arguments:
        ----------
        pdf_list : list
            The names of the tasks to create PDFs of
        bins : int, optional
            The number of bins the PDF should have
        **kwargs : additional keyword arguments for the self._get_interpolated_slices() function.
        """
        bounds = self._get_bounds(pdf_list)

        histograms = OrderedDict()
        bin_edges  = OrderedDict()
        for field in pdf_list:
            histograms[field] = np.zeros(bins)
            bin_edges[field] = np.zeros(bins+1)

        with self.my_sync:
            if self.idle : return

            for i, f in enumerate(self.files):
                if self.reader.comm.rank == 0:
                    print('reading file {}/{}...'.format(i+1, len(self.files)))
                    stdout.flush()
                if threeD:
                    file_data = self._get_interpolated_volumes(f, pdf_list, **kwargs)
                else:
                    file_data = self._get_interpolated_slices(f, pdf_list, **kwargs)

                # Create histograms of data
                for field in pdf_list:
                    hist, bin_vals = np.histogram(file_data[field], bins=bins, range=bounds[field])
                    histograms[field] += hist
                    bin_edges[field] = bin_vals


            for field in pdf_list:
                loc_hist    = np.array(histograms[field], dtype=np.float64)
                global_hist = np.zeros_like(loc_hist, dtype=np.float64)
                self.dist_comm.Allreduce(loc_hist, global_hist, op=MPI.SUM)

                dx = bin_edges[field][1]-bin_edges[field][0]
                x_vals  = bin_edges[field][:-1] + dx/2
                pdf     = global_hist/np.sum(global_hist)/dx
                self.pdfs[field] = (pdf, x_vals, dx)

            self._calculate_pdf_statistics()
        

    def plot_pdfs(self, dpi=150, **kwargs):
        """
        Plot the probability distribution functions and save them to file.

        Arguments:
        ----------
        dpi : int, optional
            Pixel density of output image.
        **kwargs : additional keyword arguments for PlotGrid()
        """
        with self.my_sync:
            if self.reader.comm.rank != 0: return

            grid = PlotGrid(1,1, **kwargs)
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
        if self.reader.comm.rank == 0:
            with h5py.File('{:s}/pdf_data.h5'.format(self.out_dir), 'w') as f:
                for k, data in self.pdfs.items():
                    pdf, xs, dx = data
                    this_group = f.create_group(k)
                    for d, n in ((pdf, 'pdf'), (xs, 'xs')):
                        dset = this_group.create_dataset(name=n, shape=d.shape, dtype=np.float64)
                        f['{:s}/{:s}'.format(k, n)][:] = d
                    dset = this_group.create_dataset(name='dx', shape=(1,), dtype=np.float64)
                    f['{:s}/dx'.format(k)][0] = dx

