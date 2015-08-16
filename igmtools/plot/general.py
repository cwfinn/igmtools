"""
General plotting recipes.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astroML.density_estimation import knuth_bin_width, bayesian_blocks

from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyQt4.QtGui import QApplication, QWidget, QVBoxLayout

from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT, FigureManagerQT)

from matplotlib.ticker import AutoMinorLocator
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib import rc

import numpy as np

import sys
import os

os.environ['PATH'] += ':/usr/texbin'


class Plot(Figure):
    """
    Defines the layout and style of a plotting space, with class inheritance
    from `matplotlib.figure.Figure`. Includes methods for labelling and
    white-space optimisation.

    Parameters
    ----------
    nrows : int, optional
        Number of figure rows (default = 1).

    ncols : int, optional
        Number of figure columns (default = 1).

    npar : int, optional
        Number of subplots (default = 1).

    width : float, optional
        Width of the figure in inches (default = 8.0).

    aspect : float, optional
        Aspect ratio for the figure (default = 0.8).

    gridspec : `gridspec.GridSpec`, optional
        Gridspec object, defining subplot grid geometry (default = None).

    blank : bool, optional
        If True, initialise a blank plotting space (default = False).

    fontsize : int, optional
        Font size for the axis labels and title (default = 16).

    legend_fontsize : int, optional
        Font size for the legend (default = 14).

    family : str, optional
        Font family (default = `serif`).

    style : str, optional
        Font style (default = `Times`).

    weight : str, optional
        Font weight (default = `normal`).

    usetex : bool, optional
        Option to use TeX fonts (default = False).

    """

    def __init__(self, nrows=1, ncols=1, npar=1, width=8.0, aspect=0.8,
                 gridspec=None, blank=False, fontsize=16, legend_fontsize=14,
                 family='serif', style='Times', weight='normal', usetex=False,
                 mathstyle='stix', tick_major_length=7, tick_major_width=1.2,
                 tick_minor_length=4, tick_minor_width=1):

        self.nrows = nrows
        self.ncols = ncols
        self.npar = npar
        self.aspect = aspect

        if usetex:
            font = {'family': family,
                    family: [style],
                    'weight': weight,
                    'size': fontsize}
        else:
            font = {'family': family,
                    'weight': weight,
                    'size': fontsize}

        rc('font', **font)
        rc('text', usetex=usetex)
        rc('legend', **{'fontsize': legend_fontsize})
        rc('mathtext', fontset=mathstyle)

        height = width * aspect
        super(Plot, self).__init__(figsize=(width, height))

        if gridspec:
            [self.add_subplot(gridspec[i]) for i in range(npar)]
        elif not blank:
            [self.add_subplot(nrows, ncols, i + 1) for i in range(npar)]

        for ax in self.axes:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(which='major', length=tick_major_length,
                           width=tick_major_width)
            ax.tick_params(which='minor', length=tick_minor_length,
                           width=tick_minor_width)

    def set_title(self, title, ax):
        """
        Adds a title to the specified subplot.

        Parameters
        ----------
        title : str
            The title.

        ax : `matplotlib.axes.Axes`
            Axes instance.

        """

        xmin = ax.get_position().xmin
        xmax = ax.get_position().xmax
        xcentre = (xmin + xmax) / 2

        y = ax.get_position().ymax + 0.02

        self.text(xcentre, y, title, ha='center', va='center')

    def set_xlabel(self, xlabel, ax, location='bottom'):
        """
        Adds a label to the x-axis of the specified subplot.

        Parameters
        ----------
        xlabel : str
            Label for the x-axis.

        ax : `matplotlib.axes.Axes`
            Axes instance.

        location : {`bottom`, `top`}, optional
            Placement for the label (default = `bottom`)

        """

        xmin = ax.get_position().xmin
        xmax = ax.get_position().xmax
        xcentre = (xmin + xmax) / 2

        if location == 'bottom':
            self.text(xcentre, 0.03 / self.aspect, xlabel, ha='center',
                      va='center')

        elif location == 'top':
            self.text(xcentre, 0.97, xlabel, ha='center',
                      va='center')

    def set_ylabel(self, ylabel, ax, location='left'):
        """
        Adds a label to the y-axis of the specified subplot.

        Parameters
        ----------
        ylabel : str
            Label for the x-axis.

        ax : `matplotlib.axes.Axes`
            Axes instance.

        location : {`left`, `right`}, optional
            Placement for the label (default = `left`)

        """

        ymin = ax.get_position().ymin
        ymax = ax.get_position().ymax
        ycentre = (ymin + ymax) / 2

        if location == 'left':
            self.text(0.03, ycentre, ylabel, rotation=90, va='center',
                      ha='center')

        elif location == 'right':
            self.text(0.97, ycentre, ylabel, rotation=90, va='center',
                      ha='center')

    def labels(self, xlabel=None, ylabel=None, x2label=None, y2label=None):
        """
        Adds figure-wide axis labels.

        Parameters
        ----------
        xlabel : str, optional
            Label for the x-axis.

        ylabel : str, optional
            Label for the y-axis.

        x2label : str, optional
            Label for the opposite x-axis (also acts as a title).

        y2label : str, optional
            Label for the opposite y-axis.

        """

        xmin = min(ax.get_position().xmin for ax in self.axes)
        xmax = max(ax.get_position().xmax for ax in self.axes)
        xcentre = (xmin + xmax) / 2

        ymin = min(ax.get_position().ymin for ax in self.axes)
        ymax = max(ax.get_position().ymax for ax in self.axes)
        ycentre = (ymin + ymax) / 2

        if xlabel:
            self.text(xcentre, 0.03 / self.aspect, xlabel, ha='center',
                      va='center')

        if ylabel:
            self.text(0.03, ycentre, ylabel, rotation=90, va='center',
                      ha='center')

        if x2label:
            self.text(xcentre, 0.97, x2label, ha='center',
                      va='center')

        if y2label:
            self.text(0.97, ycentre, y2label, rotation=90, va='center',
                      ha='center')

    def histogram(self, data, bin_width='knuth', weights=None, density=None,
                  norm=None, ax=None, **kwargs):
        """
        Plots a histogram.

        Parameters
        ----------
        data : list or array
            Data to plot.

        bin_width : {'knuth', 'bayesian'} or float, optional
            Automatically determine the bin width using Knuth's rule
            (2006physics...5197K), with Bayesian blocks (2013ApJ...764..167S),
            or manually, choosing a floating point value.

        weights : array, optional
            An array of weights, of the same shape as `a`.  Each value in `a`
            only contributes its associated weight towards the bin count
            (instead of 1).  If `density` is True, the weights are normalized,
            so that the integral of the density over the range remains 1.

        density : bool, optional
            If False, the result will contain the number of samples
            in each bin.  If True, the result is the value of the
            probability *density* function at the bin, normalised such that
            the *integral* over the range is 1. Note that the sum of the
            histogram values will not be equal to 1 unless bins of unity
            width are chosen; it is not a probability *mass* function.

        norm : int or float
            Custom normalisation.

        ax : `matplotlib.axes.Axes`, optional
            Axes instance.

        """

        # Axes instance:
        if ax is None:
            ax = self.axes[0]

        elif not isinstance(ax, Axes):
            raise TypeError('ax must be of type `matplotlib.axes.Axes`')

        # Convert list to array:
        if isinstance(data, list):
            data = np.array(data)

        if bin_width == 'knuth':
            _, bins = knuth_bin_width(data, return_bins=True)

        elif bin_width == 'bayesian':
            bins = bayesian_blocks(data)

        elif isinstance(bin_width, (int, float)):
            bins = np.arange(data.min(), data.max(), bin_width)

        else:
            raise ValueError('bin_width must be a number, or one of'
                             '(`knuth`, `bayesian`)')

        # Ensure padding with empty bins:
        dx = np.diff(bins).min()
        bins = np.pad(bins, (1, 2), mode='linear_ramp',
                      end_values=(bins[0] - dx, bins[-1] + 2 * dx))

        # Calculate histogram:
        histogram, bins = np.histogram(
            data, bins, weights=weights, density=density)
        if norm:
            histogram /= norm

        # Plot data:
        ax.plot(bins[:-1] + np.diff(bins) / 2, histogram,
                drawstyle='steps-mid', **kwargs)

    def colourmap(self, xbins, ybins, data, orientation='vertical',
                  extend='both', ticks=None, tick_labels=None, ax=None,
                  **kwargs):
        """
        Plots a colourmap.

        Parameters
        ----------
        xbins : list or array, shape (N,)
            x-axis bins.

        ybins : list or array, shape (M,)
            y-axis bins.

        data : array, shape (N, M)
            Data to plot.

        orientation : {`vertical`, `horizontal`}, optional
            Orientation of the colourbar (default = `vertical`).

        extend : {`neither`, `both`, `min`, `max`}, optional
            Make pointed end(s) on the colourbar for out-of-range values
            (default = `both`).

        ticks : list, optional
            Custom tick marker positions.

        tick_labels : list, optional
            Custom tick labels.

        ax : `matplotlib.axes.Axes`, optional
            Axes instance.

        """

        # Axes instance:
        if ax is None:
            ax = self.axes[0]

        elif not isinstance(ax, Axes):
            raise TypeError('ax must be of type matplotlib.axes.Axes')

        # Convert lists to arrays:
        if isinstance(xbins, list):
            xbins = np.array(xbins)

        if isinstance(ybins, list):
            ybins = np.array(ybins)

        # Make axis for colourbar:
        divider = make_axes_locatable(self.gca())
        position = 'right' if orientation == 'vertical' else 'bottom'
        cax = divider.append_axes(position, '5%', pad='3%')

        # Plot data:
        im = ax.pcolormesh(xbins, ybins, data, **kwargs)
        cbar = self.colorbar(im, cax=cax, ax=ax, orientation=orientation,
                             extend=extend, ticks=ticks)

        # Option for custom tick labels:
        if tick_labels is not None:
            cbar.ax.set_yticklabels(tick_labels)

        # Adjust axes:
        ax.set_xlim(xbins.min(), xbins.max())
        ax.set_ylim(ybins.min(), ybins.max())

    def tidy(self, border_space=2.5, horizontal_space=0.0,
             vertical_space=0.0, shared_axes=False):
        """
        Cleans up the plotting space, removing excess white-space between
        subplots.

        Parameters
        ----------
        border_space : float, optional
            Space around the border, in units of the fontsize (default = 2.5).

        horizontal_space : float, optional
            Horizontal space between subplots, in units of the fontsize
            (default = 0.0).

        vertical_space : float, optional
            Vertical space between subplots, in units of the fontsize
            (default = 0.0).

        shared_axes : bool, optional
            Option to remove unnecessary axis labels, for axes that are
            shared across all subplots (default = False).

        """

        if shared_axes:

            extra = self.nrows * self.ncols - self.npar

            for i in range(self.nrows):
                for j in range(self.ncols - 1):
                    if (i == self.nrows - 1) & (j >= self.ncols - 1 - extra):
                        break
                    self.axes[j + (self.ncols * i) + 1].set_yticklabels([])

            for i in range(self.npar - self.ncols):
                self.axes[i].set_xticklabels([])

        self.tight_layout(pad=border_space, h_pad=horizontal_space,
                          w_pad=vertical_space)

    def display(self):
        """
        Show the plot in an interactive window.

        """

        # Host widget for show():
        app = QApplication(sys.argv)
        app.aboutToQuit.connect(app.deleteLater)
        main_frame = QWidget()

        # Attach canvas to host widget:
        canvas = FigureCanvasQTAgg(self)
        canvas.setParent(main_frame)

        # Make navigation toolbar:
        mpl_toolbar = NavigationToolbar2QT(self.canvas, main_frame)

        # Set up layout:
        vbox = QVBoxLayout()
        vbox.addWidget(mpl_toolbar)
        vbox.addWidget(self.canvas)

        # Set the layout to the host widget:
        main_frame.setLayout(vbox)

        # Set figure manager:
        FigureManagerQT(self.canvas, 1)

        # Show plot:
        self.show()
        app.exec_()
