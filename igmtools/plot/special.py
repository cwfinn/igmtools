"""
Specialist plots.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .general import Plot

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes

from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper

from matplotlib.axes import Axes

import numpy as np


class ConePlot(Plot):
    """
    Defines the layout of a cone plot.

    """

    def __init__(self, rotation, ra_min, ra_max, z_min, z_max, stretch=1,
                 nrows=1, ncols=1, npar=1, width=8.0, aspect=0.8,
                 gridspec=None, blank=True, fontsize=16, legend_fontsize=14,
                 family='serif', style='Times', weight='normal', usetex=False):

        super(ConePlot, self).__init__(
            nrows, ncols, npar, width, aspect, gridspec, blank, fontsize,
            legend_fontsize, family, style, weight, usetex)

        # Rotate for better orientation:
        rotate = Affine2D().translate(rotation, 0)

        # Scale degree to radians:
        scale = Affine2D().scale(np.pi * stretch / 180, 1)

        transform = rotate + scale + PolarAxes.PolarTransform()

        grid_locator1 = angle_helper.LocatorHMS(4)
        grid_locator2 = MaxNLocator(5)
        tick_formatter1 = angle_helper.FormatterHMS()

        self.grid_helper = floating_axes.GridHelperCurveLinear(
            transform, extremes=(ra_min, ra_max, z_min, z_max),
            grid_locator1=grid_locator1, grid_locator2=grid_locator2,
            tick_formatter1=tick_formatter1, tick_formatter2=None)

        ax = floating_axes.FloatingSubplot(
            self, 111, grid_helper=self.grid_helper)

        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['right'].set_axis_direction('top')

        ax.axis['bottom'].set_visible(False)
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')

        ax.axis['left'].label.set_text('Redshift')
        ax.axis['top'].label.set_text('RA (J2000)')

        aux_ax = ax.get_aux_axes(transform)
        aux_ax.patch = ax.patch
        ax.patch.zorder = 0.9

        self.add_subplot(ax)
        self.aux_ax = aux_ax
