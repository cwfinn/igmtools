"""
Plotting utilities.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.constants import c

from matplotlib.artist import Artist
from matplotlib.colors import LinearSegmentedColormap

import astropy.units as u

import numpy as np

__all__ = ['get_nrows_ncols', 'truncate_colormap', 'GrowFilter',
           'FilteredArtistList']

c_kms = c.to(u.km / u.s).value  # speed of light in km/s


def get_nrows_ncols(npar):
    """
    Optimises the subplot grid layout.

    Parameters
    ----------
    npar : int
        Number of subplots.

    Returns
    -------
    nrows : int
        Number of figure rows.

    ncols : int
        Number of figure columns.

    """

    ncols = max(int(np.sqrt(npar)), 1)
    nrows = ncols

    while npar > (nrows * ncols):
        nrows += 1

    return nrows, ncols


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a colourmap.

    Parameters
    ----------
    cmap : `matplotlib.colors.LinearSegmentedColormap`
        Input colourmap.

    minval, maxval : float
        Interval to sample (minval >= 0, maxval <= 1)

    n : int
        Sampling density.

    Returns
    -------
    new_cmap : `matplotlib.colors.LinearSegmentedColormap`
        Truncated colourmap.

    """

    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(
            n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap


def smooth1d(x, window_len):
    s = np.r_[2 * x[0] - x[window_len:1:-1], x,
              2 * x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window_len - 1:-window_len + 1]


def smooth2d(a, sigma=3):
    window_len = max(int(sigma), 3) * 2 + 1
    a1 = np.array([smooth1d(x, window_len) for x in np.asarray(a)])
    a2 = np.transpose(a1)
    a3 = np.array([smooth1d(x, window_len) for x in a2])
    a4 = np.transpose(a3)

    return a4


class BaseFilter(object):
    @staticmethod
    def prepare_image(src_image, pad):
        ny, nx, depth = src_image.shape
        padded_src = np.zeros([pad * 2 + ny, pad * 2 + nx, depth], dtype='d')
        padded_src[pad:-pad, pad:-pad, :] = src_image[:, :, :]

        return padded_src

    @staticmethod
    def get_pad():
        return 0

    def __call__(self, im, dpi):
        pad = self.get_pad()
        padded_src = self.prepare_image(im, pad)
        tgt_image = self.process_image(padded_src, dpi)

        return tgt_image, -pad, -pad


class GrowFilter(BaseFilter):
    def __init__(self, pixels, color=None):
        self.pixels = pixels
        if color is None:
            self.color = (1, 1, 1)
        else:
            self.color = color

    def __call__(self, im, dpi):
        pad = self.pixels
        ny, nx, depth = im.shape
        new_im = np.empty([pad * 2 + ny, pad * 2 + nx, depth], dtype='d')
        alpha = new_im[:, :, 3]
        alpha.fill(0)
        alpha[pad:-pad, pad:-pad] = im[:, :, -1]
        alpha2 = np.clip(
            smooth2d(alpha, int(self.pixels / 72. * dpi)) * 5, 0, 1)
        new_im[:, :, -1] = alpha2
        new_im[:, :, :-1] = self.color
        offsetx, offsety = -pad, -pad

        return new_im, offsetx, offsety


class FilteredArtistList(Artist):
    def __init__(self, artist_list, filter0):
        self._artist_list = artist_list
        self._filter = filter0
        super(FilteredArtistList, self).__init__()

    def draw(self, renderer, *args, **kwargs):
        renderer.start_rasterizing()
        renderer.start_filter()

        for a in self._artist_list:
            a.draw(renderer, *args, **kwargs)

        renderer.stop_filter(self._filter)
        renderer.stop_rasterizing()
