"""
Core photometry utilities.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..data.core import Data
from ..data.spectral import Spectrum1D

from ..plot.general import Plot

from astropy.io import ascii, fits
from astropy.units import angstrom, erg, cm, s, Hz, Jy, spectral_density

from math import sqrt, log10

import numpy as np

import os

__all__ = ['get_bands', 'Passband', 'mag2flux', 'flux2mag']

data_path = os.path.abspath(__file__).rsplit('/', 1)[0] + '/data'

extinction_map = dict(VIMOS='paranal_atmos.dat',
                      HawkI='paranal_atmos.dat',
                      KPNO_Mosaic='kpno_atmos.dat',
                      CTIO_Mosaic='ctio_atmos.dat',
                      CFH12K='mko_atmos.dat',
                      CFHT_Megacam='mko_atmos.dat')

# Vega:
tab = ascii.read('{}/reference/Vega_bohlin2006'.format(data_path),
                 names=['wavelength', 'flux'])
tab['wavelength'].unit = angstrom
tab['flux'].unit = erg / s / cm ** 2 / angstrom
VEGA = Spectrum1D.from_table(tab, 'wavelength', 'flux')

# AB SED has constant flux density (f_nu) 3631 Jy, see
# http://www.sdss.org/dr5/algorithms/fluxcal.html
fnuAB = 3631 * Jy   # erg/s/cm^2/Hz
wave = np.logspace(1, 10, 100000) * angstrom
fluxAB = fnuAB.to(erg / s / cm ** 2 / angstrom,
                  equivalencies=spectral_density(wave))
AB = Spectrum1D(wave, fluxAB)

# Don't clutter the namespace:
del wave, fnuAB, fluxAB


def _listfiles(top_directory):

    names = [n for n in os.listdir(top_directory)
             if os.path.isdir(top_directory + n)]
    files = dict([(n, []) for n in names])

    for name in sorted(names):
        for n in sorted(os.listdir(top_directory + name)):
            if (n != 'README' and not os.path.isdir(
                    top_directory + name + '/' + n)):
                files[name].append(n)

    return files


PASSBANDS = _listfiles('{}/passbands/'.format(data_path))


def get_bands(instrument=None, names=None):
    """
    Get one or more passbands by giving the instrument and filename.

    If `names` is not given, then every passband for that instrument is
    returned. This can be a list, a single string, or a comma-separated
    string of values.

    Passband instruments and filenames are listed in the dictionary
    `PASSBANDS` which is returned by calling get_bands().

    Parameters
    ----------
    instrument : str
        The instrument.

    names : str or list
        Passband names.

    """

    if instrument is None:
        return PASSBANDS

    if isinstance(names, basestring):

        if ',' in names:
            names = [n.strip() for n in names.split(',')]

        else:
            return Passband('{0}/{1}'.format(instrument, names))

    elif names is None:
        names = PASSBANDS[instrument]

    return [Passband('{0}/{1}'.format(instrument, n)) for n in names]


class Passband(object):
    """
    Describes a filter transmission curve.

    Parameters
    ----------
    filename : str
        Passband filename. Should be an ASCII or FITS table containing
        wavelength in Angstrom in the first column and relative transmission
        efficiency in the second.

    Attributes
    ----------
    wavelength : array
        Wavelength (Angstrom).

    transmission : array
        Normalised transmission, including atmospheric extinction.
        May or may not include extinction from the optical path.

    effective_wavelength : float
        Effective wavelength of the passband.

    """

    def __init__(self, filename):

        if not filename.startswith('{}/passbands/'.format(data_path)):
            file_path = '{0}/passbands/{1}'.format(data_path, filename)

        else:
            file_path = filename

        if file_path.endswith('.fits'):
            t = fits.getdata(file_path, 1)
            self.wavelength = t['wa'].data
            self.transmission = t['tr'].data

        else:
            t = ascii.read(file_path)
            self.wavelength = t['col1'].data
            self.transmission = t['col2'].data

        isort = self.wavelength.argsort()
        self.wavelength = self.wavelength[isort]
        self.transmission = self.transmission[isort]

        prefix, filtername = os.path.split(filename)
        _, instrument = os.path.split(prefix)
        self.name = filename

        if instrument in extinction_map:
            data = ascii.read('{0}/extinction/{1}'.format(
                data_path, extinction_map[instrument]))
            wavelength, extinction = data['col1'], data['col2']
            self.atmosphere = np.interp(
                self.wavelength, wavelength, 10 ** (-0.4 * extinction))
            self.transmission *= self.atmosphere

        else:
            self.atmosphere = None

        # Trim away areas where band transmission is negligibly small
        # (< 0.01% of the peak transmission):
        isort = self.transmission.argsort()
        sorted_transmission = self.transmission[isort]
        max_transmission = sorted_transmission[-1]
        imax = isort[-1]
        indicies = isort[sorted_transmission < 1e-4 * max_transmission]

        if len(indicies) > 0:

            i = 0
            c0 = indicies < imax

            if c0.any():
                i = indicies[c0].max()

            j = len(self.wavelength) - 1
            c0 = indicies > imax

            if c0.any():
                j = indicies[c0].min()

            i = min(abs(i - 2), 0)
            j += 1

            self.wavelength = self.wavelength[i:j] * angstrom
            self.transmission = self.transmission[i:j]

            if self.atmosphere is not None:
                self.atmosphere = self.atmosphere[i:j]

        # Normalise:
        self.normalised_transmission = (
            self.transmission / np.trapz(self.transmission, self.wavelength))

        # Calculate the effective wavelength for the passband. This is
        # the same as equation (3) of Carter et al. (2009):
        a = np.trapz(self.transmission * self.wavelength.value)
        b = np.trapz(self.transmission / self.wavelength.value)
        self.effective_wavelength = sqrt(a / b) * angstrom

        # AB and Vega conversions:
        self.flux = {'Vega': VEGA.calculate_flux(self),
                     'AB': AB.calculate_flux(self)}

    def __repr__(self):
        return 'Passband({0})'.format(self.name)

    def plot(self, atmosphere=False, **kwargs):
        """
        Plots the passband. This will be the non-normalised transmission,
        which may or may not include losses from the atmosphere and telescope
        optics.

        Parameters
        ----------
        atmosphere : bool, optional
            Option to plot the applied atmospheric extinction.

        """

        p = Plot()
        p.axes[0].plot(self.wavelength, self.transmission, **kwargs)

        if self.atmosphere is not None and atmosphere:
            p.axes[0].plot(self.wavelength, self.atmosphere,
                           label='Applied atmospheric transmission', **kwargs)

        p.tidy()
        p.labels('Wavelength ($\\AA$)', 'Transmission')

        if atmosphere:
            p.axes[0].legend(fancybox=True)

        p.display()


def mag2flux(magnitude, band):
    """
    Converts a given AB magnitude into flux in the given band in
    erg / s / cm^2 / Angstrom.

    Parameters
    ----------
    magnitude : float
        AB magnitude.

    band : `igmtools.photometry.Passband`
        The passband.

    Returns
    -------
    flux : `astropy.units.Quantity`
        Flux in erg / s / cm^2 / Angstrom.

    """

    fnu = 10 ** (-(magnitude + 48.6) / 2.5) * erg / s / cm ** 2 / Hz
    flux = fnu.to(erg / s / cm ** 2 / angstrom,
                  equivalencies=spectral_density(band.effective_wavelength))

    return flux


def flux2mag(flux, band):
    """
    Converts flux in erg / s / cm^2 / Angstrom to AB magnitudes.

    flux : float
        Flux (erg / s / cm^2 / Angstrom).

    band : `igmtools.photometry.Passband`
        The passband.

    Returns
    -------
    magnitude : float
        AB magnitude.

    """

    if hasattr(flux, 'unit'):
        flux = flux.to(erg / s / cm ** 2 / angstrom)

    else:
        flux = flux * erg / s / cm ** 2 / angstrom

    fnu = flux.to(erg / s / cm ** 2 / Hz,
                  equivalencies=spectral_density(band.effective_wavelength))
    magnitude = -2.5 * log10(fnu.value) - 48.6

    return magnitude
