"""
Spectral data representation.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .core import Data
from .atomic import get_atomdat

from astropy.units import (erg, km, cm, s, angstrom, spectral,
                           spectral_density, Quantity, UnitsError)
from astropy.constants import m_e, c, e
from astropy.table import Table, Column

from math import pi, sqrt, exp, log10
from warnings import warn

import numpy as np

__all__ = ['Spectrum2D', 'Spectrum1D', 'Absorber', 'EmissionLine']

e2_me_c = (e.esu ** 2 / (m_e.cgs * c.cgs)).to(cm ** 2 / s)
c_kms = c.to(km / s)

atomdat = None


def find_bin_edges(bin_centres):
    """
    Find the bin edges given the bin centres.

    Parameters
    ----------
    bin_centres : array, shape (N,)
        The bin centres.

    Returns
    -------
    bins : array, shape (N + 1,)
        The bin edges.

    """

    if not isinstance(bin_centres, np.ndarray):
        bin_centres = np.asarray(bin_centres)

    edges = bin_centres[:-1] + 0.5 * (bin_centres[1:] - bin_centres[:-1])
    bins = np.concatenate(([2 * bin_centres[0] - edges[0]], edges,
                           [2 * bin_centres[-1] - edges[-1]]))

    return bins


class Spectrum2D(object):
    """
    A 2D spectrum.

    Parameters
    ----------
    dispersion : `astropy.units.Quantity` or array, shape (N,)
        Spectral dispersion axis.

    data : array, shape (N, M)
        The spectral data.

    unit : `astropy.units.UnitBase` or str, optional
        Unit for the dispersion axis.

    """

    def __init__(self, dispersion, data, unit=None):

        self.dispersion = Quantity(dispersion, unit=unit)

        if unit is not None:
            self.wavelength = self.dispersion.to(angstrom)

        else:
            self.wavelength = self.dispersion

        self.data = data


class Spectrum1D(object):
    """
    A 1D spectrum. Assumes wavelength units unless otherwise specified.

    Parameters
    ----------
    dispersion : `astropy.units.Quantity` or array
        Spectral dispersion axis.

    flux : `igmtools.data.Data`, `astropy.units.Quantity` or array
        Spectral flux. Should have the same length as `dispersion`.

    error : `astropy.units.Quantity` or array, optional
        Error on each flux value.

    continuum : `astropy.units.Quantity` or array, optional
        An estimate of the continuum flux.

    mask : array, optional
        Mask for the spectrum. The values must be False where valid and True
        where not.

    unit : `astropy.units.UnitBase` or str, optional
        Spectral unit.

    dispersion_unit : `astropy.units.UnitBase` or str, optional
        Unit for the dispersion axis.

    meta : dict, optional
        Meta data for the spectrum.

    """

    def __init__(self, dispersion, flux, error=None, continuum=None,
                 mask=None, unit=None, dispersion_unit=None, meta=None):

        _unit = flux.unit if unit is None and hasattr(flux, 'unit') else unit

        if isinstance(error, (Quantity, Data)):
            if error.unit != _unit:
                raise UnitsError('The error unit must be the same as the '
                                 'flux unit.')
            error = error.value

        elif isinstance(error, Column):
            if error.unit != _unit:
                raise UnitsError('The error unit must be the same as the '
                                 'flux unit.')
            error = error.data

        # Set zero error elements to NaN:
        if error is not None:
            zero = error == 0
            error[zero] = np.nan

            # Mask these elements:
            if mask is not None:
                self.mask = mask | np.isnan(error)

            else:
                self.mask = np.isnan(error)

        # If dispersion is a `Quantity`, `Data`, or `Column` instance with the
        # unit attribute set, that unit is preserved if `dispersion_unit` is
        # None, but overriden otherwise
        self.dispersion = Quantity(dispersion, unit=dispersion_unit)

        if dispersion_unit is not None:
            self.wavelength = self.dispersion.to(angstrom)

        else:
            # Assume wavelength units:
            self.wavelength = self.dispersion

        self.flux = Data(flux, error, unit)

        if continuum is not None:
            self.continuum = Quantity(continuum, unit=unit)

        else:
            self.continuum = None

        self.meta = meta

    @classmethod
    def from_table(cls, table, dispersion_column, flux_column,
                   error_column=None, continuum_column=None, unit=None,
                   dispersion_unit=None):
        """
        Initialises a `Spectrum1D` object from an `astropy.table.Table`
        instance.

        Parameters
        ----------
        table : `astropy.table.Table`
            Contains information used to construct the spectrum. Must have
            columns for the dispersion axis and the spectral flux.

        dispersion_column : str
            Name for the dispersion column.

        flux_column : str
            Name for the flux column.

        error_column : str, optional
            Name for the error column.

        continuum_column : str, optional
            Name for the continuum column.

        unit : `astropy.units.UnitBase` or str, optional
            Spectral unit.

        dispersion_unit : `astropy.units.UnitBase` or str, optional
            Unit for the dispersion axis.

        """

        dispersion = Quantity(table[dispersion_column])
        flux = Quantity(table[flux_column])

        if error_column is not None:
            error = Quantity(table[error_column])
        else:
            error = None

        if continuum_column is not None:
            continuum = Quantity(table[continuum_column])
        else:
            continuum = None

        meta = table.meta
        mask = table.mask

        return cls(dispersion, flux, error, continuum, mask, unit,
                   dispersion_unit, meta)

    def write(self, *args, **kwargs):
        """
        Write the spectrum to a file. Accepts the same arguments as
        `astropy.table.Table.write`

        """

        if self.dispersion.unit is None:
            label_string = 'WAVELENGTH'

        else:
            if self.dispersion.unit.physical_type == 'length':
                label_string = 'WAVELENGTH'

            elif self.dispersion.unit.physical_type == 'frequency':
                label_string = 'FREQUENCY'

            elif self.dispersion.unit.physical_type == 'energy':
                label_string = 'ENERGY'

            else:
                raise ValueError('unrecognised unit type')

        t = Table([self.dispersion, self.flux, self.flux.uncertainty.value],
                  names=[label_string, 'FLUX', 'ERROR'])
        t['ERROR'].unit = t['FLUX'].unit

        if self.continuum is not None:
            t['CONTINUUM'] = self.continuum

        t.write(*args, **kwargs)

    def plot(self, **kwargs):
        """
        Plot the spectrum. Accepts the same arguments as
        `igmtools.plot.Plot`.

        """

        from ..plot import Plot

        p = Plot(1, 1, 1, **kwargs)

        p.axes[0].plot(self.dispersion.value, self.flux.value,
                       drawstyle='steps-mid')

        if self.flux.uncertainty is not None:
            p.axes[0].plot(self.dispersion.value, self.flux.uncertainty.value,
                           drawstyle='steps-mid')

        p.tidy()
        p.display()

    def normalise_to_magnitude(self, magnitude, band):
        """
        Normalises the spectrum to match the flux equivalent to the
        given AB magnitude in the given passband.

        Parameters
        ----------
        magnitude : float
            AB magnitude.

        band : `igmtools.photometry.Passband`
            The passband.

        """

        from ..photometry import mag2flux

        mag_flux = mag2flux(magnitude, band)
        spec_flux = self.calculate_flux(band)
        norm = mag_flux / spec_flux
        self.flux *= norm

    def calculate_flux(self, band):
        """
        Calculate the mean flux for a passband, weighted by the response
        and wavelength in the given passband.

        Parameters
        ----------
        band : `igmtools.photometry.Passband`
            The passband.

        Returns
        -------
        flux : `astropy.units.Quantity`
            The mean flux in erg / s / cm^2 / Angstrom.

        Notes
        -----
        This function does not calculate an uncertainty.

        """

        if (self.wavelength[0] > band.wavelength[0] or
                self.wavelength[-1] < band.wavelength[-1]):

            warn('Spectrum does not cover the whole bandpass, '
                 'extrapolating...')
            dw = np.median(np.diff(self.wavelength.value))
            spec_wavelength = np.arange(
                band.wavelength.value[0],
                band.wavelength.value[-1] + dw, dw) * angstrom
            spec_flux = np.interp(spec_wavelength, self.wavelength,
                                  self.flux.value)

        else:
            spec_wavelength = self.wavelength
            spec_flux = self.flux.value

        i, j = spec_wavelength.searchsorted(
            Quantity([band.wavelength[0], band.wavelength[-1]]))
        wavelength = spec_wavelength[i:j]
        flux = spec_flux[i:j]

        dw_band = np.median(np.diff(band.wavelength))
        dw_spec = np.median(np.diff(wavelength))

        if dw_spec.value > dw_band.value > 20:

            warn('Spectrum wavelength sampling interval {0:.2f}, but bandpass'
                 'sampling interval {1:.2f}'.format(dw_spec, dw_band))

            # Interpolate the spectrum to the passband wavelengths:
            flux = np.interp(band.wavelength, wavelength, flux)
            band_transmission = band.transmission
            wavelength = band.wavelength

        else:
            # Interpolate the band transmission to the spectrum wavelengths:
            band_transmission = np.interp(
                wavelength, band.wavelength, band.transmission)

        # Weight by the response and wavelength, appropriate when we're
        # counting the number of photons within the band:
        flux = (np.trapz(band_transmission * flux * wavelength, wavelength) /
                np.trapz(band_transmission * wavelength, wavelength))
        flux *= erg / s / cm ** 2 / angstrom

        return flux

    def calculate_magnitude(self, band, system='AB'):
        """
        Calculates the magnitude in a given passband.

        band : `igmtools.photometry.Passband`
            The passband.

        system : {`AB`, `Vega`}
            Magnitude system.

        Returns
        -------
        magnitude : float
            Magnitude in the given system.

        """

        if system not in ('AB', 'Vega'):
            raise ValueError('`system` must be one of `AB` or `Vega`')

        f1 = self.calculate_flux(band)

        if f1 > 0:
            magnitude = -2.5 * log10(f1 / band.flux[system])

            if system == 'Vega':
                # Add 0.026 because Vega has V = 0.026:
                magnitude += 0.026

        else:
            magnitude = np.inf

        return magnitude

    def apply_extinction(self, EBmV):
        """
        Apply Milky Way extinction.

        Parameters
        ----------
        EBmV : float
            Colour excess.

        """

        from astro.extinction import MWCardelli89

        tau = MWCardelli89(self.wavelength, EBmV=EBmV).tau
        self.flux *= np.exp(-tau)

        if self.continuum is not None:
            self.continuum *= np.exp(-tau)

    def rebin(self, dispersion):
        """
        Rebin the spectrum onto a new dispersion axis.

        Parameters
        ----------
        dispersion : float, `astropy.units.Quantity` or array
            The dispersion for the rebinned spectrum. If a float, assumes a
            linear scale with that bin size.

        """

        if isinstance(dispersion, float):
            dispersion = np.arange(
                self.dispersion.value[0], self.dispersion.value[-1],
                dispersion)

        old_bins = find_bin_edges(self.dispersion.value)
        new_bins = find_bin_edges(dispersion)

        widths = np.diff(old_bins)

        old_length = len(self.dispersion)
        new_length = len(dispersion)

        i = 0  # index of old array
        j = 0  # index of new array

        # Variables used for rebinning:
        df = 0.0
        de2 = 0.0
        nbins = 0.0

        flux = np.zeros_like(dispersion)
        error = np.zeros_like(dispersion)

        # Sanity check:
        if old_bins[-1] < new_bins[0] or new_bins[-1] < old_bins[0]:
            raise ValueError('Dispersion scales do not overlap!')

        # Find the first contributing old pixel to the rebinned spectrum:
        if old_bins[i + 1] < new_bins[0]:

            # Old dispersion scale extends lower than the new one. Find the
            # first old bin that overlaps with the new scale:
            while old_bins[i + 1] < new_bins[0]:
                i += 1

            i -= 1

        elif old_bins[0] > new_bins[j + 1]:

            # New dispersion scale extends lower than the old one. Find the
            # first new bin that overlaps with the old scale:
            while old_bins[0] > new_bins[j + 1]:
                flux = np.nan
                error = np.nan
                j += 1

            j -= 1

        l0 = old_bins[i]  # lower edge of contributing old bin

        while True:

            h0 = old_bins[i + 1]  # upper edge of contributing old bin
            h1 = new_bins[j + 1]  # upper edge of jth new bin

            if h0 < h1:
                # Count up the decimal number of old bins that contribute to
                # the new one and start adding up fractional flux values:
                if self.flux.uncertainty.value[i] > 0:
                    bin_fraction = (h0 - l0) / widths[i]
                    nbins += bin_fraction

                    # We don't let `Data` handle the error propagation here
                    # because a sum of squares will not give us what we
                    # want, i.e. 0.25**2 + 0.75**2 != 0.5**2 + 0.5**2 != 1**2
                    df += self.flux.value[i] * bin_fraction
                    de2 += self.flux.uncertainty.value[i] ** 2 * bin_fraction

                l0 = h0
                i += 1

                if i == old_length:
                    break

            else:
                # We have all but one of the old bins that contribute to the
                # new one, so now just add the remaining fraction of the new
                # bin to the decimal bin count and add the remaining
                # fractional flux value to the sum:
                if self.flux.uncertainty.value[i] > 0:
                    bin_fraction = (h1 - l0) / widths[i]
                    nbins += bin_fraction
                    df += self.flux.value[i] * bin_fraction
                    de2 += self.flux.uncertainty.value[i] ** 2 * bin_fraction

                if nbins > 0:
                    # Divide by the decimal bin count to conserve flux density:
                    flux[j] = df / nbins
                    error[j] = sqrt(de2) / nbins

                else:
                    flux[j] = 0.0
                    error[j] = 0.0

                df = 0.0
                de2 = 0.0
                nbins = 0.0

                l0 = h1
                j += 1

                if j == new_length:
                    break

        if hasattr(self.dispersion, 'unit'):
            dispersion = Quantity(dispersion, self.dispersion.unit)

        if hasattr(self.flux, 'unit'):
            flux = Data(flux, error, self.flux.unit)

        # Linearly interpolate the continuum onto the new dispersion scale:
        if self.continuum is not None:
            continuum = np.interp(dispersion, self.dispersion, self.continuum)
        else:
            continuum = None

        return self.__class__(dispersion, flux, continuum=continuum)


class Absorber(object):
    """
    Class representation of an absorber.

    Parameters
    ----------
    identifier : str
        Name of the ion, molecule or isotope, e.g. `HI`.

    redshift : float, optional
        Redshift of the absorber.

    logn : float, optional
        Log10 column density (cm^-2).

    b : float, optional
        Doppler broadening parameter (km/s).

    covering_fraction : float, optional
        Covering fraction.

    atom : `igmtools.data.AtomDat`, optional
        Atomic data.

    """

    def __init__(self, identifier, redshift=None, logn=None, b=None,
                 covering_fraction=1, atom=None):

        if atom is None:
            atom = get_atomdat()

        self.identifier = identifier
        self.transitions = atom[identifier]
        self.redshift = redshift
        self.logn = logn
        self.b = b
        self.covering_fraction = covering_fraction

    def __repr__(self):

        return 'Absorber({0}, z={1:.2f}, logN={2:.2f}, b={3})'.format(
            self.identifier, self.redshift, self.logn, int(self.b))

    @classmethod
    def from_tau_peak(cls, transition, tau, b):
        """
        Initialise an absorber from the optical depth at line centre and
        Doppler broadining parameter of a given transition.

        Parameters
        ----------
        transition : str
            Name of the transition, e.g. `HI 1215'

        tau : float
            Optical depth at the line centre.

        b : float
            Doppler broadening parameter (km/s).

        """

        atom = get_atomdat()
        transition = atom.get_transition(transition)

        if isinstance(b, Quantity):
            b = b.to(cm / s)
        else:
            b = (b * km / s).to(cm / s)

        wavelength = transition.wavelength.to(cm)
        osc = transition.osc

        column = tau * b / (sqrt(pi) * e2_me_c * osc * wavelength)
        logn = log10(column.value)

        return cls(identifier=transition.parent, logn=logn, b=b)

    def optical_depth(self, dispersion):
        """
        Calculates the optical depth profile for a given spectral
        dispersion array.

        Parameters
        ----------
        dispersion : array
            Spectral dispersion.

        Returns
        -------
        tau : array
            The optical depth profile.

        """

        from ..calculations import optical_depth, tau_peak

        if isinstance(dispersion, Quantity):
            dispersion = dispersion.to(angstrom)

        elif hasattr(dispersion, 'unit'):
            if dispersion.unit is not None:
                dispersion = dispersion.to(angstrom)

        else:
            dispersion = Quantity(dispersion, unit=angstrom)

        velocity_range = ([-20000, 20000] * km / s if self.logn > 18
                          else [-1000, 1000] * km / s)

        # Select only transitions with redshifted central wavelengths inside
        # `dispersion` +/- 500 km/s:
        rest_wavelengths = Quantity([t.wavelength for t in self.transitions])
        observed_wavelengths = rest_wavelengths * (1 + self.redshift)

        wmin = dispersion[0] * (1 - 500 * km / s / c_kms)
        wmax = dispersion[-1] * (1 - 500 * km / s / c_kms)

        in_range = ((observed_wavelengths >= wmin) &
                    (observed_wavelengths <= wmax))
        transitions = np.array(self.transitions)[in_range]

        tau = np.zeros_like(dispersion.value)

        for i, transition in enumerate(transitions):

            tau_max = tau_peak(transition, self.logn, self.b)

            if 1 - exp(-tau_max) < 1e-3:
                continue

            observed_wavelength = transition.wavelength * (1 + self.redshift)
            dv = ((dispersion - observed_wavelength) /
                  observed_wavelength * c_kms)

            i0, i1 = dv.searchsorted(velocity_range)
            tau0 = optical_depth(dv[i0:i1], transition, self.logn, self.b)
            tau[i0:i1] += tau0

        return tau


class EmissionLine(object):
    """
    Class representation of an emission line and its properties.

    Parameters
    ----------
    wavelength : float
        Rest frame wavelength of the line in Angstrom.

    redshift : float
        Redshift of the emission line.

    flux : `igmtools.data.Data`, optional
        Integrated line flux.

    cont : `igmtools.data.Data`, optional
        Continuum flux at the line centre.

    eqw : `igmtools.data.Data`, optional
        Equivalent width of the line.

    """

    def __init__(self, wavelength, redshift, flux=None, cont=None, eqw=None):

        from ..calculations import comoving_distance

        if flux and not isinstance(flux, Data):
            raise ValueError('flux must be an instance of a Data object')

        if cont and not isinstance(cont, Data):
            raise ValueError('cont must be an instance of a Data object')

        if eqw and not isinstance(eqw, Data):
            raise ValueError('eqw must be an instance of a Data object')

        if isinstance(wavelength, Quantity):
            self.wavelength = wavelength.to(angstrom)

        else:
            self.wavelength = wavelength * angstrom

        self.redshift = redshift
        self.wavelength_observed = self.wavelength * (1 + self.redshift)

        if flux:

            self._flux = flux.to(erg / cm ** 2 / s, equivalencies=spectral())
            self.rflux = self._flux * (1 + redshift) ** 2

            distance = comoving_distance(self.redshift).cgs
            self.luminosity = 4 * pi * distance ** 2 * self._flux

            if cont and eqw:

                self._cont = cont.to(
                    erg / cm ** 2 / s / angstrom,
                    equivalencies=spectral_density(self.wavelength_observed))
                self.rcont = self._cont * (1 + redshift) ** 3

                self._eqw = eqw.to(angstrom)
                self.reqw = self._eqw / (1 + redshift)

            elif cont and not eqw:

                self._cont = cont.to(
                    erg / cm ** 2 / s / angstrom,
                    equivalencies=spectral_density(self.wavelength_observed))
                self.rcont = self._cont * (1 + redshift) ** 3

                self._eqw = self._flux / self._cont
                self.reqw = self._eqw / (1 + redshift)

            elif eqw and not cont:

                self._eqw = eqw.to(angstrom)
                self.reqw = self._eqw / (1 + redshift)

                self._cont = self._flux / self._eqw
                self.rcont = self._cont * (1 + redshift) ** 3

            else:

                self._eqw = eqw
                self.reqw = None
                self._cont = cont
                self.rcont = None

        elif cont:

            self._cont = cont.to(
                erg / cm ** 2 / s / angstrom,
                equivalencies=spectral_density(self.wavelength_observed))
            self.rcont = self._cont * (1 + redshift) ** 3

            if eqw:

                self._eqw = eqw.to(angstrom)
                self.reqw = self._eqw / (1 + redshift)

                self._flux = self._cont * self._eqw
                self.rflux = self._flux * (1 + redshift) ** 2

                distance = comoving_distance(self.redshift).cgs
                self.luminosity = 4 * pi * distance ** 2 * self._flux

            else:

                self._eqw = eqw
                self.reqw = None
                self._flux = flux
                self.rflux = None
                self.luminosity = None

        elif eqw:

            self._eqw = eqw.to(angstrom)
            self.reqw = self._eqw / (1 + redshift)

            self._flux = flux
            self.rflux = None
            self._cont = cont
            self.rcont = None
            self.luminosity = None

        else:

            self._flux = flux
            self.rflux = None
            self._cont = cont
            self.rcont = None
            self._eqw = eqw
            self.reqw = None
            self.luminosity = None

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, value):

        from ..calculations import comoving_distance

        if not isinstance(value, Data):
            raise ValueError('flux must be an instance of a Data object')

        self._flux = value.to(erg / cm ** 2 / s, equivalencies=spectral())
        self.rflux = self._flux * (1 + self.redshift) ** 2

        distance = comoving_distance(self.redshift).cgs
        self.luminosity = 4 * pi * distance ** 2 * self._flux

    @property
    def cont(self):
        return self._cont

    @cont.setter
    def cont(self, value):

        if not isinstance(value, Data):
            raise ValueError('cont must be an instance of a Data object')

        self._cont = value.to(
            erg / cm ** 2 / s / angstrom,
            equivalencies=spectral_density(self.wavelength_observed))
        self.rcont = self._cont * (1 + self.redshift) ** 3

    @property
    def eqw(self):
        return self._eqw

    @eqw.setter
    def eqw(self, value):

        if not isinstance(value, Data):
            raise ValueError('eqw must be an instance of a Data object')

        self._eqw = value.to(angstrom)
        self.reqw = self._eqw / (1 + self.redshift)
