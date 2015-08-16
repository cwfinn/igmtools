"""
Calculations relating to absorption lines.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .voigt import voigt

from ..data.atomic import get_atomdat

from astropy.units import km, s, angstrom, cm, Quantity
from astropy.constants import m_e, c, e

from math import pi, sqrt, log

import numpy as np

__all__ = ['optical_depth', 'tau_peak', 'logn_from_tau_peak',
           'equivalent_width_from_voigt', 'limiting_equivalent_width']

ion_cache = {}
atomdat = None

# cgs units:
c = c.to(cm / s)
c_kms = c.to(km / s)
e2_me_c = (e.esu ** 2 / (m_e * c)).to(cm ** 2 / s)

sqrt_ln2 = 0.8325546111576977  # sqrt(ln(2))

# Atomic weights:
Ar = dict(H=1.00794,
          He=4.002602,
          C=12.0107,
          N=14.0067,
          O=15.9994,
          Ne=20.1798,
          Na=22.98976928,
          Mg=24.3050,
          Al=26.9815386,
          Si=28.0855,
          P=30.973762,
          S=32.065,
          Ar=39.948,
          Ca=40.078,
          Fe=55.845)


def optical_depth(velocity, transition, logn, b, verbose=False):
    """
    Returns the optical depth as a function of velocity (Voigt profile)
    for a transition.

    Parameters
    ----------
    velocity : array
        Velocities in km / s.

    transition : str or `igmtools.data.Transition`
        The transition. Can be a string, e.g. `HI 1215`

    logn : float
        Log10 column density (cm^-2).

    b : float
        Doppler broadening parameter (km / s).

    verbose : bool, optional
        If True, warn when the optical depth profile is undersampled
        (default = True).

    Returns
    -------
    tau : array of floats, shape (N,)
        The optical depth as a function of velocity.

    Notes
    -----
    The step size for `velocity` must be small enough to properly sample the
    profile.

    """

    if isinstance(velocity, Quantity):
        velocity = velocity.to(cm / s)
    else:
        velocity = (velocity * km / s).to(cm / s)

    if isinstance(transition, str):
        atom = get_atomdat()
        transition = atom.get_transition(transition)

    wavelength = transition.wavelength.to(cm)
    osc = transition.osc
    gamma = transition.gamma

    if isinstance(b, Quantity):
        b = b.to(cm / s)
    else:
        b = (b * km / s).to(cm / s)

    column = 10 ** logn / cm ** 2
    frequency = c / wavelength

    # Doppler relation between velocity and frequency assuming
    # gamma << frequency:
    gamma_v = gamma / frequency * c.to(cm / s)

    if verbose:

        fwhm_gaussian = 2 * sqrt_ln2 * b

        ic = np.searchsorted(velocity, 0)

        if ic == len(velocity):
            ic -= 1

        if ic == 0:
            ic += 1

        vstep = (velocity[ic] - velocity[ic - 1]).to(km / s).value
        fwhm = max(gamma_v, fwhm_gaussian).to(km / s).value

        if vstep > fwhm:
            print('WARNING: tau profile undersampled!\n'
                  '  Pixel width: {0:.3f} km/s, '
                  'transition FWHM: {1:.3f} km/s'.format(vstep, fwhm))

    u = velocity / b
    a = gamma_v / (4 * pi * b)
    vp = voigt(a, u)

    sigma = pi * e2_me_c * wavelength / (sqrt(pi) * b) * vp
    tau = (column * osc) * sigma

    return tau.value


def tau_peak(transition, logn, b):
    """
    Find the optical depth of a transition at the line centre.

    Parameters
    ----------
    transition : str or `igmtools.data.Transition`
        The transition. Can be a string, e.g. `HI 1215`

    logn : float or array
        Log10 column density (cm^-2).

    b : float or array
        Doppler broadening parameter (km/s).

    Returns
    -------
    tau : float or array
        Optical depth at the line centre.

    """

    if isinstance(transition, str):
        atom = get_atomdat()
        transition = atom.get_transition(transition)

    if isinstance(b, Quantity):
        b = b.to(cm / s)
    else:
        b = (b * km / s).to(cm / s)

    wavelength = transition.wavelength.to(cm)
    osc = transition.osc
    column = 10 ** logn / cm ** 2

    tau = sqrt(pi) * e2_me_c * column * osc * wavelength / b

    return tau.value


def logn_from_tau_peak(transition, tau, b):
    """
    Calculate column density based on the optical depth at the line
    centre of a transition and the Doppler broadening parameter.

    transition : str or `igmtools.data.Transition`
        The transition. Can be a string e.g. `HI 1215`

    tau : float or array
        Optical depth at the line centre.

    b : float or array
        Doppler broadening parameter (km / s).

    Returns
    -------
    logn : float or array
        Log10 column density (cm^-2).

    """

    if isinstance(transition, str):
        atom = get_atomdat()
        transition = atom.get_transition(transition)

    if isinstance(b, Quantity):
        b = b.to(cm / s)
    else:
        b = (b * km / s).to(cm / s)

    wavelength = transition.wavelength.to(cm)
    osc = transition.osc

    column = tau * b / (sqrt(pi) * e2_me_c * osc * wavelength)
    logn = np.log10(column.value)

    return logn


def equivalent_width_from_voigt(transition, logn, b):
    """
    Calculate the rest-frame equivalent width of a transition from its
    column density and doppler broadening parameter.

    Parameters
    ----------
    transition : `igmtools.data.Transition`
        Transition.

    logn : float or array
        Log10 column density (cm^-2).

    b : float or array
        Doppler broadening parameter (km/s).

    Returns
    -------
    reqw : `astropy.units.Quantity`
        Rest-frame equivalent width (Angstrom).

    """

    if isinstance(b, Quantity):
        b = b.to(cm / s)
    else:
        b = (b * km / s).to(cm / s)

    tau = tau_peak(transition, logn, b)
    wavelength = transition.wavelength.to(cm)
    gamma = transition.gamma

    if tau <= 1.25393:
        reqw = sqrt(pi) * (b / c) * tau / (1 + tau / 2 / sqrt(2))

    else:
        reqw = sqrt((2 * b / c) ** 2 * log(tau / log(2)) + (b / c) *
                    (gamma * wavelength / c) * (tau - 1.25393) / sqrt(pi))

    reqw *= transition.wavelength

    return reqw


def limiting_equivalent_width(significance, wavelength, b, snpix, dispersion,
                              fwhm_lsf, smoothing=1):

    """
    Determines the limiting equivalent width (to some significance level)
    for an absorption line assuming a Gaussian line shape (a reasonable
    approximation to a Voigt profile for weak lines).

    Parameters
    ----------
    significance : int
        Significance level of the limit (number of standard deviations).

    wavelength : float
        Observed wavelength of the line (Angstrom).

    b : float
        An estimate of the Doppler broadening parameter (km/s).

    snpix : float
        Signal-to-noise ratio per spectral pixel.

    dispersion : float
        Dispersion of the spectrograph in Angstrom/pixel.

    fhwm_lsf : float
        FWHM of the (assumed Gaussian) LSF (Angstrom).

    smoothing : float, optional
        The number of pixels the spectrum has been re-binned by
        (default = 1 (no re-binning)).

    Returns
    -------
    lim_eqw : `astropy.units.Quantity`
        Limiting equivalent width (Angstrom).

    """

    if isinstance(wavelength, Quantity):
        wavelength = wavelength.to(angstrom)

    else:
        wavelength = wavelength * angstrom

    if isinstance(dispersion, Quantity):
        dispersion = dispersion.to(angstrom)

    else:
        dispersion = dispersion * angstrom

    if isinstance(b, Quantity):
        b = b.to(km / s)

    else:
        b = b * km / s

    if isinstance(fwhm_lsf, Quantity):
        fwhm_lsf = fwhm_lsf.to(angstrom)

    else:
        fwhm_lsf = fwhm_lsf * angstrom

    # Doppler line width:
    dx = b * wavelength / (c_kms * dispersion)  # in pixels

    # Gaussian line profile comes from a convolution of the intrinsic
    # line profile with the LSF:
    stddev1 = dx / sqrt(2)
    stddev2 = fwhm_lsf / (2 * sqrt(2) * sqrt_ln2) / dispersion  # in pixels
    stddev = np.sqrt(stddev1 ** 2 + stddev2 ** 2)

    # Optimal integration width is approximately the line FWHM:
    xopt = 2 * sqrt(2) * sqrt_ln2 * stddev

    # Convert S/N per re-binned pixel to S/N per native pixel assuming
    # Poissonian noise properties, i.e. error per pixel = sqrt(counts):
    sn1 = snpix / smoothing ** 0.5

    lim_eqw = significance * (dispersion / sn1) * xopt

    return lim_eqw
