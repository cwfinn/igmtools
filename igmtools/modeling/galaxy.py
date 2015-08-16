"""
Set of functions for measuring the spectral properties of galaxies.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..plot import Plot

from astropy.stats import sigma_clip
from astropy.constants import c
from astropy.units import km, s, erg, cm, angstrom
from astropy.modeling.models import Gaussian1D, GaussianAbsorption1D
from astropy.modeling.fitting import LevMarLSQFitter

from math import fabs

import matplotlib.pyplot as pl
import numpy as np

import random

__all__ = ['line_measurements']

c_kms = c.to(km / s).value

transition_wavelengths = {'OII': 3728.484,
                          'Hd': 4102.890,
                          'Hg': 4341.680,
                          'Hb': 4862.680,
                          'OIIIB': 4960.295,
                          'OIIIR': 5008.240,
                          'NIIB': 6549.840,
                          'Ha': 6564.610,
                          'NIIR': 6585.230,
                          'SIIB': 6718.320,
                          'SIIR': 6732.710}  # Vacuum wavelengths

o2_doublet = (3727.09, 3729.88)
o2 = (7580, 7680)
oh = (8600, 8700)

sigma_max = 10

bbox = dict(facecolor='w', edgecolor='none')

TwoGaussians = (Gaussian1D + Gaussian1D).rename('TwoGaussians')
ThreeGaussians = (Gaussian1D + Gaussian1D +
                  Gaussian1D).rename('ThreeGaussians')


def tie_sigma(model):
    stddev = model.stddev_0
    return stddev


def tie_sii(model):
    amplitude = model.amplitude_0 / 1.4
    return amplitude


def tie_nii(model):
    amplitude = model.amplitude_0 / 2.95
    return amplitude


def tie_oiii(model):
    amplitude = model.amplitude_0 / 2.98
    return amplitude


def clean_spectrum(wavelength, flux, error):

    inf = np.isinf(error)
    zero = flux == 0
    nan = np.isnan(flux)
    cond = ~inf & ~zero & ~nan

    return wavelength[cond], flux[cond], error[cond], cond


def monte_carlo_error(wavelength, flux, error, continuum, fitter, init,
                      absorption=False, n=100):

    perturbed_g = []

    for i in range(0, n):

        if absorption:
            perturb = np.array([random.gauss(0, 1) * error[i] / continuum[i]
                                for i in range(len(error))])
            flux_perturbed = (flux / continuum) + perturb

        else:
            perturb = np.array([random.gauss(0, 1) * error[i]
                                for i in range(len(error))])
            flux_perturbed = flux - continuum + perturb

        perturbed_g.append(fitter(init, wavelength, flux_perturbed))

    params = {}
    for i, name in enumerate(init.param_names):
        params[name] = []

    for g in perturbed_g:
        for i, name in enumerate(g.param_names):
            params[name].append(g.parameters[i])

    errors = {}
    for key in params.keys():
        errors[key] = np.std(np.array(params[key]))

    return errors


def line_measurements(name, spec1d, z, sky=None, spec2d=None, resolution=200,
                      yposition=None, sky_threshold=None, fit_o2_doublet=False,
                      plot=False, show_plot=False, plot_directory=None,
                      sky_in_counts=False):
    """
    Measures emission line fluxes and equivalent widths from a galaxy
    spectrum with redshift z.

    Parameters
    ----------
    name : str
        Object ID.

    spec1d : `igmtools.data.Spectrum1D`
        1D spectrum.

    z : float
        Galaxy redshift.

    sky : array
        1D sky spectrum.

    spec2d : `igmtools.data.Spectrum2D`
        2D spectrum.

    R : int, optional
        Spectral resolution (default = 200).

    yposition : tuple, optional
        y-coordinates of the object on the 2D spectrum
        (edge 1, centre, edge 2).

    sky_threshold : float, optional
        Sky counts/flux value above which flag 3 is raised (useful for
        eliminating zero-order contamination and regions of bad sky
        subtraction).

    fit_o2_doublet : bool, optional
        Option to fit two Gaussian components to the OII line
        (default = False).

    plot : bool, optional
        Option to plot continuum estimates across bands where a measurement
        is executed, and the Gaussian fit, if performed (default = False).

    show_plot : bool, optional
        Option to show each plot in an interactive window (default = False).

    plot_directory : str, optional
        If specified, plots will be saved in this directory as PNG files.

    sky_in_counts : bool, optional
        Set to True if the sky spectrum is in counts rather than flux units.

    Returns
    -------
    measurements : `astropy.table.Table`
        The line measurements.

    Notes
    -----
    Measurements are made using the following line indicies:

    OII : (3655.0, 3705.0, 3708.5, 3748.5, 3750.0, 3800.0)
    Hd : (4030.0, 4080.0, 4082.0, 4122.0, 4125.0, 4170.0)
    Hg : (4230.0, 4270.0, 4321.5, 4361.5, 4365.0, 4400.0)
    Hb : (4785.0, 4820.0, 4842.5, 4882.5, 5030.0, 5100.0)
    OIII : (4785.0, 4820.0, 4988.0, 5028.0, 5030.0, 5100.0)
    Ha : (6460.0, 6520.0, 6544.5, 6584.5, 6610.0, 6670.0),
    SII : (6640.0, 6700.0, 6713.0, 6753.0, 6760.0, 6810.0)

    These line indicies are optimised for spectra with R ~ 200, and have
    measurement windows 20 Angstroms wide. We assume the maximum intrinsic
    line width to be that of a Gaussian with a standard deviation of 10
    Angstroms in the rest frame. Convolved with the instrument line spread
    function, this corresponds to measurement windows of width close to
    the maximum expected standard deviation of the Gaussian line profile.
    For specified instrument resolutions different to the default value of
    200, measurement windows are scaled to preserve this feature.

    Lines with integrated fluxes measured at greater than 3 sigma
    significance are fitted with Gaussians. The measurements are then
    taken from the Gaussian fitting parameters and their errors computed
    from a Monte-Carlo type estimation. The maximum allowed Gaussian
    standard deviation corresponds to 10 Angstroms in the intrinsic line
    profile, and the minimum to 0.5 times that of that instrumental line
    spread function.

    If Hdelta is absorption dominated at a greater than 3 sigma level,
    a Gaussian absorption profile is fitted. This is motivated by the idea
    that Hdelta may be used as a proxy for the Balmer absorption correction.

    Equivalent widths are positive for emission lines, and negative for
    absorption lines.

    Warning flags are defined as follows:

    0 : No warnings.
    1 : Measurement may be affected by the OH forest between 8600 and 8700
        Angstrom.
    2 : Line was fit with the maximum/minimum allowed Gaussian standard
        deviation.
    3 : Line coincides with region above the specified sky threshold.
    4 : Line may be affected by O2 telluric absorption (7580 - 7680 Angstrom).
    5 : Bad continuum reduced chi squared (> 10).
    6 : No spectral coverage, or human verification failed.

    No measurement is recorded for flag 6 - all values are set to -99.0.

    """

    plot = True if show_plot else plot

    bands = {'OII': [3655.0, 3705.0, 3708.5, 3748.5, 3750.0, 3800.0],
             'Hd': [4030.0, 4080.0, 4082.0, 4122.0, 4125.0, 4170.0],
             'Hg': [4230.0, 4270.0, 4321.5, 4361.5, 4365.0, 4400.0],
             'Hb': [4785.0, 4820.0, 4842.5, 4882.5, 5030.0, 5100.0],
             'OIII': [4785.0, 4820.0, 4988.0, 5028.0, 5030.0, 5100.0],
             'Ha': [6460.0, 6520.0, 6544.5, 6584.5, 6610.0, 6670.0],
             'SII': [6640.0, 6700.0, 6713.0, 6753.0, 6760.0, 6810.0]}

    # Modify measurement windows if appropriate:
    if resolution != 200:

        sigma_max200 = 18.8  # Max Gaussian sigma for R = 200 (approx)
        dlambda = 7500 / resolution
        sigma_lsf = dlambda / 2.35482
        sigma_max_convolved = np.sqrt(sigma_lsf ** 2 + sigma_max ** 2)
        scale_factor = sigma_max_convolved / sigma_max200

        for key in bands.keys():
            window = bands[key][3] - bands[key][2]
            window0 = window * scale_factor
            bands[key][1] += (window - window0) / 2
            bands[key][2] += (window - window0) / 2
            bands[key][3] -= (window - window0) / 2
            bands[key][4] -= (window - window0) / 2

    # Initialise dictionaries:
    (line_flux, continuum_flux, eqw, sn,
     continuum_params, line_params, flags) = {}, {}, {}, {}, {}, {}, {}

    # 1D spectrum arrays:
    wavelength = spec1d.wavelength.value
    flux = spec1d.flux.value
    error = spec1d.flux.uncertainty.value

    # Clean the spectrum:
    wavelength, flux, error, cond = clean_spectrum(wavelength, flux, error)

    if sky is not None:
        sky = sky[cond]

    # Do measurements:
    for key in bands.keys():

        # Initialise dictionary for continuum parameters:
        continuum_params[key] = {}

        # Line groupings:
        if (key == 'OII') and fit_o2_doublet:
            lines = ['OIIR', 'OIIB']
            rest_wavelengths = o2_doublet

        elif key == 'OIII':
            lines = ['OIIIR', 'OIIIB']
            rest_wavelengths = [transition_wavelengths['OIIIR'],
                                transition_wavelengths['OIIIB']]

        elif key == 'Ha':
            lines = ['NIIR', 'Ha', 'NIIB']
            rest_wavelengths = [transition_wavelengths['NIIR'],
                                transition_wavelengths['Ha'],
                                transition_wavelengths['NIIB']]

        elif key == 'SII':
            lines = ['SIIR', 'SIIB']
            rest_wavelengths = [transition_wavelengths['SIIR'],
                                transition_wavelengths['SIIB']]

        else:
            lines = [key]
            rest_wavelengths = [transition_wavelengths[key]]

        # Initialise dictionaries for line parameters:
        for line in lines:
            line_params[line] = {}

        # Observed wavelengths of the lines:
        observed_wavelengths = [item * (1 + z) for item in rest_wavelengths]

        # Fitting/measurement regions:
        co_blue = ((wavelength >= bands[key][0] * (1 + z)) &
                   (wavelength < bands[key][1] * (1 + z)))
        co_red = ((wavelength >= bands[key][4] * (1 + z)) &
                  (wavelength < bands[key][5] * (1 + z)))
        co_region = co_red | co_blue
        line_region = ((wavelength >= bands[key][2] * (1 + z)) &
                       (wavelength <= bands[key][3] * (1 + z)))

        # Extended region around the measurement. Used for excluding
        # measurements affected by zero orders:
        centre = ((bands[key][2] * (1 + z)) + (bands[key][3] * (1 + z))) / 2
        centre_band = ((wavelength >= centre - 100) &
                       (wavelength <= centre + 100))

        # The full fitting region:
        region = ((wavelength >= bands[key][0] * (1 + z)) &
                  (wavelength < bands[key][5] * (1 + z)))

        # Masks to identify regions potentially affected by 7600A O2 telluric
        # absorption:
        o2_blue = ((wavelength[co_blue] >= o2[0]) &
                   (wavelength[co_blue] <= o2[1]))
        o2_red = (wavelength[co_red] >= o2[0]) & (wavelength[co_red] <= o2[1])
        o2_line = ((wavelength[line_region] >= o2[0]) &
                   (wavelength[line_region] <= o2[1]))

        # Masks to identify regions potentially affected by the OH forest:
        oh_blue = ((wavelength[co_blue] >= oh[0]) &
                   (wavelength[co_blue] <= oh[1]))
        oh_red = (wavelength[co_red] >= oh[0]) & (wavelength[co_red] <= oh[1])
        oh_line = ((wavelength[line_region] >= oh[0]) &
                   (wavelength[line_region] <= oh[1]))

        # Assume the measurement will be good at first:
        flags[key] = 0

        # Check that we have spectral coverage:
        if ((np.sum(co_blue) < 5) | (np.sum(co_red) < 5) |
                (flux[co_blue] == 0).all() | (flux[co_red] == 0).all()):

            # If no coverage, mark all measurements as -99.0 and assign flag
            # 6, then go to next iteration of the loop:
            for line in lines:
                line_flux[line] = (-99.0, -99.0)
                continuum_flux[line] = (-99.0, -99.0)
                eqw[line] = (-99.0, -99.0)
                line_params[line]['amplitude'] = -99.0
                line_params[line]['mean'] = -99.0
                line_params[line]['stddev'] = -99.0

            continuum_params[key]['gradient'] = -99.0
            continuum_params[key]['intercept'] = -99.0
            continuum_params[key]['chi2norm'] = -99.0
            sn[key] = -99.0
            flags[key] = 6
            continue

        # See if we're affected by 7600A O2 telluric absorption:
        if ((np.sum(o2_blue) > 0) | (np.sum(o2_red) > 0) |
                (np.sum(o2_line) > 0)):
            flags[key] = 4

        # See if we're affected by OH forest:
        if ((np.sum(oh_blue) > 0) | (np.sum(oh_red) > 0) |
                (np.sum(oh_line) > 0)):
            flags[key] = 1

        # Assign sky threshold flag if a value is specified and it exceeds
        # this:
        if sky_threshold is not None:
            if any(sky0 > sky_threshold for sky0 in sky[centre_band]):
                flags[key] = 3

        # Sigma clip the continuum, to ensure it's not affected by nearby
        # absorption features:
        filtered_blue = sigma_clip(flux[co_blue], 1.5)
        filtered_red = sigma_clip(flux[co_red], 1.5)

        # Take the mean value of the sigma clipped continuum either side of the
        # line:
        co_level1 = np.mean(filtered_blue)
        co_level1_error = np.std(filtered_blue)
        co_level2 = np.mean(filtered_red)
        co_level2_error = np.std(filtered_red)

        # Linearly interpolate between these values:
        continuum = ((co_level2 - co_level1) /
                     (np.mean(wavelength[co_red]) -
                      np.mean(wavelength[co_blue])) *
                     (wavelength - np.mean(wavelength[co_blue])) + co_level1)
        continuum_error = np.sqrt(
            co_level1_error ** 2 + co_level2_error ** 2) / 2

        # Continuum gradient:
        gradient = ((continuum[1] - continuum[0]) /
                    (wavelength[1] - wavelength[0]))
        continuum_params[key]['gradient'] = gradient

        # Continuum intercept:
        intercept = continuum[0] - gradient * wavelength[0]
        continuum_params[key]['intercept'] = intercept

        # Flag if normalised continuum chi squared > 10:
        cont_chi2norm = (np.sum(
            (flux[co_region] - continuum[co_region]) ** 2 /
            error[co_region] ** 2) / (len(flux[co_region]) - 3))

        if cont_chi2norm > 10:
            flags[key] = 5

        continuum_params[key]['chi2norm'] = cont_chi2norm

        # Estimate integrated line flux and equivalent width (observed,
        # not rest frame):
        dl = np.mean(wavelength[line_region][1:] -
                     wavelength[line_region][:-1])
        n = np.sum(line_region)

        line_flux_value = dl * np.sum(
            flux[line_region] - continuum[line_region])
        line_flux_error = dl * np.sqrt(
            np.sum(error[line_region] ** 2) + n * continuum_error ** 2)

        eqw_value = dl * np.sum(
            flux[line_region] / continuum[line_region]) - n
        eqw_error = dl * np.sqrt(
            np.sum(error[line_region] ** 2 / continuum[line_region] ** 2) +
            n * continuum_error ** 2 *
            np.sum(flux[line_region] ** 2 / continuum[line_region] ** 4))

        # Continuum flux at the line centre:
        ind = np.abs(wavelength - observed_wavelengths[0]).argmin()
        centre_flux = continuum[ind]
        centre_flux_error = error[ind]

        # Estimate signal-to-noise ratio around the line:
        sn_blue = (filtered_blue[~filtered_blue.mask] /
                   error[co_blue][~filtered_blue.mask])
        sn_red = (filtered_red[~filtered_red.mask] /
                  error[co_red][~filtered_red.mask])
        sn_value = np.average(np.concatenate([sn_blue, sn_red]))

        # Calculate minimum and maximum allowed Gaussian standard
        # deviations:
        dlambda = rest_wavelengths[0] / resolution
        sigma_lsf = dlambda / 2.35482
        min_stddev = sigma_lsf / 2
        max_stddev = np.sqrt(sigma_lsf ** 2 + sigma_max ** 2) * (1 + z)

        # Fit Gaussian component(s) if the integrated line flux is
        # positive and has greater than 3 sigma significance:
        if (line_flux_value > 0) & ((line_flux_value / line_flux_error) > 3):

            amplitude = np.max(flux[line_region] - continuum[line_region])

            if ((key in ('Hg', 'Hd')) |
                    ((key == 'OII') and not fit_o2_doublet)):

                # One component Gaussian fit for Hg, Hd, OII:
                # -------------------------------------------
                mean = observed_wavelengths[0]

                g_init = Gaussian1D(amplitude, mean, min_stddev)

                g_init.amplitude.min = 0.0

                g_init.mean.min = mean - (1000 * mean / c_kms)
                g_init.mean.max = mean + (1000 * mean / c_kms)

                g_init.stddev.min = min_stddev
                g_init.stddev.max = max_stddev
                # -------------------------------------------

            elif (key == 'OII') and fit_o2_doublet:

                # Optional two component Gaussian fit for OII:
                # --------------------------------------------
                mean_0 = observed_wavelengths[0]
                mean_1 = observed_wavelengths[1]

                tied_params = {'stddev_1': tie_sigma}

                g_init = TwoGaussians(
                    amplitude, mean_0, min_stddev, amplitude, mean_1,
                    min_stddev, tied=tied_params)

                g_init.amplitude_0.min = 0.0
                g_init.amplitude_1.min = 0.0

                g_init.mean_0.min = mean_0 - (1000 * mean_0 / c_kms)
                g_init.mean_0.max = mean_0 + (1000 * mean_0 / c_kms)
                g_init.mean_1.min = mean_1 - (1000 * mean_1 / c_kms)
                g_init.mean_1.max = mean_1 + (1000 * mean_1 / c_kms)

                g_init.stddev_0.min = min_stddev
                g_init.stddev_0.max = max_stddev
                g_init.stddev_1.min = min_stddev
                g_init.stddev_1.max = max_stddev
                # --------------------------------------------

            elif key == 'SII':

                # Two component Gaussian fit for SII:
                # -----------------------------------
                mean_0 = observed_wavelengths[0]
                mean_1 = observed_wavelengths[1]

                tied_params = {'amplitude_1': tie_sii,
                               'stddev_1': tie_sigma}

                g_init = TwoGaussians(
                    amplitude, mean_0, min_stddev, amplitude, mean_1,
                    min_stddev, tied=tied_params)

                g_init.amplitude_0.min = 0.0
                g_init.amplitude_1.min = 0.0

                g_init.mean_0.min = mean_0 - (1000 * mean_0 / c_kms)
                g_init.mean_0.max = mean_0 + (1000 * mean_0 / c_kms)
                g_init.mean_1.min = mean_1 - (1000 * mean_1 / c_kms)
                g_init.mean_1.max = mean_1 + (1000 * mean_1 / c_kms)

                g_init.stddev_0.min = min_stddev
                g_init.stddev_0.max = max_stddev
                g_init.stddev_1.min = min_stddev
                g_init.stddev_1.max = max_stddev
                # -----------------------------------

            elif key in ('Hb', 'OIII'):

                # Three component fit over Hb/OIII region:
                # ----------------------------------------
                mean_0 = observed_wavelengths[0]

                if key == 'Hb':
                    mean_1 = transition_wavelengths['OIIIB'] * (1 + z)
                    mean_2 = transition_wavelengths['OIIIR'] * (1 + z)

                    tied_params = {'stddev_1': tie_sigma,
                                   'stddev_2': tie_sigma}

                else:
                    mean_1 = transition_wavelengths['OIIIB'] * (1 + z)
                    mean_2 = transition_wavelengths['Hb'] * (1 + z)

                    tied_params = {'amplitude_1': tie_oiii,
                                   'stddev_1': tie_sigma,
                                   'stddev_2': tie_sigma}

                g_init = ThreeGaussians(
                    amplitude, mean_0, min_stddev, amplitude, mean_1,
                    min_stddev, amplitude, mean_2, min_stddev,
                    tied=tied_params)

                g_init.amplitude_0.min = 0.0
                g_init.amplitude_1.min = 0.0
                g_init.amplitude_2.min = 0.0

                g_init.mean_0.min = mean_0 - (1000 * mean_0 / c_kms)
                g_init.mean_0.max = mean_0 + (1000 * mean_0 / c_kms)
                g_init.mean_1.min = mean_1 - (1000 * mean_1 / c_kms)
                g_init.mean_1.max = mean_1 + (1000 * mean_1 / c_kms)
                g_init.mean_2.min = mean_2 - (1000 * mean_2 / c_kms)
                g_init.mean_2.max = mean_2 + (1000 * mean_2 / c_kms)

                g_init.stddev_0.min = min_stddev
                g_init.stddev_0.max = max_stddev
                g_init.stddev_1.min = min_stddev
                g_init.stddev_1.max = max_stddev
                g_init.stddev_2.min = min_stddev
                g_init.stddev_2.max = max_stddev
                # ----------------------------------------

            else:

                # Try one and three component fit over Ha/NII region:
                # ---------------------------------------------------
                mean_1 = observed_wavelengths[1]

                g_init = Gaussian1D(amplitude, mean_1, min_stddev)

                g_init.amplitude.min = 0.0

                g_init.mean.min = mean_1 - (1000 * mean_1 / c_kms)
                g_init.mean.max = mean_1 + (1000 * mean_1 / c_kms)

                g_init.stddev.min = min_stddev
                g_init.stddev.max = max_stddev

                mean_0 = observed_wavelengths[0]
                mean_2 = observed_wavelengths[2]

                tied_params = {'amplitude_2': tie_nii,
                               'stddev_1': tie_sigma,
                               'stddev_2': tie_sigma}

                g_init2 = ThreeGaussians(
                    amplitude, mean_0, min_stddev, amplitude, mean_1,
                    min_stddev, amplitude, mean_2, min_stddev,
                    tied=tied_params)

                g_init2.amplitude_0.min = 0.0
                g_init2.amplitude_1.min = 0.0
                g_init2.amplitude_2.min = 0.0

                g_init2.mean_0.min = mean_0 - (1000 * mean_0 / c_kms)
                g_init2.mean_0.max = mean_0 + (1000 * mean_0 / c_kms)
                g_init2.mean_1.min = mean_1 - (1000 * mean_1 / c_kms)
                g_init2.mean_1.max = mean_1 + (1000 * mean_1 / c_kms)
                g_init2.mean_2.min = mean_2 - (1000 * mean_2 / c_kms)
                g_init2.mean_2.max = mean_2 + (1000 * mean_2 / c_kms)

                g_init2.stddev_0.min = min_stddev
                g_init2.stddev_0.max = max_stddev
                g_init2.stddev_1.min = min_stddev
                g_init2.stddev_1.max = max_stddev
                g_init2.stddev_2.min = min_stddev
                g_init2.stddev_2.max = max_stddev
                # ---------------------------------------------------

            # Do the fitting:
            fit_g = LevMarLSQFitter()
            g = fit_g(g_init, wavelength[region],
                      flux[region] - continuum[region])

            # Chi2 on the fit:
            line_chi2 = np.sum(
                (flux[region] - continuum[region] -
                 g(wavelength)[region]) ** 2 / error[region] ** 2)

            # Monte carlo error estimation:
            g_errors = monte_carlo_error(
                wavelength[region], flux[region], error[region],
                continuum[region], fit_g, g_init)

            ha_3comp = False

            # Compare chi squared values for the two Ha/NII fits and adopt
            # the one that has the minimum chi squared:
            if key == 'Ha':
                # Three component fit of Ha/NII region:
                fit_g2 = LevMarLSQFitter()
                g2 = fit_g2(g_init2, wavelength[region],
                            flux[region] - continuum[region])

                # Monte carlo error estimation:
                g2_errors = monte_carlo_error(
                    wavelength[region], flux[region], error[region],
                    continuum[region], fit_g2, g_init2)

                # Chi2 on the fit:
                line_chi2_2 = np.sum(
                    (flux[region] - continuum[region] -
                     g2(wavelength[region])) ** 2 /
                    (g2(wavelength[region]) + continuum[region]))

                # Compare chi2:
                if line_chi2 > line_chi2_2:
                    g = g2
                    g_errors = g2_errors
                    ha_3comp = True

            # Get lists of best-fit Gaussian parameters:
            if ((key in ('Hg', 'Hd')) |
                    ((key == 'OII') and not fit_o2_doublet)):
                amplitudes = [g.amplitude.value]
                amplitude_errors = [g_errors['amplitude']]
                means = [g.mean.value]
                stddevs = [g.stddev.value]
                stddev_errors = [g_errors['stddev']]

            elif ((key == 'OII') and fit_o2_doublet) | (key == 'SII'):
                amplitudes = [g.amplitude_0.value, g.amplitude_1.value]
                amplitude_errors = [g_errors['amplitude_0'],
                                    g_errors['amplitude_1']]
                means = [g.mean_0.value, g.mean_1.value]
                stddevs = [g.stddev_0.value, g.stddev_1.value]
                stddev_errors = [g_errors['stddev_0'],
                                 g_errors['stddev_1']]

            elif ((key == 'Ha') and ha_3comp) | (key in ('Hb', 'OIII')):
                amplitudes = [g.amplitude_0.value, g.amplitude_1.value,
                              g.amplitude_2.value]
                amplitude_errors = [g_errors['amplitude_0'],
                                    g_errors['amplitude_1'],
                                    g_errors['amplitude_2']]
                means = [g.mean_0.value, g.mean_1.value, g.mean_2.value]
                stddevs = [g.stddev_0.value, g.stddev_1.value,
                           g.mean_2.value]
                stddev_errors = [g_errors['stddev_0'],
                                 g_errors['stddev_1'],
                                 g_errors['stddev_2']]

            else:
                amplitudes = [g.amplitude.value, -99.0, -99.0]
                amplitude_errors = [g_errors['amplitude'], -99.0, -99.0]
                means = [g.mean.value, -99.0, -99.0]
                stddevs = [g.stddev.value, -99.0, -99.0]
                stddev_errors = [g_errors['stddev'], -99.0, -99.0]

            # Log these line by line:
            for i, line in enumerate(lines):

                # Log the line fitting parameters:
                line_params[line]['amplitude'] = amplitudes[i]
                line_params[line]['mean'] = means[i]
                line_params[line]['stddev'] = stddevs[i]

                # Only adopt the measurements if the fitted amplitude is
                # non-zero, otherwise, measurements from direct integration
                # of pixels are retained:
                if amplitudes[i] != 0:

                    # Integrated line flux:
                    line_flux_value = (amplitudes[i] * stddevs[i] *
                                       np.sqrt(2 * np.pi))

                    # Error on the integrated line flux:
                    line_flux_error = line_flux_value * np.sqrt(
                        (amplitude_errors[i] / amplitudes[i]) ** 2 +
                        (stddev_errors[i] / stddevs[i]) ** 2)

                    # Re-evaluate the continuum flux at the line centre:
                    ind = np.abs(wavelength - means[i]).argmin()
                    centre_flux = continuum[ind]
                    centre_flux_error = error[ind]

                    # Equivalent width:
                    eqw_value = line_flux_value / centre_flux

                    # Error on the equivalent width:
                    eqw_error = eqw_value * np.sqrt(
                        (line_flux_error / line_flux_value) ** 2 +
                        (centre_flux_error / centre_flux) ** 2)

                # Log the line flux, continuum flux and equivalent width:
                line_flux[line] = (line_flux_value, line_flux_error)
                continuum_flux[line] = (centre_flux, centre_flux_error)
                eqw[line] = (eqw_value, eqw_error)

            fit = True
            fit_hd = False

        # Fit single Gaussian absorption component to Hd if the integrated
        # line flux is negative and has greater than 3 sigma significance:
        elif ((key == 'Hd') & (line_flux_value < 0) &
                ((line_flux_value / line_flux_error) < -3)):

            amplitude = np.max(1 - flux[line_region] / continuum[line_region])
            mean = transition_wavelengths[key] * (1 + z)
            dm = 1000 * mean / c_kms

            g_init = GaussianAbsorption1D(amplitude, mean, min_stddev)

            g_init.mean.min = mean - dm
            g_init.mean.max = mean + dm

            g_init.stddev.min = min_stddev
            g_init.stddev.max = max_stddev

            # Do the fitting:
            fit_g = LevMarLSQFitter()
            g = fit_g(g_init, wavelength[region],
                      flux[region] / continuum[region])

            # Monte carlo error estimation:
            g_errors = monte_carlo_error(
                wavelength[region], flux[region], error[region],
                continuum[region], fit_g, g_init, absorption=True)

            # Equivalent width:
            eqw_value = (-g.amplitude.value * g.stddev.value *
                         np.sqrt(2 * np.pi) / (1 + z))

            # Error on the equivalent width:
            eqw_error = fabs(eqw_value) * np.sqrt(
                (g_errors['amplitude'] / g.amplitude.value) ** 2 +
                (g_errors['stddev'] / g.stddev.value) ** 2)

            for line in lines:

                # Log the line fitting parameters:
                line_params[line]['amplitude'] = g.amplitude.value
                line_params[line]['mean'] = g.mean.value
                line_params[line]['stddev'] = g.stddev.value

                # Log the line flux, continuum flux and equivalent width:
                line_flux[line] = (line_flux_value, line_flux_error)
                continuum_flux[line] = (centre_flux, centre_flux_error)
                eqw[line] = (eqw_value, eqw_error)

            fit = False
            fit_hd = True

        # Otherwise we won't do any line fitting:
        else:

            for line in lines:

                # Set all line fitting parameters to -99:
                line_params[line]['amplitude'] = -99.0
                line_params[line]['mean'] = -99.0
                line_params[line]['stddev'] = -99.0

                # Log the line flux, continuum flux and equivalent width:
                line_flux[line] = (line_flux_value, line_flux_error)
                continuum_flux[line] = (centre_flux, centre_flux_error)
                eqw[line] = (eqw_value, eqw_error)

            fit = False
            fit_hd = False

        sn[key] = sn_value

        # Make plots if that option is turned on:
        if plot:

            if sky is not None and spec2d is not None:
                n = 3
                p = Plot(n, 1, n, aspect=1, width=5.9, fontsize=12)

            elif ((sky is not None and spec2d is None) |
                    (spec2d is not None and sky is None)):
                n = 2
                p = Plot(n, 1, n, aspect=0.8, width=5.9, fontsize=12)

            else:
                n = 1
                p = Plot(n, 1, n, aspect=0.6, width=5.9, fontsize=12)

            centre = (bands[key][0] * (1 + z) + bands[key][5] * (1 + z)) / 2
            cond = (wavelength > centre - 250) & (wavelength < centre + 250)

            if spec2d is not None:
                cond2 = ((spec2d.wavelength.value > centre - 250) &
                         (spec2d.wavelength.value < centre + 250))

            # 2D spectrum plot:
            if spec2d is not None:

                n = 3 if n == 3 else 2

                # 2D spectrum parameters for plotting:
                i = min(spec2d.data.shape[0] // 2, 3)
                v1 = np.percentile(spec2d.data[i:-1, :].ravel(), 90)
                wdelt = spec2d.wavelength.value[1] - spec2d.wavelength.value[0]
                yvals = np.arange(spec2d.data.shape[0]) * wdelt

                p.axes[n - n].pcolormesh(
                    spec2d.wavelength.value[cond2], yvals,
                    spec2d.data[:, cond2], vmin=-v1 / 5, vmax=2 * v1,
                    cmap=pl.cm.hot)

                if yposition is not None:
                    p.axes[n - n].axhline(
                        wdelt * yposition[0], ls='--', lw=2, color='LawnGreen')
                    p.axes[n - n].axhline(
                        wdelt * yposition[2], ls='--', lw=2, color='LawnGreen')

            # 1D spectrum plot:
            if (n == 3) | ((n == 2) & (sky is not None and spec2d is None)):
                n = 2

            else:
                n = 1

            p.axes[n - n].plot(
                wavelength[cond], flux[cond] / 1e-16, drawstyle='steps-mid',
                color='k')
            p.axes[n - n].plot(
                wavelength[cond], error[cond] / 1e-16, drawstyle='steps-mid',
                color='r')
            p.axes[n - n].plot(
                wavelength[region], continuum[region] / 1e-16, lw=3,
                color='RoyalBlue')

            if fit:
                p.axes[n - n].plot(
                    wavelength[region],
                    (g(wavelength[region]) + continuum[region]) / 1e-16,
                    color='m', lw=2)

            if fit_hd:
                p.axes[n - n].plot(
                    wavelength[region],
                    (g(wavelength[region]) * continuum[region]) / 1e-16,
                    color='m', lw=2)

            p.axes[n - n].axvspan(
                bands[key][2] * (1 + z), bands[key][3] * (1 + z),
                facecolor='g', edgecolor='none', alpha=0.5)
            p.axes[n - n].annotate(
                key, xy=(0.05, 0.8),
                xycoords='axes fraction', horizontalalignment='left',
                fontsize=12, bbox=bbox, color='k')
            p.axes[n - n].annotate(
                name, xy=(0.95, 0.8), xycoords='axes fraction',
                horizontalalignment='right', fontsize=12, bbox=bbox, color='k')

            # Sky spectrum plot:
            if sky is not None:

                if sky_in_counts:
                    p.axes[n - 1].plot(
                        wavelength[cond], sky[cond], drawstyle='steps-mid',
                        color='MidnightBlue')

                else:
                    p.axes[n - 1].plot(
                        wavelength[cond], sky[cond] / 1e-16,
                        drawstyle='steps-mid', color='MidnightBlue')

                p.axes[n - 1].annotate(
                    'sky', xy=(0.05, 0.8), xycoords='axes fraction',
                    horizontalalignment='left', fontsize=12, bbox=bbox,
                    color='k')

            # Axis limits and labels, tidy up and display:
            region_min = np.min(error[region])
            region_max = np.max(flux[region])
            sn_spec = np.median(flux / error)

            for i in range(0, n):
                p.axes[i].set_xlim(wavelength[cond][0], wavelength[cond][-1])

            p.axes[n - 2].set_ylim(
                (region_min - 0.2 * sn_spec * region_min) / 1e-16,
                (region_max + 0.8 * region_max) / 1e-16)

            xlabel = 'Wavelength ($\AA$)'

            if spec1d.flux.unit == erg / s / cm ** 2 / angstrom:
                ylabel = 'Flux ($10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA$)'

            else:
                ylabel = 'Flux ({0})'.format(
                    spec1d.flux.unit.to_string(format='latex'))

            p.tidy(shared_axes=True)
            p.labels(xlabel, ylabel)

            if plot_directory is not None:
                p.savefig('{0}/{1}_{2}.png'.format(plot_directory, name, key))

            if show_plot:
                p.display()

    return (line_flux, continuum_flux, eqw, sn, continuum_params, line_params,
            flags)
