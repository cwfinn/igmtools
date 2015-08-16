"""
A set of utilities for making velocity plots.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .general import Plot
from .utils import get_nrows_ncols

from ..data.atomic import AtomDat
from ..data.spectral import Absorber
from ..modeling.vpfit import read_f26
from ..calculations.absorption import tau_peak

from configobj import ConfigObj
from validate import Validator

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.table import Table, vstack
from astropy.constants import c
from astropy.units import km, s
from astropy.io import ascii

from matplotlib.transforms import blended_transform_factory

from collections import defaultdict

from math import ceil

import numpy as np

import sys
import os

c_kms = c.to(km / s).value  # speed of light in km/s
atom = AtomDat()  # atomic data

# Panel label inset format:
bbox = dict(facecolor='w', edgecolor='None')


def make_velocity_scale(vmin, vmax, observed_wavelength, wavelength):
    """
    Makes a velocity scale for a spectrum, with the zero velocity
    defined by the observed wavelength of some spectral feature.

    Parameters
    ----------
    vmin, vmax : float
        Minimum and maximum velocity for the scale.

    observed_wavelength : float
        Observed wavelength for the spectral feature in Angstrom.

    wavelength : array
        The spectral dispersion in Angstrom.

    Returns
    -------
    velocity : array
        Velocity scale.

    mask : array
        Mask for the spectral elements covered by the velocity scale.

    """

    wmin = observed_wavelength * (1 + 1.1 * vmin / c_kms)
    wmax = observed_wavelength * (1 + 1.1 * vmax / c_kms)

    mask = (wavelength > wmin) & (wavelength < wmax)
    velocity = (wavelength[mask] / observed_wavelength - 1) * c_kms

    return velocity, mask


class VelocityPlot(Plot):
    """
    Defines the layout of a velocity plot for absorption profiles.

    Parameters
    ----------
    transitions : list
        A list of transitions. Each entry should be something like 'HI 1215'.

    vmin : float
        The velocity scale minimum in km / s.

    vmax : float
        The velocity scale maximum in km / s.

    nrows : int, optional
        Number of figure rows. Determined automatically from the number
        of transitions if unspecified.

    ncols : int, optional
        Number of figure columns. Determined automatically from the number
        of transitions if unspecified.

    width : float, optional
        Width of the figure in inches (default = 8.0).

    aspect : float, optional
        Aspect ratio for the figure (default = 0.3).

    fontsize : int, optional
        Font size for the axis labels and title (default = 16).

    label_fontsize : int, optional
        Font size for the panel labels (default = 12).

    family : str, optional
        Font family (default = `serif`).

    style : str, optional
        Font style (default = `Times`).

    weight : str, optional
        Font weight (default = `normal`).

    usetex : bool, optional
        Option to use TeX fonts (default = False).

    Methods
    -------
    plot_data()
    plot_models()

    """

    def __init__(self, transitions, vmin=-400, vmax=400, nrows=None,
                 ncols=None, label_fontsize=12, usetex=False, **kwargs):

        # Transitions:
        self.transitions = []

        for transition in transitions:
            transition = transition.strip()
            transition = atom.get_transition(transition)
            self.transitions.append(transition)

        # Figure properties
        npar = len(self.transitions)
        self.vmin = vmin
        self.vmax = vmax
        self.label_fontsize = label_fontsize
        self.usetex = usetex

        # Automatically get the number of rows and columns if not already
        # specified:
        if not nrows and not ncols:
            nrows, ncols = get_nrows_ncols(npar)
        elif not nrows:
            nrows = int(ceil(npar / ncols))
        elif not ncols:
            ncols = int(ceil(npar / nrows))

        # Initialise:
        super(VelocityPlot, self).__init__(
            nrows, ncols, npar, usetex=usetex, **kwargs)

        # Set to fill columns first:
        ind = np.arange(0, nrows * ncols).reshape(nrows, ncols)
        self._ind = ind.transpose().flatten()

        # Ensure we have the correct number of indicies:
        if len(self._ind) > npar:
            number = len(self._ind) - npar
            extra = ind[-1, ::-1][:number]
            self._ind = np.array([i for i in self._ind if i not in extra])

        # Loop over transitions:
        for i, transition in enumerate(self.transitions):

            ax = self.axes[self._ind[i]]

            # Plot guide-lines:
            ax.axvline(0, color='k', lw=0.5, ls='--')
            ax.axhline(0, color='gray', lw=0.5)
            ax.axhline(1, color='gray', lw=0.5, ls='dashed')

            # Set axis limits and ticks:
            ax.set_xlim(self.vmin, self.vmax)
            ax.set_ylim(-0.6, 1.6)
            ax.set_yticks([0.0, 0.5, 1.0])

        # Clean excess whitespace:
        self.tidy(shared_axes=True)
        self.subplots_adjust(wspace=0, hspace=0)

        # Axis labels:
        self.labels('Velocity offset (km s$^{-1}$)', 'Transmission')

        # Initialise spectrum information:
        self.wavelength = None
        self.flux = None
        self.error = None
        self.continuum = None
        self.transmission = None
        self.redshift = None

        # Initialise models:
        self.model_wavelength = None
        self.model = None
        self.coarse_model = None
        self.system_components = None
        self.blend_components = None
        self.ticks = None

        # Initialise lines and labels:
        self.data = None
        self.models = None

    def plot_data(self, wavelength, flux, error, continuum, redshift,
                  colour='0.5', linewidth=1.0):
        """
        Plot the spectral data.

        Parameters
        ----------
        wavelength : array
            The spectral dispersion in Angstrom.

        flux : array
            The spectrum flux in arbitrary units.

        error : array
            The spectrum flux error.

        continuum : array
            The psuedo continuum - must have the same units as the flux array.

        redshift : float
            Redshift of the absorption system.

        colour : str, optional
            The line colour (default = '0.5').

        linewidth : float, optional
            The line width (default = 0.5).

        """

        # Divide only by real, positive, continuum values:
        valid = ~np.isnan(continuum) & (continuum > 0)

        self.wavelength = np.array(wavelength).astype('float64')[valid]
        self.flux = np.array(flux).astype('float64')[valid]
        self.error = np.array(error).astype('float64')[valid]
        self.continuum = np.array(continuum).astype('float64')[valid]
        self.transmission = self.flux / self.continuum
        self.redshift = redshift

        self.data = []

        # Loop over transitions:
        for i, transition in enumerate(self.transitions):

            ax = self.axes[self._ind[i]]

            # Get the velocity scale:
            observed_wavelength = transition.wavelength.value * (1 + redshift)
            velocity, mask = make_velocity_scale(
                self.vmin, self.vmax, observed_wavelength,
                self.wavelength)

            # Plot the data:
            self.data.append(
                ax.plot(velocity, self.transmission[mask], color=colour,
                        lw=linewidth, ls='steps-mid'))

            # Panel label:
            transf = blended_transform_factory(ax.transAxes, ax.transData)
            name = transition.name

            # Use TeX fonts if specified:
            if self.usetex:
                name = name.split()
                if name[0][1].islower():
                    name = name[0][:2] + '$\;$\\textsc{' + \
                        name[0][2:].lower() + '} $\lambda$' + name[1]
                else:
                    name = name[0][:1] + '$\;$\\textsc{' + \
                        name[0][1:].lower() + '} $\lambda$' + name[1]

            # Plot panel label:
            ax.text(0.03, 0.5, name, fontsize=self.label_fontsize,
                    bbox=bbox, transform=transf)

    def _get_models(self, absorbers, system_width, resolution,
                    convolve_with_cos_fuv, convolve_with_cos_nuv, oversample,
                    plot_components):

        # Subdivide wavelength array if we are oversampling:
        if oversample > 1:
            dw = ((self.wavelength[1] - self.wavelength[0]) /
                  oversample)
            self.model_wavelength = np.arange(
                self.wavelength[0], self.wavelength[-1], dw)
        else:
            self.model_wavelength = self.wavelength

        # Gaussian kernel with a pixel velocity width 4 times smaller than
        # the FWHM:
        ndiv = 4
        kernel = Gaussian1DKernel(ndiv / 2.354820046)

        # Calculate quantities for Gaussian convolution if they are required:
        if not convolve_with_cos_fuv and not convolve_with_cos_nuv:

            if isinstance(resolution, dict):
                scales = []
                low = 0
                wmax = resolution.keys().sort()

                for w in wmax:
                    high = self.model_wavelength.searchsorted(w)
                    dlogw = np.log10(1 + resolution[w] / (ndiv * c_kms))
                    npts = int(np.log10(self.model_wavelength[high] /
                               self.model_wavelength[low]) / dlogw)
                    scales.append(self.model_wavelength[low] *
                                  10 ** (np.arange(npts) * dlogw))
                    low = high

                scale = np.concatenate(scales)

            else:
                dlogw = np.log10(1 + resolution / (ndiv * c_kms))
                npts = int(np.log10(self.model_wavelength[-1] /
                           self.model_wavelength[0]) / dlogw)
                scale = (self.model_wavelength[0] *
                         10 ** (np.arange(npts) * dlogw))

        self.system_components, self.blend_components = [], []
        self.ticks = defaultdict(list)

        # We want to create models for absorbers with different covering
        # fractions separately, so we'll store them with key access in a
        # dictionary. We'll take the product of all models with the same
        # covering fractions in the loop below, then apply zero offsets
        # (for covering fractions < 1) to those models at the end, before
        # combining them into a single model:
        models = dict()

        # Loop over absorbers:
        for absorber in absorbers:

            # Calculate the optical depth profile:
            tau = absorber.optical_depth(
                self.model_wavelength).astype('float64')

            # Add this to the total model appropriate for the covering
            # fraction of this absorber. Don't apply any zero offsets yet -
            # we'll do this at the end:
            if absorber.covering_fraction not in models.keys():
                models[absorber.covering_fraction] = np.exp(-tau)
            else:
                models[absorber.covering_fraction] *= np.exp(-tau)

            # Log wavelengths of transitions that contribute to the model if
            # the optical depth at the line centre is > 0.001:
            for transition in absorber.transitions:
                if tau_peak(transition, absorber.logn, absorber.b) > 0.001:
                    self.ticks['wavelength'].append(
                        transition.wavelength.value * (1 + absorber.redshift))
                    self.ticks['transition'].append(transition.name)

            # Only calculate the individual model components if we are
            # plotting them (expensive otherwise):
            if plot_components:

                # Ensure zero offsets are applied (for covering fractions < 1):
                component = (np.exp(-tau) * absorber.covering_fraction + 1 -
                             absorber.covering_fraction)

                # Convolve with the instrument LSF:
                if convolve_with_cos_fuv or convolve_with_cos_nuv:
                    try:
                        from COS import convolve_with_COS_FOS

                    except:
                        raise ImportError('COS module not installed')

                    if convolve_with_cos_nuv:

                        if self.model_wavelength[0] > 1600:
                            component = convolve_with_COS_FOS(
                                component, self.model_wavelength,
                                use_cos_nuv=True, cos_nuv_only=True)

                        else:
                            component = convolve_with_COS_FOS(
                                component, self.model_wavelength,
                                use_cos_nuv=True)

                    else:
                        component = convolve_with_COS_FOS(
                            component, self.model_wavelength)

                else:
                    # Interpolate onto constant velocity scale:
                    component0 = np.interp(
                        scale, self.model_wavelength, component)

                    # Do the convolution:
                    component = np.interp(
                        self.model_wavelength, scale,
                        convolve(component0, kernel))

                dz = system_width * (1 + self.redshift) / c_kms

                # Add components to the appropriate lists, depending on the
                # adopted system width:
                if ((absorber.redshift > self.redshift - dz / 2) &
                        (absorber.redshift < self.redshift + dz / 2)):
                    self.system_components.append(component)
                else:
                    self.blend_components.append(component)

        # Ensure that zero offsets are applied (for covering fractions < 1):
        for key in models.keys():
            covering_fraction = key
            models[key] = (models[key] * covering_fraction +
                           1 - covering_fraction)

        # Combine the models:
        model = np.product([models[key] for key in models.keys()], axis=0)

        # Cast line lists as NumPy arrays:
        self.ticks['wavelength'] = np.array(self.ticks['wavelength'])
        self.ticks['transition'] = np.array(self.ticks['transition'])

        # Convolve the models with the instrument LSF:
        if convolve_with_cos_fuv or convolve_with_cos_nuv:
            try:
                from COS import convolve_with_COS_FOS

            except:
                raise ImportError('convolve_with_COS_FOS not available')

            if convolve_with_cos_nuv:

                if self.model_wavelength[0] > 1600:
                    self.model = convolve_with_COS_FOS(
                        model, self.model_wavelength, use_cos_nuv=True,
                        cos_nuv_only=True)

                else:
                    self.model = convolve_with_COS_FOS(
                        model, self.model_wavelength, use_cos_nuv=True)

            else:
                self.model = convolve_with_COS_FOS(
                    model, self.model_wavelength)

        else:
            # Interpolate onto a constant velocity scale:
            model0 = np.interp(scale, self.model_wavelength, model)

            # Do the convolution:
            self.model = np.interp(
                self.model_wavelength, scale, convolve(model0, kernel))

        # Combined model on coarse wavelength grid
        # (same as `model` if `oversample` = 1):
        if oversample == 1:
            self.coarse_model = self.model
        else:
            self.coarse_model = np.interp(
                self.wavelength, self.model_wavelength, self.model)

    def plot_models(self, absorbers, system_width=200, resolution=10,
                    convolve_with_cos_fuv=False, convolve_with_cos_nuv=False,
                    oversample=1, plot_ticks=True, plot_components=False,
                    plot_residuals=False, system_colour='b', blend_colour='r',
                    linewidth=1.0, tick_linewidth=1.0, component_linewidth=0.5,
                    residuals_colour='g', residuals_markersize=2.0):
        """
        Plot the model profiles.

        Parameters
        ----------
        absorbers : list of `igmtools.data.Absorber` instances
            The absorbers that contribute to the models.

        system_width : float, optional
            Defines the width of the absorption system of interest in km/s.
            All model components at velocities outside this range are
            regarded as blends (default = 200)

        resolution : float or dict, optional
            FWHM of the LSF of the spectrograph in km/s (default = 10).
            This can be a dictionary with key:value pairs in the format
            wmax:fwhm, where wmax is the maximum wavelength for which the
            specified fwhm holds.

        convolve_with_cos_fuv : bool, optional
            Option to convolve with the COS FUV LSF (default = False).

        convolve_with_cos_nuv : bool, optional
            Option to convolve with the COS NUV LSF (default = False).

        oversample : int, optional
            Oversample the model profiles by this factor (default = 1).

        plot_ticks : bool, optional
            Option to plot tick markers above the absorption lines
            (default = True).

        plot_components : bool, optional
            Option to plot individual model components (default = False).

        plot_residuals : bool, optional
            Option to plot residuals on the fitted profiles (default = False).

        system_colour : str, optional
            Colour for the model profiles at the system redshift.
            (default = 'b').

        blend_colour : str, optional
            Colour for the blended model profiles at other redshifts.
            (default = 'r').

        linewidth : float, optional
            Line width for the model profiles (default = 1.0).

        tick_linewidth : float, optional
            Line width for the tick markers (default = 1.5).

        component_linewidth : float, optional
            Line width for the individual model components (default = 0.5).

        residuals_colour : str, optional
            Marker colour for the residuals (default = 'g').

        residuals_markersize : float, optional
            Marker size for the residuals (default = 3.0).

        """

        self.models = []

        # Calculate models:
        self._get_models(
            absorbers, system_width, resolution, convolve_with_cos_fuv,
            convolve_with_cos_nuv, oversample, plot_components)

        # Loop over transitions:
        for i, transition in enumerate(self.transitions):

            # Get the velocity scale:
            observed_wavelength = (transition.wavelength.value *
                                   (1 + self.redshift))
            velocity, mask = make_velocity_scale(
                self.vmin, self.vmax, observed_wavelength,
                self.model_wavelength)

            ax = self.axes[self._ind[i]]

            # Option to plot individual model components:
            if plot_components:

                # Loop over model components:
                for j, component in enumerate(self.system_components):

                    # Plot the model component:
                    self.models.append(
                        ax.plot(velocity, component[mask], color=system_colour,
                                lw=component_linewidth))

                # Loop over blend components:
                for j, component in enumerate(self.blend_components):

                    # Plot the blend component:
                    self.models.append(
                        ax.plot(velocity, component[mask], color=blend_colour,
                                lw=component_linewidth))

            # Plot the model:
            self.models.append(
                ax.plot(velocity, self.model[mask], color=system_colour,
                        lw=linewidth))

            # Option to plot residuals:
            if plot_residuals:

                # Make another velocity scale if needed:
                if oversample > 1:
                    velocity, mask = make_velocity_scale(
                        self.vmin, self.vmax, observed_wavelength,
                        self.wavelength)

                # Divide only by real, positive error values:
                valid = (~np.isnan(self.error[mask]) &
                         ~np.isinf(self.error[mask]) &
                         (self.error[mask] > 0))

                # Calculate and scale residuals:
                residual = ((self.transmission[mask][valid] -
                             self.coarse_model[mask][valid]) /
                            self.error[mask][valid])
                scaling = np.std(residual) / 0.05
                residual = residual / scaling - 0.2

                # Plot the residuals:
                self.models.append(
                    ax.axhline(-0.2 + 0.05, color='k', lw=0.5, alpha=0.5))
                self.models.append(
                    ax.axhline(-0.2 - 0.05, color='k', lw=0.5, alpha=0.5))
                self.models.append(
                    ax.plot(velocity[valid], residual, 'o',
                            ms=residuals_markersize, mfc=residuals_colour,
                            mec=residuals_colour))

            # Option to plot tick markers:
            if plot_ticks:

                # Mask for the ticks covered by the plot:
                wmin = observed_wavelength * (1 + self.vmin / c_kms)
                wmax = observed_wavelength * (1 + self.vmax / c_kms)

                # Wavelength range covered by the absorption system:
                system_wmin = (observed_wavelength *
                               (1 - system_width / 2 / c_kms))
                system_wmax = (observed_wavelength *
                               (1 + system_width / 2 / c_kms))

                in_system = ((self.ticks['wavelength'] > system_wmin) &
                             (self.ticks['wavelength'] < system_wmax) &
                             (self.ticks['transition'] == transition.name))

                blend = ((self.ticks['wavelength'] > wmin) &
                         (self.ticks['wavelength'] < wmax) &
                         (self.ticks['transition'] != transition.name))

                # Plot blend tick markers:
                positions = ((self.ticks['wavelength'][blend] /
                              observed_wavelength - 1) * c_kms)

                for position in positions:
                    self.models.append(
                        ax.plot([position, position], [1.15, 1.3],
                                color=blend_colour, lw=tick_linewidth))

                # Plot system tick markers:
                positions = ((self.ticks['wavelength'][in_system] /
                              observed_wavelength - 1) * c_kms)

                for position in positions:
                    self.models.append(
                        ax.plot([position, position], [1.15, 1.3],
                                color=system_colour, lw=tick_linewidth))


def main(args=None):
    """
    This is the main function called by the `velplot` script.

    """

    from astropy.utils.compat import argparse
    from astropy.extern.configobj import configobj, validate

    from pkg_resources import resource_stream

    parser = argparse.ArgumentParser(
        description='Creates a stacked velocity plot.\nTo dump a default '
                    'configuration file: velplot -d\nTo dump an extended '
                    'default configuration file: velplot -dd',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('config', help='path to the configuration file')

    config = resource_stream(__name__, '/config/velplot.cfg')
    config_extended = resource_stream(__name__, '/config/velplot_extended.cfg')
    spec = resource_stream(__name__, '/config/velplot_specification.cfg')

    if len(sys.argv) > 1:

        if sys.argv[1] == '-d':
            cfg = ConfigObj(config)
            cfg.filename = '{0}/velplot.cfg'.format(os.getcwd())
            cfg.write()
            return

        elif sys.argv[1] == '-dd':
            cfg = ConfigObj(config_extended)
            cfg.filename = '{0}/velplot.cfg'.format(os.getcwd())
            cfg.write()
            return

    args = parser.parse_args(args)

    try:
        cfg = configobj.ConfigObj(args.config, configspec=spec)
        validator = validate.Validator()
        cfg.validate(validator)

    except:
        raise IOError('Configuration file could not be read')

    figname = cfg['FIGURE'].pop('filename')

    # Create list of transitions:
    fname = cfg['FIGURE'].pop('transitions')
    print('Reading transitions from ', fname)

    fh = open(fname)
    transitions = list(fh)
    fh.close()

    # Don't include transitions that are commented out:
    transitions = [transition for transition in transitions
                   if not transition.startswith('#')]

    # Initialise figure:
    velplot = VelocityPlot(transitions, **cfg['FIGURE'])
    fname = cfg['DATA'].pop('filename')

    if not fname:
        raise IOError('no data to plot!')

    # Get spectrum information and plot:
    spectrum = (Table.read(fname) if fname.endswith('fits')
                else ascii.read(fname))
    wavelength = spectrum[cfg['DATA'].pop('wavelength_column')]
    flux = spectrum[cfg['DATA'].pop('flux_column')]
    error = spectrum[cfg['DATA'].pop('error_column')]
    continuum = spectrum[cfg['DATA'].pop('continuum_column')]
    velplot.plot_data(wavelength, flux, error, continuum, **cfg['DATA'])

    # Get model information and plot if specified:
    fname = cfg['MODEL'].pop('filename')

    if fname:

        ion = cfg['MODEL'].pop('ion_column')
        redshift = cfg['MODEL'].pop('redshift_column')
        logn = cfg['MODEL'].pop('logn_column')
        b = cfg['MODEL'].pop('b_column')

        if fname.endswith('f26'):
            model = read_f26(fname)
            absorbers = model.absorbers

        else:
            table = (Table.read(fname) if fname.endswith('fits')
                     else ascii.read(fname))
            absorbers = [Absorber(row[ion], row[redshift], row[logn], row[b])
                         for row in table]

        velplot.plot_models(absorbers, **cfg['MODEL'])

    # Save:
    if figname:
        print('Saving to {0}'.format(figname))
        velplot.savefig(figname)

    # Display:
    velplot.display()
