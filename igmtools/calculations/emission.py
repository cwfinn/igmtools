"""
Calculations relating to emission lines.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..data import EmissionLine

from astropy.units import erg, s, solMass, yr

from math import log10

import numpy as np

# Kennicutt (1998) H-alpha luminosity to SFR conversion:
halpha_sfr = [(erg / s, solMass / yr,
               lambda x: 7.9e-42 * x,
               lambda x: x / 7.9e-42)]

# Kennicutt (1998) OII luminosity to SFR conversion:
oii_sfr = [(erg / s, solMass / yr,
            lambda x: 1.4e-41 * x,
            lambda x: x / 1.4e-41)]

# Moustakas (2006) Hb luminosity to SFR conversions
# (dependent on B-band luminosity):
hb_sfr_7p25 = [(erg / s, solMass / yr,
                lambda x: 2.723e-41 * x,
                lambda x: x / 2.723e-41)]
hb_sfr_7p75 = [(erg / s, solMass / yr,
                lambda x: 2.529e-41 * x,
                lambda x: x / 2.529e-41)]
hb_sfr_8p25 = [(erg / s, solMass / yr,
                lambda x: 2.698e-41 * x,
                lambda x: x / 2.698e-41)]
hb_sfr_8p75 = [(erg / s, solMass / yr,
                lambda x: 2.594e-41 * x,
                lambda x: x / 2.594e-41)]
hb_sfr_9p25 = [(erg / s, solMass / yr,
                lambda x: 3.357e-41 * x,
                lambda x: x / 3.357e-41)]
hb_sfr_9p75 = [(erg / s, solMass / yr,
                lambda x: 3.828e-41 * x,
                lambda x: x / 3.828e-41)]
hb_sfr_10p25 = [(erg / s, solMass / yr,
                lambda x: 5.957e-41 * x,
                lambda x: x / 5.957e-41)]
hb_sfr_10p75 = [(erg / s, solMass / yr,
                lambda x: 8.453e-41 * x,
                lambda x: x / 8.453e-41)]

# Moustakas (2006) OII luminosity to SFR conversions
# (dependent on B-band luminosity):
oii_sfr_7p75 = [(erg / s, solMass / yr,
                 lambda x: 1.432e-41 * x,
                 lambda x: x / 1.432e-41)]
oii_sfr_8p25 = [(erg / s, solMass / yr,
                 lambda x: 1.285e-41 * x,
                 lambda x: x / 1.285e-41)]
oii_sfr_8p75 = [(erg / s, solMass / yr,
                 lambda x: 0.962e-41 * x,
                 lambda x: x / 0.962e-41)]
oii_sfr_9p25 = [(erg / s, solMass / yr,
                 lambda x: 1.368e-41 * x,
                 lambda x: x / 1.368e-41)]
oii_sfr_9p75 = [(erg / s, solMass / yr,
                 lambda x: 1.824e-41 * x,
                 lambda x: x / 1.824e-41)]
oii_sfr_10p25 = [(erg / s, solMass / yr,
                 lambda x: 3.508e-41 * x,
                 lambda x: x / 3.508e-41)]
oii_sfr_10p75 = [(erg / s, solMass / yr,
                 lambda x: 5.610e-41 * x,
                 lambda x: x / 5.610e-41)]

hb_estimator = [hb_sfr_7p25, hb_sfr_7p75, hb_sfr_8p25, hb_sfr_8p75,
                hb_sfr_9p25, hb_sfr_9p75, hb_sfr_10p25, hb_sfr_10p75]

oii_estimator = [oii_sfr_7p75, oii_sfr_8p25, oii_sfr_8p75, oii_sfr_9p25,
                 oii_sfr_9p75, oii_sfr_10p25, oii_sfr_10p75]

b_luminosities_hb = np.array([7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5])

b_luminosities_oii = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5])


class HAlpha(EmissionLine):
    """
    Class representation of the H-alpha emission line. Includes a method
    for calculating the star-formation rate.

    Parameters
    ----------
    redshift : float
        Redshift of the emission line.

    flux : `igmtools.data.Data`, optional
        Integrated line flux.

    cont : `igmtools.data.Data`, optional
        Continuum flux at the line centre.

    eqw : `igmtools.data.Data`, optional
        Equivalent width of the line.

    absorption_correction : `igmtools.data.Data, optional
        Balmer absorption correction.

    """

    def __init__(self, redshift, flux=None, cont=None, eqw=None,
                 absorption_correction=None):

        wavelength = 6564.61  # vacuum

        super(HAlpha, self).__init__(wavelength, redshift, flux, cont, eqw)

        if absorption_correction:

            if not isinstance(absorption_correction, Data):
                raise ValueError('absorption_correction must be an instance '
                                 'of a Data object')

            if not self.eqw or not self.flux:
                raise AttributeError('eqw and flux attributes must be set to '
                                     'apply an absorption correction')

            self.flux = ((self.eqw + absorption_correction) *
                         self.flux / self.eqw)
            self.eqw += absorption_correction

    def compute_sfr(self, balmer_decrement=None):
        """
        Compute the star formation rate.

        Parameters
        ----------
        balmer_decrement : float, optional
            Absorption corrected Balmer decrement.

        Returns
        -------
        sfr : `igmtools.data.Data`
            Star formation rate in solar masses per year.

        """

        if not self.flux:
            raise AttributeError('flux attribute must be set to calculate a '
                                 'star formation rate')

        if balmer_decrement:
            # Assume obscuration curve of Cardelli et al. (1989):
            self.luminosity *= (balmer_decrement / 2.86) ** 2.114

        sfr = self.luminosity.convert_unit_to(
            solMass / yr, equivalencies=halpha_sfr)

        return sfr


class HBeta(EmissionLine):
    """
    Class representation of the H-beta emission line. Includes a method
    for calculating the star-formation rate.

    Parameters
    ----------
    redshift : float
        Redshift of the emission line.

    flux : `igmtools.data.Data`, optional
        Integrated line flux.

    cont : `igmtools.data.Data`, optional
        Continuum flux at the line centre.

    eqw : `igmtools.data.Data`, optional
        Equivalent width of the line.

    absorption_correction : `igmtools.data.Data, optional
        Balmer absorption correction.

    """

    def __init__(self, redshift, flux=None, cont=None, eqw=None,
                 absorption_correction=None):

        wavelength = 4862.68  # vacuum

        super(HBeta, self).__init__(wavelength, redshift, flux, cont, eqw)

        if absorption_correction:

            if not isinstance(absorption_correction, Data):
                raise ValueError('absorption_correction must be an instance '
                                 'of a Data object')

            if not self.eqw or not self.flux:
                raise AttributeError('eqw and flux attributes must be set to '
                                     'apply an absorption correction')

            self.flux = ((self.eqw + absorption_correction) *
                         self.flux / self.eqw)
            self.eqw += absorption_correction

    def compute_sfr(self, b_luminosity, balmer_decrement=None):
        """
        Compute the star formation rate.

        Parameters
        ----------
        b_luminosity : float
            B-band luminosity (erg / s).

        balmer_decrement : float, optional
            Absorption corrected Balmer decrement.

        Returns
        -------
        sfr : `igmtools.data.Data`
            Star formation rate in solar masses per year.

        """

        if isinstance(b_luminosity, Quantity):
            b_luminosity = b_luminosity.to(erg / s).value

        elif not isinstance(b_luminosity, float):
            raise ValueError('`b_luminosity` must be specified as a float or '
                             '`Quantity` instance')

        if not self.flux:
            raise AttributeError('flux attribute must be set to calculate a '
                                 'star formation rate')

        if balmer_decrement:
            # Assume obscuration curve of Cardelli et al. (1989):
            self.luminosity *= (balmer_decrement / 2.86) ** 2.114

        log_luminosity = log10(b_luminosity)
        i = np.searchsorted(b_luminosities_hb, log_luminosity)
        estimator = hb_estimator[i]

        sfr = self.luminosity.convert_unit_to(
            solMass / yr, equivalencies=estimator)

        return sfr


class OII(EmissionLine):
    """
    Class representation of the OII emission line. Includes a method
    for calculating the star-formation rate.

    Parameters
    ----------
    redshift : float
        Redshift of the emission line.

    flux : `igmtools.data.Data`, optional
        Integrated line flux.

    cont : `igmtools.data.Data`, optional
        Continuum flux at the line centre.

    eqw : `igmtools.data.Data`, optional
        Equivalent width of the line.

    """

    def __init__(self, redshift, flux=None, cont=None, eqw=None):

        wavelength = 3728.48  # vacuum, average of the doublet

        super(OII, self).__init__(wavelength, redshift, flux, cont, eqw)

    def compute_sfr(self, b_luminosity=None, balmer_decrement=None):
        """
        Compute the star formation rate.

        Parameters
        ----------
        b_luminosity : float, optional
            B-band luminosity (erg / s).

        balmer_decrement : float, optional
            Absorption corrected Balmer decrement.

        Returns
        -------
        sfr : `igmtools.data.Data`
            Star formation rate in solar masses per year.

        """

        if not self.flux:
            raise AttributeError('flux attribute must be set to calculate a '
                                 'star formation rate')

        if balmer_decrement:
            # Assume obscuration curve of Cardelli et al. (1989):
            self.luminosity *= (balmer_decrement / 2.86) ** 2.114

        if b_luminosity:
            log_luminosity = log10(b_luminosity)
            i = np.searchsorted(b_luminosities_oii, log_luminosity)
            estimator = oii_estimator[i]

            sfr = self.luminosity.convert_unit_to(
                solMass / yr, equivalencies=estimator)

        else:
            sfr = self.luminosity.convert_unit_to(
                solMass / yr, equivalencies=oii_sfr)

        return sfr
