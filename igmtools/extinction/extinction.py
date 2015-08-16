"""
Core representation for extinction curves.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

__all__ = ['ExtinctionCurve', 'MWCardelli89']


class ExtinctionCurve(object):
    """
    Describes an extinction curve.

    """

    def __init__(self, wavelength, Rv, AlamAv, EBmV=None,
                 name='ExtinctionCurve'):

        self.name = name
        self.wavelength = wavelength
        self.Rv = Rv
        self.AlamAv = AlamAv
        self._ElamV = None

        if EBmV is not None:
            self._EBmV = EBmV

        else:
            self._EBmV = None
            self.Av = None
            self.tau = None

    def __repr__(self):
        s = '< {0._name}: R(V)={0.Rv}, E(B-V)={0._EBmV}, A(V)={0._Av} >'
        return s.format(self)

    @property
    def EBmV(self):
        return self._EBmV

    @EBmV.setter
    def EBmV(self, value):

        self._EBmV = value
        self.Av = value * self.Rv

        self.tau = self.AlamAv * self.Av / (2.5 * np.log10(np.e))

    @property
    def ElamV(self):

        if self._ElamV is None:
            self._ElamV = self.AlamAv * self.Rv - self.Rv

        return self._ElamV


class MWCardelli89(ExtinctionCurve):
    """
    Milky Way extinction law from Cardelli et al. (1989).

    Parameters
    ----------
    wavelength : array
        Wavelengths at which to evaluate the extinction (Angstrom).

    Rv : float, optional
        R(V). The default is for the diffuse ISM. A value of 5 is generally
        assumed for dense molecular clouds.

    Notes
    -----
    A power law extrapolation is used if there are any wavelengths past the
    IR or far UV limits of the Cardelli law.

    References
    ----------
    http://adsabs.harvard.edu/abs/1989ApJ...345..245C

    """

    def __init__(self, wavelength, EBmV=None, Rv=3.1):

        x = 1e4 / wavelength

        a = np.ones_like(x)
        b = np.ones_like(x)

        ir = (0.3 <= x) & (x <= 1.1)
        vis = (1.1 <= x) & (x <= 3.3)
        nuv1 = (3.3 <= x) & (x <= 5.9)
        nuv2 = (5.9 <= x) & (x <= 8)
        fuv = (8 <= x) & (x <= 10)

        # Infrared
        if ir.any():
            temp = x[ir] ** 1.61
            a[ir] = 0.574 * temp
            b[ir] = -0.527 * temp

        # NIR/optical
        if vis.any():
            co1 = (0.32999, -0.7753, 0.01979, 0.72085, -0.02427,
                   -0.50447, 0.17699, 1.0)
            a[vis] = np.polyval(co1, x[vis] - 1.82)
            co2 = (-2.09002, 5.3026, -0.62251, -5.38434, 1.07233,
                   2.28305, 1.41338, 0.0)
            b[vis] = np.polyval(co2, x[vis] - 1.82)

        # NUV
        if nuv1.any():
            a[nuv1] = (1.752 - 0.316 * x[nuv1] -
                       0.104 / ((x[nuv1] - 4.67) ** 2 + 0.341))
            b[nuv1] = (-3.09 + 1.825 * x[nuv1] +
                       1.206 / ((x[nuv1] - 4.62) ** 2 + 0.263))

        if nuv2.any():
            y = x[nuv2] - 5.9
            Fa = -0.04473 * y ** 2 - 0.009779 * y ** 3
            Fb = 0.2130 * y ** 2 + 0.1207 * y ** 3
            a[nuv2] = (1.752 - 0.316 * x[nuv2] -
                       0.104 / ((x[nuv2] - 4.67) ** 2 + 0.341) + Fa)
            b[nuv2] = (-3.09 + 1.825 * x[nuv2] +
                       1.206 / ((x[nuv2] - 4.62) ** 2 + 0.263) + Fb)

        # FUV
        if fuv.any():
            a[fuv] = np.polyval((-0.070, 0.137, -0.628, -1.073), x[fuv] - 8)
            b[fuv] = np.polyval((0.374, -0.42, 4.257, 13.67), x[fuv] - 8)

        AlamAv = a + b / Rv

        # Extrapolate in log space (i.e. a power law) if there are any
        # wavelengths outside the Cardelli range.
        ir_extrap = x < 0.3
        if ir_extrap.any():
            coeff = np.polyfit(np.log(x[ir][-2:]), np.log(AlamAv[ir][-2:]), 1)
            AlamAv[ir_extrap] = np.exp(np.polyval(coeff, np.log(x[ir_extrap])))

        uv_extrap = x > 10
        if uv_extrap.any():
            coeff = np.polyfit(np.log(x[fuv][:2]), np.log(AlamAv[fuv][:2]), 1)
            AlamAv[uv_extrap] = np.exp(np.polyval(coeff, np.log(x[uv_extrap])))

        super(MWCardelli89, self).__init__(
            wavelength, Rv, AlamAv, EBmV=EBmV, name='MWCardelli89')
