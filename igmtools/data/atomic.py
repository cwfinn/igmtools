"""
Atomic data.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict

from astropy.table import Table
from astropy.units import angstrom, s, Quantity

import numpy as np

import os

__all__ = ['Ar', 'Transition', 'AtomDat']

data_path = os.path.abspath(__file__).rsplit('/', 1)[0] + '/data'

atomdat = None

Ar = dict(H=1.00794,
          He=4.002602,
          C=12.0107,
          N=14.0067,
          O=15.9994,
          Mg=24.3050,
          Al=26.9815386,
          Si=28.0855,
          P=30.973762,
          S=32.065,
          Ca=40.078,
          Fe=55.845,
          Ti=47.867,
          Zn=65.38,
          Cr=51.9961)


def get_atomdat():
    """
    Function to cache atom.dat.

    """

    global atomdat

    if atomdat is None:
        atomdat = AtomDat()

    return atomdat


class Transition(object):
    """
    An atomic transition.

    Parameters
    ----------
    parent : str
        Name of the ion, molecule or isotope, e.g. `HI`.

    wavelength : float
        Rest frame wavelength (Angstrom).

    osc : float
        Oscillator strength (dimensionless).

    gamma : float
        Gamma parameter (s^-1).

    """

    def __init__(self, parent, wavelength, osc, gamma):

        self.name = '{0} {1:.2f}'.format(parent, wavelength)
        self.parent = parent
        self.wavelength = wavelength * angstrom
        self.osc = Quantity(osc)
        self.gamma = gamma / s

    def __repr__(self):

        return 'Transition({0})'.format(self.name)


class AtomDat(defaultdict):
    """
    Atomic transition data from a VPFIT-style atom.dat file.

    Parameters
    ----------
    filename : str, optional
        The name of the atom.dat-style file. If not given, then the version
        bundled with this package is used.

    """

    def __init__(self, filename=None):

        if filename is None:
            filename = '{0}/atom.dat'.format(data_path)

        if filename.endswith('.gz'):
            import gzip
            fh = gzip.open(filename, 'rb')

        else:
            fh = open(filename, 'rb')

        specials = ('??', '__', '>>', '<<', '<>')

        super(AtomDat, self).__init__(list)

        rows = []

        for line in fh:

            line = line.decode()

            if not line[0].isupper() and line[:2] not in specials:
                continue

            identifier = line[:7].replace(' ', '')

            wavelength, osc, gamma = [
                float(item) for item in line[7:].split()[:3]]

            transition = Transition(identifier, wavelength, osc, gamma)

            self[identifier].append(transition)
            rows.append((identifier, wavelength, osc, gamma))

        self.table = Table(
            rows=rows, names=['ID', 'WAVELENGTH', 'OSC', 'GAMMA'])
        self.table['WAVELENGTH'].unit = angstrom
        self.table['GAMMA'].unit = 1 / s

        fh.close()

    def get_transition(self, name):
        """
        Gets information on a given transition.

        Parameters
        ----------
        name : str
            Name of the transition, something like `HI 1215`.

        Returns
        -------
        transition : `igmtools.data.atomic.Transition`
            Transition data.

        """

        i = 0
        name = name.strip()

        if name[:4] in set(['H2J0', 'H2J1', 'H2J2', 'H2J3', 'H2J4', 'H2J5',
                            'H2J6', 'H2J7', 'COJ0', 'COJ1', 'COJ2', 'COJ3',
                            'COJ4', 'COJ5']):
            i = 4

        elif name[:3] == 'C3I':
            i = 3

        else:
            while i < len(name) and (name[i].isalpha() or name[i] == '*'):
                i += 1

        identifier = name[:i]

        # Get all transition wavelengths and sorted indicies:
        wavelengths = np.array(
            [item.wavelength.value for item in self[identifier]])
        isort = wavelengths.argsort()

        try:
            wavelength = float(name[i:])

        except:
            raise ValueError('Possible transitions for {0}:\n'
                             '  {1}'.format(identifier, self.ion[identifier]))

        index = np.searchsorted(wavelengths[isort], wavelength)

        if index == len(wavelengths):
            index -= 1

        else:
            difference1 = np.abs(
                np.array(self[identifier])[isort][index].wavelength.value -
                wavelength)
            difference2 = np.abs(
                np.array(self[identifier])[isort][index - 1].wavelength.value -
                wavelength)
            if difference2 < difference1:
                index -= 1

        transition = np.array(self[identifier])[isort][index]

        return transition
