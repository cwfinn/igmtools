"""
Set of tools for running VPFIT and parsing output.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..data.spectral import Absorber
from ..data.atomic import AtomDat

from astropy.table import Table
from astropy.io import ascii

from textwrap import wrap

import numpy as np

import subprocess
import bisect
import shutil
import time
import os

__all__ = ['VpfitModel', 'read_f26', 'parse_entry', 'parse_lines',
           'parse_regions', 'run_vpfit']

atom = AtomDat()


def parse_entry(entry):
    """
    Separates an entry into a numeric value and a tied/fixed flag,
    if present.

    Parameters
    ----------
    entry : str
        An f26 parameter entry.

    Returns
    -------
    value : float
        The entry value.

    flag : str
        The fitting flag associated with the entry.

    """

    if entry.startswith('nan'):
        value = float(entry[:3])
        flag = entry[3:]

    else:
        i = -1

        while not entry[i].isdigit():
            i -= 1

        if i != -1:
            value = float(entry[:i + 1])
            flag = entry[i + 1:]

        else:
            value = float(entry)
            flag = ''

    return value, flag


def parse_lines(parameters):
    """
    Separates the parameters from their tied/fixed/special
    characters.

    Parameters
    ----------
    parameters : list
        A list of rows (each a tuple) containing the line parameters from an
        f26 file.

    Returns
    -------
    lines : `astropy.table.Table`
        Tabulated fit parameters.

    """

    temp = []

    for identifier, z, b, logn, z_error, b_error, logn_error in parameters:

        z, z_flag = parse_entry(z)
        b, b_flag = parse_entry(b)
        logn, logn_flag = parse_entry(logn)

        try:
            z_error = float(z_error)
        except ValueError:
            z_error = -1

        try:
            b_error = float(b_error)
        except ValueError:
            b_error = -1

        try:
            logn_error = float(logn_error)
        except ValueError:
            logn_error = -1

        temp.append((identifier, z, z_flag, z_error, b, b_flag, b_error,
                     logn, logn_flag, logn_error))

    lines = Table(rows=temp,
                  names=['ID', 'Z', 'Z_FLAG', 'Z_ERR', 'B', 'B_FLAG', 'B_ERR',
                         'LOGN', 'LOGN_FLAG', 'LOGN_ERR'])

    return lines


def parse_regions(rows, resolution=None):
    """
    Parses the region information from an f26 file.

    Parameters
    ----------
    regions : list
        A list of rows (each a tuple) describing the fitting regions in an
        f26 file.

    resolution : str
        The resolution information for the spectra fitted, e.g. `vfwhm=10.0`.

    Returns
    -------
    regions : `astropy.table.Table`
        Tabulated fitting regions info.

    """

    if resolution is None:
        resolution = ''

    regions = None
    row_info = []

    for row in rows:

        r = row.split('!')[0].lstrip().lstrip('%%').split()
        n_items = len(r)
        r[2] = float(r[2])
        r[3] = float(r[3])

        if n_items == 4:
            row_info.append(tuple(r + [resolution]))

        elif n_items > 4:
            r = r[:4] + [' '.join(r[4:])]
            row_info.append(tuple(r))

        else:
            raise Exception('bad format in fitting regions:\n {0}'.format(row))

    if row_info:
        regions = Table(rows=row_info,
                        names=['FILENAME', 'SPECTRUM', 'WMIN', 'WMAX',
                               'RESOLUTION'])

    return regions


class VpfitModel(object):
    """
    Holds all the information about a VPFIT model. Can write out the
    model as a fort.26 style file.

    Parameters
    ----------
    identifiers : array, optional
        Ion, molecule and/or isotope identifiers.

    z : array, optional
        Redshift.

    b : array, optional
        Doppler broadening parameter (km/s).

    logn : array, optional
        Log10 column density (cm^-2).

    z_flags : array, optional
        Redshift fitting flags.

    b_flags : array, optional
        Doppler broadening parameter fitting flags.

    logn_flags : array, optional
        Log10 column density fitting flags.

    filenames : array, optional
        Spectrum filenames for each fitting region.

    wmin, wmax : array, optional
        Lower and upper wavelength bounds on the fitting regions.

    resolution : array, optional
        The resolution information for each of the spectra fitted,
        e.g. `vfwhm=10.0`.

    spectrum : array, optional
        Unique index for each spectrum.

    Attributes
    ----------
    lines : `astropy.table.Table`
        Table holding parameters relating to absorption lines.

    absorbers : list
        A list of `igmtools.data.Absorber` instances for each of the absorbers
        defined by the absorption line parameters.

    regions : `astropy.table.Table`
        Table holding parameters relating to fitting regions.

    stats : dict
        Dictionary holding the statistics on the fit performed by VPFIT
        (initialised to None).

    Methods
    -------
    write_f26()
    copy()

    Notes
    -----
    A VpfitModel instance may be initialised with no parameters, simply as
    VpfitModel(). Parameters can be specified at initialisation or later. To
    initialise the model from en existing VPFIT f26 file, use the `read_f26()`
    function provided as part of this module.

    """

    def __init__(self, identifiers=None, z=None, b=None, logn=None,
                 z_flags=None, b_flags=None, logn_flags=None, filenames=None,
                 wmin=None, wmax=None, resolution=None, spectrum=None):

        if None in (identifiers, logn, z, b):
            self.lines = None
            self.absorbers = None

        else:
            assert len(z) == len(logn) == len(b) == len(identifiers)
            n_components = len(z)

            if z_flags is None:
                z_flags = [''] * n_components

            if b_flags is None:
                b_flags = [''] * n_components

            if logn_flags is None:
                logn_flags = [''] * n_components

            z_error = [-1] * n_components
            b_error = [-1] * n_components
            logn_error = [-1] * n_components

            self.lines = Table(
                [identifiers, z, z_flags, z_error, b, b_flags, b_error, logn,
                 logn_flags, logn_error],
                names=['ID', 'Z', 'Z_FLAG', 'Z_ERR', 'B', 'B_FLAG', 'B_ERR',
                       'LOGN', 'LOGN_FLAG', 'LOGN_ERR'])

            specials = ('__', '<<', '>>', '<>')

            self.absorbers = []

            for line in self.lines:

                if line['ID'].strip() in specials:
                    continue

                absorber = Absorber(
                    line['ID'].replace(' ', ''), line['Z'], line['LOGN'],
                    line['B'])

                self.absorbers.append(absorber)

        if None in (filenames, wmin, wmax):
            self.regions = None

        else:
            if resolution is None:
                resolution = [''] * len(filenames)

            if spectrum is None:
                spectrum = ['1'] * len(filenames)

            assert all((len(n) < filename_length) for n in filenames)

            self.regions = Table(
                [filenames, spectrum, wmin, wmax, resolution],
                names=['FILENAME', 'SPECTRUM', 'WMIN', 'WMAX', 'RESOLUTION'])

        self.stats = None

    def __repr__(self):
        temp = ', '.join(sorted(str(attr) for attr in self.__dict__ if not
                         str(attr).startswith('_')))

        return 'VpfitModel({0})'.format('\n      '.join(wrap(temp, width=69)))

    def write_f26(self, filename, write_regions=True):
        """
        Writes out the model to an f26-style file.

        """

        temp = []

        if write_regions and self.regions is not None:
            for r in self.regions:
                temp.append('%%%% {0}  {1}  {2:7.2f}  {3:7.2f}  {4}\n'.format(
                            r['FILENAME'], r['SPECTRUM'], r['WMIN'],
                            r['WMAX'], r['RESOLUTION']))

        if self.lines is not None:
            for l in self.lines:
                temp.append(' {0}     {1:11.8f}{2} {3:11.8f} {4:6.2f}{5} '
                            '{6:6.2f} {7:7.4f}{8} {9:7.4f}\n'.format(
                                l['ID'], l['Z'], l['Z_FLAG'], l['Z_ERR'],
                                l['B'], l['B_FLAG'], l['B_ERR'], l['LOGN'],
                                l['LOGN_FLAG'], l['LOGN_ERR']))

        open(filename, 'wb').writelines(temp)

    def copy(self):
        """
        Makes a copy of the `VpfitModel` instance.

        """
        from copy import deepcopy
        return deepcopy(self)


def read_f26(f26, resolution=None):
    """
    Reads a f26 style file and returns a VpfitModel instance.

    Parameters
    ----------
    f26 : str
      Path to the f26 file.

    resolution : str
      This provides the resolution information for the spectra fitted,
      e.g. `vfwhm=10.0`.

    Returns
    -------
    model : `igmtools.modeling.VpfitModel`
        The VPFIT model.

    """

    fh = open(f26, 'rb')
    lines = fh.readlines()
    fh.close()

    model = VpfitModel()

    if len(lines) == 0:
        return model

    lines = [row for row in lines if not row.lstrip().startswith('!')
             or 'Stats' in row]

    region_rows = [row for row in lines if row.lstrip().startswith('%%')]
    parameter_rows = [row for row in lines if '%%' not in row and
                      'Stats' not in row and row.lstrip()]

    keys = 'iterations nchisq npts dof prob ndropped info'.split()
    stats_row = [row for row in lines if 'Stats' in row]

    if stats_row:

        if stats_row[0].split()[-1] == 'BAD':
            status = 'BAD'

        else:
            status = 'OK'

        values = stats_row[0].split()[2:8] + [status]
        model.stats = dict(zip(keys, values))

    elif parameter_rows:

        # Older style f26 file:
        stats = parameter_rows[0]
        status = ('BAD' if stats.split()[-1] == 'BAD' else 'OK')
        values = [stats[66:71], stats[71:85], stats[85:90], stats[90:95],
                  stats[95:102], stats[102:107], status]
        model.stats = dict(zip(keys, values))

    model.regions = parse_regions(region_rows, resolution=resolution)

    if len(parameter_rows) == 0:
        return model

    parameter_rows = [row.lstrip() for row in parameter_rows]
    parameters = []
    molecule_names = set(('H2J0 H2J1 H2J2 H2J3 H2J4 H2J5 H2J6 '
                          'COJ0 COJ1 COJ2 COJ3 COJ4 COJ5 COJ6 '
                          'HDJ0 HDJ1 HDJ2').split())

    for row in parameter_rows:

        if 'nan' in row:
            i = row.index('nan')
            parameters.append([row[:i]] + row[i:].split())
            continue

        if row[:4] in molecule_names:
            i = 4

        else:
            i = 0
            while not row[i].isdigit() and row[i] != '-':
                i += 1

        parameters.append([row[:i]] + row[i:].split())

    parameters = [[p[0], p[1], p[3], p[5], p[2], p[4], p[6]]
                  for p in parameters]
    model.lines = parse_lines(parameters)

    izeros = np.array(
        [row.index for row in model.lines if row['ID'].strip() == '__'])
    zeros = model.lines[izeros]
    isort = model.regions.argsort('WMIN')

    zero_offset_regions = []

    for zero in zeros:

        wavelength = atom['__'][0].wavelength * (1 + zero['Z'])
        i0 = model.regions['WMIN'][isort].searchsorted(wavelength)
        i1 = model.regions['WMAX'][isort].searchsorted(wavelength)
        assert i0 - 1 == i1

        wmin = model.regions['WMIN'][isort[i1]]
        wmax = model.regions['WMAX'][isort[i1]]

        zero_offset_regions.append((wmin, wmax))

    specials = ('__', '<<', '>>', '<>')
    absorbers = []

    for line in model.lines:

        if line['ID'].strip() in specials:
            continue

        covering_factor = 1.0

        if len(izeros) != 0:

            if line.index < izeros[0]:
                identifier = atom.table['ID'] == line['ID'].replace(' ', '')
                info = atom.table[identifier]
                info.sort('OSC')

                for i, wavelength in enumerate(info['WAVELENGTH']):

                    in_region = np.array(
                        [(wavelength * (1 + line['Z']) >= region[0]) &
                         (wavelength * (1 + line['Z']) <= region[1])
                         for region in zero_offset_regions])

                    if any(in_region):
                        offset = zeros['LOGN'][in_region].item()
                        covering_factor = 1 - offset
                        break

        absorber = Absorber(
            line['ID'].replace(' ', ''), line['Z'], line['LOGN'], line['B'],
            covering_factor)
        absorbers.append(absorber)

    model.absorbers = absorbers

    return model


def run_vpfit(f26, inc=None, fwhm=10, cos_fuv=False, cos_nuv=False):
    """
    Run VPFIT with the specified parameters.

    Parameters
    ----------
    f26 : str
        f26 filename.

    inc : str, optional
        f26 files to include.

    fwhm : float or dict, optional
        FWHM of the LSF of the spectrograph in km/s (default = 10).
        VPFIT will assume a Gaussian LSF with this FWHM. This can be a
        dictionary with key:value pairs in the format wmax:fwhm, where wmax
        is the maximum wavelength for which the specified fwhm holds.

    cos_fuv : bool, optional
        Set to True to use the COS FUV LSF (default = False). This overrides
        whatever is specified for `fwhm` at wavelengths < 1795 Angstrom.

    cos_nuv : bool, optional
        Set to True to use the COS NUV LSF (default = False). This overrides
        whatever is specified for `fwhm` at wavelengths between 1795 and 3200
        Angstrom.

    Notes
    -----
    If cos_fuv is True, this function currently assumes that the `M` gratings
    are used. Support for the FUV `L` gratings may be implemented at some
    point in the future.

    """

    fh = open(f26, 'rb')
    wmid, old_row, params, spectrum = [], [], [], []

    for row in fh:

        if not row.lstrip().startswith('%%'):
            params.append(row)
            continue

        temp = row.split()

        old = row.split('!')[0].rsplit(None, 1)[0]
        old_row.append(old)

        spectrum.append(temp[1])

        wmin, wmax = float(temp[3]), float(temp[4])
        wmid.append(0.5 * (wmin + wmax))

    fh.close()

    # Keep version history:
    i = 0
    old_prefix = '{}.old'.format(f26)
    old_name = old_prefix

    while os.path.lexists(old_name):
        i += 1
        old_name = '{0}.{1}'.format(old_prefix, i)

    shutil.move(f26, old_name)

    fh = open(f26, 'wb')

    for i, w in enumerate(wmid):

        if cos_fuv and (w < 1795) or cos_nuv and ((w >= 1795) & (w < 3200)):

            try:
                from COS import read_lsf

            except:
                raise ImportError('COS is not installed!')

            try:
                sp = ascii.read(spectrum[i])

            except:
                raise TypeError('the input spectrum file should be of type '
                                'ascii')

            wavelength = sp.colnames[0]
            index = sp[wavelength].searchsorted(w)
            dw = sp[wavelength][index + 1] - sp[wavelength][index]
            dw /= 9  # we subdivide by 9 in VPFIT

            outname = 'LSF/LSF_{:.1f}.txt'.format(w)
            fh.write(old_row[i] + ' pfin={}\n'.format(outname))

            if w < 1460:
                lsf = read_lsf('G130M')

            elif (w >= 1460) & (w < 1795):
                lsf = read_lsf('G160M')

            else:
                lsf = read_lsf('G230L')

            lsf.write(w, dw)

        else:

            if isinstance(fwhm, (int, float)):
                outname = ' vfwhm={:.1f}\n'.format(fwhm)

            elif isinstance(fwhm, dict):
                wmax = fhwm.keys().sort()
                if w >= wmax[-1]:
                    raise ValueError('LSF FWHM has not been specified for '
                                     'wavelengths > {} Angstrom'.format(w))
                index = bisect.bisect_left(w, wmax)
                outname = ' vfwhm={:.1f}\n'.format(fwhm[wmax[index]])

            else:
                raise TypeError('`fwhm` should be of type float or dict')

            fh.write(old_row[i] + outname)

    for line in params:
        fh.write(line)

    fh.close()
    fh = open('.vpfit_input', 'wb')

    if inc is not None:
        fh.write('f inc {0}\n\n{1}\nn\n\n'.format(inc, f26))

    else:
        fh.write('f\n\n{}\nn\n\n'.format(f26))

    fh.close()

    out = subprocess.check_output(
        ['/usr/local/VPFIT10/vpfit < .vpfit_input'], shell=True)
    subprocess.call(['echo n'], shell=True)
    time.sleep(0.5)

    result = '{}.result'.format(f26)
    out_file = sorted(os.listdir('.'), key=os.path.getctime)[-1]
    shutil.move(out_file, result)

    log = '{}.log'.format(f26)
    with open(log, 'wb') as log_file:
        log_file.write(out)


def main(args=None):
    """
    This is the main function called by the `runvpfit` script.

    """

    from astropy.utils.compat import argparse

    parser = argparse.ArgumentParser(
        description='Run VPFIT using a specified f26 file as input.')

    parser.add_argument('f26', help='f26 input filename')
    parser.add_argument('--include', help='path to the f26 file to include')
    parser.add_argument('--fwhm', type=float, default=10,
                        help='FWHM of the LSF of the spectrograph in km/s')
    parser.add_argument('--cos-fuv', help='option to use the HST/COS FUV LSF',
                        action='store_true')
    parser.add_argument('--cos-nuv', help='option to use the HST/COS NUV LSF',
                        action='store_true')

    args = parser.parse_args(args)

    run_vpfit(args.f26, inc=args.include, fwhm=args.fwhm, cos_fuv=args.cos_fuv,
              cos_nuv=args.cos_nuv)
