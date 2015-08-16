"""
Tools useful for fitting models to data.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .vpfit import read_f26

from ..data.atomic import AtomDat
from ..plot.velplot_utils import VelocityPlot

from PyQt4.QtGui import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                         QGridLayout, QInputDialog, QLabel, QTextEdit,
                         QTextCursor, QPushButton, QCheckBox, QFileDialog,
                         QDialog, QDialogButtonBox, QLineEdit,
                         QDoubleValidator, QValidator, QMessageBox, qApp)
from PyQt4.QtCore import Qt, QObject, pyqtSignal, pyqtSlot

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.transforms import blended_transform_factory

from astropy.table import Table
from astropy.constants import c
from astropy.units import km, s
from astropy.io import ascii

from math import ceil

import numpy as np

import sys
import os

c_kms = c.to(km / s).value
atom = AtomDat()

# Panel label inset format:
bbox = dict(facecolor='w', edgecolor='None')

info = """
The following keypress options are available:

space   Set the zero velocity at the position of the cursor and re-draw.
b       Print a VPFIT fitting region based on two cursor positions
        (requires pressing 'b' twice, once at each extrema).
l       Print a VPFIT line entry at the position of the cursor, guessing the
        column density (assumes b = 20 km/s).
z       Move the plot to a new redshift.
S       Save the figure.

"""


class OutputStream(QObject):

    text_written = pyqtSignal(str)

    def __init__(self):
        super(OutputStream, self).__init__()

    def write(self, text):
        self.text_written.emit(str(text))


class SpectralResolutionDialog(QDialog):

    def __init__(self, parent=None):

        super(SpectralResolutionDialog, self).__init__(parent)

        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.setSpacing(10)

        label1, label2 = QLabel(self), QLabel(self)
        label1.setText('Max wavelength')
        label2.setText('LSF FWHM (km/s)')

        self.wmax1, self.fwhm1 = QLineEdit(self), QLineEdit(self)
        self.wmax2, self.fwhm2 = QLineEdit(self), QLineEdit(self)
        self.wmax3, self.fwhm3 = QLineEdit(self), QLineEdit(self)
        self.wmax4, self.fwhm4 = QLineEdit(self), QLineEdit(self)

        validator = QDoubleValidator()
        self.wmax1.setValidator(validator)
        self.wmax2.setValidator(validator)
        self.wmax3.setValidator(validator)
        self.wmax4.setValidator(validator)
        self.fwhm1.setValidator(validator)
        self.fwhm2.setValidator(validator)
        self.fwhm3.setValidator(validator)
        self.fwhm4.setValidator(validator)

        grid.addWidget(label1, 1, 1)
        grid.addWidget(label2, 1, 2)
        grid.addWidget(self.wmax1, 2, 1)
        grid.addWidget(self.fwhm1, 2, 2)
        grid.addWidget(self.wmax2, 3, 1)
        grid.addWidget(self.fwhm2, 3, 2)
        grid.addWidget(self.wmax3, 4, 1)
        grid.addWidget(self.fwhm3, 4, 2)
        grid.addWidget(self.wmax4, 5, 1)
        grid.addWidget(self.fwhm4, 5, 2)

        self.wmax1.textChanged.connect(self.check_state)
        self.wmax1.textChanged.emit(self.wmax1.text())
        self.wmax2.textChanged.connect(self.check_state)
        self.wmax2.textChanged.emit(self.wmax2.text())
        self.wmax3.textChanged.connect(self.check_state)
        self.wmax3.textChanged.emit(self.wmax3.text())
        self.wmax4.textChanged.connect(self.check_state)
        self.wmax4.textChanged.emit(self.wmax4.text())

        self.fwhm1.textChanged.connect(self.check_state)
        self.fwhm1.textChanged.emit(self.fwhm1.text())
        self.fwhm2.textChanged.connect(self.check_state)
        self.fwhm2.textChanged.emit(self.fwhm2.text())
        self.fwhm3.textChanged.connect(self.check_state)
        self.fwhm3.textChanged.emit(self.fwhm3.text())
        self.fwhm4.textChanged.connect(self.check_state)
        self.fwhm4.textChanged.emit(self.fwhm4.text())

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addLayout(grid)
        layout.addWidget(buttons)

    def check_state(self):

        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]

        if state == QValidator.Acceptable:
            colour = '#c4df9b'  # green

        elif state == QValidator.Intermediate:
            colour = '#fff79a'  # yellow

        else:
            colour = '#f6989d'  # red

        sender.setStyleSheet('QLineEdit { background-color: %s }' % colour)

    def spectral_resolution(self):

        resolution = dict()

        wmax = [self.wmax1.text().toFloat()[0], self.wmax2.text().toFloat()[0],
                self.wmax3.text().toFloat()[0], self.wmax4.text().toFloat()[0]]
        fwhm = [self.fwhm1.text().toFloat()[0], self.fwhm2.text().toFloat()[0],
                self.fwhm3.text().toFloat()[0], self.fwhm4.text().toFloat()[0]]

        for i in range(len(wmax)):
            if wmax[i] != 0.0:
                resolution[wmax[i]] = fwhm[i]

        resolution = None if len(resolution.keys()) == 0 else resolution

        return resolution

    @staticmethod
    def get_spectral_resolution(parent=None):

        dialog = SpectralResolutionDialog(parent)
        result = dialog.exec_()

        spectral_resolution = dialog.spectral_resolution()

        return spectral_resolution, result == QDialog.Accepted


class InteractiveVelocityPlot(VelocityPlot):
    """
    Visual tool to help with fitting Voigt profiles to absorption lines
    in QSO spectra.

    """

    def __init__(self, filename, transitions, wavelength, flux, error,
                 continuum, redshift, **kwargs):

        # Main window:
        self.window = QMainWindow()

        # Host widget for plot, push button, and checkboxes:
        self.main_frame = QWidget()

        # Spectrum filename:
        self.filename = filename

        # Plotting options:
        self.options = kwargs

        # Optimise screen usage:
        if len(transitions) > 20:
            ncols = int(ceil(len(transitions) / 10))
        else:
            ncols = int(ceil(len(transitions) / 5))

        # Initialise plot:
        super(InteractiveVelocityPlot, self).__init__(
            transitions, ncols=ncols, aspect=0.45, **self.options['WINDOW'])

        # Attach canvas to host widget:
        self.canvas = FigureCanvasQTAgg(self)
        self.canvas.setParent(self.main_frame)

        # Text input/output:
        self.text_output = QTextEdit()

        # Push buttons:
        self.load_button = QPushButton('&Load')
        self.save_button = QPushButton('&Save')
        self.preview_button = QPushButton('&Preview')
        self.clear_button = QPushButton('&Clear')
        self.refresh_plot_button = QPushButton('&Refresh plot')
        self.plot_model_button = QPushButton('&Plot model')
        self.plot_all_models_button = QPushButton('&Plot all models')
        self.clear_models_button = QPushButton('&Clear models')
        self.runvpfit_button = QPushButton('&Run VPFIT')
        self.set_resolution = QPushButton('&Set resolution')
        self.help = QPushButton('&Help')
        self.quit = QPushButton('&Quit')

        # Checkboxes:
        self.cos_fuv_checkbox = QCheckBox('&use COS FUV LSF')
        self.cos_nuv_checkbox = QCheckBox('&use COS NUV LSF')
        self.include_checkbox = QCheckBox('&include previous')

        # Push button `clicked` connections:
        self.load_button.clicked.connect(self.on_load)
        self.save_button.clicked.connect(self.on_save)
        self.preview_button.clicked.connect(self.on_preview)
        self.clear_button.clicked.connect(self.on_clear)
        self.refresh_plot_button.clicked.connect(self.on_refresh)
        self.plot_model_button.clicked.connect(self.on_plot_model)
        self.plot_all_models_button.clicked.connect(self.on_plot_all_models)
        self.clear_models_button.clicked.connect(self.on_clear_models)
        self.runvpfit_button.clicked.connect(self.on_runvpfit)
        self.set_resolution.clicked.connect(self.on_resolution)
        self.help.clicked.connect(self.on_help)
        self.quit.clicked.connect(self.window.close)

        # Checkbox `stateChanged` connections:
        self.cos_fuv_checkbox.stateChanged.connect(self.on_cos_fuv_checkbox)
        self.cos_nuv_checkbox.stateChanged.connect(self.on_cos_nuv_checkbox)
        self.include_checkbox.stateChanged.connect(self.on_include_checkbox)

        # Set up grid layout:
        grid = QGridLayout()
        grid.setSpacing(10)

        # Add widgets:
        grid.addWidget(self.text_output, 1, 1, 4, 3)
        grid.addWidget(self.load_button, 1, 4)
        grid.addWidget(self.save_button, 2, 4)
        grid.addWidget(self.preview_button, 3, 4)
        grid.addWidget(self.clear_button, 4, 4)
        grid.addWidget(self.refresh_plot_button, 1, 5)
        grid.addWidget(self.plot_model_button, 2, 5)
        grid.addWidget(self.plot_all_models_button, 3, 5)
        grid.addWidget(self.clear_models_button, 4, 5)
        grid.addWidget(self.runvpfit_button, 1, 6)
        grid.addWidget(self.cos_fuv_checkbox, 2, 6)
        grid.addWidget(self.cos_nuv_checkbox, 3, 6)
        grid.addWidget(self.include_checkbox, 4, 6)
        grid.addWidget(self.set_resolution, 1, 7)
        grid.addWidget(self.help, 3, 7)
        grid.addWidget(self.quit, 4, 7)

        # Place plotting canvas above the options grid:
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addLayout(grid)

        # Set the layout to the host widget:
        self.main_frame.setLayout(vbox)

        # Store all static elements (can be very slow to re-draw these):
        self.canvas.draw()
        self.backgrounds = [self.canvas.copy_from_bbox(ax.bbox)
                            for ax in self.axes]

        # Plot the data:
        self.plot_data(wavelength, flux, error, continuum, redshift,
                       **self.options['DATA'])

        # Set the window title:
        self.window.setWindowTitle('vpguess z = {0:.3f}'.format(self.redshift))

        # Give keyboard focus to the figure:
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        # Variable defaults:
        self.cos_fuv = False
        self.cos_nuv = False
        self.include = False
        self.last_loaded = None
        self.resolution = None

        # Keypress variables:
        self.previous_keypress = None
        self.previous_wavelength = None

        # Set up the key-press events:
        self.canvas.mpl_connect('key_press_event', self.on_keypress)

        # Set the main frame as the central widget:
        self.window.setCentralWidget(self.main_frame)

        # Resize the window so it will display the canvas with the
        # requested size:
        layout_height = vbox.sizeHint().height()
        layout_width = vbox.sizeHint().width()
        status_bar_height = self.window.statusBar().sizeHint().height()
        height = layout_height + status_bar_height
        self.window.resize(layout_width, height)

        # Re-do labels:
        del self.texts[:]
        self.text(0.45, 0.02, 'Velocity offset (km s$^{-1}$)')
        self.text(0.01, 0.57, 'Transmission', rotation=90)

        # Disable any existing callbacks:
        self.cids = dict()
        cids1 = list(self.canvas.callbacks.callbacks['key_press_event'])
        cids2 = list(self.canvas.callbacks.callbacks['button_press_event'])
        cids = cids1 + cids2
        for cid in cids:
            self.canvas.callbacks.disconnect(cid)

        # Connect new callbacks:
        self.connect()

    def update_plot(self, redshift):

        # Restore background regions:
        [self.canvas.restore_region(background)
         for background in self.backgrounds]

        # Plot data:
        [data.pop(0).remove() for data in self.data]
        self.plot_data(
            self.wavelength, self.flux, self.error, self.continuum,
            redshift, **self.options['DATA'])

        # Draw the artists:
        for artists in self.data:
            for element in artists:
                ax = element.get_axes()
                ax.draw_artist(element)

        # Plot the models (if any):
        if self.options['MODEL']['absorbers'] is not None:
            self.plot_models(resolution=self.resolution,
                             convolve_with_cos_fuv=self.cos_fuv,
                             convolve_with_cos_nuv=self.cos_nuv,
                             **self.options['MODEL'])

            # Draw the artists:
            for artists in self.models:
                for element in artists:
                    ax = element.get_axes()
                    ax.draw_artist(element)

        # Draw new panel labels:
        for i, transition in enumerate(self.transitions):

            ax = self.axes[self._ind[i]]
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

            label = ax.text(0.03, 0.5, name, fontsize=self.label_fontsize,
                            bbox=bbox, transform=transf)
            ax.draw_artist(label)

        # Update:
        self.canvas.update()
        self.window.setWindowTitle(
            'vpguess z = {0:.3f}'.format(self.redshift))

    @staticmethod
    def concatenate_results():

        from glob import glob
        results = glob('*.result')

        with open('.all_results', 'wb') as combined:
            for result in results:
                with open(result, 'rb') as single:
                    combined.write(single.read())

    @pyqtSlot(str)
    def on_output(self, output):

        self.text_output.moveCursor(QTextCursor.End)
        self.text_output.insertPlainText(output)

    def on_load(self):

        self.text_output.clear()

        directory = os.getcwd()
        filename = QFileDialog.getOpenFileName(
            self.window, 'Select f26 file', directory)

        with open(filename, 'rb') as f26:
            self.text_output.setText(f26.read())

        self.last_loaded = filename

    def on_save(self):

        directory = os.getcwd()
        filename, ok = QFileDialog.getSaveFileName(
            self.window, 'Save f26 file', directory)

        with open(filename, 'w') as f26:
            f26.write(str(self.text_output.toPlainText()))

        self.text_output.clear()
        self.last_loaded = None

    def on_preview(self):

        self.concatenate_results()

        with open('.f26_preview', 'wb') as preview:
            with open('.all_results', 'rb') as results:
                preview.write(str(self.text_output.toPlainText()))
                preview.write(results.read())

        f26 = read_f26('.f26_preview')

        self.options['MODEL']['absorbers'] = f26.absorbers
        self.update_plot(self.redshift)

    def on_clear(self):

        self.text_output.clear()
        self.last_loaded = None

    def on_refresh(self):

        self.update_plot(self.redshift)

    def on_plot_model(self):

        directory = os.getcwd()
        filename = QFileDialog.getOpenFileName(
            self.window, 'Select f26 file', directory)

        f26 = read_f26(filename)
        self.options['MODEL']['absorbers'] = f26.absorbers

        if f26.absorbers is not None:
            self.update_plot(self.redshift)

    def on_plot_all_models(self):

        self.concatenate_results()

        f26 = read_f26('.all_results')
        self.options['MODEL']['absorbers'] = f26.absorbers

        if f26.absorbers is not None:
            self.update_plot(self.redshift)

    def on_clear_models(self):

        self.options['MODEL']['absorbers'] = None
        self.update_plot(self.redshift)

    def on_runvpfit(self):

        from .vpfit import run_vpfit

        directory = os.getcwd()

        if self.last_loaded is not None:

            filename = self.last_loaded

            if self.text_output.document().isModified():
                with open(filename, 'w') as f26:
                    f26.write(str(self.text_output.toPlainText()))

        else:

            filename = QFileDialog.getSaveFileName(
                self.window, 'Save f26 file', directory)

            with open(filename, 'w') as f26:
                f26.write(str(self.text_output.toPlainText()))

        self.concatenate_results()
        inc = '.all_results' if self.include else None

        self.text_output.clear()
        fwhm = self.resolution if self.resolution is not None else 10
        run_vpfit(filename, inc, fwhm=fwhm, cos_fuv=self.cos_fuv,
                  cos_nuv=self.cos_nuv)

        self.concatenate_results()
        f26 = read_f26('.all_results')
        self.options['MODEL']['absorbers'] = f26.absorbers

        if f26.absorbers is not None:
            self.update_plot(self.redshift)

        self.last_loaded = None

    def on_cos_fuv_checkbox(self, state):

        if state == Qt.Checked:
            self.cos_fuv = True

        else:
            self.cos_fuv = False

    def on_cos_nuv_checkbox(self, state):

        if state == Qt.Checked:
            self.cos_nuv = True

        else:
            self.cos_nuv = False

    def on_include_checkbox(self, state):

        if state == Qt.Checked:
            self.include = True

        else:
            self.include = False

    def on_resolution(self):

        resolution, ok = SpectralResolutionDialog.get_spectral_resolution(
            self.main_frame)

        if ok:
            self.resolution = resolution

    def on_help(self):

        QMessageBox.information(self.main_frame, 'Help', info, False)

    def on_buttonpress(self, event):

        if event.inaxes is None:
            return

        i = self.axes.index(event.inaxes)
        transition = self.transitions[np.where(self._ind == i)[0]]

        z = (1 + event.xdata / c_kms) * (1 + self.redshift) - 1
        wavelength = transition.wavelength.value * (1 + z)

        isort = np.argsort(self.ticks['wavelength'])
        wavelengths = self.ticks['wavelength'][isort]
        transitions = self.ticks['transition'][isort]

        idx = wavelengths.searchsorted(wavelength)
        wavelength = wavelengths[idx]
        transition = atom.get_transition(transitions[idx])
        z = wavelength / transition.wavelength.value - 1

        message = '{0}, z = {1:.3f}'.format(transition.name, z)
        QMessageBox.information(self.main_frame, 'Transition', message, False)

    def on_keypress(self, event):

        if event.key == ' ' and event.inaxes is not None:

            z = self.redshift

            # Get amount to add to redshift:
            dz = (event.xdata / c_kms) * (1 + z)

            # Get new axis limits, if any:
            vmin, vmax = event.inaxes.get_xlim()
            self.vmin = min(0, vmin)
            self.vmax = max(0, vmax)
            self.update_plot(z + dz)

        if event.key == 'z':

            redshift, ok = QInputDialog.getText(
                self.main_frame, 'New Redshift', 'Enter redshift: ', False)

            if ok:
                self.update_plot(float(redshift))

        if event.key == 'b':

            i = self.axes.index(event.inaxes)
            transition = self.transitions[np.where(self._ind == i)[0]]

            z = (1 + event.xdata / c_kms) * (1 + self.redshift) - 1
            wavelength = transition.wavelength.value * (1 + z)

            if (self.previous_keypress == 'b' and
                    self.previous_wavelength is not None):

                wmin = self.previous_wavelength
                wmax = wavelength

                if wmin > wmax:
                    wmin, wmax = wmax, wmin

                print('%% {0} 1 {1:.3f} {2:.3f} vfwhm=10.0'.format(
                    self.filename, wmin, wmax))

                self.previous_keypress = None
                self.previous_wavelength = None

            else:
                self.previous_wavelength = wavelength

        if event.key == 'l':

            from ..calculations.absorption import logn_from_tau_peak

            i = self.axes.index(event.inaxes)
            transition = self.transitions[np.where(self._ind == i)[0]]

            z = (1 + event.xdata / c_kms) * (1 + self.redshift) - 1
            wavelength = transition.wavelength.value * (1 + z)
            index = self.wavelength.searchsorted(wavelength)

            flux = self.flux[index - 1:index + 1]
            error = self.error[index - 1:index + 1]
            continuum = self.continuum[index - 1:index + 1]
            flux_norm = flux / continuum
            error_norm = error / continuum

            valid = (error_norm > 0) & ~np.isnan(flux_norm)

            if not any(valid):
                print('No good pixels!')
                return

            flux_norm = np.median(flux_norm[valid])
            error_norm = np.median(error_norm[valid])

            if flux_norm < error_norm:
                flux_norm = error_norm

            elif flux_norm > 1 - error_norm:
                flux_norm = 1 - error_norm

            b = 20  # assume 20 km/s
            tau = -np.log(flux_norm)
            logn = logn_from_tau_peak(transition, tau, b)

            print('{0:6s} {1:8.6f} 0.0 {2:4.1f} 0.0 {3:4.1f} 0.0'.format(
                transition.parent, z, b, logn))

        if event.key == 'S':

            filename, ok = QInputDialog.getText(
                self.main_frame, 'Save Figure', 'Enter filename: ', False)

            if ok:
                self.savefig(filename)

        self.previous_keypress = event.key

    def connect(self):
        cids = dict()
        cids['key_press_event'] = self.canvas.mpl_connect(
            'key_press_event', self.on_keypress)
        cids['button_press_event'] = self.canvas.mpl_connect(
            'button_press_event', self.on_buttonpress)
        self.cids.update(cids)


def main(args=None):
    """
    This is the main function called by the `ivelplot` script.

    """

    from astropy.utils.compat import argparse
    from astropy.extern.configobj import configobj, validate

    from pkg_resources import resource_stream

    parser = argparse.ArgumentParser(
        description='An interactive environment for absorption line '
                    'identification and Voigt profile \nfitting with VPFIT.\n'
                    '\nTo dump a default configuration file: ivelplot -d'
                    '\nTo dump an extended default configuration file: '
                    'ivelplot -dd',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('config', help='path to the configuration file')
    parser.add_argument('-z', '--redshift', help='redshift')
    parser.add_argument('--search', action='store_true',
                        help='display a general search list of ions')
    parser.add_argument('--lyman', action='store_true',
                        help='display the Lyman series transitions')
    parser.add_argument('--galactic', action='store_true',
                        help='display the common Galactic transitions')
    parser.add_argument('--agn', action='store_true',
                        help='display the common AGN associated transitions')

    config = resource_stream(__name__, '/config/ivelplot.cfg')
    config_extended = resource_stream(
        __name__, '/config/ivelplot_extended.cfg')
    spec = resource_stream(__name__, '/config/ivelplot_specification.cfg')

    if len(sys.argv) > 1:

        if sys.argv[1] == '-d':
            cfg = configobj.ConfigObj(config)
            cfg.filename = '{0}/ivelplot.cfg'.format(os.getcwd())
            cfg.write()
            return

        elif sys.argv[1] == '-dd':
            cfg = configobj.ConfigObj(config_extended)
            cfg.filename = '{0}/ivelplot.cfg'.format(os.getcwd())
            cfg.write()
            return

    args = parser.parse_args(args)

    try:
        cfg = configobj.ConfigObj(args.config, configspec=spec)
        validator = validate.Validator()
        cfg.validate(validator)

    except:
        raise IOError('Configuration file could not be read')

    fname = cfg['WINDOW'].pop('transitions')

    if args.search:
        fh = resource_stream(__name__, '/data/search.dat')
        transitions = list(fh)
        fh.close()

    elif args.lyman:
        fh = resource_stream(__name__, '/data/lyman.dat')
        transitions = list(fh)
        fh.close()

    elif args.galactic:
        fh = resource_stream(__name__, '/data/galactic.dat')
        transitions = list(fh)
        fh.close()

    elif args.agn:
        fh = resource_stream(__name__, '/data/agn.dat')
        transitions = list(fh)
        fh.close()

    else:
        print('Reading transitions from ', fname)
        fh = open(fname)
        transitions = list(fh)
        fh.close()

    transitions = [transition for transition in transitions
                   if not transition.startswith('#')]

    fname = cfg['DATA'].pop('filename')
    if not fname:
        raise IOError('no data to plot!')

    spectrum = Table.read(fname) if fname[-4:] == 'fits' else ascii.read(fname)
    wavelength = spectrum[cfg['DATA'].pop('wavelength_column')]
    flux = spectrum[cfg['DATA'].pop('flux_column')]
    error = spectrum[cfg['DATA'].pop('error_column')]
    continuum = spectrum[cfg['DATA'].pop('continuum_column')]
    redshift = float(args.redshift) if args.redshift is not None else 0

    cfg['MODEL']['system_width'] = (cfg['WINDOW']['vmax'] -
                                    cfg['WINDOW']['vmin'])
    cfg['MODEL']['absorbers'] = None

    print(info)

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)

    desktop = app.desktop()
    screen = desktop.screenGeometry()
    width = screen.width() / desktop.physicalDpiX() * 0.88

    fontsize = 0.7 * width
    label_fontsize = 0.6 * width

    cfg['WINDOW']['width'] = width
    cfg['WINDOW']['fontsize'] = fontsize
    cfg['WINDOW']['label_fontsize'] = label_fontsize

    velocity_plot = InteractiveVelocityPlot(
        fname, transitions, wavelength, flux, error, continuum, redshift,
        **cfg)
    velocity_plot.window.show()

    output_stream = OutputStream()
    output_stream.text_written.connect(velocity_plot.on_output)

    sys.stdout = output_stream
    sys.exit(app.exec_())
