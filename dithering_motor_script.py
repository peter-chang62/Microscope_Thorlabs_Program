import matplotlib.pyplot as plt
import scipy.constants as sc
import PyQt5.QtWidgets as qt
from Error_Window import Ui_Form
from scipy.constants import c as c_mks
import PyQt5.QtCore as qtc
import MotorClassFromAptProtocolConnor as apt
import ProcessingFunctions as pf
import numpy as np
import mkl_fft

edge_limit_buffer_mm = 0.0  # 1 um
COM1 = "COM8"
COM2 = "COM9"


def fft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def ifft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def dist_um_to_T_fs(value_um):
    """
    :param value_um: delta x in micron
    :return value_fs: delta t in femtosecond
    """
    return (2 * value_um / c_mks) * 1e9


def T_fs_to_dist_um(value_fs):
    """
    :param value_fs: delta t in femtosecond
    :return value_um: delta x in micron
    """
    return (c_mks * value_fs / 2) * 1e-9


class ErrorWindow(qt.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def set_text(self, text):
        self.textBrowser.setText(text)


def raise_error(error_window, text):
    error_window.set_text(text)
    error_window.show()


class Signal(qtc.QObject):
    started = qtc.pyqtSignal(object)
    progress = qtc.pyqtSignal(object)
    finished = qtc.pyqtSignal(object)


class MotorInterface:
    """To help with integrating other pieces of hardware, I was thinking to
    keep classes in utilities.py more bare bone, and focus on hardware
    communication there. Here I will add more things I would like the Motor
    class to have. This class expects an instance of util.Motor class from
    utilities.py """

    def __init__(self, motor):
        motor: apt.KDC101
        self.motor = motor

        self.T0_um = 0  # T0 position of the motor in micron

        # don't let the stage come closer than this to the stage limits.
        self._safety_buffer_mm = edge_limit_buffer_mm  # 1um

    @property
    def pos_um(self):
        return self.motor.position * 1e3

    @pos_um.setter
    def pos_um(self, value_um):
        # move the motor to the new position, assuming they give the motor
        # position in mm
        self.motor.position = value_um * 1e-3

    @property
    def pos_fs(self):
        # pos_fs is taken from pos_um and T0_um
        return dist_um_to_T_fs(self.pos_um - self.T0_um)

    @pos_fs.setter
    def pos_fs(self, value_fs):
        # pos_fs is taken from pos_um, so just set pos_um
        # setting pos_um moves the motor
        self.pos_um = T_fs_to_dist_um(value_fs) + self.T0_um

    @property
    def trigger_on(self):
        return self.motor.trigger_on

    @trigger_on.setter
    def trigger_on(self, flag):
        self.motor.trigger_on = flag  # there is already an assert statement to check if flag is a bool in motor

    @property
    def is_in_motion(self):
        return self.motor.is_in_motion

    @property
    def step_um(self):
        return self.motor.step_mm * 1e3

    @step_um.setter
    def step_um(self, val_um):
        val_mm = val_um * 1e-3
        self.motor.step_mm = val_mm

    @property
    def pulse_width_ms(self):
        return self.motor.pulse_width_ms

    @pulse_width_ms.setter
    def pulse_width_ms(self, width_ms):
        self.motor.pulse_width_ms = width_ms

    def stop(self):
        self.motor.stop_profiled()

    def home(self):
        return self.motor.move_home()

    def move_by_fs(self, value_fs):
        # obtain the distance to move in micron and meters
        value_um = T_fs_to_dist_um(value_fs)
        value_mm = value_um * 1e-3

        # move the motor to the new position and update the position in micron
        self.motor.move_by(value_mm)

    def move_by_um(self, value_um):
        value_mm = value_um * 1e-3

        # move the motor to the new position and update the position in micron
        self.motor.move_by(value_mm)

    def set_max_vel(self, m_s):
        self.motor.set_max_vel(m_s)


m = MotorInterface(apt.KDC101(COM1))
m.set_max_vel(.5)
pos = [10.5e3, 11.2e3]
b = 0
while True:
    m.pos_um = pos[b]
    while m.is_in_motion:
        pass
    b = not b
