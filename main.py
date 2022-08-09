import PyQt5.QtWidgets as qt
from Error import Ui_Form
import utilities as util
from scipy.constants import c as c_mks
import PyQt5.QtCore as qtc

edge_limit_buffer_mm = 0.0  # 1 um


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
        motor: util.Motor
        self.motor = motor

        self.T0_um = 0  # T0 position of the motor in micron

        # don't let the stage come closer than this to the stage limits.
        self._safety_buffer_mm = edge_limit_buffer_mm  # 1um

        self.error_window = ErrorWindow()

    @property
    def pos_um(self):
        return self.motor.position_mm * 1e3

    @property
    def pos_fs(self):
        # pos_fs is taken from pos_um and T0_um
        return dist_um_to_T_fs(self.pos_um - self.T0_um)

    @pos_um.setter
    def pos_um(self, value_um):
        # move the motor to the new position, assuming they give the motor
        # position in mm
        self.motor.position_mm = value_um * 1e-3

    @pos_fs.setter
    def pos_fs(self, value_fs):
        # pos_fs is taken from pos_um, so just set pos_um
        # setting pos_um moves the motor
        self.pos_um = T_fs_to_dist_um(value_fs) + self.T0_um

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

    def value_exceeds_limits(self, value_um):
        predicted_pos_um = value_um + self.pos_um
        max_limit_um = self.motor.max_pos_mm * 1e3
        min_limit_um = self.motor.min_pos_mm * 1e3
        buffer_um = self._safety_buffer_mm * 1e3

        if (predicted_pos_um < min_limit_um + buffer_um) or (
                predicted_pos_um > max_limit_um - buffer_um):
            raise_error(self.error_window,
                        "too close to stage limits (within 1um)")
            return True
        else:
            return False
