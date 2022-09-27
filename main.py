import threading
import time
import matplotlib.pyplot as plt
import scipy.constants as sc
import PyQt5.QtWidgets as qt
from Error_Window import Ui_Form
from scipy.constants import c as c_mks
import PyQt5.QtCore as qtc
import MotorClassFromAptProtocolConnor as apt
import ProcessingFunctions as pf
import numpy as np
import sys
import RUN_DataStreamApplication as dsa
import mkl_fft
import PlotWidgets as pw
import PyQt5.QtGui as qtg

edge_limit_buffer_mm = 0.0  # 1 um
COM1 = "COM4"
COM2 = "COM6"


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


# ______________________________________________________________________________________________________________________
# This class is essentially the imaging version of the StreamWithGui class from RUN_DataStreamApplication.py
# ______________________________________________________________________________________________________________________
class StreamWithGui(dsa.StreamWithGui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GuiTwoCards(qt.QMainWindow, dsa.Ui_MainWindow):
    def __init__(self):
        qt.QMainWindow.__init__(self)
        dsa.Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.shared_info = dsa.SharedInfo()

        self.stream1 = StreamWithGui(self, index=1, inifile_stream='include/Stream2Analysis_CARD1.ini',
                                     inifile_acquire='include/Acquire_CARD1.ini',
                                     shared_info=self.shared_info)
        self.stream2 = StreamWithGui(self, index=2, inifile_stream='include/Stream2Analysis_CARD2.ini',
                                     inifile_acquire='include/Acquire_CARD2.ini',
                                     shared_info=self.shared_info)

        self.show()

        self.ErrorWindow = ErrorWindow()

        self._card_index = 1
        self.active_stream = self.stream1
        self.Nyquist_Window = 1
        self.frep = 1e9

        self.lcd_ptscn_pos_um_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_ptscn_pos_um_2.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_ptscn_pos_fs_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_ptscn_pos_fs_2.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_lscn_pos_um_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_lscn_pos_um_2.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_lscn_pos_fs_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_lscn_pos_fs_2.setSegmentStyle(qt.QLCDNumber.Flat)

        self.lcd_ptscn_pos_um_1.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_um_2.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_fs_1.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_fs_2.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_um_1.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_um_2.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_fs_1.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_fs_2.setSmallDecimalPoint(True)

        self.le_nyq_window.setValidator(qtg.QIntValidator())
        self.le_frep.setValidator(qtg.QDoubleValidator())
        self.le_pos_um_1.setValidator(qtg.QDoubleValidator())
        self.le_pos_um_2.setValidator(qtg.QDoubleValidator())
        self.le_pos_fs_1.setValidator(qtg.QDoubleValidator())
        self.le_pos_fs_2.setValidator(qtg.QDoubleValidator())
        self.le_step_size_um_1.setValidator(qtg.QDoubleValidator())
        self.le_step_size_um_2.setValidator(qtg.QDoubleValidator())
        self.le_step_size_fs_1.setValidator(qtg.QDoubleValidator())
        self.le_step_size_fs_2.setValidator(qtg.QDoubleValidator())
        self.le_lscn_start_um_1.setValidator(qtg.QDoubleValidator())
        self.le_lscn_start_fs_1.setValidator(qtg.QDoubleValidator())
        self.le_lscn_start_um_2.setValidator(qtg.QDoubleValidator())
        self.le_lscn_start_fs_2.setValidator(qtg.QDoubleValidator())
        self.le_lscn_end_um_1.setValidator(qtg.QDoubleValidator())
        self.le_lscn_end_fs_1.setValidator(qtg.QDoubleValidator())
        self.le_lscn_end_um_2.setValidator(qtg.QDoubleValidator())
        self.le_lscn_end_fs_2.setValidator(qtg.QDoubleValidator())

        self.plot_ptscn = pw.PlotWindow(self.le_ptscn_xmin,
                                        self.le_ptscn_xmax,
                                        self.le_ptscn_ymin,
                                        self.le_ptscn_ymax,
                                        self.gv_ptscn)
        self.plot_lscn = pw.PlotWindow(self.le_lscn_xmin,
                                       self.le_lscn_xmax,
                                       self.le_lscn_ymin,
                                       self.le_lscn_ymax,
                                       self.gv_lscn)
        self.curve_ptscn = pw.create_curve()
        self.plot_ptscn.plotwidget.addItem(self.curve_ptscn)
        self.curve_lscn = pw.create_curve()
        self.plot_lscn.plotwidget.addItem(self.curve_lscn)

        self.stage_1 = MotorInterface(apt.KDC101(COM1))
        self.stage_2 = MotorInterface(apt.KDC101(COM2))
        self.stage_1.T0_um = float(np.loadtxt("T0_um_1.txt"))
        self.stage_2.T0_um = float(np.loadtxt("T0_um_2.txt"))
        self.stage_1.set_max_vel(1)
        self.stage_2.set_max_vel(1)
        self.pos_um_1 = None
        self.pos_um_2 = None
        self.update_lcd_pos_1(self.stage_1.pos_um)
        self.update_lcd_pos_2(self.stage_2.pos_um)

        self.motor_moving_1 = threading.Event()
        self.motor_moving_2 = threading.Event()
        self.lscn_running = threading.Event()
        self.stop_lscn = threading.Event()

        self.target_ptscn_um_1 = None
        self.target_ptscn_um_2 = None
        self.target_lscn_strt_um_1 = None
        self.target_lscn_strt_um_2 = None
        self.target_lscn_end_um_1 = None
        self.target_lscn_end_um_2 = None
        self.step_size_ptscn_um_1 = None
        self.step_size_ptscn_um_2 = None
        self.step_size_lscn_um = None
        self.update_stepsize_ptscn_um_1()
        self.update_stepsize_ptscn_um_2()
        self.update_stepsize_lscn_um()

        self.update_motor_thread_1 = None
        self.update_motor_thread_2 = None

        self.connect()

        # temporary storage variables __________________________________________________________________________________
        self._x2 = None
        self._y2 = None

        self._step_um = None
        self._step_x = None
        self._step_y = None
        self._npts = None
        self._n = None
        self._X = None
        self._Y = None
        self._FT = None
        self._WL = None
        # ______________________________________________________________________________________________________________

    @property
    def step_size_ptscn_fs_1(self):
        return dist_um_to_T_fs(self.step_size_ptscn_um_1)

    @step_size_ptscn_fs_1.setter
    def step_size_ptscn_fs_1(self, val):
        self.step_size_ptscn_um_1 = T_fs_to_dist_um(val)

    @property
    def step_size_ptscn_fs_2(self):
        return dist_um_to_T_fs(self.step_size_ptscn_um_2)

    @step_size_ptscn_fs_2.setter
    def step_size_ptscn_fs_2(self, val):
        self.step_size_ptscn_um_2 = T_fs_to_dist_um(val)

    @property
    def step_size_lscn_fs(self):
        return dist_um_to_T_fs(self.step_size_lscn_um)

    @step_size_lscn_fs.setter
    def step_size_lscn_fs(self, val):
        self.step_size_lscn_um = T_fs_to_dist_um(val)

    @property
    def target_ptscn_fs_1(self):
        dx = self.target_ptscn_um_1 - self.stage_1.T0_um
        return dist_um_to_T_fs(dx)

    @target_ptscn_fs_1.setter
    def target_ptscn_fs_1(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_1.T0_um + dx
        self.target_ptscn_um_1 = x

    @property
    def target_ptscn_fs_2(self):
        dx = self.target_ptscn_um_2 - self.stage_2.T0_um
        return dist_um_to_T_fs(dx)

    @target_ptscn_fs_2.setter
    def target_ptscn_fs_2(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_2.T0_um + dx
        self.target_ptscn_um_2 = x

    @property
    def target_lscn_strt_fs_1(self):
        dx = self.target_lscn_strt_um_1 - self.stage_1.T0_um
        return dist_um_to_T_fs(dx)

    @target_lscn_strt_fs_1.setter
    def target_lscn_strt_fs_1(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_1.T0_um + dx
        self.target_lscn_strt_um_1 = x

    @property
    def target_lscn_strt_fs_2(self):
        dx = self.target_lscn_strt_um_2 - self.stage_2.T0_um
        return dist_um_to_T_fs(dx)

    @target_lscn_strt_fs_2.setter
    def target_lscn_strt_fs_2(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_2.T0_um + dx
        self.target_lscn_strt_um_2 = x

    @property
    def target_lscn_end_fs_1(self):
        dx = self.target_lscn_end_um_1 - self.stage_1.T0_um
        return dist_um_to_T_fs(dx)

    @target_lscn_end_fs_1.setter
    def target_lscn_end_fs_1(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_1.T0_um + dx
        self.target_lscn_end_um_1 = x

    @property
    def target_lscn_end_fs_2(self):
        dx = self.target_lscn_end_um_2 - self.stage_2.T0_um
        return dist_um_to_T_fs(dx)

    @target_lscn_end_fs_2.setter
    def target_lscn_end_fs_2(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_2.T0_um + dx
        self.target_lscn_end_um_2 = x

    def connect(self):
        self.radbtn_card1.clicked.connect(self.select_card_1)
        self.radbtn_card2.clicked.connect(self.select_card_2)
        self.radbtn_trigon_1.clicked.connect(self.update_trigon_1)
        self.radbtn_trigon_2.clicked.connect(self.update_trigon_2)

        self.le_nyq_window.editingFinished.connect(self.setNyquistWindow)
        self.le_frep.editingFinished.connect(self.setFrep)
        self.le_pos_um_1.editingFinished.connect(self.update_target_um_1)
        self.le_pos_um_2.editingFinished.connect(self.update_target_um_2)
        self.le_pos_fs_1.editingFinished.connect(self.update_target_fs_1)
        self.le_pos_fs_2.editingFinished.connect(self.update_target_fs_2)
        self.le_step_size_um_1.editingFinished.connect(self.update_stepsize_ptscn_um_1)
        self.le_step_size_fs_1.editingFinished.connect(self.update_stepsize_ptscn_fs_1)
        self.le_step_size_um_2.editingFinished.connect(self.update_stepsize_ptscn_um_2)
        self.le_step_size_fs_2.editingFinished.connect(self.update_stepsize_ptscn_fs_2)
        self.le_lscn_step_size_um.editingFinished.connect(self.update_stepsize_lscn_um)
        self.le_lscn_step_size_fs.editingFinished.connect(self.update_stepsize_lscn_fs)

        self.le_lscn_start_um_1.editingFinished.connect(self.update_target_lscn_strt_um_1)
        self.le_lscn_start_um_2.editingFinished.connect(self.update_target_lscn_strt_um_2)
        self.le_lscn_start_fs_1.editingFinished.connect(self.update_target_lscn_strt_fs_1)
        self.le_lscn_start_fs_2.editingFinished.connect(self.update_target_lscn_strt_fs_2)
        self.le_lscn_end_um_1.editingFinished.connect(self.update_target_lscn_end_um_1)
        self.le_lscn_end_um_2.editingFinished.connect(self.update_target_lscn_end_um_2)
        self.le_lscn_end_fs_1.editingFinished.connect(self.update_target_lscn_end_fs_1)
        self.le_lscn_end_fs_2.editingFinished.connect(self.update_target_lscn_end_fs_2)

        self.btn_acquire_pt_scn.clicked.connect(self.acquire_and_get_spectrum)
        self.btn_set_T0_1.clicked.connect(self.set_T0_1)
        self.btn_set_T0_2.clicked.connect(self.set_T0_2)
        self.btn_move_to_pos_1.clicked.connect(self.move_to_pos_1)
        self.btn_move_to_pos_2.clicked.connect(self.move_to_pos_2)
        self.btn_home_stage_1.clicked.connect(self.home_stage_1)
        self.btn_home_stage_2.clicked.connect(self.home_stage_2)
        self.btn_step_left_1.clicked.connect(self.step_left_1)
        self.btn_step_right_1.clicked.connect(self.step_right_1)
        self.btn_step_left_2.clicked.connect(self.step_left_2)
        self.btn_step_right_2.clicked.connect(self.step_right_2)
        self.btn_lscn_start.clicked.connect(self.start_line_scan_notrigger)

    def connect_update_motor_1(self):
        self.update_motor_thread_1: UpdateMotorThread
        self.update_motor_thread_1.signal.progress.connect(self.update_lcd_pos_1)
        self.update_motor_thread_1.signal.finished.connect(self.end_of_move_motor_1)

    def connect_update_motor_2(self):
        self.update_motor_thread_2: UpdateMotorThread
        self.update_motor_thread_2.signal.progress.connect(self.update_lcd_pos_2)
        self.update_motor_thread_2.signal.finished.connect(self.end_of_move_motor_2)

    def end_of_move_motor_1(self):
        self.btn_move_to_pos_1.setText("move to position")
        self.btn_home_stage_1.setText("home stage")

    def end_of_move_motor_2(self):
        self.btn_move_to_pos_2.setText("move to position")
        self.btn_home_stage_2.setText("home stage")

    def select_card_1(self, flag):
        if flag:
            self._card_index = 1
            self.active_stream = self.stream1
            print("selecting card 1")

    def select_card_2(self, flag):
        if flag:
            self._card_index = 2
            self.active_stream = self.stream2
            print("selecting card 2")

    def setNyquistWindow(self):
        nyq_window = int(self.le_nyq_window.text())
        if nyq_window < 1:
            raise_error(self.ErrorWindow, "nyquist window must be >= 1")
            self.le_nyq_window.setText(str(int(self.Nyquist_Window)))
            return
        else:
            self.Nyquist_Window = nyq_window

    def setFrep(self):
        frep = float(self.le_frep.text())
        if frep <= 0:
            raise_error(self.ErrorWindow, "frep must be > 0")
            self.le_frep.setText(str(float(self.frep)))
            return
        else:
            self.frep = frep

    def update_lcd_pos_1(self, pos_um):
        self.pos_um_1 = pos_um
        pos_fs = dist_um_to_T_fs(pos_um - self.stage_1.T0_um)
        self.lcd_ptscn_pos_um_1.display('%.3f' % pos_um)
        self.lcd_ptscn_pos_fs_1.display('%.3f' % pos_fs)
        self.lcd_lscn_pos_um_1.display('%.3f' % pos_um)
        self.lcd_lscn_pos_fs_1.display('%.3f' % pos_fs)

    def update_lcd_pos_2(self, pos_um):
        self.pos_um_2 = pos_um
        pos_fs = dist_um_to_T_fs(pos_um - self.stage_2.T0_um)
        self.lcd_ptscn_pos_um_2.display('%.3f' % pos_um)
        self.lcd_ptscn_pos_fs_2.display('%.3f' % pos_fs)
        self.lcd_lscn_pos_um_2.display('%.3f' % pos_um)
        self.lcd_lscn_pos_fs_2.display('%.3f' % pos_fs)

    def update_target_um_1(self):
        target_um = float(self.le_pos_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_ptscn_um_1 = target_um

        self.le_pos_fs_1.setText('%.3f' % self.target_ptscn_fs_1)

    def update_target_fs_1(self):
        target_fs = float(self.le_pos_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_ptscn_um_1 = target_um

        self.le_pos_um_1.setText('%.3f' % self.target_ptscn_um_1)

    def update_target_um_2(self):
        target_um = float(self.le_pos_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_ptscn_um_2 = target_um

        self.le_pos_fs_2.setText('%.3f' % self.target_ptscn_fs_2)

    def update_target_fs_2(self):
        target_fs = float(self.le_pos_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_ptscn_um_2 = target_um

        self.le_pos_um_2.setText('%.3f' % self.target_ptscn_um_2)

    def update_target_lscn_strt_um_1(self):
        target_um = float(self.le_lscn_start_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_strt_um_1 = target_um

        self.le_lscn_start_fs_1.setText('%.3f' % self.target_lscn_strt_fs_1)

    def update_target_lscn_strt_fs_1(self):
        target_fs = float(self.le_lscn_start_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_strt_um_1 = target_um

        self.le_lscn_start_um_1.setText('%.3f' % self.target_lscn_strt_um_1)

    def update_target_lscn_strt_um_2(self):
        target_um = float(self.le_lscn_start_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_strt_um_2 = target_um

        self.le_lscn_start_fs_2.setText('%.3f' % self.target_lscn_strt_fs_2)

    def update_target_lscn_strt_fs_2(self):
        target_fs = float(self.le_lscn_start_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_strt_um_2 = target_um

        self.le_lscn_start_um_2.setText('%.3f' % self.target_lscn_strt_um_2)

    def update_target_lscn_end_um_1(self):
        target_um = float(self.le_lscn_end_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_end_um_1 = target_um

        self.le_lscn_end_fs_1.setText('%.3f' % self.target_lscn_end_fs_1)

    def update_target_lscn_end_fs_1(self):
        target_fs = float(self.le_lscn_end_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_end_um_1 = target_um

        self.le_lscn_end_um_1.setText('%.3f' % self.target_lscn_end_um_1)

    def update_target_lscn_end_um_2(self):
        target_um = float(self.le_lscn_end_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_end_um_2 = target_um

        self.le_lscn_end_fs_2.setText('%.3f' % self.target_lscn_end_fs_2)

    def update_target_lscn_end_fs_2(self):
        target_fs = float(self.le_lscn_end_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_lscn_end_um_2 = target_um

        self.le_lscn_end_um_2.setText('%.3f' % self.target_lscn_end_um_2)

    def update_stepsize_ptscn_um_1(self):
        step_size_um = float(self.le_step_size_um_1.text())
        self.step_size_ptscn_um_1 = step_size_um
        self.le_step_size_fs_1.setText('%.3f' % self.step_size_ptscn_fs_1)

    def update_stepsize_ptscn_fs_1(self):
        step_size_fs = float(self.le_step_size_fs_1.text())
        self.step_size_ptscn_fs_1 = step_size_fs
        self.le_step_size_um_1.setText('%.3f' % self.step_size_ptscn_um_1)

    def update_stepsize_ptscn_um_2(self):
        step_size_um = float(self.le_step_size_um_2.text())
        self.step_size_ptscn_um_2 = step_size_um
        self.le_step_size_fs_2.setText('%.3f' % self.step_size_ptscn_fs_2)

    def update_stepsize_ptscn_fs_2(self):
        step_size_fs = float(self.le_step_size_fs_2.text())
        self.step_size_ptscn_fs_2 = step_size_fs
        self.le_step_size_um_2.setText('%.3f' % self.step_size_ptscn_um_2)

    def update_stepsize_lscn_um(self):
        step_size_um = float(self.le_lscn_step_size_um.text())
        self.step_size_lscn_um = step_size_um
        self.le_lscn_step_size_fs.setText('%.3f' % self.step_size_lscn_fs)

    def update_stepsize_lscn_fs(self):
        step_size_fs = float(self.le_lscn_step_size_fs.text())
        self.step_size_lscn_fs = step_size_fs
        self.le_lscn_step_size_um.setText('%.3f' % self.step_size_lscn_um)

    def move_to_pos_1(self, *args, target_um=None):
        if self.motor_moving_1.is_set():
            self.update_motor_thread_1: UpdateMotorThread
            self.update_motor_thread_1.stop()
            return

        if target_um is None:
            target_um = self.target_ptscn_um_1

        ll_mm, ul_mm = self.stage_1.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([target_um < ll_um, target_um > ul_um]):
            raise_error(self.ErrorWindow, "value exceeds stage limits")
            return

        self.stage_1.pos_um = target_um  # start moving the motor

        self.btn_home_stage_1.setText("stop motor")
        self.btn_move_to_pos_1.setText("stop motor")
        self.update_motor_thread_1 = UpdateMotorThread(self.stage_1, self.motor_moving_1)
        self.connect_update_motor_1()
        thread = threading.Thread(target=self.update_motor_thread_1.run)
        self.motor_moving_1.set()
        thread.start()

    def move_to_pos_2(self, *args, target_um=None):
        if self.motor_moving_2.is_set():
            self.update_motor_thread_2: UpdateMotorThread
            self.update_motor_thread_2.stop()
            return

        if target_um is None:
            target_um = self.target_ptscn_um_2

        ll_mm, ul_mm = self.stage_2.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([target_um < ll_um, target_um > ul_um]):
            raise_error(self.ErrorWindow, "value exceeds stage limits")
            return

        self.stage_2.pos_um = target_um  # start moving the motor

        self.btn_home_stage_2.setText("stop motor")
        self.btn_move_to_pos_2.setText("stop motor")
        self.update_motor_thread_2 = UpdateMotorThread(self.stage_2, self.motor_moving_2)
        self.connect_update_motor_2()
        thread = threading.Thread(target=self.update_motor_thread_2.run)
        self.motor_moving_2.set()
        thread.start()

    def home_stage_1(self):
        if self.motor_moving_1.is_set():
            self.update_motor_thread_1: UpdateMotorThread
            self.update_motor_thread_1.stop()
            return

        self.stage_1.home()

        self.btn_home_stage_1.setText("stop motor")
        self.btn_move_to_pos_1.setText("stop motor")
        self.update_motor_thread_1 = UpdateMotorThread(self.stage_1, self.motor_moving_1)
        self.connect_update_motor_1()
        thread = threading.Thread(target=self.update_motor_thread_1.run)
        self.motor_moving_1.set()
        thread.start()

    def home_stage_2(self):
        if self.motor_moving_2.is_set():
            self.update_motor_thread_2: UpdateMotorThread
            self.update_motor_thread_2.stop()
            return

        self.stage_2.home()

        self.btn_home_stage_2.setText("stop motor")
        self.btn_move_to_pos_2.setText("stop motor")
        self.update_motor_thread_2 = UpdateMotorThread(self.stage_2, self.motor_moving_2)
        self.connect_update_motor_2()
        thread = threading.Thread(target=self.update_motor_thread_2.run)
        self.motor_moving_2.set()
        thread.start()

    def set_T0_1(self):
        pos_um = self.stage_1.pos_um  # read position from stage 1
        self.stage_1.T0_um = pos_um

        with open("T0_um_1.txt", "w") as file:
            file.write(str(pos_um))

        self.update_lcd_pos_1(pos_um)
        self.le_pos_fs_1.setText('0')
        self.update_target_fs_1()

    def set_T0_2(self):
        pos_um = self.stage_2.pos_um  # read position from stage 1
        self.stage_2.T0_um = pos_um

        with open("T0_um_2.txt", "w") as file:
            file.write(str(pos_um))

        self.update_lcd_pos_2(pos_um)
        self.le_pos_fs_2.setText('0')
        self.update_target_fs_2()

    def update_trigon_1(self, flag):
        if flag:
            print("turning ON trigger for stage 1")
            self.stage_1.trigger_on = True
        else:
            print("turning OFF trigger for stage 1")
            self.stage_1.trigger_on = False

    def update_trigon_2(self, flag):
        if flag:
            print("turning ON trigger for stage 2")
            self.stage_2.trigger_on = True
        else:
            print("turning OFF trigger for stage 2")
            self.stage_2.trigger_on = False

    def acquire_and_get_spectrum(self, *args, active_correct=False):
        # acquire
        try:
            self.active_stream.acquire(set_ppifg=False)
        except:
            raise_error(self.ErrorWindow, "FAILED TO ACQUIRE :(")
            return  # exit

        if self.active_stream.ppifg is None:
            raise_error(self.ErrorWindow, "ESTABLISH A PPIFG IN THE OSCILLOSCOPE TAB FIRST")
            return  # exit

        x = self.active_stream.single_acquire_array

        if active_correct:
            x = x[np.argmax(x[:self.active_stream.ppifg]):][self.active_stream.ppifg // 2:]
            N = len(x) // self.active_stream.ppifg
            x = x[:N * self.active_stream.ppifg]
            x.resize((N, self.active_stream.ppifg))

            # __________________________________________________________________________________________________________
            # below I just shift correct by overlapping the maxima of all the interferograms
            # __________________________________________________________________________________________________________

            ind_ref = np.argmax(x[0])  # maximum of first interferogram
            ind_diff = ind_ref - np.argmax(x, axis=1)  # maximum of first - maximum of all the rest

            for n, i in enumerate(x):
                x[n] = np.roll(i, ind_diff[n])
            x = np.mean(x, 0)

            ft = fft(x).__abs__()

            # __________________________________________________________________________________________________________
            # calculate the wavelength axis
            # __________________________________________________________________________________________________________
            center = self.active_stream.ppifg // 2
            Nyq_Freq = center * self.frep
            translation = (self.Nyquist_Window - 1) * Nyq_Freq
            nu = np.linspace(0, Nyq_Freq, center) + translation
            wl = np.where(nu > 0, sc.c * 1e6 / nu, np.nan)

            if self.Nyquist_Window % 2 == 0:
                ft = ft[:center]  # negative frequency side
            else:
                ft = ft[center:]  # positive frequency side

            # __________________________________________________________________________________________________________
            # update the plot
            # __________________________________________________________________________________________________________
            self.curve_ptscn.setData(wl, ft)
            return wl, ft

        else:
            return None, x

    def step_right_1(self, *args, step_um=None):
        if self.motor_moving_1.is_set():
            self.update_motor_thread_1: UpdateMotorThread
            self.update_motor_thread_1.stop()
            return

        if step_um is None:
            step_um = self.step_size_ptscn_um_1

        pos_um = self.stage_1.pos_um  # retrieve position from stage
        target_um = pos_um + step_um
        ll_mm, ul_mm = self.stage_1.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([target_um < ll_um, target_um > ul_um]):
            raise_error(self.ErrorWindow, "value exceeds stage limits")
            return

        self.stage_1.move_by_um(step_um)  # start moving the motor in relative mode

        self.update_motor_thread_1 = UpdateMotorThread(self.stage_1, self.motor_moving_1)
        self.connect_update_motor_1()
        thread = threading.Thread(target=self.update_motor_thread_1.run)
        self.motor_moving_1.set()
        thread.start()

    def step_left_1(self, *args, step_um=None):
        if self.motor_moving_1.is_set():
            self.update_motor_thread_1: UpdateMotorThread
            self.update_motor_thread_1.stop()
            return

        if step_um is None:
            step_um = self.step_size_ptscn_um_1

        pos_um = self.stage_1.pos_um  # retrieve position from stage
        target_um = pos_um - step_um  # step left -> subtract
        ll_mm, ul_mm = self.stage_1.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([target_um < ll_um, target_um > ul_um]):
            raise_error(self.ErrorWindow, "value exceeds stage limits")
            return

        self.stage_1.move_by_um(-step_um)  # start moving the motor in relative mode, note the minus sign

        self.update_motor_thread_1 = UpdateMotorThread(self.stage_1, self.motor_moving_1)
        self.connect_update_motor_1()
        thread = threading.Thread(target=self.update_motor_thread_1.run)
        self.motor_moving_1.set()
        thread.start()

    def step_right_2(self, *args, step_um=None):
        if self.motor_moving_2.is_set():
            self.update_motor_thread_2: UpdateMotorThread
            self.update_motor_thread_2.stop()
            return

        if step_um is None:
            step_um = self.step_size_ptscn_um_2

        pos_um = self.stage_2.pos_um  # retrieve position from stage
        target_um = pos_um + step_um
        ll_mm, ul_mm = self.stage_2.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([target_um < ll_um, target_um > ul_um]):
            raise_error(self.ErrorWindow, "value exceeds stage limits")
            return

        self.stage_2.move_by_um(step_um)  # start moving the motor in relative mode

        self.update_motor_thread_2 = UpdateMotorThread(self.stage_2, self.motor_moving_2)
        self.connect_update_motor_2()
        thread = threading.Thread(target=self.update_motor_thread_2.run)
        self.motor_moving_2.set()
        thread.start()

    def step_left_2(self, *args, step_um=None):
        if self.motor_moving_2.is_set():
            self.update_motor_thread_2: UpdateMotorThread
            self.update_motor_thread_2.stop()
            return

        if step_um is None:
            step_um = self.step_size_ptscn_um_2

        pos_um = self.stage_2.pos_um  # retrieve position from stage
        target_um = pos_um - step_um  # step left -> subtract
        ll_mm, ul_mm = self.stage_2.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([target_um < ll_um, target_um > ul_um]):
            raise_error(self.ErrorWindow, "value exceeds stage limits")
            return

        self.stage_2.move_by_um(-step_um)  # start moving the motor in relative mode, note the minus sign

        self.update_motor_thread_2 = UpdateMotorThread(self.stage_2, self.motor_moving_2)
        self.connect_update_motor_2()
        thread = threading.Thread(target=self.update_motor_thread_2.run)
        self.motor_moving_2.set()
        thread.start()

    def start_line_scan_notrigger(self):
        x1 = self.target_lscn_strt_um_1
        y1 = self.target_lscn_strt_um_2
        x2 = self.target_lscn_end_um_1
        y2 = self.target_lscn_end_um_2
        step_um = self.step_size_lscn_um

        self.line_scan_notrigger(x1, y1, x2, y2, step_um)

    # __________________________________________________________________________________________________________________
    # Below we do all scans without trigger. This is significantly easier than continuous move because no streaming is
    # necessary
    # __________________________________________________________________________________________________________________
    def line_scan_notrigger(self, x1, y1, x2, y2, step_um):
        if self.lscn_running.is_set():  # scan already running
            self.stop_lscn.set()  # stop the line scan
            return
        elif self.motor_moving_1.is_set():  # motor 1 currently in use
            raise_error(self.ErrorWindow, "stop stage 1 first")
            return
        elif self.motor_moving_2.is_set():  # motor 2 curently in use
            raise_error(self.ErrorWindow, "stop stage 2 first")
            return

        # treat motor 1 as x and motor 2 as y
        self.move_to_pos_1(target_um=x1)  # move to start position
        self.move_to_pos_2(target_um=y1)

        # store variables
        self._x2 = x2
        self._y2 = y2
        self._step_um = step_um

        # connect motor
        self.update_motor_thread_1.signal.finished.connect(self._check_if_at_start)

    def _check_if_at_start(self):
        if self.motor_moving_2.is_set():
            self.update_motor_thread_2.signal.finished.connect(self._check_if_at_start)
        else:
            self._line_scan_notrigger_2(self._x2, self._y2, self._step_um)

    def _line_scan_notrigger_2(self, x2, y2, step_um):
        # acquire the first spectrum and record the position
        wl, ft = self.acquire_and_get_spectrum()
        FT = [ft]
        x1 = self.stage_1.pos_um
        y1 = self.stage_2.pos_um
        X = [x1]
        Y = [y1]

        # calculate the step size in x and step size in y
        dx = x2 - x1
        dy = y2 - y1
        r = np.sqrt(dx ** 2 + dy ** 2)
        rx = dx / r  # rhat = (rx, ry), note that rx and / or ry can be negative
        ry = dy / r
        step_x = step_um * rx  # step_x, step_y can be negative -> we will only call step_right in the loop below!
        step_y = step_um * ry
        npts = np.ceil(r / step_um)

        # store variables
        self._step_x = step_x
        self._step_y = step_y
        self._step_um = step_um
        self._npts = npts
        self._n = 0
        self._X = X
        self._Y = Y
        self._FT = FT
        self._WL = wl

        # connect motor
        self.lscn_running.set()
        self.btn_lscn_start.setText("stop scan")
        self._step_one()

        # old for loop version (would cause GUI to freeze) _____________________________________________________________
        # Scan!
        # for n in range(npts):
        #     self.step_right_1(step_x)
        #     self.step_right_2(step_y)
        #
        #     X.append(self.stage_1.pos_um)
        #     Y.append(self.stage_2.pos_um)
        #     FT.append(self.acquire_and_get_spectrum()[1])
        #
        #     print(X[n], Y[n], f'acquired point {n} of {npts}')
        #
        # # convert lists to arrays and return
        # X = np.array(X)
        # Y = np.array(Y)
        # FT = np.array(FT)
        # return X, Y, wl, FT

    def _check_if_ready_for_next(self):
        if self.motor_moving_2.is_set():
            self.update_motor_thread_2.signal.finished.connect(self._check_if_ready_for_next)
        else:
            self._step_two()

    def lscn_stop_finished(self):
        self.stop_lscn.clear()
        self.lscn_running.clear()
        self.btn_lscn_start.setText("start scan")
        self._X = np.array(self._X)
        self._Y = np.array(self._Y)
        self._FT = np.array(self._FT)

    def _step_one(self):
        if self.stop_lscn.is_set():  # check for stop event
            self.lscn_stop_finished()
            return

        if self._n < self._npts:
            if abs(self._step_x) > 0:
                self.step_right_1(step_um=self._step_x)
            if abs(self._step_y) > 0:
                self.step_right_2(step_um=self._step_y)

            if abs(self._step_x) > 0:
                self.update_motor_thread_1.signal.finished.connect(self._check_if_ready_for_next)
            else:
                self.update_motor_thread_2.signal.finished.connect(self._check_if_ready_for_next)
        else:
            self.lscn_stop_finished()

    def _step_two(self):
        if self.stop_lscn.is_set():  # check for stop event
            self.lscn_stop_finished()
            return

        self._X.append(self.stage_1.pos_um)
        self._Y.append(self.stage_2.pos_um)
        self._FT.append(self.acquire_and_get_spectrum()[1])
        print(f'acquired point {self._n} of {self._npts}')

        self._n += 1
        self._step_one()


# %%____________________________________________________________________________________________________________________
# Runnable classes
# %%____________________________________________________________________________________________________________________
class UpdateMotorThread:
    def __init__(self, motor_interface, threading_event):
        motor_interface: MotorInterface
        self.motor_interface = motor_interface
        self.signal = Signal()
        threading_event: threading.Event
        self.event = threading_event
        self.stop_flag = threading.Event()

    def stop(self):
        self.stop_flag.set()
        print("will attempt stop in next loop")

    def run(self):
        while self.motor_interface.is_in_motion:
            if self.stop_flag.is_set():
                self.motor_interface.stop()
                print("attempted a stop")

            pos = self.motor_interface.pos_um
            self.signal.progress.emit(pos)

        self.event.clear()

        pos = self.motor_interface.pos_um
        self.signal.progress.emit(pos)
        self.signal.finished.emit(None)


if __name__ == '__main__':
    app = qt.QApplication([])
    hey = GuiTwoCards()
    app.exec()
