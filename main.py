# %% package imports
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
import RUN_DataStreamApplication as rdsa
import mkl_fft
import PlotWidgets as pw
import PyQt5.QtGui as qtg
import DataStreamApplication as dsa
from scipy.integrate import simps

# %% global variables
edge_limit_buffer_mm = 0.0  # 1 um
COM1 = "COM8"  # will change whenever usb's reconnect to computer
COM2 = "COM9"
active_correct_line_scan = True
databackup_path = r"D:\\Microscope\\databackup/"

extra_steps_linescan = 10
# The imaging with trigger is acquired by repeated linescans. Originally, I had
# it set to scan the stage that would minimize the number of linescans.
# However, this sometimes resulted in me having to constantly switch the
# trigger on the stages. So, if you like you can set it to only ever scan one
# of the stages (I chose stage 1).
trigger_stage_1_only = True


# %% function defs
def fft(x, axis=None):
    """
    calculates the 1D fft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(
            mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
        )


def ifft(x, axis=None):
    """
    calculates the 1D ifft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    if axis is None:
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x)))
    else:
        return np.fft.fftshift(
            mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
        )


def rfft(x, axis=None):
    """
    calculates the 1D rfft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    # ifftshift the input, the output is already fftshifted
    if axis is None:
        return mkl_fft.rfft_numpy(np.fft.ifftshift(x))
    else:
        return mkl_fft.rfft_numpy(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=None):
    """
    calculates the 1D irfft of the numpy array x
    if x is not 1D you need to specify the axis
    """

    # fftshift the output, the input is already fftshifted
    if axis is None:
        return np.fft.fftshift(mkl_fft.irfft_numpy(x))
    else:
        return np.fft.fftshift(mkl_fft.irfft_numpy(x, axis=axis), axes=axis)


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


def raise_error(error_window, text):
    error_window.set_text(text)
    error_window.show()


# %% utility classes
class ErrorWindow(qt.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def set_text(self, text):
        self.textBrowser.setText(text)


class Signal(qtc.QObject):
    started = qtc.pyqtSignal(object)
    progress = qtc.pyqtSignal(object)
    finished = qtc.pyqtSignal(object)


# %% motor interface
class MotorInterface:
    """To help with integrating other pieces of hardware, I was thinking to
    keep classes in utilities.py more bare bone, and focus on hardware
    communication there. Here I will add more things I would like the Motor
    class to have. This class expects an instance of util.Motor class from
    utilities.py"""

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


# %% GUI
# _____________________________________________________________________________
# This class is essentially the imaging version of the StreamWithGui class from
# RUN_DataStreamApplication.py
# _____________________________________________________________________________
class StreamWithGui(rdsa.StreamWithGui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GuiTwoCards(qt.QMainWindow, rdsa.Ui_MainWindow):
    def __init__(self):
        qt.QMainWindow.__init__(self)
        rdsa.Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.shared_info = rdsa.SharedInfo()

        self.stream1 = StreamWithGui(
            self,
            index=1,
            inifile_stream="include/Stream2Analysis_CARD1.ini",
            inifile_acquire="include/Acquire_CARD1.ini",
            shared_info=self.shared_info,
        )
        self.stream2 = StreamWithGui(
            self,
            index=2,
            inifile_stream="include/Stream2Analysis_CARD2.ini",
            inifile_acquire="include/Acquire_CARD2.ini",
            shared_info=self.shared_info,
        )

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
        self.lcd_img_pos_um_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_img_pos_um_2.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_img_pos_fs_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_img_pos_fs_2.setSegmentStyle(qt.QLCDNumber.Flat)

        self.lcd_ptscn_pos_um_1.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_um_2.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_fs_1.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_fs_2.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_um_1.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_um_2.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_fs_1.setSmallDecimalPoint(True)
        self.lcd_lscn_pos_fs_2.setSmallDecimalPoint(True)
        self.lcd_img_pos_um_1.setSmallDecimalPoint(True)
        self.lcd_img_pos_um_2.setSmallDecimalPoint(True)
        self.lcd_img_pos_fs_1.setSmallDecimalPoint(True)
        self.lcd_img_pos_fs_2.setSmallDecimalPoint(True)

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
        self.le_img_start_um_1.setValidator(qtg.QDoubleValidator())
        self.le_img_start_fs_1.setValidator(qtg.QDoubleValidator())
        self.le_img_start_um_2.setValidator(qtg.QDoubleValidator())
        self.le_img_start_fs_2.setValidator(qtg.QDoubleValidator())
        self.le_img_end_um_1.setValidator(qtg.QDoubleValidator())
        self.le_img_end_fs_1.setValidator(qtg.QDoubleValidator())
        self.le_img_end_um_2.setValidator(qtg.QDoubleValidator())
        self.le_img_end_fs_2.setValidator(qtg.QDoubleValidator())

        self.plot_ptscn = pw.PlotWindow(
            self.le_ptscn_xmin,
            self.le_ptscn_xmax,
            self.le_ptscn_ymin,
            self.le_ptscn_ymax,
            self.gv_ptscn,
        )
        self.plot_lscn = pw.PlotWindow(
            self.le_lscn_xmin,
            self.le_lscn_xmax,
            self.le_lscn_ymin,
            self.le_lscn_ymax,
            self.gv_lscn,
        )
        self.plot_image = pw.PlotWindow(
            self.le_img_xmin,
            self.le_img_xmax,
            self.le_img_ymin,
            self.le_img_ymax,
            self.gv_img,
        )

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
        self.img_running = threading.Event()
        self.stop_lscn = threading.Event()
        self.stop_img = threading.Event()
        self.calling_from_image = threading.Event()

        self.target_ptscn_um_1 = None
        self.target_ptscn_um_2 = None
        self.target_lscn_strt_um_1 = None
        self.target_lscn_strt_um_2 = None
        self.target_lscn_end_um_1 = None
        self.target_lscn_end_um_2 = None
        self.target_img_strt_um_1 = None
        self.target_img_strt_um_2 = None
        self.target_img_end_um_1 = None
        self.target_img_end_um_2 = None
        self.step_size_ptscn_um_1 = None
        self.step_size_ptscn_um_2 = None
        self.step_size_lscn_um = None
        self.step_size_img_um = None
        self.update_stepsize_ptscn_um_1()
        self.update_stepsize_ptscn_um_2()
        self.update_stepsize_lscn_um()
        self.update_stepsize_img_um()

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

        self._N_linescans = None
        self._n_img = None
        self._x1_img = None
        self._x2_img = None
        self._y1_img = None
        self._y2_img = None
        self._step_um_img = None
        self._scan_img = None
        self._vel_mm_s = None
        self._IMG = None
        self._h = None

        self._card_stream_progress_fcts = None
        self._card_stream_finished_fcts = None
        # _____________________________________________________________________

        # a signal to use inside the main Gui
        self.signal = Signal()

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
    def step_size_img_fs(self):
        return dist_um_to_T_fs(self.step_size_img_um)

    @step_size_img_fs.setter
    def step_size_img_fs(self, val):
        self.step_size_img_um = T_fs_to_dist_um(val)

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

    @property
    def target_img_strt_fs_1(self):
        dx = self.target_img_strt_um_1 - self.stage_1.T0_um
        return dist_um_to_T_fs(dx)

    @target_img_strt_fs_1.setter
    def target_img_strt_fs_1(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_1.T0_um + dx
        self.target_img_strt_um_1 = x

    @property
    def target_img_strt_fs_2(self):
        dx = self.target_img_strt_um_2 - self.stage_2.T0_um
        return dist_um_to_T_fs(dx)

    @target_img_strt_fs_2.setter
    def target_img_strt_fs_2(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_2.T0_um + dx
        self.target_img_strt_um_2 = x

    @property
    def target_img_end_fs_1(self):
        dx = self.target_img_end_um_1 - self.stage_1.T0_um
        return dist_um_to_T_fs(dx)

    @target_img_end_fs_1.setter
    def target_img_end_fs_1(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_1.T0_um + dx
        self.target_img_end_um_1 = x

    @property
    def target_img_end_fs_2(self):
        dx = self.target_img_end_um_2 - self.stage_2.T0_um
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
        self.le_img_step_size_um.editingFinished.connect(self.update_stepsize_img_um)
        self.le_img_step_size_fs.editingFinished.connect(self.update_stepsize_img_fs)

        self.le_lscn_start_um_1.editingFinished.connect(
            self.update_target_lscn_strt_um_1
        )
        self.le_lscn_start_um_2.editingFinished.connect(
            self.update_target_lscn_strt_um_2
        )
        self.le_lscn_start_fs_1.editingFinished.connect(
            self.update_target_lscn_strt_fs_1
        )
        self.le_lscn_start_fs_2.editingFinished.connect(
            self.update_target_lscn_strt_fs_2
        )
        self.le_lscn_end_um_1.editingFinished.connect(self.update_target_lscn_end_um_1)
        self.le_lscn_end_um_2.editingFinished.connect(self.update_target_lscn_end_um_2)
        self.le_lscn_end_fs_1.editingFinished.connect(self.update_target_lscn_end_fs_1)
        self.le_lscn_end_fs_2.editingFinished.connect(self.update_target_lscn_end_fs_2)

        self.le_img_start_um_1.editingFinished.connect(self.update_target_img_strt_um_1)
        self.le_img_start_um_2.editingFinished.connect(self.update_target_img_strt_um_2)
        self.le_img_start_fs_1.editingFinished.connect(self.update_target_img_strt_fs_1)
        self.le_img_start_fs_2.editingFinished.connect(self.update_target_img_strt_fs_2)
        self.le_img_end_um_1.editingFinished.connect(self.update_target_img_end_um_1)
        self.le_img_end_um_2.editingFinished.connect(self.update_target_img_end_um_2)
        self.le_img_end_fs_1.editingFinished.connect(self.update_target_img_end_fs_1)
        self.le_img_end_fs_2.editingFinished.connect(self.update_target_img_end_fs_2)

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
        self.btn_img_start.clicked.connect(self.start_image_no_trigger)

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
        self.lcd_ptscn_pos_um_1.display("%.3f" % pos_um)
        self.lcd_ptscn_pos_fs_1.display("%.3f" % pos_fs)
        self.lcd_lscn_pos_um_1.display("%.3f" % pos_um)
        self.lcd_lscn_pos_fs_1.display("%.3f" % pos_fs)
        self.lcd_img_pos_um_1.display("%.3f" % pos_um)
        self.lcd_img_pos_fs_1.display("%.3f" % pos_fs)

    def update_lcd_pos_2(self, pos_um):
        self.pos_um_2 = pos_um
        pos_fs = dist_um_to_T_fs(pos_um - self.stage_2.T0_um)
        self.lcd_ptscn_pos_um_2.display("%.3f" % pos_um)
        self.lcd_ptscn_pos_fs_2.display("%.3f" % pos_fs)
        self.lcd_lscn_pos_um_2.display("%.3f" % pos_um)
        self.lcd_lscn_pos_fs_2.display("%.3f" % pos_fs)
        self.lcd_img_pos_um_2.display("%.3f" % pos_um)
        self.lcd_img_pos_fs_2.display("%.3f" % pos_fs)

    def update_target_um_1(self):
        target_um = float(self.le_pos_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_ptscn_um_1 = target_um

        self.le_pos_fs_1.setText("%.3f" % self.target_ptscn_fs_1)

    def update_target_fs_1(self):
        target_fs = float(self.le_pos_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_ptscn_um_1 = target_um

        self.le_pos_um_1.setText("%.3f" % self.target_ptscn_um_1)

    def update_target_um_2(self):
        target_um = float(self.le_pos_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_ptscn_um_2 = target_um

        self.le_pos_fs_2.setText("%.3f" % self.target_ptscn_fs_2)

    def update_target_fs_2(self):
        target_fs = float(self.le_pos_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_ptscn_um_2 = target_um

        self.le_pos_um_2.setText("%.3f" % self.target_ptscn_um_2)

    def update_target_lscn_strt_um_1(self):
        target_um = float(self.le_lscn_start_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_strt_um_1 = target_um

        self.le_lscn_start_fs_1.setText("%.3f" % self.target_lscn_strt_fs_1)

    def update_target_lscn_strt_fs_1(self):
        target_fs = float(self.le_lscn_start_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_strt_um_1 = target_um

        self.le_lscn_start_um_1.setText("%.3f" % self.target_lscn_strt_um_1)

    def update_target_lscn_strt_um_2(self):
        target_um = float(self.le_lscn_start_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_strt_um_2 = target_um

        self.le_lscn_start_fs_2.setText("%.3f" % self.target_lscn_strt_fs_2)

    def update_target_lscn_strt_fs_2(self):
        target_fs = float(self.le_lscn_start_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_strt_um_2 = target_um

        self.le_lscn_start_um_2.setText("%.3f" % self.target_lscn_strt_um_2)

    def update_target_lscn_end_um_1(self):
        target_um = float(self.le_lscn_end_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_end_um_1 = target_um

        self.le_lscn_end_fs_1.setText("%.3f" % self.target_lscn_end_fs_1)

    def update_target_lscn_end_fs_1(self):
        target_fs = float(self.le_lscn_end_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_end_um_1 = target_um

        self.le_lscn_end_um_1.setText("%.3f" % self.target_lscn_end_um_1)

    def update_target_lscn_end_um_2(self):
        target_um = float(self.le_lscn_end_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_end_um_2 = target_um

        self.le_lscn_end_fs_2.setText("%.3f" % self.target_lscn_end_fs_2)

    def update_target_lscn_end_fs_2(self):
        target_fs = float(self.le_lscn_end_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_lscn_end_um_2 = target_um

        self.le_lscn_end_um_2.setText("%.3f" % self.target_lscn_end_um_2)

    def update_target_img_strt_um_1(self):
        target_um = float(self.le_img_start_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_strt_um_1 = target_um

        self.le_img_start_fs_1.setText("%.3f" % self.target_img_strt_fs_1)

    def update_target_img_strt_fs_1(self):
        target_fs = float(self.le_img_start_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_strt_um_1 = target_um

        self.le_img_start_um_1.setText("%.3f" % self.target_img_strt_um_1)

    def update_target_img_strt_um_2(self):
        target_um = float(self.le_img_start_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_strt_um_2 = target_um

        self.le_img_start_fs_2.setText("%.3f" % self.target_img_strt_fs_2)

    def update_target_img_strt_fs_2(self):
        target_fs = float(self.le_img_start_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_strt_um_2 = target_um

        self.le_img_start_um_2.setText("%.3f" % self.target_img_strt_um_2)

    def update_target_img_end_um_1(self):
        target_um = float(self.le_img_end_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_end_um_1 = target_um

        self.le_img_end_fs_1.setText("%.3f" % self.target_img_end_fs_1)

    def update_target_img_end_fs_1(self):
        target_fs = float(self.le_img_end_fs_1.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_1.T0_um
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_end_um_1 = target_um

        self.le_img_end_um_1.setText("%.3f" % self.target_img_end_um_1)

    def update_target_img_end_um_2(self):
        target_um = float(self.le_img_end_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_end_um_2 = target_um

        self.le_img_end_fs_2.setText("%.3f" % self.target_img_end_fs_2)

    def update_target_img_end_fs_2(self):
        target_fs = float(self.le_img_end_fs_2.text())
        target_um = T_fs_to_dist_um(target_fs) + self.stage_2.T0_um
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f"target position must be <= {upper_limit}")
            return
        self.target_img_end_um_2 = target_um

        self.le_img_end_um_2.setText("%.3f" % self.target_img_end_um_2)

    def update_stepsize_ptscn_um_1(self):
        step_size_um = float(self.le_step_size_um_1.text())
        self.step_size_ptscn_um_1 = step_size_um
        self.le_step_size_fs_1.setText("%.3f" % self.step_size_ptscn_fs_1)

    def update_stepsize_ptscn_fs_1(self):
        step_size_fs = float(self.le_step_size_fs_1.text())
        self.step_size_ptscn_fs_1 = step_size_fs
        self.le_step_size_um_1.setText("%.3f" % self.step_size_ptscn_um_1)

    def update_stepsize_ptscn_um_2(self):
        step_size_um = float(self.le_step_size_um_2.text())
        self.step_size_ptscn_um_2 = step_size_um
        self.le_step_size_fs_2.setText("%.3f" % self.step_size_ptscn_fs_2)

    def update_stepsize_ptscn_fs_2(self):
        step_size_fs = float(self.le_step_size_fs_2.text())
        self.step_size_ptscn_fs_2 = step_size_fs
        self.le_step_size_um_2.setText("%.3f" % self.step_size_ptscn_um_2)

    def update_stepsize_lscn_um(self):
        step_size_um = float(self.le_lscn_step_size_um.text())
        self.step_size_lscn_um = step_size_um
        self.le_lscn_step_size_fs.setText("%.3f" % self.step_size_lscn_fs)

    def update_stepsize_lscn_fs(self):
        step_size_fs = float(self.le_lscn_step_size_fs.text())
        self.step_size_lscn_fs = step_size_fs
        self.le_lscn_step_size_um.setText("%.3f" % self.step_size_lscn_um)

    def update_stepsize_img_um(self):
        step_size_um = float(self.le_img_step_size_um.text())
        self.step_size_img_um = step_size_um
        self.le_img_step_size_fs.setText("%.3f" % self.step_size_img_fs)

    def update_stepsize_img_fs(self):
        step_size_fs = float(self.le_img_step_size_fs.text())
        self.step_size_img_fs = step_size_fs
        self.le_img_step_size_um.setText("%.3f" % self.step_size_img_um)

    def move_to_pos_1(
        self,
        *args,
        target_um=None,
        connect_to_progress_fcts=None,
        connect_to_finish_fcts=None,
    ):
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
        self.update_motor_thread_1 = UpdateMotorThread(
            self.stage_1, self.motor_moving_1
        )
        self.connect_update_motor_1()

        if connect_to_progress_fcts is not None:
            assert isinstance(connect_to_progress_fcts, list)
            [
                self.update_motor_thread_1.signal.progress.connect(i)
                for i in connect_to_progress_fcts
            ]

        if connect_to_finish_fcts is not None:
            assert isinstance(connect_to_finish_fcts, list)
            [
                self.update_motor_thread_1.signal.finished.connect(i)
                for i in connect_to_finish_fcts
            ]

        thread = threading.Thread(target=self.update_motor_thread_1.run)
        self.motor_moving_1.set()
        thread.start()

    def move_to_pos_2(
        self,
        *args,
        target_um=None,
        connect_to_progress_fcts=None,
        connect_to_finish_fcts=None,
    ):
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
        self.update_motor_thread_2 = UpdateMotorThread(
            self.stage_2, self.motor_moving_2
        )
        self.connect_update_motor_2()

        if connect_to_progress_fcts is not None:
            assert isinstance(connect_to_progress_fcts, list)
            [
                self.update_motor_thread_2.signal.progress.connect(i)
                for i in connect_to_progress_fcts
            ]

        if connect_to_finish_fcts is not None:
            assert isinstance(connect_to_finish_fcts, list)
            [
                self.update_motor_thread_2.signal.finished.connect(i)
                for i in connect_to_finish_fcts
            ]

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
        self.update_motor_thread_1 = UpdateMotorThread(
            self.stage_1, self.motor_moving_1
        )
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
        self.update_motor_thread_2 = UpdateMotorThread(
            self.stage_2, self.motor_moving_2
        )
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
        self.le_pos_fs_1.setText("0")
        self.update_target_fs_1()

    def set_T0_2(self):
        pos_um = self.stage_2.pos_um  # read position from stage 1
        self.stage_2.T0_um = pos_um

        with open("T0_um_2.txt", "w") as file:
            file.write(str(pos_um))

        self.update_lcd_pos_2(pos_um)
        self.le_pos_fs_2.setText("0")
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

    def acquire_and_get_spectrum(self, *args, active_correct=active_correct_line_scan):
        # acquire
        try:
            self.active_stream.acquire(set_ppifg=False)
        except:
            raise_error(self.ErrorWindow, "FAILED TO ACQUIRE :(")
            return  # exit

        if self.active_stream.ppifg is None:
            raise_error(
                self.ErrorWindow, "ESTABLISH A PPIFG IN THE OSCILLOSCOPE TAB FIRST"
            )
            return  # exit

        x = self.active_stream.single_acquire_array

        if active_correct:
            x = x[np.argmax(x[: self.active_stream.ppifg]) :][
                self.active_stream.ppifg // 2 :
            ]
            N = len(x) // self.active_stream.ppifg
            x = x[: N * self.active_stream.ppifg]
            x.resize((N, self.active_stream.ppifg))

            # _________________________________________________________________
            # below I just shift correct by overlapping the maxima of all the
            # interferograms comment out this block if you just want to
            # straight up average
            # ind_ref = np.argmax(x[0])  # maximum of first interferogram
            # ind_diff = ind_ref - np.argmax(
            #     x, axis=1
            # )  # maximum of first - maximum of all the rest
            # for n, i in enumerate(x):
            #     x[n] = np.roll(i, ind_diff[n])
            # _________________________________________________________________

            x = np.mean(x, 0)
            ft = abs(rfft(x))

            # _________________________________________________________________
            # calculate the wavelength axis
            # _________________________________________________________________
            nu = np.fft.rfftfreq(len(x), d=1e-9) * len(x)
            nu += nu[-1] * (self.Nyquist_Window - 1)
            wl = np.where(nu > 0, sc.c * 1e6 / nu, np.nan)
            if self.Nyquist_Window % 2 == 0:
                ft = ft[::-1]  # negative frequency side

            # _________________________________________________________________
            # update the plot
            # _________________________________________________________________
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

        self.update_motor_thread_1 = UpdateMotorThread(
            self.stage_1, self.motor_moving_1
        )
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

        self.stage_1.move_by_um(
            -step_um
        )  # start moving the motor in relative mode, note the minus sign

        self.update_motor_thread_1 = UpdateMotorThread(
            self.stage_1, self.motor_moving_1
        )
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

        self.update_motor_thread_2 = UpdateMotorThread(
            self.stage_2, self.motor_moving_2
        )
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

        self.stage_2.move_by_um(
            -step_um
        )  # start moving the motor in relative mode, note the minus sign

        self.update_motor_thread_2 = UpdateMotorThread(
            self.stage_2, self.motor_moving_2
        )
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

    # _________________________________________________________________________
    # Line scan without trigger
    # _________________________________________________________________________
    def line_scan_notrigger(self, x1, y1, x2, y2, step_um):
        if self.img_running.is_set():  # imaging already running
            if not self.calling_from_image.is_set():
                self.stop_img.set()
                # the line scan will return with the finished signal which will
                # go into the imaging loop. At that point the stop_img flag
                # will be detected and will send the program to
                # img_stop_finished() where the finished signal will be
                # disconnected
                if self.lscn_running.is_set():  # scan already running
                    self.stop_lscn.set()  # stop the line scan
                    return
        elif self.lscn_running.is_set():  # scan already running
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
            if self.radbtn_lscn_trigon.isChecked():  # triggered line scan
                self.stage_1.trigger_on = True
                self.stage_2.trigger_on = True
                self.radbtn_trigon_1.setChecked(True)
                self.radbtn_trigon_2.setChecked(True)

                self._line_scan_withtrigger(self._x2, self._y2, self._step_um)

            else:  # point by point scan (no trigger)
                self._line_scan_notrigger_2(self._x2, self._y2, self._step_um)

    def _line_scan_notrigger_2(self, x2, y2, step_um):
        # acquire the first spectrum and record the position
        wl, ft = self.acquire_and_get_spectrum()

        if not active_correct_line_scan:
            np.save(databackup_path + "spectra/" + "0.npy", ft)
            self._N_loop = 1

        x1 = self.stage_1.pos_um
        y1 = self.stage_2.pos_um

        # calculate the step size in x and step size in y
        dx = x2 - x1
        dy = y2 - y1
        r = np.sqrt(dx**2 + dy**2)
        rx = dx / r  # rhat = (rx, ry), note that rx and / or ry can be negative
        ry = dy / r
        step_x = (
            step_um * rx
        )  # step_x, step_y can be negative -> we will only call step_right in the loop below!
        step_y = step_um * ry
        npts = np.floor(r / step_um)

        # round off: if the step is less than 10 nm then it's just stage error.
        # This caused me a lot of issues because it would tell the stage to
        # step, and in the next line connect the finished signal. but because
        # 10 nm ~ 0, the finished flag returned before it was connected
        if abs(step_x * 1e3) < 10:
            step_x = 0.0
        if abs(step_y * 1e3) < 10:
            step_y = 0.0

        # store variables
        self._step_x = step_x
        self._step_y = step_y
        self._step_um = step_um
        self._npts = int(npts)
        self._n = 0

        self._X = np.zeros(self._npts + 1)
        self._Y = np.zeros(self._npts + 1)
        self._FT = np.zeros((self._npts + 1, len(ft)))
        self._X[0] = x1
        self._Y[0] = y1
        self._FT[0] = ft
        self._WL = wl

        print(f"acquired point 0 of {self._npts}")
        self.curve_lscn.setData(wl, ft)

        # connect motor
        self.lscn_running.set()
        self.btn_lscn_start.setText("stop scan")
        self._lscn_step_one()

    def _check_if_ready_for_next(self):
        if self.motor_moving_2.is_set():
            self.update_motor_thread_2.signal.finished.connect(
                self._check_if_ready_for_next
            )
        else:
            self._lscn_step_two()

    def lscn_stop_finished(self):
        self.stop_lscn.clear()
        self.lscn_running.clear()
        self.btn_lscn_start.setText("start scan")
        self.signal.finished.emit(None)

    def _lscn_step_one(self):
        if self.stop_lscn.is_set():  # check for stop event
            self.lscn_stop_finished()
            return

        if self._n < self._npts:
            if abs(self._step_x) > 0:
                self.step_right_1(step_um=self._step_x)
            if abs(self._step_y) > 0:
                self.step_right_2(step_um=self._step_y)

            if abs(self._step_x) > 0:
                self.update_motor_thread_1.signal.finished.connect(
                    self._check_if_ready_for_next
                )
            else:
                self.update_motor_thread_2.signal.finished.connect(
                    self._check_if_ready_for_next
                )
        else:
            self.lscn_stop_finished()

    def _lscn_step_two(self):
        if self.stop_lscn.is_set():  # check for stop event
            self.lscn_stop_finished()
            return

        self._X[self._n + 1] = self.stage_1.pos_um
        self._Y[self._n + 1] = self.stage_2.pos_um

        if active_correct_line_scan:
            self._FT[self._n + 1] = self.acquire_and_get_spectrum()[1]
            self.curve_lscn.setData(self._WL, self._FT[self._n + 1])
        else:
            np.save(
                databackup_path + "spectra/" + f"{self._N_loop}.npy",
                self.acquire_and_get_spectrum()[1],
            )
            self._N_loop += 1

        print(f"acquired point {self._n + 1} of {self._npts}")

        self._n += 1
        self._lscn_step_one()

    # _________________________________________________________________________
    # Line scan with trigger
    # _________________________________________________________________________
    # this function will be called from _check_if_at_start instead of
    # _line_scan_notrigger_2
    def _line_scan_withtrigger(self, x2, y2, step_um):
        if self.calling_from_image.is_set():

            def func():
                self._line_scan_withtrigger(x2, y2, step_um)

            if self.motor_moving_1.is_set():
                self.update_motor_thread_1.signal.finished.connect(func)
            elif self.motor_moving_2.is_set():
                self.update_motor_thread_2.signal.finished.connect(func)

        x1 = self.stage_1.pos_um
        y1 = self.stage_2.pos_um

        # calculate the step size in x and step size in y
        dx = x2 - x1
        dy = y2 - y1
        r = np.sqrt(dx**2 + dy**2)
        rx = dx / r  # rhat = (rx, ry), note that rx and / or ry can be negative
        ry = dy / r
        step_x = (
            step_um * rx
        )  # step_x, step_y can be negative -> we will only call step_right in the loop below!
        step_y = step_um * ry
        npts = np.floor(r / step_um)

        # round off: if the step is less than 10 nm then it's just stage error.
        # This caused me a lot of issues because it would tell the stage to
        # step, and in the next line connect the finished signal. but because
        # 10 nm ~ 0, the finished flag returned before it was connected
        if abs(step_x * 1e3) < 10:
            step_x = 0.0
        if abs(step_y * 1e3) < 10:
            step_y = 0.0

        if np.all([step_x, step_y]):
            raise_error(
                self.ErrorWindow, "for triggered linescans, only one stage can move"
            )
            return

        # store variables
        self._step_x = step_x
        self._step_y = step_y
        self._step_um = step_um
        self._npts = int(npts)
        self._n = 0
        self._h = 0

        self._X = np.zeros(self._npts + 1)
        self._Y = np.zeros(self._npts + 1)

        self._X[0] = x1
        self._Y[0] = y1

        ppifg = self.active_stream.ppifg

        nu = np.fft.rfftfreq(ppifg, d=1e-9) * ppifg
        nu += nu[-1] * (self.Nyquist_Window - 1)
        wl = np.where(nu > 0, sc.c * 1e6 / nu, np.nan)
        self._FT = np.zeros((self._npts + 1, len(nu)))
        self._WL = wl

        # 1)
        # this sets the buffer size in bytes
        # the time stamp is 64 samples (128 bytes) long!

        # samples = self.active_stream.acquire_npts + 64
        samples = self.active_stream.acquire_npts + 32
        self.active_stream.apply_ppifg(
            target_NBYTES=samples * 2, prep_walk_correction=False
        )

        # 2)
        # this sets the sample size of each segment (not byte size!)
        # segmentsize = self.active_stream.acquire_npts
        segmentsize = self.active_stream.acquire_npts * 2
        dsa.setExtTrigger(self.active_stream.inifile_stream, 1)
        dsa.setSegmentSize(self.active_stream.inifile_stream, segmentsize)

        # 3)
        self._card_stream_progress_fcts = [self.DoAnalysis]
        self._card_stream_finished_fcts = [self.lscn_with_trigger_stop_finished]

        # 4)
        self.lscn_running.set()
        self.btn_lscn_start.setText("stop scan")

        # T = self.active_stream.acquire_npts * 1e-9
        # self._vel_mm_s = min([step_um * 1e-3, 1e-3 / T])
        self._vel_mm_s = step_um * 1e-3 * 8  # two pixels per second

        if abs(step_x) > 0:
            self.stage_1.step_um = step_x  # set trigger interval for stage 1
            self.step_left_1(
                step_um=step_x
            )  # back up step_x so first data point is taken at the start line
            self.update_motor_thread_1.signal.finished.connect(
                self._line_scan_withtrigger_2
            )
        elif abs(step_y) > 0:
            self.stage_2.step_um = step_y  # set trigger interval for stage 2
            self.step_left_2(
                step_um=step_y
            )  # back up step_y so first data point is taken at the start line
            self.update_motor_thread_2.signal.finished.connect(
                self._line_scan_withtrigger_2
            )
        else:
            raise_error(self.ErrorWindow, "step_x and step_y are zero!")

    def _line_scan_withtrigger_2(self):
        # 6)
        if abs(self._step_x) > 0:

            def func():
                self.stage_1.set_max_vel(
                    self._vel_mm_s
                )  # set scan velocity for stage 1
                self.move_to_pos_1(
                    target_um=self._x2 + self._step_x * extra_steps_linescan,
                    # connect_to_finish_fcts=[self.lscn_with_trigger_end_of_motion]
                )

        elif abs(self._step_y) > 0:

            def func():
                self.stage_2.set_max_vel(
                    self._vel_mm_s
                )  # set scan velocity for stage 2
                self.move_to_pos_2(
                    target_um=self._y2 + self._step_y * extra_steps_linescan,
                    # connect_to_finish_fcts=[self.lscn_with_trigger_end_of_motion]
                )

        else:
            raise_error(self.ErrorWindow, "both step_x and step_y are zero!")
            return

        # 7)
        self.active_stream.stream_data(
            connect_to_progress_fcts=self._card_stream_progress_fcts,
            connect_to_finish_fcts=self._card_stream_finished_fcts,
            live_update_plot=False,
            fcts_call_before_stream=[func],
        )

    def DoAnalysis(self, X):
        ppifg = self.active_stream.ppifg
        center = ppifg // 2
        n_skip = ppifg // 400

        # skip the first data point (due to the stage triggering where I don't
        # want it to)
        if self._n == 1:
            self._n += 1
            return  # skip

        # skip all odd loop counts
        if self._n % 2 == 0:
            x = np.frombuffer(X, "<h")
            pass
        else:
            self._n += 1
            return  # skip

        if self._h >= len(self._FT):
            self._n += 1
            self._h += 1
            return  # skip

        # ------------------ analysis -----------------------------------------
        x = x[np.argmax(x[:ppifg]) :][center:]
        N = len(x) // self.active_stream.ppifg
        x = x[: N * self.active_stream.ppifg]
        x.resize((N, self.active_stream.ppifg))

        x = np.mean(x, 0)
        ft = abs(rfft(x))

        if self.Nyquist_Window % 2 == 0:
            ft = ft[::-1]  # negative frequency side

        self._FT[self._h] = ft
        self.curve_lscn.setData(self._WL[::n_skip], ft[::n_skip])
        print(self._h, "out of", len(self._FT))
        # ---------------------------------------------------------------------

        if self._h == len(self._FT) - 1 or self.stop_lscn.is_set():
            print("attempted to terminate stream")
            self.active_stream.terminate()

        # self._n incremented always, self._h incremented only if checks pass
        self._n += 1
        self._h += 1

    def lscn_with_trigger_stop_finished(self):
        dsa.setExtTrigger(self.active_stream.inifile_stream, 0)
        dsa.setSegmentSize(self.active_stream.inifile_stream, -1)
        self.stop_lscn.clear()
        self.lscn_running.clear()
        self.btn_lscn_start.setText("start scan")

        if self.motor_moving_1.is_set():
            self.update_motor_thread_1.signal.finished.connect(
                self.lscn_with_trigger_end_of_motion
            )
            self.update_motor_thread_1.stop()
        elif self.motor_moving_2.is_set():
            self.update_motor_thread_2.signal.finished.connect(
                self.lscn_with_trigger_end_of_motion
            )
            self.update_motor_thread_2.stop()
        else:
            self.lscn_with_trigger_end_of_motion()

    def lscn_with_trigger_end_of_motion(self):
        self.stage_1.set_max_vel(1)
        self.stage_2.set_max_vel(1)
        self.signal.finished.emit(None)

    # _________________________________________________________________________
    # Image with or without trigger
    # _________________________________________________________________________

    def start_image_no_trigger(self):
        x1 = self.target_img_strt_um_1
        y1 = self.target_img_strt_um_2
        x2 = self.target_img_end_um_1
        y2 = self.target_img_end_um_2
        step_um = self.step_size_img_um

        self.image_no_trigger(x1, y1, x2, y2, step_um)

    def image_no_trigger(self, x1, y1, x2, y2, step_um):
        """
        :param x1: x1 coordinate (um)
        :param y1: y1 coordinate (um)
        :param x2: x2 coordinate (um)
        :param y2: y2 coordinate (um)
        :param step_um: step size (um)
        :return:

        (x1, y1) ________________ **
        |                         |
        |                         |
        |                         |
        |                         |
        ** __________________ (x2, y2)

        The idea here is to do repetitive line scans, with an update of the
        spectrum on the line scan tab, and an update of the image on the
        imaging tab
        """
        if self.img_running.is_set():  # imaging already running
            self.stop_img.set()
            # the line scan will return with the finished signal which will go
            # into the imaging loop. At that point the stop_img flag will be
            # detected and will send the program to img_stop_finished() where
            # the finished signal will be disconnected
            if self.lscn_running.is_set():  # scan already running
                self.stop_lscn.set()  # stop the line scan
                return
        elif self.lscn_running.is_set():  # scan already running
            self.stop_lscn.set()  # stop the line scan
            return
        elif self.motor_moving_1.is_set():  # motor 1 currently in use
            raise_error(self.ErrorWindow, "stop stage 1 first")
            return
        elif self.motor_moving_2.is_set():  # motor 2 curently in use
            raise_error(self.ErrorWindow, "stop stage 2 first")
            return

        dx = x2 - x1
        dy = y2 - y1

        if not trigger_stage_1_only:
            if dx >= dy:  # doing line scans along x, so dy / step_um line scans
                self._N_linescans = int(np.floor(dy / step_um))
                self._scan_img = 0
            else:  # doing line scans along y, so dx / step_um line scans
                self._N_linescans = int(np.floor(dx / step_um))
                self._scan_img = 1

        else:
            self._N_linescans = int(np.floor(dy / step_um))
            self._scan_img = 0

        if self._scan_img == 0:  # x scan
            self.line_scan_notrigger(x1, y1, x2, y1, step_um)
        else:  # y scan
            self.line_scan_notrigger(x1, y1, x1, y2, step_um)

        # store variables
        self._n_img = 0
        self._x1_img = x1
        self._y1_img = y1
        self._x2_img = x2
        self._y2_img = y2
        self._step_um_img = step_um

        # connect signal
        self.signal.finished.connect(self._initialize_IMG_array)
        self.img_running.set()
        self.calling_from_image.set()
        self.btn_img_start.setText("stop scan")

    def _initialize_IMG_array(self):
        if self._scan_img == 0:  # x scan
            self._IMG = np.zeros(
                (len(self._FT), self._N_linescans + 1, len(self._FT[0]))
            )
            self._XPOS = np.zeros((len(self._FT), self._N_linescans + 1))
            self._YPOS = np.zeros((len(self._FT), self._N_linescans + 1))

        else:  # y scan
            self._IMG = np.zeros(
                (self._N_linescans + 1, len(self._FT), len(self._FT[0]))
            )
            self._XPOS = np.zeros((self._N_linescans + 1, len(self._FT)))
            self._YPOS = np.zeros((self._N_linescans + 1, len(self._FT)))

        self.signal.finished.disconnect(
            self._initialize_IMG_array
        )  # disconnect this function after execution
        self.signal.finished.connect(
            self._img_next_step
        )  # and connect to the new function
        self._img_next_step()

    def _img_next_step(self):
        if self.stop_img.is_set():
            self.img_stop_finished()
            return

        # save data from the last scan
        if self._scan_img == 0:  # x scan
            self._IMG[:, self._n_img, :] = self._FT
            self._XPOS[:, self._n_img] = self._X
            self._YPOS[:, self._n_img] = self._Y
            print(
                f"________________ acquired image point {self._n_img} of {self._N_linescans} ________________"
            )

            # update image plot
            self.update_image(self._IMG[:, : self._n_img + 1])

        else:  # Y scan
            self._IMG[self._n_img, :, :] = self._FT
            self._XPOS[self._n_img, :] = self._X
            self._YPOS[self._n_img, :] = self._Y
            print(
                f"________________ acquired image point {self._n_img} of {self._N_linescans} ________________"
            )

            # update image plot
            self.update_image(self._IMG[: self._n_img + 1])

        # start the next scan (if there is one)
        if self._n_img < self._N_linescans:
            if self._scan_img == 0:  # x scan
                self.line_scan_notrigger(
                    self._x1_img,
                    self._y1_img + self._step_um_img,
                    self._x2_img,
                    self._y1_img + self._step_um_img,
                    self._step_um_img,
                )
                self._y1_img += self._step_um_img

            else:  # y scan
                self.line_scan_notrigger(
                    self._x1_img + self._step_um_img,
                    self._y1_img,
                    self._x1_img + self._step_um_img,
                    self._y2_img,
                    self._step_um_img,
                )
                self._x1_img += self._step_um_img

            self._n_img += 1
        else:
            self.img_stop_finished()

    def img_stop_finished(self):
        self.stop_img.clear()
        self.img_running.clear()
        self.calling_from_image.clear()
        self.btn_img_start.setText("start scan")
        self.signal.finished.disconnect(self._img_next_step)

    def update_image(self, arr):
        arr = simps(arr, axis=-1)
        self.plot_image.plotwidget.plot_image(arr)


# %% runnable classes
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


# %% run call
if __name__ == "__main__":
    app = qt.QApplication([])
    hey = GuiTwoCards()
    app.exec()
