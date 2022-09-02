import threading
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
COM1 = "COM3"
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

        self.error_window = ErrorWindow()

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

    def value_exceeds_limits(self, value_um):
        predicted_pos_um = value_um + self.pos_um  # target position
        min_limit_um = self.motor.get_stage_axis_info()[0] * 1e3
        max_limit_um = self.motor.get_stage_axis_info()[1] * 1e3
        buffer_um = self._safety_buffer_mm * 1e3

        if (predicted_pos_um < min_limit_um + buffer_um) or (
                predicted_pos_um > max_limit_um - buffer_um):
            raise_error(self.error_window,
                        "exceeding stage limits")
            return True
        else:
            return False


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

        self.lcd_ptscn_pos_um_1.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_um_2.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_fs_1.setSmallDecimalPoint(True)
        self.lcd_ptscn_pos_fs_2.setSmallDecimalPoint(True)

        self.le_nyq_window.setValidator(qtg.QIntValidator())
        self.le_frep.setValidator(qtg.QDoubleValidator())
        self.le_pos_um_1.setValidator(qtg.QDoubleValidator())
        self.le_pos_um_2.setValidator(qtg.QDoubleValidator())
        self.le_pos_fs_1.setValidator(qtg.QDoubleValidator())
        self.le_pos_fs_2.setValidator(qtg.QDoubleValidator())

        self.plot_ptscn = pw.PlotWindow(self.le_ptscn_xmin,
                                        self.le_ptscn_xmax,
                                        self.le_ptscn_ymin,
                                        self.le_ptscn_ymax,
                                        self.gv_ptscn)
        self.curve_ptscn = pw.create_curve()
        self.plot_ptscn.plotwidget.addItem(self.curve_ptscn)

        self.stage_1 = MotorInterface(apt.KDC101(COM1))
        self.stage_2 = MotorInterface(apt.KDC101(COM2))
        self.stage_1.T0_um = float(np.loadtxt("T0_um_1.txt"))
        self.stage_2.T0_um = float(np.loadtxt("T0_um_2.txt"))
        self.pos_um_1 = None
        self.pos_um_2 = None
        self.update_lcd_pos_1(self.stage_1.pos_um)
        self.update_lcd_pos_2(self.stage_2.pos_um)

        self.motor_moving_1 = threading.Event()
        self.motor_moving_2 = threading.Event()
        self.target_um_1 = None
        self.target_um_2 = None

        self.update_motor_thread_1 = None
        self.update_motor_thread_2 = None

        self.connect()

    @property
    def target_fs_1(self):
        dx = self.target_um_1 - self.stage_1.T0_um
        return dist_um_to_T_fs(dx)

    @target_fs_1.setter
    def target_fs_1(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_1.T0_um + dx
        self.target_um_1 = x

    @property
    def target_fs_2(self):
        dx = self.target_um_2 - self.stage_2.T0_um
        return dist_um_to_T_fs(dx)

    @target_fs_2.setter
    def target_fs_2(self, val):
        dx = T_fs_to_dist_um(val)
        x = self.stage_2.T0_um + dx
        self.target_um_2 = x

    def connect(self):
        self.radbtn_card1.clicked.connect(self.select_card_1)
        self.radbtn_card2.clicked.connect(self.select_card_2)
        self.radbtn_trigon_1.clicked.connect(self.update_trigon_1)
        self.radbtn_trigon_2.clicked.connect(self.update_trigon_2)
        self.btn_acquire_pt_scn.clicked.connect(self.acquire_and_get_spectrum)
        self.le_nyq_window.editingFinished.connect(self.setNyquistWindow)
        self.le_frep.editingFinished.connect(self.setFrep)
        self.le_pos_um_1.editingFinished.connect(self.update_target_um_1)
        self.le_pos_um_2.editingFinished.connect(self.update_target_um_2)
        self.le_pos_fs_1.editingFinished.connect(self.update_target_fs_1)
        self.le_pos_fs_2.editingFinished.connect(self.update_target_fs_2)
        self.btn_set_T0_1.clicked.connect(self.set_T0_1)
        self.btn_set_T0_2.clicked.connect(self.set_T0_2)

        self.btn_move_to_pos_1.clicked.connect(self.move_to_pos_1)
        self.btn_move_to_pos_2.clicked.connect(self.move_to_pos_2)
        self.btn_home_stage_1.clicked.connect(self.home_stage_1)
        self.btn_home_stage_2.clicked.connect(self.home_stage_2)

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

    def update_lcd_pos_2(self, pos_um):
        self.pos_um_2 = pos_um
        pos_fs = dist_um_to_T_fs(pos_um - self.stage_2.T0_um)
        self.lcd_ptscn_pos_um_2.display('%.3f' % pos_um)
        self.lcd_ptscn_pos_fs_2.display('%.3f' % pos_fs)

    def update_target_um_1(self):
        target_um = float(self.le_pos_um_1.text())
        upper_limit = self.stage_1.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_um_1 = target_um

        self.le_pos_fs_1.setText('%.3f' % self.target_fs_1)

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
        self.target_um_1 = target_um

        self.le_pos_um_1.setText('%.3f' % self.target_um_1)

    def update_target_um_2(self):
        target_um = float(self.le_pos_um_2.text())
        upper_limit = self.stage_2.motor.get_stage_axis_info()[1] * 1e3

        if target_um < 0:
            raise_error(self.ErrorWindow, "target position must be >= 0")
            return
        elif target_um > upper_limit:
            raise_error(self.ErrorWindow, f'target position must be <= {upper_limit}')
            return
        self.target_um_2 = target_um

        self.le_pos_fs_2.setText('%.3f' % self.target_fs_2)

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
        self.target_um_2 = target_um

        self.le_pos_um_2.setText('%.3f' % self.target_um_2)

    def move_to_pos_1(self, *args, target_um=None):
        if self.motor_moving_1.is_set():
            self.update_motor_thread_1: UpdateMotorThread
            self.update_motor_thread_1.stop()
            return

        if target_um is None:
            target_um = self.target_um_1

        pos_um = self.stage_1.pos_um  # retrieve current position from stage
        ll_mm, ul_mm = self.stage_1.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([pos_um < ll_um, pos_um > ul_um]):
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
            target_um = self.target_um_2

        pos_um = self.stage_2.pos_um  # retrieve current position from stage
        ll_mm, ul_mm = self.stage_2.motor.get_stage_axis_info()[:2]
        ll_um, ul_um = ll_mm * 1e3, ul_mm * 1e3
        if any([pos_um < ll_um, pos_um > ul_um]):
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

    def acquire_and_get_spectrum(self):
        # acquire
        try:
            self.active_stream.acquire()
        except:
            raise_error(self.ErrorWindow, "FAILED TO ACQUIRE :(")
            return  # exit

        if self.active_stream.ppifg is None:
            raise_error(self.ErrorWindow, "ESTABLISH A PPIFG IN THE OSCILLOSCOPE TAB FIRST")
            return  # exit

        x = self.active_stream.single_acquire_array
        x = x[self.active_stream.ppifg // 2:]  # assuming self-triggered : throw out first PPIFG // 2
        x = pf.adjust_data_and_reshape(x, self.stream1.ppifg)  # didn't acquire NPTS = integer x ppifg

        # ______________________________________________________________________________________________________________
        # below I just shift correct by overlapping the maxima of all the interferograms
        # ______________________________________________________________________________________________________________

        ind_ref = np.argmax(x[0])  # maximum of first interferogram
        ind_diff = ind_ref - np.argmax(x, axis=1)  # maximum of first - maximum of all the rest
        r, c = np.ogrid[:x.shape[0], :x.shape[1]]
        c_shift = c + (ind_diff - len(c.flatten()))[:, np.newaxis]
        x = x[r, c_shift]
        x = np.mean(x, 0)
        ft = fft(x).__abs__()[self.active_stream.ppifg // 2:]  # center -> end : pick the positive frequency side

        # ______________________________________________________________________________________________________________
        # calculate the wavelength axis
        # ______________________________________________________________________________________________________________
        center = self.active_stream.ppifg // 2
        Nyq_Freq = center * self.frep
        translation = (self.Nyquist_Window - 1) * Nyq_Freq
        nu = np.linspace(0, Nyq_Freq, center) + translation
        wl = np.where(nu > 0, sc.c * 1e6 / nu, np.nan)

        if self.Nyquist_Window % 2 == 0:
            ft = ft[::-1]

        # ______________________________________________________________________________________________________________
        # update the plot
        # ______________________________________________________________________________________________________________
        lims_wl = np.array([min(wl), max(wl)])
        lims_ft = np.array([0, max(ft)])
        self.plot_ptscn.format_to_xy_data(lims_wl, lims_ft)
        self.curve_ptscn.setData(wl, ft)


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
