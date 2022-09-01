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
        self.motor.position_mm = value_um * 1e-3

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
        predicted_pos_um = value_um + self.pos_um
        max_limit_um = self.motor.get_stage_axis_info()[0] * 1e3
        min_limit_um = self.motor.get_stage_axis_info()[1] * 1e3
        buffer_um = self._safety_buffer_mm * 1e3

        if (predicted_pos_um < min_limit_um + buffer_um) or (
                predicted_pos_um > max_limit_um - buffer_um):
            raise_error(self.error_window,
                        "too close to stage limits (within 1um)")
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

        self.lcd_cnt_update_current_pos_um_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_cnt_update_current_pos_um_2.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_cnt_update_current_pos_fs_1.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd_cnt_update_current_pos_fs_2.setSegmentStyle(qt.QLCDNumber.Flat)

        self.le_nyq_window.setValidator(qtg.QIntValidator())
        self.le_frep.setValidator(qtg.QDoubleValidator())

        self.plot_ptscn = pw.PlotWindow(self.le_ptscn_xmin,
                                        self.le_ptscn_xmax,
                                        self.le_ptscn_ymin,
                                        self.le_ptscn_ymax,
                                        self.gv_ptscn)
        self.curve_ptscn = pw.create_curve()
        self.plot_ptscn.plotwidget.addItem(self.curve_ptscn)
        self.connect()

    def connect(self):
        self.radbtn_card1.clicked.connect(self.select_card_index)
        self.radbtn_card2.clicked.connect(self.select_card_index)
        self.btn_acquire_pt_scn.clicked.connect(self.acquire_and_get_spectrum)
        self.le_nyq_window.editingFinished.connect(self.setNyquistWindow)
        self.le_frep.editingFinished.connect(self.setFrep)

    def select_card_index(self):
        if self.radbtn_card1.isChecked():
            self._card_index = 1
            self.active_stream = self.stream1
            print("selecting card 1")
        elif self.radbtn_card2.isChecked():
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


if __name__ == '__main__':
    app = qt.QApplication([])
    hey = GuiTwoCards()
    app.exec()
