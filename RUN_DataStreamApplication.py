"""
All the PyGage commands sends a command to the digitizer, the digitizer will send back a success or error
signal. If the command was a query it will either return the requested information, or an error signal.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import gc
import sys

sys.path.append("include")
from builtins import int
from configparser import ConfigParser  # needed to read ini files
import threading
import sys
import time
import itertools  # for infinite for loop
import PyGage3_64 as PyGage
import GageSupport as gs
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qt
import PlotWidgets as pw
import PyQt5.QtGui as qtg
import ProcessingFunctions as pf
import pyfftw
import DataStreamApplication as dsa
import Acquire

NUMBER_OF_CARDS = 2
assert ((NUMBER_OF_CARDS == 1) or (NUMBER_OF_CARDS == 2)), f"NUMBER_OF_CARDS must be 1 or 2 but got {NUMBER_OF_CARDS}"

if NUMBER_OF_CARDS == 1:
    from GuiDesigner import Ui_MainWindow
else:
    from GuiDesigner_TWO_CARDS import Ui_MainWindow

extTrigger = False
if extTrigger:
    ActiveSaveData = True
else:
    ActiveSaveData = False


def isnumeric(s):
    s: str
    if s.isnumeric():
        return True
    elif s.startswith('-') or s.startswith('+'):
        return s[1:].isnumeric()
    else:
        return False


class GuiWindow(qt.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.show()


class StreamWithGui(dsa.Stream):
    def __init__(self, gui,
                 index=1,
                 inifile_stream=dsa.inifile_default,
                 inifile_acquire='include/Acquire.ini',
                 shared_info=None):
        dsa.Stream.__init__(self, inifile_stream)
        gui: Ui_MainWindow
        shared_info: SharedInfo
        self.gui = gui
        self.shared_info = shared_info
        self.inifile_acquire = inifile_acquire

        """I want to be able to specify the instance, so the Gui class cannot be inherited"""

        # conditional gui attributes to inherit ----------------------------------------------------------------------
        assert ((index == 1) or (index == 2)), f"index needs to be 1 or 2 but got {index}"
        if index == 1:
            self.le_ifgplot_xmin = self.gui.le_ifgplot_xmin
            self.le_ifgplot_xmax = self.gui.le_ifgplot_xmax
            self.le_ifgplot_ymin = self.gui.le_ifgplot_ymin
            self.le_ifgplot_ymax = self.gui.le_ifgplot_ymax
            self.gv_ifgplot = self.gui.gv_ifgplot

            self.le_ppifg = self.gui.le_ppifg
            self.progressBar = self.gui.progressBar
            self.txtbws_stream_rate = self.gui.txtbws_stream_rate

            self.btn_start_stream = self.gui.btn_start_stream
            self.btn_stop_stream = self.gui.btn_stop_stream
            self.btn_single_acquire = self.gui.btn_single_acquire

            self.chkbx_save_data = self.gui.chkbx_save_data
            self.tableWidget = self.gui.tableWidget
            self.btn_plot = self.gui.btn_plot
            self.btn_apply_ppifg = self.gui.btn_apply_ppifg
            self.actionSave = self.gui.actionSave

        else:
            self.le_ifgplot_xmin = self.gui.le_ifgplot_xmin_2
            self.le_ifgplot_xmax = self.gui.le_ifgplot_xmax_2
            self.le_ifgplot_ymin = self.gui.le_ifgplot_ymin_2
            self.le_ifgplot_ymax = self.gui.le_ifgplot_ymax_2
            self.gv_ifgplot = self.gui.gv_ifgplot_2

            self.le_ppifg = self.gui.le_ppifg_2
            self.progressBar = self.gui.progressBar_2
            self.txtbws_stream_rate = self.gui.txtbws_stream_rate_2

            self.btn_start_stream = self.gui.btn_start_stream_2
            self.btn_stop_stream = self.gui.btn_stop_stream_2
            self.btn_single_acquire = self.gui.btn_single_acquire_2

            self.chkbx_save_data = self.gui.chkbx_save_data_2
            self.tableWidget = self.gui.tableWidget_2
            self.btn_plot = self.gui.btn_plot_2
            self.btn_apply_ppifg = self.gui.btn_apply_ppifg_2
            self.actionSave = self.gui.actionSave2

        # all the other gui attributes to inherit -----------------------------------
        self.le_npts_post_trigger = self.gui.le_npts_post_trigger
        self.le_buffer_size_MB = self.gui.le_buffer_size_MB
        self.le_npts_to_plot = self.gui.le_npts_to_plot
        self.btn_start_both_streams = self.gui.btn_start_stream_3
        self.btn_stop_both_streams = self.gui.btn_stop_stream_3

        # ---------------------------------------------------------------------------

        self.plotwindow = pw.PlotWindow(self.le_ifgplot_xmin, self.le_ifgplot_xmax,
                                        self.le_ifgplot_ymin, self.le_ifgplot_ymax,
                                        self.gv_ifgplot)
        self.curve = pw.create_curve()
        self.plotwindow.plotwidget.addItem(self.curve)
        self.general_connects()

        self.contPlotUpdate = None

        # these line edits can only be set to integer or double
        self.le_buffer_size_MB.setValidator(qtg.QDoubleValidator())
        self.le_ppifg.setValidator(qtg.QIntValidator())
        self.le_npts_to_plot.setValidator(qtg.QIntValidator())
        self.le_npts_post_trigger.setValidator(qtg.QIntValidator())

        self.progressBar.setValue(0)
        self.progressBar.setRange(0, 100)

        self._plot_npts_cap = 400
        self._nplot = 400
        self.le_npts_to_plot.setText(str(self._nplot))
        self.acquire_npts = 30000000
        self.le_npts_post_trigger.setText(str(self.acquire_npts))

        self.single_acquire_array = None
        self.ppifg = None

        self.adjusted_buffer_to_ppifg = False

        self.data_storage_size = None
        self.data_storage_buffer = None
        self.N_ifgs_to_fill_buffer = None

        self.loop_count = 0

        self.card_index = index

        # set the cell widget for the table to QLineEdits so that we can employ a QIntValidator
        config = ConfigParser()
        config.read(self.inifile_stream)
        level = config['Trigger1']['Level']
        plotchecklevel = config['PlotCheckLevel']['plotchecklevel']
        segmentsize = config['Acquisition']['SegmentSize']
        extclk = config['Acquisition']['extclk']
        ext_trigger = config['Trigger1']['source']
        if ext_trigger == -1:
            ext_trigger = 1
        else:
            ext_trigger = 0
        self.tableWidget.item(0, 0).setText(str(level))
        self.tableWidget.item(1, 0).setText(str(plotchecklevel))
        self.tableWidget.item(2, 0).setText(str(segmentsize))
        self.tableWidget.item(3, 0).setText(str(extclk))
        self.tableWidget.item(4, 0).setText(str(ext_trigger))
        self.saved_table_widget_item_text = 'hello world'

        self.wait_time = 100

        # don't forget to set ActiveSaveData to True before starting the real shocktube experiment
        self.save_data_loopcount = ActiveSaveData

        # if self.card_index is 1, set the data back up path to the data back up for card 1 folder
        if self.card_index == 1:
            self.databackup_path = 'DataBackup/card1/'

        # otherwise, if card_index is two, set it to the data backup folder for card 2
        elif self.card_index == 2:
            self.databackup_path = 'DataBackup/card2/'

    @property
    def plotchecklevel(self):
        config = ConfigParser()
        config.read(self.inifile_stream)
        plotchecklevel = config['PlotCheckLevel']['plotchecklevel']
        return float(plotchecklevel)

    def save_table_item(self, row, col):
        self.saved_table_widget_item_text = self.tableWidget.item(row, col).text()

    def slot_for_table_widget(self, row, col):
        if (row, col) == (0, 0):
            self.set_new_trigger_level(row, col)

        if (row, col) == (1, 0):
            self.set_new_plotchecklevel(row, col)

        if (row, col) == (2, 0):
            self.setSegmentSize(row, col)

        if (row, col) == (3, 0):
            self.setExtClk(row, col)

        if (row, col) == (4, 0):
            self.setTriggerSource(row, col)

    def set_new_plotchecklevel(self, row, col):
        if not self.tableWidget.item(row, col).text().isnumeric():
            dsa.raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        plotchecklevel = int(self.tableWidget.item(row, col).text())

        if plotchecklevel <= 0:
            dsa.raise_error(self.ErrorWindow, "trigger level needs to be >= 0 ")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        dsa.setNewPlotCheckLevel(self.inifile_stream, plotchecklevel)

    def set_new_trigger_level(self, row, col):
        if not self.tableWidget.item(row, col).text().isnumeric():
            dsa.raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        level_percent = int(self.tableWidget.item(row, col).text())

        if level_percent < 0:
            dsa.raise_error(self.ErrorWindow, "trigger level needs to be >= 0 ")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        dsa.setNewTriggerLevel(self.inifile_stream, level_percent)
        dsa.setNewTriggerLevel(self.inifile_acquire, level_percent)

    def setSegmentSize(self, row, col):
        if not isnumeric(self.tableWidget.item(row, col).text()):
            dsa.raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        segmentsize = int(self.tableWidget.item(row, col).text())

        if (segmentsize == 0) or (segmentsize < -1):
            dsa.raise_error(self.ErrorWindow, "segment size must be >= 1 or else be -1")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        dsa.setSegmentSize(self.inifile_stream, segmentsize)

    def setExtClk(self, row, col):
        if not isnumeric(self.tableWidget.item(row, col).text()):
            dsa.raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        extclk = int(self.tableWidget.item(row, col).text())

        if not any([extclk == 0, extclk == 1]):
            dsa.raise_error(self.ErrorWindow, "extclk must be 0 or 1")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        dsa.setExtClk(self.inifile_stream, extclk)
        dsa.setExtClk(self.inifile_acquire, extclk)

    def setTriggerSource(self, row, col):
        if not self.tableWidget.item(row, col).text().isnumeric():
            dsa.raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        ext_trigger = int(self.tableWidget.item(row, col).text())

        if not any([ext_trigger == 0, ext_trigger == 1]):
            dsa.raise_error(self.ErrorWindow, "external trigger must be 0 or 1")
            self.tableWidget.item(row, col).setText(str(self.saved_table_widget_item_text))
            return

        dsa.setExtTrigger(self.inifile_stream, ext_trigger)  # only change trigger source for streaming

    def plot(self):
        if self.single_acquire_array is None:
            dsa.raise_error(self.ErrorWindow, "no data acquired yet")
            return

        plt.figure()
        N = 30000000 // 2
        plt.plot(dsa.normalize(self.single_acquire_array[:N]))
        plt.show()

    def initialization_before_streaming(self):
        # initialization common amongst all sample programs:
        # ________________________________________________________________________________________
        if self.card_index == 2:
            if self.shared_info.handle1_initialized:
                self.handle = dsa.get_handle(1)
            else:
                self.handle = dsa.get_handle(self.card_index)

        else:
            self.handle = dsa.get_handle(1)
            self.shared_info.handle1_initialized = True

        if self.handle < 0:
            # get error string
            error_string = PyGage.GetErrorString(self.handle)
            print("Error: ", error_string)

            dsa.raise_error(self.ErrorWindow, error_string)
            return

        self.system_info = PyGage.GetSystemInfo(self.handle)
        if not isinstance(self.system_info, dict):  # if it's not a dict, it's an int indicating an error
            error_string = PyGage.GetErrorString(self.system_info)
            print("Error: ", error_string)
            PyGage.FreeSystem(self.handle)

            dsa.raise_error(self.ErrorWindow, error_string)
            return
        print("\nBoard Name: ", self.system_info["BoardName"])

        # get streaming parameters
        self.app = self.load_stm_configuration()

        # configure system
        status = self.configure_system()
        if status < 0:
            # get error string
            error_string = PyGage.GetErrorString(status)
            print("Error: ", error_string)
            PyGage.FreeSystem(self.handle)

            dsa.raise_error(self.ErrorWindow, error_string)
            return

        # initialize the stream
        status = self.initialize_stream()
        if status < 0:
            # error string is printed out in initialize_stream
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # This function sends configuration parameters that are in the driver to the CompuScope system associated
        # with the handle. The parameters are sent to the driver via SetAcquisitionConfig. SetChannelConfig and
        # SetTriggerConfig. The call to Commit sends these values to the hardware. If successful, the function
        # returns CS_SUCCESS (1). Otherwise, a negative integer representing a CompuScope error code is returned.
        status = PyGage.Commit(self.handle)
        if status < 0:
            # get error string
            error_string = PyGage.GetErrorString(status)
            print("Error: ", error_string)
            PyGage.FreeSystem(self.handle)

            dsa.raise_error(self.ErrorWindow, error_string)
            return
            # raise SystemExit

        # _____________________________________________________________________________________________
        # initialization done

        if self.streaming_buffer_size_to_set is not None:
            self.app["BufferSize"] = self.streaming_buffer_size_to_set

    def stream_data(self):
        """
        This overrides the stream_data method in Stream(). It adds the option to save data by running
        CardStreamSaveData instead of CardStream. It is otherwise the same as stream_data in Stream.
        """

        if self.card_stream_running.is_set():
            print("stream is already running, cannot start another")
            return

        self.stream_started_event.clear()
        self.ready_for_stream_event.clear()
        self.stream_aborted_event.clear()
        self.stream_error_event.clear()
        self.workBuffer_initiated_event.clear()

        self.g_cardTotalData = []
        self.g_segmentCounted = []

        # initialize
        self.initialization_before_streaming()

        # Returns the frequency of the timestamp counter in Hertz for the CompuScope system associated with the
        # handle. negative if an error occurred
        self.g_tickFrequency = PyGage.GetTimeStampFrequency(self.handle)

        if self.g_tickFrequency < 0:
            print("Error: ", PyGage.GetErrorString(self.g_tickFrequency))
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # after commit the sample size may change
        # get the big acquisition configuration dict
        acq_config = PyGage.GetAcquisitionConfig(self.handle)

        # get total amount of data we expect to receive in bytes, negative if an error occurred
        total_samples = PyGage.GetStreamTotalDataSizeInBytes(self.handle)

        if total_samples < 0 and total_samples != acq_config['SegmentSize']:
            print("Error: ", PyGage.GetErrorString(total_samples))
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # convert from bytes -> samples and print it to screen
        if total_samples != -1:
            total_samples = total_samples // self.system_info['SampleSize']
            print("total samples is: ", total_samples)

        if self.chkbx_save_data.isChecked():
            self.calc_data_storage_buffer_size()
            pass

        """We first initialize and start the card stream thread. The card stream thread initializes two buffers and 
        handles the data transfer to the buffers. Then we send the command to the Gage Card to start the data capture. 

        After that, we initialize and start the thread that tracks the progress of the data stream. This thread emits 
        signals that are used to plot data on the GUI. """

        # annoys me that I don't know if this is necessary but whatever
        del self.card_stream
        gc.collect()
        if self.chkbx_save_data.isChecked():
            self.card_stream = dsa.CardStreamSaveData(self.handle, self.card_index,
                                                      self.system_info['SampleSize'],
                                                      self.app,
                                                      self.stream_started_event,
                                                      self.ready_for_stream_event,
                                                      self.stream_aborted_event,
                                                      self.stream_error_event,
                                                      self.g_segmentCounted,
                                                      self.g_cardTotalData, self)
        else:
            self.card_stream = dsa.CardStream(self.handle, self.card_index,
                                              self.system_info['SampleSize'],
                                              self.app,
                                              self.stream_started_event,
                                              self.ready_for_stream_event,
                                              self.stream_aborted_event,
                                              self.stream_error_event,
                                              self.g_segmentCounted,
                                              self.g_cardTotalData, self)
        self.connect_card_stream_update()
        self._a1 = self.card_stream._a1
        self._a2 = self.card_stream._a2
        thread_cardstream = threading.Thread(target=self.card_stream.run)
        thread_cardstream.start()  # this starts the card stream while loop

        # the card_stream function should have set the self.ready_for_stream_event to true, if it is not true
        # then an error occurred
        set = self.ready_for_stream_event.wait(5)
        if not set:
            print("\nThread initialization error on card ", self.card_index)
            self.stream_aborted_event.set()
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # won't work anymore now that I've thrown the update onto a separate thread
        # (a thread that is not main)
        # print("\nStarting streaming. Press CTRL-C to abort\n\n")

        # if set passed, then CardStream.run finished the buffer preparation, and we can start the capture!
        status = PyGage.StartCapture(self.handle)
        if status < 0:
            # get error string
            print("Error: ", PyGage.GetErrorString(status))
            PyGage.FreeSystem(self.handle)
            raise SystemExit  # ??

        # ______________________________________________________________________________________________________________
        # The following the function calls cause the GUI to freeze in the case that the stream is waiting
        # on data to come in:
        #   1. self.stream_started_event.set() which is waited on by CardStream.run to start the data transfer loop
        #   2. self.connect_tracking_stream_update() not sure why this is, but it does and as a result all the other
        #      thread_trackstream stuff that depends on it can't go either
        #
        # Just to note, it can still get by the self.stream_started_event.set() call and go on to call the next
        # function, such as telling the motor to move or something. I've verified that even if the GUI freezes,
        # it does enter the while loop of CardStream.run
        # ______________________________________________________________________________________________________________

        # CardStream.run waits for this flag before running the loop that transfers data to RAM
        self.stream_started_event.set()

        # print("I can still get by setting this flag and call the next function though ...")

        # annoys me that I don't know if this is necessary but whatever
        del self.trackingstream
        gc.collect()
        self.trackingstream = dsa.TrackStreamProgress(self, thread_cardstream)
        self.connect_tracking_stream_update()
        thread_trackstream = threading.Thread(target=self.trackingstream.run)
        thread_trackstream.start()  # this loop updates the amount of data streamed to the Gui, and checks for errors
        # or stream abort

    def calc_data_storage_buffer_size(self):
        if self.data_storage_size is None:
            self.update_storage_buffer_size()

        # I'm expecting this function to be called by stream_data after self.app has been initialized
        buffersize = self.app['BufferSize']
        N = int(np.floor(self.data_storage_size / buffersize))
        self.N_ifgs_to_fill_buffer = N

        self.data_storage_size = N * buffersize
        self.le_buffer_size_MB.setText(str(self.data_storage_size / 1e6))

    def update_storage_buffer_size(self):
        num = int(float(self.le_buffer_size_MB.text()) * 1e6)

        if self.streaming_buffer_size_to_set is not None:
            if num < self.streaming_buffer_size_to_set:
                dsa.raise_error(self.ErrorWindow, "needs to at least be the streaming buffer size. "
                                                  "I am setting this instead to " +
                                str(self.streaming_buffer_size_to_set / 1e6))
                num = self.streaming_buffer_size_to_set
                self.le_buffer_size_MB.setText(str(num / 1e6))

        elif self.app is not None:
            if num < self.app["BufferSize"]:
                dsa.raise_error(self.ErrorWindow, "needs to at least be the streaming buffer size. "
                                                  "I am setting this instead to " +
                                str(self.app["BufferSize"] / 1e6))
                num = self.app["BufferSize"]
                self.le_buffer_size_MB.setText(str(num / 1e6))

        # have default be 13 MB
        else:
            if num < 0:
                dsa.raise_error(self.ErrorWindow, "can't be negative")
                num = int(13e6)
                self.le_buffer_size_MB.setText(str(num / 1e6))

        self.data_storage_size = num

    def update_npts_toplot(self):
        num = int(self.le_npts_to_plot.text())
        if not (1 <= num <= 400):
            dsa.raise_error(self.ErrorWindow, "must be between 1 and 400")
            self.le_npts_to_plot.setText(str(self._nplot))
            return

        self._nplot = num

    def closeEvent(self, *args):
        if not self.card_stream_running.is_set():
            # nothing's happened yet
            return
        else:
            # set the terminate flag so that the thread gets terminated
            # and the handle is released
            self.terminate()

    def connect_tracking_stream_update(self):
        # this needs to be called every time the card_stream is initialized
        # so it's easier to place it in a separate function
        # the original one prints the update out to terminal / console

        # super().connect_tracking_stream_update_signals()
        self.trackingstream.signal.progress.connect(self.displaymsg)
        self.trackingstream.signal.finished.connect(self.displaymsg)

        if not self.chkbx_save_data.isChecked():
            self.workBuffer_initiated_event.wait()

            self.contPlotUpdate = dsa.UpdateDisplay(self, self.wait_time)
            self.contPlotUpdate.start()

            self.contPlotUpdate.signal.progress.connect(self.updateDisplay)

    def connect_card_stream_update(self):
        if self.chkbx_save_data:
            self.card_stream.signal.finished.connect(self.update_progress_bar)
            self.card_stream.signal.progress.connect(self.update_progress_bar)

    def update_progress_bar(self):
        if self.chkbx_save_data.isChecked():
            num = int(np.round(self.loop_count * 100 / self.N_ifgs_to_fill_buffer))
            self.progressBar.setValue(num)

    def displaymsg(self, s):
        self.txtbws_stream_rate.setText(s)

    def general_connects(self):
        self.btn_start_stream.clicked.connect(self.stream_data)
        self.btn_stop_stream.clicked.connect(self.terminate)

        self.le_npts_post_trigger.editingFinished.connect(self.update_acquire_post_trigger_npts_from_le)
        self.btn_single_acquire.clicked.connect(self.acquire)

        self.btn_apply_ppifg.clicked.connect(self.apply_ppifg)

        self.le_ppifg.editingFinished.connect(self.update_ppifg_from_le)

        self.le_npts_to_plot.editingFinished.connect(self.update_npts_toplot)

        self.le_buffer_size_MB.editingFinished.connect(self.update_storage_buffer_size)

        self.actionSave.triggered.connect(self.save)

        self.btn_start_both_streams.clicked.connect(self.stream_data)
        self.btn_stop_both_streams.clicked.connect(self.terminate)

        # table widget connections
        self.tableWidget.cellClicked.connect(self.save_table_item)
        self.tableWidget.cellChanged.connect(self.slot_for_table_widget)

        # plotting for plotchecklevel
        self.btn_plot.clicked.connect(self.plot)

    def updateDisplay(self, X):
        center_ind_chnged_by_loop = False

        x, y = X
        # y = np.frombuffer(y, dtype='h') <- already implemented this in the UpdateDisplay class
        n_plot = self._nplot
        if not self.adjusted_buffer_to_ppifg:
            y = self.adc_to_volts(y)
            self.curve.setData(x=x[:n_plot], y=y[:n_plot])
        else:
            # y buffer converted from ADC values to volts
            y = self.adc_to_volts(y)

            if not self.gui.rbtn_dont_correct.isChecked():
                # if this instance runs card 2 and the walking check is referenced for card 1
                if np.all([self.gui.rbtn_walkon_1.isChecked(), self.card_index == 2,
                           self.shared_info.center_ind is not None]):
                    self.center_ind = self.shared_info.center_ind

                # if this instance runs card 1 and the walking check is referenced for card 2
                elif np.all([self.gui.rbtn_walkon_2.isChecked(), self.card_index == 1,
                             self.shared_info.center_ind is not None]):
                    self.center_ind = self.shared_info.center_ind

                else:
                    center_ind_chnged_by_loop = True

                    # section of y that will be plotted to screen
                    section = y[self.center_ind - n_plot // 2:self.center_ind + n_plot // 2]

                    # the indices of values in section that are above the level threshold indicating the presence of an
                    # interferogram
                    ind = (abs(section - np.mean(section)) > self._level - np.mean(section)).nonzero()[0]

                    # if the number of indices is less than half the original value (set when we know there was an
                    # interferogram in there)
                    if 0 < len(ind) < self._N_ind * 0.25:
                        avg = np.mean(ind)

                        # if the interferogram is walking out right of the screen, the correction should be to move it left (
                        # the plot window would move right)
                        if avg > n_plot // 2:
                            correction = n_plot

                        # otherwise it is walking out to the left of the screen, and the correction should be to move it
                        # right (the plot window would move left)
                        else:
                            correction = -n_plot

                        # an easy way to implement this correction is to move to the next interferogram in the buffer
                        self.center_ind += self.ppifg + correction

                        # when we reach the end of the buffer, wrap around
                        if self.center_ind > len(y):
                            self.center_ind -= len(y)

                    # same thing as above, except if the number of indices is already 0, use the previous set of indices
                    elif len(ind) == 0 and self._ind_old is not None:
                        avg = np.mean(self._ind_old)
                        if avg > n_plot // 2:
                            correction = n_plot
                        else:
                            correction = -n_plot

                        self.center_ind += self.ppifg + correction
                        if self.center_ind > len(y):
                            self.center_ind -= len(y)

                    # save the current set of indices in case this set passed without correction, but the next set
                    # has no elements
                    self._ind_old = ind

                # if not walking independently and we got this far, then this card stream instance must
                # be referenced for center_ind by the other card_stream instance
                if (not self.gui.rbtn_walk_independently.isChecked()) and center_ind_chnged_by_loop:
                    self.shared_info.center_ind = self.center_ind
                    # print('setting shared center_ind')

            self.curve.setData(x=x[self.center_ind - n_plot // 2:self.center_ind + n_plot // 2],
                               y=y[self.center_ind - n_plot // 2:self.center_ind + n_plot // 2])

    def change_stream_buffer_size(self, num, mB=False):
        if mB:
            self.streaming_buffer_size_to_set = int(num * 1e6)
        else:
            self.streaming_buffer_size_to_set = int(num)

    def update_acquire_post_trigger_npts_from_le(self):
        num = int(self.le_npts_post_trigger.text())
        if num < 20000:
            dsa.raise_error(self.ErrorWindow, "npts has to at least be 20,000")
            return
        self.acquire_npts = num

    def update_ppifg_from_le(self):
        print("I'm updating ppifg")
        num = int(self.le_ppifg.text())
        if num < 1000:
            dsa.raise_error(self.ErrorWindow, "surely ppifg is at least 1000")
            self.le_ppifg.setText(str(1000))
            num = 1000
        self.ppifg = num

        # this slows it down so much it basically freezes
        # if self.single_acquire_array is not None:
        #     self.apply_ppifg()

    def acquire(self):
        if self.card_stream_running.is_set():
            dsa.raise_error(self.ErrorWindow, "stop card stream first")
            return

        if self.card_index == 1:
            self.single_acquire_array = Acquire.acquire(self.acquire_npts,
                                                        inifile=self.inifile_acquire)
        else:
            # call initialize twice, to get the second card in the registry
            handle1 = Acquire.initialize()
            handle2 = Acquire.initialize()
            PyGage.FreeSystem(handle1)

            # handle2 will be freed by Acquire.acquire
            self.single_acquire_array = Acquire.acquire(self.acquire_npts, handle=handle2,
                                                        inifile=self.inifile_acquire)

        gc.collect()

        npts_int, npts_float, level = pf.find_npts(self.single_acquire_array, level_percent=self.plotchecklevel)
        self._level = level

        self.le_ppifg.setText(str(npts_float))
        self.ppifg = npts_int

    def apply_ppifg(self):
        if self.ppifg is None:
            dsa.raise_error(self.ErrorWindow, "no ppifg yet")
            return
        buffer_size_bytes = 2 * self.ppifg * 2

        # have the buffer be at least 13 MB
        if buffer_size_bytes < 13e6:
            N = int(np.ceil(13e6 / buffer_size_bytes))
            buffer_size_bytes *= N

        self.change_stream_buffer_size(buffer_size_bytes, mB=False)

        print("applied points per interferogram, {N} interferograms per buffer".format(
            N=buffer_size_bytes / (2 * self.ppifg * 2)))

        self.adjusted_buffer_to_ppifg = True

        self.le_ppifg.setText(str(self.ppifg))
        self.center_ind = self.ppifg

        section = self.single_acquire_array[self.ppifg - self._nplot // 2:self.ppifg + self._nplot // 2]
        ind = (abs(section - np.mean(section)) > self._level - np.mean(section)).nonzero()[0]
        self._ind_old = None
        self._N_ind = len(ind)

    def save(self):
        if self.data_storage_buffer is None:
            dsa.raise_error(self.ErrorWindow, "no data saved to storage buffer yet")
            return

        filename, _ = qt.QFileDialog.getSaveFileName(caption=f"Save Data for Card {self.card_index}")
        if filename == '':
            return

        N_ifgs = self.data_storage_buffer.size // (self.ppifg * 2)
        filename += "_{Nifgs}x{ppifg}".format(ppifg=int(self.ppifg), Nifgs=int(N_ifgs))
        filename += ".bin"

        self.data_storage_buffer.tofile(filename)

    def terminate(self):
        super().terminate()
        if self.card_index == 1:
            self.shared_info.handle1_initialized = False


class GuiTwoCards(qt.QMainWindow, Ui_MainWindow):
    def __init__(self):
        qt.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.shared_info = SharedInfo()

        # if you want to use the Two Cards Gui, but only running one of the cards, comment out the appropriate
        # stream1 or stream2 lines below
        if extTrigger:
            self.stream1 = StreamWithGui(self, index=1, inifile_stream='include/Stream2Analysis_exttrigger.ini',
                                         inifile_acquire='include/Acquire_CARD1.ini',
                                         shared_info=self.shared_info)
            self.stream2 = StreamWithGui(self, index=2, inifile_stream='include/Stream2Analysis_exttrigger.ini',
                                         inifile_acquire='include/Acquire_CARD2.ini',
                                         shared_info=self.shared_info)
        else:
            self.stream1 = StreamWithGui(self, index=1, inifile_stream='include/Stream2Analysis_CARD1.ini',
                                         inifile_acquire='include/Acquire_CARD1.ini',
                                         shared_info=self.shared_info)
            self.stream2 = StreamWithGui(self, index=2, inifile_stream='include/Stream2Analysis_CARD2.ini',
                                         inifile_acquire='include/Acquire_CARD2.ini',
                                         shared_info=self.shared_info)

        self.show()


class SharedInfo:
    __slots__ = ["center_ind", "handle1_initialized"]

    def __init__(self):
        self.center_ind = None
        self.handle1_initialized = False

# if __name__ == '__main__':
#     app = qt.QApplication([])
#     if NUMBER_OF_CARDS == 1:
#         hey = dsa.Gui()
#     else:
#         hey = GuiTwoCards()
#     app.exec()
