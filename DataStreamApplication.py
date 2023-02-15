"""
All the PyGage commands sends a command to the digitizer, the digitizer will
send back a success or error signal. If the command was a query it will either
return the requested information, or an error signal.
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
from GuiDesigner import Ui_MainWindow
import PlotWidgets as pw
import PyQt5.QtGui as qtg
import ProcessingFunctions as pf
import pyfftw
from datetime import datetime

from GageConstants import (
    CS_CURRENT_CONFIGURATION,
    CS_ACQUISITION_CONFIGURATION,
    CS_STREAM_TOTALDATA_SIZE_BYTES,
    CS_DATAPACKING_MODE,
    CS_MASKED_MODE,
    CS_GET_DATAFORMAT_INFO,
    CS_BBOPTIONS_STREAM,
    CS_MODE_USER1,
    CS_MODE_USER2,
    CS_EXTENDED_BOARD_OPTIONS,
    STM_TRANSFER_ERROR_FIFOFULL,
    CS_SEGMENTTAIL_SIZE_BYTES,
    CS_TIMESTAMP_TICKFREQUENCY,
)
from GageErrors import (
    CS_MISC_ERROR,
    CS_INVALID_PARAMS_ID,
    CS_STM_TRANSFER_TIMEOUT,
    CS_STM_COMPLETED,
)
import array
from Error_Window import Ui_Form
import Acquire

# default parameters
TRANSFER_TIMEOUT = -1  # milliseconds
STREAM_BUFFERSIZE = 0x200000  # 2097152
MAX_SEGMENT_COUNT = 25000
inifile_default = "include/Stream2Analysis.ini"
inifile_acquire_default = "include/Acquire.ini"
plotchecklevel = 25
show_walking = True


# class used to hold streaming information, attributes listed in __slots__
# below are assigned later in the card_stream function. The purpose of
# __slots__ is that it does not allow the user to add new attributes not listed
# in __slots__ (a kind of neat implementation I hadn't known about).
class StreamInfo:
    __slots__ = [
        "WorkBuffer",
        "TimeStamp",
        "BufferSize",
        "SegmentSize",
        "TailSize",
        "LeftOverSize",
        "BytesToEndSegment",
        "BytesToEndTail",
        "DeltaTime",
        "LastTimeStamp",
        "Segment",
        "SegmentCountDown",
        "SplitTail",
    ]


class ErrorWindow(qt.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def displaytxt(self, s):
        self.textBrowser.setText(s)


def raise_error(error_window, text):
    error_window: ErrorWindow
    error_window.displaytxt(text)
    error_window.show()


def setNewTriggerLevel(inifile=inifile_default, level_percent=2):
    config = ConfigParser()
    config.read(inifile)
    config["Trigger1"]["Level"] = str(level_percent)
    with open(inifile, "w") as configfile:
        config.write(configfile)
    print(f"overwrote trigger level in {inifile} to {level_percent}")


def setNewPlotCheckLevel(inifile=inifile_default, level_percent=2):
    config = ConfigParser()
    config.read(inifile)
    config["PlotCheckLevel"]["plotchecklevel"] = str(level_percent)
    with open(inifile, "w") as configfile:
        config.write(configfile)
    print(f"overwrote plotchecklevel in {inifile} to {level_percent}")


def setSegmentSize(inifile=inifile_default, segmentsize=30000000):
    config = ConfigParser()
    config.read(inifile)
    config["Acquisition"]["segmentsize"] = str(segmentsize)
    config["Acquisition"]["depth"] = str(segmentsize)

    if segmentsize != -1:
        config["Acquisition"]["segmentcount"] = str(-1)
    else:
        config["Acquisition"]["segmentcount"] = str(1)

    with open(inifile, "w") as configfile:
        config.write(configfile)
    print(f"overwrote segment size in {inifile} to {segmentsize}")


def setExtClk(inifile=inifile_default, extclk=1):
    assert (
        extclk == 1 or extclk == 0
    ), f"external clock needs to be 0 or 1 but got {extclk}"

    config = ConfigParser()
    config.read(inifile)
    config["Acquisition"]["extclk"] = str(extclk)

    with open(inifile, "w") as configfile:
        config.write(configfile)
    print(f"overwrote external clock in {inifile} to {extclk}")


def setExtTrigger(inifile=inifile_default, exttrigger=0):  # self-trigger by default
    assert (
        exttrigger == 1 or exttrigger == 0
    ), f"external trigger needs to be 0 or 1 but got {exttrigger}"

    config = ConfigParser()
    config.read(inifile)
    if exttrigger:
        source = -1
        config["Trigger1"]["source"] = str(-1)  # external trigger is True
    else:
        source = 1
        config["Trigger1"]["source"] = str(
            1
        )  # external trigger is False ->  self-trigger (channel 1)

    with open(inifile, "w") as configfile:
        config.write(configfile)
    print(f"overwrote Trigger1 source in {inifile} to {source}")


# finding a gauge card and returning the handle
# the handle is just an integer used to identify the card,
# and sort of "is" the gage card
def get_handle(index=1):
    assert (index == 1) or (index == 2), f"index needs to be 1 or 2 but got {index}"

    if index == 1:
        status = PyGage.Initialize()
        if status < 0:
            return status
        else:
            handle = PyGage.GetSystem(0, 0, 0, 0)
            return handle

    else:
        status = PyGage.Initialize()
        if status < 0:
            return status
        else:
            handle1 = PyGage.GetSystem(0, 0, 0, 0)
            handle2 = PyGage.GetSystem(0, 0, 0, 0)
            PyGage.FreeSystem(handle1)
            return handle2


def convert_adc_to_volts(adc, sampleoffset, sampleres, scale_factor, offset):
    return (((sampleoffset - adc) / sampleres) * scale_factor) + offset


def normalize(vec):
    return vec / np.max(abs(vec))


class Stream:
    """
    This class was made by modifying the stream function in Gage's example
    code. It is inherited by the GUI class which incorporates Stream into a GUI
    application. In addition to inheriting this class, GUI also inherits
    QMainWindow and UI_MainWindow.
    """

    def __init__(
        self,
        inifile=inifile_default,
        inifile_acquire=inifile_acquire_default,
        *args,
        **kwargs,
    ):
        self.g_tickFrequency = 0
        self.g_cardTotalData = []
        self.g_segmentCounted = []
        self.systemTotalData = 0
        self.inifile_stream = inifile
        self.inifile_acquire = inifile_acquire

        self.totalElapsedTime = 0
        self.totalBytes = 0

        # we will create a thread for data streaming, and the following events:
        self.stream_started_event = threading.Event()
        self.ready_for_stream_event = threading.Event()
        self.stream_aborted_event = threading.Event()
        self.stream_error_event = threading.Event()
        self.workBuffer_initiated_event = threading.Event()

        self.handle = None
        self.system_info = None
        self.app = None
        self.copyOfWorkBuffer1 = None
        self.copyOfWorkBuffer2 = None
        self.plotCopyOfWorkBuffer = None

        self.card_stream = None
        self._a1 = None
        self._a2 = None
        self._terminate = False
        self.trackingstream = None
        # self.card_stream_running = False
        self.card_stream_running = threading.Event()
        self.card_stream_stopped = threading.Event()
        self.card_stream_stopped.set()  # initially true (stream not running)

        self.ErrorWindow = ErrorWindow()

        self.streaming_buffer_size_to_set = None
        self.trigger_hold_off = None

        self._level = None
        self.center_ind = None
        self._N_ind = None
        self._ind_old = None

        self.card_index = 1

    def terminate(self):
        if not self.card_stream_running.is_set():
            return
        self._terminate = True

    def connect_tracking_stream_update(self):
        self.trackingstream: TrackStreamProgress
        self.trackingstream.signal.progress.connect(
            self.print_transfer_update_to_screen
        )
        self.trackingstream.signal.finished.connect(self.print_transfer_ended_to_screen)

    def connect_card_stream_update(self):
        pass

    def dataBufferUpdate(self, arr):
        # if you have a good idea for how to analyze data extremely fast you
        # can go for it. otherwise it's not going to do anything no matter how
        # hard you try it will never be fast enough
        pass

    def adc_to_volts(self, arr):
        return self._a1 + self._a2 * arr

    def print_transfer_update_to_screen(self, s):
        sys.stdout.write(s)
        sys.stdout.flush()

    def print_transfer_ended_to_screen(self, msg):
        print("\n" + msg + "\n")

    def initialization_before_streaming(self):
        # initialization common amongst all sample programs:
        # _____________________________________________________________________
        self.handle = get_handle()
        if self.handle < 0:
            # get error string
            error_string = PyGage.GetErrorString(self.handle)
            print("Error: ", error_string)

            raise_error(self.ErrorWindow, error_string)
            return

        self.system_info = PyGage.GetSystemInfo(self.handle)
        if not isinstance(
            self.system_info, dict
        ):  # if it's not a dict, it's an int indicating an error
            error_string = PyGage.GetErrorString(self.system_info)
            print("Error: ", error_string)
            PyGage.FreeSystem(self.handle)

            raise_error(self.ErrorWindow, error_string)
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

            raise_error(self.ErrorWindow, error_string)
            return

        # initialize the stream
        status = self.initialize_stream()
        if status < 0:
            # error string is printed out in initialize_stream
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # This function sends configuration parameters that are in the driver
        # to the CompuScope system associated with the handle. The parameters
        # are sent to the driver via SetAcquisitionConfig. SetChannelConfig and
        # SetTriggerConfig. The call to Commit sends these values to the
        # hardware. If successful, the function returns CS_SUCCESS (1).
        # Otherwise, a negative integer representing a CompuScope error code is
        # returned.
        status = PyGage.Commit(self.handle)
        if status < 0:
            # get error string
            error_string = PyGage.GetErrorString(status)
            print("Error: ", error_string)
            PyGage.FreeSystem(self.handle)

            raise_error(self.ErrorWindow, error_string)
            return
            # raise SystemExit

        # _____________________________________________________________________
        # initialization done

        if self.streaming_buffer_size_to_set is not None:
            self.app["BufferSize"] = self.streaming_buffer_size_to_set

    def stream_data(self):
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

        # Returns the frequency of the timestamp counter in Hertz for the
        # CompuScope system associated with the handle. Returns a negative
        # number if an error occurred
        self.g_tickFrequency = PyGage.GetTimeStampFrequency(self.handle)

        if self.g_tickFrequency < 0:
            print("Error: ", PyGage.GetErrorString(self.g_tickFrequency))
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # after commit the sample size may change
        # get the big acquisition configuration dict
        acq_config = PyGage.GetAcquisitionConfig(self.handle)

        # get total amount of data we expect to receive in bytes, negative if
        # an error occurred
        total_samples = PyGage.GetStreamTotalDataSizeInBytes(self.handle)

        if total_samples < 0 and total_samples != acq_config["SegmentSize"]:
            print("Error: ", PyGage.GetErrorString(total_samples))
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # convert from bytes -> samples and print it to screen
        if total_samples != -1:
            total_samples = total_samples // self.system_info["SampleSize"]
            print("total samples is: ", total_samples)

        # annoys me that I don't know if this is necessary but whatever
        del self.card_stream
        gc.collect()
        self.card_stream = CardStream(
            self.handle,
            self.card_index,
            self.system_info["SampleSize"],
            self.app,
            self.stream_started_event,
            self.ready_for_stream_event,
            self.stream_aborted_event,
            self.stream_error_event,
            self.g_segmentCounted,
            self.g_cardTotalData,
            self,
        )
        self.connect_card_stream_update()
        self._a1 = self.card_stream._a1
        self._a2 = self.card_stream._a2
        thread_cardstream = threading.Thread(target=self.card_stream.run)
        thread_cardstream.start()

        # the card_stream function should have set the
        # self.ready_for_stream_event to true, if it is not true then an error
        # occurred
        set = self.ready_for_stream_event.wait(5)
        if not set:
            print("\nThread initialization error on card ", self.card_index)
            self.stream_aborted_event.set()
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # won't work anymore now that I've thrown the update onto a separate
        # thread (a thread that is not main). Only the main thread can detect a
        # keyboard interrupt print("\nStarting streaming. Press CTRL-C to
        # abort\n\n")

        # start the capture!
        status = PyGage.StartCapture(self.handle)
        if status < 0:
            # get error string
            print("Error: ", PyGage.GetErrorString(status))
            PyGage.FreeSystem(self.handle)
            raise SystemExit  # ??

        # get tick count
        self.stream_started_event.set()

        # annoys me that I don't know if this is necessary but whatever
        del self.trackingstream
        gc.collect()
        self.trackingstream = TrackStreamProgress(self, thread_cardstream)
        self.connect_tracking_stream_update()
        thread_trackstream = threading.Thread(target=self.trackingstream.run)
        thread_trackstream.start()

    def configure_system(self):
        acq, sts = gs.LoadAcquisitionConfiguration(self.handle, self.inifile_stream)

        # I verified that the acquisition config is changed, but it doesn't work
        # for streaming...
        if self.trigger_hold_off is not None:
            acq["TriggerHoldoff"] = int(self.trigger_hold_off)

        if isinstance(acq, dict) and acq:
            status = PyGage.SetAcquisitionConfig(self.handle, acq)
            if status < 0:
                return status
        else:
            print("Using defaults for acquisition parameters")
            status = PyGage.SetAcquisitionConfig(self.handle, acq)

        if sts == gs.INI_FILE_MISSING:
            print("Missing ini file, using defaults")
        elif sts == gs.PARAMETERS_MISSING:
            print(
                "One or more acquisition parameters missing, "
                "using defaults for missing values"
            )

        self.system_info = PyGage.GetSystemInfo(self.handle)

        if not isinstance(
            self.system_info, dict
        ):  # if it's not a dict, it's an int indicating an error
            return self.system_info

        channel_increment = gs.CalculateChannelIndexIncrement(
            acq["Mode"],
            self.system_info["ChannelCount"],
            self.system_info["BoardCount"],
        )

        missing_parameters = False
        for i in range(1, self.system_info["ChannelCount"] + 1, channel_increment):
            chan, sts = gs.LoadChannelConfiguration(self.handle, i, self.inifile_stream)
            if isinstance(chan, dict) and chan:
                status = PyGage.SetChannelConfig(self.handle, i, chan)
                if status < 0:
                    return status
            else:
                print("Using default parameters for channel ", i)
            if sts == gs.PARAMETERS_MISSING:
                missing_parameters = True

        if missing_parameters:
            print(
                "One or more channel parameters missing, "
                "using defaults for missing values"
            )

        missing_parameters = False
        # in this example we're only using 1 trigger source, if we use
        # self.system_info['TriggerMachineCount'] we'll get warnings about
        # using default values for the trigger engines that aren't in
        # the ini file
        trigger_count = 1
        for i in range(1, trigger_count + 1):
            trig, sts = gs.LoadTriggerConfiguration(self.handle, i, self.inifile_stream)
            if isinstance(trig, dict) and trig:
                status = PyGage.SetTriggerConfig(self.handle, i, trig)
                if status < 0:
                    return status
            else:
                print("Using default parameters for trigger ", i)

            if sts == gs.PARAMETERS_MISSING:
                missing_parameters = True

        if missing_parameters:
            print(
                "One or more trigger parameters missing, "
                "using defaults for missing values"
            )

        for i in range(self.system_info["BoardCount"]):
            self.g_cardTotalData.append(0)
            self.g_segmentCounted.append(0)

        return status

    def load_stm_configuration(self):
        app = {}
        # set reasonable defaults

        app["TimeoutOnTransfer"] = TRANSFER_TIMEOUT
        app["BufferSize"] = STREAM_BUFFERSIZE
        app["DoAnalysis"] = 0
        app["ResultsFile"] = "Result"

        config = ConfigParser()

        # parse existing file
        config.read(self.inifile_stream)
        section = "StmConfig"

        if section in config:
            for key in config[section]:
                key = key.lower()
                value = config.get(section, key)
                if key == "doanalysis":
                    if int(value) == 0:
                        app["DoAnalysis"] = False
                    else:
                        app["DoAnalysis"] = True
                elif key == "timeoutontransfer":
                    app["TimeoutOnTransfer"] = int(value)
                elif key == "buffersize":  # in bytes
                    app["BufferSize"] = int(value)  # may need to be an int64
                elif key == "resultsfile":
                    app["ResultsFile"] = value
        return app

    def initialize_stream(self):
        expert_options = CS_BBOPTIONS_STREAM

        # get the big acquisition configuration dict
        acq = PyGage.GetAcquisitionConfig(self.handle)

        if not isinstance(acq, dict):
            if not acq:
                print("Error in call to GetAcquisitionConfig")
                return CS_MISC_ERROR
            else:  # should be error code
                print("Error: ", PyGage.GetErrorString(acq))
                return acq

        extended_options = PyGage.GetExtendedBoardOptions(self.handle)
        if extended_options < 0:
            print("Error: ", PyGage.GetErrorString(extended_options))
            return extended_options

        if extended_options & expert_options:
            print("\nSelecting Expert Stream from image 1")
            acq["Mode"] |= CS_MODE_USER1
        elif (extended_options >> 32) & expert_options:
            print("\nSelecting Expert Stream from image 2")
            acq["Mode"] |= CS_MODE_USER2

        # I'm getting an unknown error signal, so comment this out:

        # comment out _________________________________________________________
        # else:
        #     print("\nCurrent system does not support Expert Streaming")
        #     print("\nApplication terminated")
        #     return CS_MISC_ERROR
        # _____________________________________________________________________

        else:
            # the eXpert image is loaded on Image1
            acq["Mode"] |= CS_MODE_USER1
            pass

        status = PyGage.SetAcquisitionConfig(self.handle, acq)
        if status < 0:
            print("Error: ", PyGage.GetErrorString(status))
        return status


class Signal(qtc.QObject):
    toggle = qtc.pyqtSignal(object)
    progress = qtc.pyqtSignal(object)
    finished = qtc.pyqtSignal(object)


class CardStream:
    def __init__(
        self,
        handle,
        card_index,
        sample_size,
        app,
        stream_started_event,
        ready_for_stream_event,
        stream_aborted_event,
        stream_error_event,
        g_segmentCounted,
        g_cardTotalData,
        parent,
    ):
        parent: Gui
        self.handle = handle
        self.card_index = card_index
        self.sample_size = sample_size
        self.app = app

        self.stream_started_event = stream_started_event
        self.ready_for_stream_event = ready_for_stream_event
        self.stream_aborted_event = stream_aborted_event
        self.stream_error_event = stream_error_event

        self.g_segmentCounted = g_segmentCounted
        self.g_cardTotalData = g_cardTotalData
        self.parent = parent

        self.signal = Signal()

        # acq is a big dict containing:
        # Sample Rate               Sample rate value in Hz
        # External Clock            External clocking status. 0 = "inactive", otherwise "active"
        # Mode                      Acquisition mode of the system
        # SampleBits                Actual vertical resolution, in bits, of the CompuScope system.
        # SampleResolution          Actual sample resolution of the system
        # SampleSize                Actual sample size, in bytes, of the CompuScope system
        # SegmentCount              Number of segments per acquisition.
        # Depth                     Number of samples to capture after the trigger event.
        # SegmentSize               Maximum number of points stored for one segment acquisition.
        # Trigger Timeout           Amount of time to wait (in 100 ns units) after start of segment acquisition
        #                           before forcing a trigger event. Timeout counter is reset for every segment in
        #                           multiple record acquisition
        # Trigger Delay             Number of samples to skip after the trigger event before starting to decrement
        #                           the depth counter.
        # TriggerHoldoff            Number of samples to acquire before enabling the trigger circuitry. The amount
        #                           of pre-trigger data is determined by TriggerHoldoff and has a maximum value of
        #                           SegmentSize – Depth.
        # SampleOffset              Actual sample offset for the CompuScope system.
        # TimeStampConfig           Time stamp mode. Multiple selections may be ORed together. Available values
        #                           (defined inGageConstants.py) are TIMESTAMP_GCLK, TIMESTAMP_MCLK,
        #                           TIMESTAMP_SEG_RESET, TIMESTAMP_FREERUN
        self.acq = PyGage.GetAcquisitionConfig(self.handle)
        self.stream_info = StreamInfo()

        chan = PyGage.GetChannelConfig(handle, 1)
        self.scale_factor = chan["InputRange"] / 2000
        self.offset = chan["DcOffset"] / 1000
        self.sampleoffset = self.acq["SampleOffset"]
        self.sampleres = self.acq["SampleResolution"]

        # for faster conversion from adc to volts:
        a = self.sampleoffset
        b = self.sampleres
        c = self.scale_factor
        d = self.offset
        self._a1 = a * c / b + d
        self._a2 = -c / b

    def run(self):
        # create the two buffers for the card stream (to see the reason for the
        # two buffers, see page 7 of the Advanced Sample Programs in CompuScope
        # SDKs manual:

        """
        The user first allocates two PC RAM contiguous buffers of identical
        size, called Buffer1 and Buffer2, that serve as targets for the data
        stream. If the operating system is unable to allocate the requested
        contiguous buffer, an allocation error will occur. The user toggles the
        two buffer targets as each one gets successively filled. While data are
        streaming to one buffer, the user analyzes ( consumes) waveform data
        from the second buffer. As long as the user is able to sustainably
        analyze (or consume) all waveform data in the buffer under analysis
        before the streaming target buffer is filled up with new waveform data,
        then the streaming acquisition can proceed indefinitely with no data
        loss

        This function requests a contiguous buffer suitable for streaming from
        the driver for the CompuScope system identified with the handle. The
        cardIndex parameter indicates the index of the board in a master /
        slave system. Board indices begin at 1. Use 1 for a single card system.
        The size parameter is the requested size of the buffer in bytes. If the
        function succeeds, the return value is a suitable buffer (from 19
        numpy) for streaming. If the function fails, the return value is a
        negative error which represents a CompuScope error code.
        """

        buffer1 = PyGage.GetStreamingBuffer(
            self.handle, self.card_index, self.app["BufferSize"]
        )
        if isinstance(buffer1, int):
            print("Error getting streaming buffer 1: ", PyGage.GetErrorString(buffer1))
            self.stream_error_event.set()
            time.sleep(1)  # to give stream_error_wait() a chance to catch it
            return False

        buffer2 = PyGage.GetStreamingBuffer(
            self.handle, self.card_index, self.app["BufferSize"]
        )
        if isinstance(buffer2, int):
            print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer2))
            PyGage.FreeStreamingBuffer(self.handle, self.card_index, buffer1)
            self.stream_error_event.set()
            time.sleep(1)  # to give stream_error_wait() a chance to catch it
            return False

        # number of samples in data segment
        data_in_segment_samples = self.acq["SegmentSize"] * (
            self.acq["Mode"] & CS_MASKED_MODE
        )

        """
        The status below is the segment tail size in bytes, otherwise status is
        an error. Now what is a tail you might ask? It is: 
        
        Retrieve the size (in bytes) of the segment tail size for the
        CompuScope system identified by the handle. Some CompuScope boards have
        some data (tail) at the end of each segment which may contain extra
        information about the capture. If successful the return value is an
        unsigned integer containing the tail size in bytes. If unsuccessful,
        the return value is a negative integer representing a CompuScope error
        code. 
        """

        status = PyGage.GetSegmentTailSizeInBytes(self.handle)
        if status < 0:
            print("Error: ", PyGage.GetErrorString(status))
            return
        segment_tail_size_in_bytes = status

        segment_size_in_bytes = data_in_segment_samples * self.sample_size
        transfer_size_in_samples = self.app["BufferSize"] // self.sample_size
        print("\nActual buffer size used for data streaming = ", self.app["BufferSize"])
        print(
            "\nActual sample size used for data streaming = ", transfer_size_in_samples
        )

        self.ready_for_stream_event.set()

        self.stream_started_event.wait()  # should also be waiting for abort
        done = False
        stream_completed_success = False
        loop_count = 0
        work_buffer_active = False
        tail_left_over = 0

        # get_handle the work_buffer to buffer_1
        self.stream_info.WorkBuffer = np.zeros_like(buffer1)
        self.stream_info.TimeStamp = array.array("q")

        self.stream_info.BufferSize = self.app["BufferSize"]
        self.stream_info.SegmentSize = segment_size_in_bytes
        self.stream_info.TailSize = segment_tail_size_in_bytes
        self.stream_info.BytesToEndSegment = segment_size_in_bytes
        self.stream_info.BytesToEndTail = segment_tail_size_in_bytes
        self.stream_info.LeftOverSize = tail_left_over
        self.stream_info.LastTimeStamp = 0
        self.stream_info.Segment = 1
        self.stream_info.SegmentCountDown = self.acq["SegmentCount"]
        self.stream_info.SplitTail = False

        # to avoid Access Violations, we need a copy of the WorkBuffer
        # for this Gui Application, the work buffer will not be larger
        # than the size of one interferogram
        del self.parent.copyOfWorkBuffer1
        del self.parent.copyOfWorkBuffer2
        gc.collect()
        self.parent.copyOfWorkBuffer1 = np.zeros_like(buffer1)
        self.parent.copyOfWorkBuffer2 = np.zeros_like(buffer1)
        self.parent.plotCopyOfWorkBuffer = self.parent.copyOfWorkBuffer1
        self.parent.workBuffer_initiated_event.set()

        # work_buffer_active is initially false. So, the first loop simply
        # acquires data into buffer1, then work_buffer_active is set to True,
        # and the work_buffer is set to buffer1
        #
        # in the next loop data is acquired into buffer2, buffer1 (the
        # work_buffer) is analyzed, and the work_buffer is set to buffer2
        #
        # in the next loop, data is acquired into buffer1 and we analyze
        # buffer2, and so on...
        while not done and not stream_completed_success:
            # check to see if the user aborted the stream
            # don't wait, just check (so set the timeout time to 0)
            set = self.stream_aborted_event.wait(0)
            if set:  # user has aborted
                break

            # if loop count is odd, the buffer is buffer2, otherwise it is
            # buffer1 so starting with loop_count = 0, we go from buffer 1 to
            # buffer 2 and so on.
            if loop_count & 1:
                buffer = buffer2

                copyOfWorkBuffer = self.parent.copyOfWorkBuffer2
                self.parent.plotCopyOfWorkBuffer = self.parent.copyOfWorkBuffer1
            else:
                buffer = buffer1

                copyOfWorkBuffer = self.parent.copyOfWorkBuffer1
                self.parent.plotCopyOfWorkBuffer = self.parent.copyOfWorkBuffer2

            """
            Transfers streaming data from the CompuScope system associated with
            the handle. The cardIndex parameter identifies which board in the
            system the transfer is for. For a single card system use 1. The
            buffer must be suitable for streaming and have been previously
            obtained by calling GetStreamingBuffer. The transferSizeInSamples
            is the requested transfer size in samples. If the function
            succeeds, the return value is CS_SUCCESS (1) and the data is
            returned in the buffer parameter. If the function fails, the return
            value is a negative integer which represents a CompuScope error
            code. """
            status = PyGage.TransferStreamingData(
                self.handle, self.card_index, buffer, transfer_size_in_samples
            )

            if status < 0:
                if status == CS_STM_COMPLETED:
                    # pass (-803 just indicates that the streaming acquisition
                    # completed)
                    stream_completed = True
                else:
                    print("Error: ", PyGage.GetErrorString(status))
                    self.stream_error_event.set()
                    time.sleep(1)  # to give stream_error_wait() a chance to catch it
                    break

            if work_buffer_active:
                # save temporarily for plotting
                # this actually copies the work buffer's content into a second array
                copyOfWorkBuffer[:] = self.stream_info.WorkBuffer[:]

                self.signal.progress.emit(copyOfWorkBuffer)
                # print("work buffer signal emitted!")

            # Wait for the DMA transfer on the current buffer to complete so we
            # can loop back around to start a new one. Calling thread will
            # sleep until the transfer completes

            # Returns the current DMA status of a streaming transfer for the
            # board identified by cardIndex in the CompuScope system identified
            # by the handle. For a single card system, cardIndex should be set
            # to 1. The waitTimeout (in milliseconds) parameter controls how to
            # long to wait before returning. If it is 0, the function returns
            # immediately with the status. If it is not 0, the function will
            # wait until the current DMA transfer is completed or the
            # waitTimeout value has expired before returning. If the function
            # fails, a negative integer representing a CompuScope error code is
            # returned. Otherwise, a tuple is returned containing the following
            # values:

            # tuple(0):    errorFlag – returns one or many error flags that may occur during streaming. Currently,
            #              STM_TRANSFER_ERROR_FIFOFULL is defined. This error indicates the self.application is not fast
            #              enough to transfer the data from on-board memory to PC RAM and that the FIFO is full, which
            #              results in data loss.

            # tuple(1):    actualLength – holds the number of valid samples in the buffer once the DMA has completed.

            # tuple(2):    endOfData – if this value is 1, all data from the current acquisition has been
            #              transferred. If it is 0, there is more to transfer. In infinite streaming mode this value
            #              is always 0 and actualLength is the requested size of the transfer.

            p = PyGage.GetStreamingTransferStatus(
                self.handle, self.card_index, self.app["TimeoutOnTransfer"]
            )

            if isinstance(p, tuple):
                # Used to be self.card_index - 1 But I use a new instance for
                # each card stream so I don't need to index it
                self.g_cardTotalData[0] += p[
                    1
                ]  # have total_data be an array, 1 for each card
                if p[2] == 0:
                    stream_completed_success = False
                else:
                    stream_completed_success = True

                if STM_TRANSFER_ERROR_FIFOFULL & p[0]:
                    print("Fifo full detected on card ", self.card_index)
                    done = True
                    self.parent.terminate()

                    # raise_error(
                    #     self.parent.ErrorWindow,
                    #     f"Fifo full detected on card {self.card_index}",
                    # )
            else:  # error detected
                done = True
                if p == CS_STM_TRANSFER_TIMEOUT:
                    print("\nStream transfer timeout on card ", self.card_index)
                else:
                    print("5 Error: ", p)
                    print("5 Error: ", PyGage.GetErrorString(p))

            self.stream_info.WorkBuffer = buffer
            work_buffer_active = True
            loop_count += 1

        # Do analysis on last buffer
        # self.signal.progress.emit(None)
        # this actually copies the work buffer's content into a second array
        # self.parent.copyOfWorkBuffer1[:] = self.stream_info.WorkBuffer[:]

        self.signal.finished.emit(None)

        status = PyGage.FreeStreamingBuffer(self.handle, self.card_index, buffer1)
        status = PyGage.FreeStreamingBuffer(self.handle, self.card_index, buffer2)
        if stream_completed_success:
            return True
        else:
            return False


class TrackStreamProgress:
    def __init__(self, parent, thread):
        parent: Stream
        self.parent = parent

        self.signal = Signal()
        self.thread = thread

    def run(self):
        self.parent.card_stream_running.set()
        self.parent.card_stream_stopped.clear()

        tickStart = time.time()

        Done = False
        aborted = False
        error_occurred = False

        # while loop to update status of the stream
        while not Done:

            # see if the card_stream function set the
            # self.parent.stream_error_event to True, if yes then exit,
            # otherwise continue
            set = self.parent.stream_error_event.wait(0.5)
            if set:  # error occured
                Done = True

            tickNow = time.time()

            # update progress (print to screen)
            self.parent.totalElapsedTime = tickNow - tickStart
            self.parent.systemTotalData = sum(self.parent.g_cardTotalData)
            self.parent.totalBytes = (
                self.parent.systemTotalData * self.parent.system_info["SampleSize"]
            )

            self.update_progress()

            # not sure what this does, it is not the same as thread.is_alive()...
            count = 0
            for t in threading.enumerate():
                if t is not threading.currentThread():
                    count += 1
            if count == 0:
                Done = True

            if self.parent._terminate:
                self.parent._terminate = False
                self.parent.stream_aborted_event.set()
                aborted = True

                # wait for thread
                self.thread.join()
                self.parent.card_stream_running.clear()
                self.parent.card_stream_stopped.set()
                Done = True

        PyGage.AbortCapture(self.parent.handle)
        PyGage.FreeSystem(self.parent.handle)

        if error_occurred:
            msg = "Stream aborted on error"
            self.signal.finished.emit(msg)
        elif aborted:
            msg = "Stream aborted by user"
            self.signal.finished.emit(msg)
        else:
            msg = "you exited for no good reason"
            self.signal.finished.emit(msg)

    def update_progress(self):
        hours = 0
        minutes = 0

        # want to report time in hours, minutes, seconds
        if self.parent.totalElapsedTime > 0:
            rate = (self.parent.totalBytes / 1000000) / self.parent.totalElapsedTime

            seconds = int(self.parent.totalElapsedTime)  # elapsed time is in seconds
            if seconds >= 60:  # seconds
                minutes = seconds // 60
                if minutes >= 60:
                    hours = minutes // 60
                    if hours > 0:
                        minutes %= 60
                seconds %= 60

            total = self.parent.totalBytes / 1000000  # mega samples

            # the string to print to screen
            s = "Total: {0:.2f} MB, Rate: {1:6.2f} MB/s Elapsed time: {2:d}:{3:02d}:{4:02d}\r".format(
                total, rate, hours, minutes, seconds
            )
            # sys.stdout.write(s)
            # sys.stdout.flush()
            self.signal.progress.emit(s)


class CardStreamSaveData(CardStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_ifgs_to_fill_buffer = self.parent.N_ifgs_to_fill_buffer

    def run(self):
        # create the two buffers for the card stream (to see the reason for the two buffers, see page 7 of the Advanced
        # Sample Programs in CompuScope SDKs manual:

        """
        The user first allocates two PC RAM contiguous buffers of identical
        size, called Buffer1 and Buffer2, that serve as targets for the data
        stream. If the operating system is unable to allocate the requested
        contiguous buffer, an allocation error will occur. The user toggles the
        two buffer targets as each one gets successively filled. While data are
        streaming to one buffer, the user analyzes ( consumes) waveform data
        from the second buffer. As long as the user is able to sustainably
        analyze (or consume) all waveform data in the buffer under analysis
        before the streaming target buffer is filled up with new waveform data,
        then the streaming acquisition can proceed indefinitely with no data
        loss

        This function requests a contiguous buffer suitable for streaming from
        the driver for the CompuScope system identified with the handle. The
        cardIndex parameter indicates the index of the board in a master /
        slave system. Board indices begin at 1. Use 1 for a single card system.
        The size parameter is the requested size of the buffer in bytes. If the
        function succeeds, the return value is a suitable buffer (from 19
        numpy) for streaming. If the function fails, the return value is a
        negative error which represents a CompuScope error code.
        """

        buffer1 = PyGage.GetStreamingBuffer(
            self.handle, self.card_index, self.app["BufferSize"]
        )
        if isinstance(buffer1, int):
            print("Error getting streaming buffer 1: ", PyGage.GetErrorString(buffer1))
            self.stream_error_event.set()
            time.sleep(1)  # to give stream_error_wait() a chance to catch it
            return False

        buffer2 = PyGage.GetStreamingBuffer(
            self.handle, self.card_index, self.app["BufferSize"]
        )
        if isinstance(buffer2, int):
            print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer2))
            PyGage.FreeStreamingBuffer(self.handle, self.card_index, buffer1)
            self.stream_error_event.set()
            time.sleep(1)  # to give stream_error_wait() a chance to catch it
            return False

        # number of samples in data segment
        data_in_segment_samples = self.acq["SegmentSize"] * (
            self.acq["Mode"] & CS_MASKED_MODE
        )

        """
        The status below is the segment tail size in bytes, otherwise status is
        an error. Now what is a tail you might ask? It is: 

        Retrieve the size (in bytes) of the segment tail size for the
        CompuScope system identified by the handle. Some CompuScope boards have
        some data (tail) at the end of each segment which may contain extra
        information about the capture. If successful the return value is an
        unsigned integer containing the tail size in bytes. If unsuccessful,
        the return value is a negative integer representing a CompuScope error
        code. 
        """

        status = PyGage.GetSegmentTailSizeInBytes(self.handle)
        if status < 0:
            print("Error: ", PyGage.GetErrorString(status))
            return
        segment_tail_size_in_bytes = status

        segment_size_in_bytes = data_in_segment_samples * self.sample_size
        transfer_size_in_samples = self.app["BufferSize"] // self.sample_size
        print("\nActual buffer size used for data streaming = ", self.app["BufferSize"])
        print(
            "\nActual sample size used for data streaming = ", transfer_size_in_samples
        )

        self.ready_for_stream_event.set()

        self.stream_started_event.wait()  # should also be waiting for abort
        done = False
        stream_completed_success = False
        loop_count = 0
        work_buffer_active = False
        tail_left_over = 0

        # get_handle the work_buffer to buffer_1
        self.stream_info.WorkBuffer = np.zeros_like(buffer1)
        self.stream_info.TimeStamp = array.array("q")

        self.stream_info.BufferSize = self.app["BufferSize"]
        self.stream_info.SegmentSize = segment_size_in_bytes
        self.stream_info.TailSize = segment_tail_size_in_bytes
        self.stream_info.BytesToEndSegment = segment_size_in_bytes
        self.stream_info.BytesToEndTail = segment_tail_size_in_bytes
        self.stream_info.LeftOverSize = tail_left_over
        self.stream_info.LastTimeStamp = 0
        self.stream_info.Segment = 1
        self.stream_info.SegmentCountDown = self.acq["SegmentCount"]
        self.stream_info.SplitTail = False

        # _____________________________________________________________________
        # this card stream now only saves the data and does not plot it, so I
        # don't need these buffers any more

        # to avoid Access Violations, we need a copy of the WorkBuffer
        # for this Gui Application, the work buffer will not be larger
        # than the size of one interferogram
        # del self.parent.copyOfWorkBuffer1
        # del self.parent.copyOfWorkBuffer2
        # gc.collect()
        # self.parent.copyOfWorkBuffer1 = np.zeros_like(buffer1)
        # self.parent.copyOfWorkBuffer2 = np.zeros_like(buffer1)
        # _____________________________________________________________________

        self.parent.workBuffer_initiated_event.set()

        # what if I did this?
        if not self.parent.save_data_loopcount:
            del self.parent.data_storage_buffer
            gc.collect()

        # if already saving data, don't bother!
        if not self.parent.save_data_loopcount:
            self.parent.data_storage_buffer = np.zeros(
                (self.N_ifgs_to_fill_buffer, self.app["BufferSize"]), dtype=np.uint8
            )

        # work_buffer_active is initially false. So, the first loop simply
        # acquires data into buffer1, then work_buffer_active is set to True,
        # and the work_buffer is set to buffer1
        #
        # in the next loop data is acquired into buffer2, buffer1 (the
        # work_buffer) is analyzed, and the work_buffer is set to buffer2
        #
        # in the next loop, data is acquired into buffer1 and we analyze
        # buffer2, and so on...
        while (
            not done
            and (not stream_completed_success)
            and (loop_count - 1 < self.N_ifgs_to_fill_buffer)
        ):
            # check to see if the user aborted the stream
            # don't wait, just check (so set the timeout time to 0)
            set = self.stream_aborted_event.wait(0)
            if set:  # user has aborted
                break

            # if loop count is odd, the buffer is buffer2, otherwise it is
            # buffer1 so starting with loop_count = 0, we go from buffer 1 to
            # buffer 2 and so on.
            if loop_count & 1:
                buffer = buffer2
            else:
                buffer = buffer1

            """
            Transfers streaming data from the CompuScope system associated with
            the handle. The cardIndex parameter identifies which board in the
            system the transfer is for. For a single card system use 1. The
            buffer must be suitable for streaming and have been previously
            obtained by calling GetStreamingBuffer. The transferSizeInSamples
            is the requested transfer size in samples. If the function
            succeeds, the return value is CS_SUCCESS (1) and the data is
            returned in the buffer parameter. If the function fails, the return
            value is a negative integer which represents a CompuScope error
            code. 
            """
            status = PyGage.TransferStreamingData(
                self.handle, self.card_index, buffer, transfer_size_in_samples
            )

            if status < 0:
                if status == CS_STM_COMPLETED:
                    # pass (-803 just indicates that the streaming acquisition
                    # completed)
                    stream_completed = True
                else:
                    print("Error: ", PyGage.GetErrorString(status))
                    self.stream_error_event.set()
                    time.sleep(1)  # to give stream_error_wait() a chance to catch it
                    break

            if work_buffer_active:
                self.signal.progress.emit(None)

                # save the data!
                if not self.parent.save_data_loopcount:
                    self.parent.data_storage_buffer[loop_count - 1][
                        :
                    ] = self.stream_info.WorkBuffer[:]

                # if active data backup is on, save data to a file in the
                # backup folder, make sure to set it differently for each card!
                else:
                    # the save path is the data backup path
                    # and the name goes as: the loop count + date time
                    # there is also a time stamp in the data segment itself
                    self.stream_info.WorkBuffer.tofile(
                        self.parent.databackup_path
                        + f"LoopCount_{loop_count}_Datetime_"
                        + datetime.now().strftime("%d%m%Y_%H_%M_%S")
                        + ".bin"
                    )
                print(loop_count)

            # Wait for the DMA transfer on the current buffer to complete so we
            # can loop back around to start a new one. Calling thread will
            # sleep until the transfer completes

            # Returns the current DMA status of a streaming transfer for the
            # board identified by cardIndex in the CompuScope system identified
            # by the handle. For a single card system, cardIndex should be set
            # to 1. The waitTimeout (in milliseconds) parameter controls how to
            # long to wait before returning. If it is 0, the function returns
            # immediately with the status. If it is not 0, the function will
            # wait until the current DMA transfer is completed or the
            # waitTimeout value has expired before returning. If the function
            # fails, a negative integer representing a CompuScope error code is
            # returned. Otherwise, a tuple is returned containing the following
            # values:

            # tuple(0):    errorFlag – returns one or many error flags that may occur during streaming. Currently,
            #              STM_TRANSFER_ERROR_FIFOFULL is defined. This error indicates the self.application is not fast
            #              enough to transfer the data from on-board memory to PC RAM and that the FIFO is full, which
            #              results in data loss.

            # tuple(1):    actualLength – holds the number of valid samples in the buffer once the DMA has completed.

            # tuple(2):    endOfData – if this value is 1, all data from the current acquisition has been
            #              transferred. If it is 0, there is more to transfer. In infinite streaming mode this value
            #              is always 0 and actualLength is the requested size of the transfer.

            p = PyGage.GetStreamingTransferStatus(
                self.handle, self.card_index, self.app["TimeoutOnTransfer"]
            )

            if isinstance(p, tuple):
                # Used to be self.card_index - 1
                # But I use a new instance for each card stream so I don't need to index it
                self.g_cardTotalData[0] += p[
                    1
                ]  # have total_data be an array, 1 for each card
                if p[2] == 0:
                    stream_completed_success = False
                else:
                    stream_completed_success = True

                if STM_TRANSFER_ERROR_FIFOFULL & p[0]:
                    print("Fifo full detected on card ", self.card_index)
                    done = True
                    self.parent.terminate()

                    # raise_error(
                    #     self.parent.ErrorWindow,
                    #     f"Fifo full detected on card {self.card_index}",
                    # )

            else:  # error detected
                done = True
                if p == CS_STM_TRANSFER_TIMEOUT:
                    print("\nStream transfer timeout on card ", self.card_index)
                else:
                    print("5 Error: ", p)
                    print("5 Error: ", PyGage.GetErrorString(p))

            self.stream_info.WorkBuffer = buffer
            work_buffer_active = True
            loop_count += 1

            self.parent.loop_count = loop_count - 1

        # Do analysis on last buffer
        # self.signal.progress.emit(None)
        # this actually copies the work buffer's content into a second array
        # self.parent.copyOfWorkBuffer1[:] = self.stream_info.WorkBuffer[:]

        self.signal.finished.emit(None)
        if loop_count - 1 == self.N_ifgs_to_fill_buffer:
            self.parent.terminate()

        status = PyGage.FreeStreamingBuffer(self.handle, self.card_index, buffer1)
        status = PyGage.FreeStreamingBuffer(self.handle, self.card_index, buffer2)
        if stream_completed_success:
            return True
        else:
            return False


class Gui(qt.QMainWindow, Ui_MainWindow, Stream):
    def __init__(self, inifile=inifile_default):
        qt.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        Stream.__init__(self, inifile)
        self.setupUi(self)
        self.show()

        self.plotwindow = pw.PlotWindow(
            self.le_ifgplot_xmin,
            self.le_ifgplot_xmax,
            self.le_ifgplot_ymin,
            self.le_ifgplot_ymax,
            self.gv_ifgplot,
        )
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

        self.card_index = 1

        # set the cell widget for the table to QLineEdits so that we can employ
        # a QIntValidator
        config = ConfigParser()
        config.read(self.inifile_stream)
        level = config["Trigger1"]["Level"]
        plotchecklevel = config["PlotCheckLevel"]["plotchecklevel"]
        segment_size = config["Acquisition"]["segmentsize"]
        extclk = config["Acquisition"]["extclk"]
        self.tableWidget.item(0, 0).setText(str(level))
        self.tableWidget.item(1, 0).setText(str(plotchecklevel))
        self.tableWidget.item(2, 0).setText(str(segment_size))
        self.tableWidget.item(3, 0).setText(str(extclk))
        self.saved_table_widget_item_text = "hello world"

        self.save_data_loopcount = False
        self.databackup_path = "DataBackup/single_card/"

    @property
    def plotchecklevel(self):
        config = ConfigParser()
        config.read(self.inifile_stream)
        plotchecklevel = config["PlotCheckLevel"]["plotchecklevel"]
        return float(plotchecklevel)

    def save_table_item(self, row, col):
        self.saved_table_widget_item_text = self.tableWidget.item(row, col).text()

    def slot_for_table_widget(self, row, col):
        if (row, col) == (0, 0):
            self.set_new_trigger_level(row, col)

        elif (row, col) == (1, 0):
            self.set_new_plotchecklevel(row, col)

        elif (row, col) == (2, 0):
            self.set_segment_size(row, col)

        elif (row, col) == (3, 0):
            self.set_extclk(row, col)

    def set_new_plotchecklevel(self, row, col):
        if not self.tableWidget.item(row, col).text().isnumeric():
            raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        plotchecklevel = int(self.tableWidget.item(row, col).text())

        if plotchecklevel <= 0:
            raise_error(self.ErrorWindow, "trigger level needs to be >= 0 ")
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        setNewPlotCheckLevel(self.inifile_stream, plotchecklevel)

    def set_new_trigger_level(self, row, col):
        if not self.tableWidget.item(row, col).text().isnumeric():
            raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        level_percent = int(self.tableWidget.item(row, col).text())

        if level_percent <= 0:
            raise_error(self.ErrorWindow, "trigger level needs to be >= 0 ")
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        setNewTriggerLevel(self.inifile_stream, level_percent)
        setNewTriggerLevel(self.inifile_acquire, level_percent)

    def set_segment_size(self, row, col):
        if not self.tableWidget.item(row, col).text().isnumeric():
            raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        segment_size = int(self.tableWidget.item(row, col).text())

        if segment_size < -1:
            raise_error(
                self.ErrorWindow,
                f"segment size cannot be less than -1 but got {segment_size}",
            )
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        setSegmentSize(self.inifile_stream, segment_size)

    def set_extclk(self, row, col):
        if not self.tableWidget.item(row, col).text().isnumeric():
            raise_error(self.ErrorWindow, "input must be an integer")
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        extclk = int(self.tableWidget.item(row, col).text())

        if not (extclk == 0 or extclk == 1):
            raise_error(
                self.ErrorWindow, f"external clock must be 0 or 1 but got {extclk}"
            )
            self.tableWidget.item(row, col).setText(
                str(self.saved_table_widget_item_text)
            )
            return

        setExtClk(self.inifile_stream, extclk)
        setExtClk(self.inifile_acquire, extclk)

    def plot(self):
        if self.single_acquire_array is None:
            raise_error(self.ErrorWindow, "no data acquired yet")
            return

        plt.figure()
        N = 30000000 // 2
        plt.plot(normalize(self.single_acquire_array[:N]))
        plt.show()

    def stream_data(self):
        """
        This overrides the stream_data method in Stream(). It adds the option
        to save data by running CardStreamSaveData instead of CardStream. It is
        otherwise the same as stream_data in Stream.
        """

        if self.card_stream_running.is_set():
            # can raise an error window, I choose not to, just print this to screen
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

        # Returns the frequency of the timestamp counter in Hertz for the
        # CompuScope system associated with the handle. negative if an error
        # occurred
        self.g_tickFrequency = PyGage.GetTimeStampFrequency(self.handle)

        if self.g_tickFrequency < 0:
            print("Error: ", PyGage.GetErrorString(self.g_tickFrequency))
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # after commit the sample size may change
        # get the big acquisition configuration dict
        acq_config = PyGage.GetAcquisitionConfig(self.handle)

        # get total amount of data we expect to receive in bytes, negative if
        # an error occurred
        total_samples = PyGage.GetStreamTotalDataSizeInBytes(self.handle)

        if total_samples < 0 and total_samples != acq_config["SegmentSize"]:
            print("Error: ", PyGage.GetErrorString(total_samples))
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # convert from bytes -> samples and print it to screen
        if total_samples != -1:
            total_samples = total_samples // self.system_info["SampleSize"]
            print("total samples is: ", total_samples)

        if self.chkbx_save_data.isChecked():
            self.calc_data_storage_buffer_size()
            pass

        """
        We first initialize and start the card stream thread. The card stream
        thread initializes two buffers and handles the data transfer to the
        buffers. Then we send the command to the Gage Card to start the data
        capture. 
        
        After that, we initialize and start the thread that tracks the progress
        of the data stream. This thread emits signals that are used to plot
        data on the GUI. """

        # annoys me that I don't know if this is necessary but whatever
        del self.card_stream
        gc.collect()
        if self.chkbx_save_data.isChecked():
            self.card_stream = CardStreamSaveData(
                self.handle,
                self.card_index,
                self.system_info["SampleSize"],
                self.app,
                self.stream_started_event,
                self.ready_for_stream_event,
                self.stream_aborted_event,
                self.stream_error_event,
                self.g_segmentCounted,
                self.g_cardTotalData,
                self,
            )
        else:
            self.card_stream = CardStream(
                self.handle,
                self.card_index,
                self.system_info["SampleSize"],
                self.app,
                self.stream_started_event,
                self.ready_for_stream_event,
                self.stream_aborted_event,
                self.stream_error_event,
                self.g_segmentCounted,
                self.g_cardTotalData,
                self,
            )
        self.connect_card_stream_update()
        self._a1 = self.card_stream._a1
        self._a2 = self.card_stream._a2
        thread_cardstream = threading.Thread(target=self.card_stream.run)
        thread_cardstream.start()

        # the card_stream function should have set the
        # self.ready_for_stream_event to true, if it is not true then an error
        # occurred
        set = self.ready_for_stream_event.wait(5)
        if not set:
            print("\nThread initialization error on card ", self.card_index)
            self.stream_aborted_event.set()
            PyGage.FreeSystem(self.handle)
            raise SystemExit

        # won't work anymore now that I've thrown the update onto a separate
        # thread (a thread that is not main)
        # print("\nStarting streaming. Press CTRL-C to abort\n\n")

        # start the capture!
        status = PyGage.StartCapture(self.handle)
        if status < 0:
            # get error string
            print("Error: ", PyGage.GetErrorString(status))
            PyGage.FreeSystem(self.handle)
            raise SystemExit  # ??

        # get tick count
        self.stream_started_event.set()

        # annoys me that I don't know if this is necessary but whatever
        del self.trackingstream
        gc.collect()
        self.trackingstream = TrackStreamProgress(self, thread_cardstream)
        self.connect_tracking_stream_update()
        thread_trackstream = threading.Thread(target=self.trackingstream.run)
        thread_trackstream.start()

    def calc_data_storage_buffer_size(self):
        if self.data_storage_size is None:
            self.update_storage_buffer_size()

        # I'm expecting this function to be called by stream_data after
        # self.app has been initialized
        buffersize = self.app["BufferSize"]
        N = int(np.ceil(self.data_storage_size / buffersize))
        self.N_ifgs_to_fill_buffer = N

        self.data_storage_size = N * buffersize
        self.le_buffer_size_MB.setText(str(self.data_storage_size / 1e6))

    def update_storage_buffer_size(self):
        num = int(float(self.le_buffer_size_MB.text()) * 1e6)

        if self.streaming_buffer_size_to_set is not None:
            if num < self.streaming_buffer_size_to_set:
                raise_error(
                    self.ErrorWindow,
                    "needs to at least be the streaming buffer size. "
                    "I am setting this instead to "
                    + str(self.streaming_buffer_size_to_set / 1e6),
                )
                num = self.streaming_buffer_size_to_set
                self.le_buffer_size_MB.setText(str(num / 1e6))

        elif self.app is not None:
            if num < self.app["BufferSize"]:
                raise_error(
                    self.ErrorWindow,
                    "needs to at least be the streaming buffer size. "
                    "I am setting this instead to " + str(self.app["BufferSize"] / 1e6),
                )
                num = self.app["BufferSize"]
                self.le_buffer_size_MB.setText(str(num / 1e6))

        # have default be 13 MB
        else:
            if num < 0:
                raise_error(self.ErrorWindow, "can't be negative")
                num = int(13e6)
                self.le_buffer_size_MB.setText(str(num / 1e6))

        self.data_storage_size = num

    def update_npts_toplot(self):
        num = int(self.le_npts_to_plot.text())
        if not (1 <= num <= 400):
            raise_error(self.ErrorWindow, "must be between 1 and 400")
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
            self.contPlotUpdate = UpdateDisplay(self)
            # thread = threading.Thread(target=self.contPlotUpdate.run)
            # thread.start()
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

        self.le_npts_post_trigger.editingFinished.connect(
            self.update_acquire_post_trigger_npts_from_le
        )
        self.btn_single_acquire.clicked.connect(self.acquire)

        self.btn_apply_ppifg.clicked.connect(self.apply_ppifg)

        self.le_ppifg.editingFinished.connect(self.update_ppifg_from_le)

        self.le_npts_to_plot.editingFinished.connect(self.update_npts_toplot)

        self.le_buffer_size_MB.editingFinished.connect(self.update_storage_buffer_size)

        self.actionSave.triggered.connect(self.save)

        # table widget connections
        self.tableWidget.cellClicked.connect(self.save_table_item)
        self.tableWidget.cellChanged.connect(self.slot_for_table_widget)

        # plotting for plotchecklevel
        self.btn_plot.clicked.connect(self.plot)

    def updateDisplay(self, X):
        x, y = X
        n_plot = self._nplot
        if not self.adjusted_buffer_to_ppifg:
            y = self.adc_to_volts(y)
            self.curve.setData(x=x[:n_plot], y=y[:n_plot])
        else:
            # y buffer converted from ADC values to volts
            y = self.adc_to_volts(y)

            if show_walking:
                # section of y that will be plotted to screen
                section = y[
                    self.center_ind - n_plot // 2 : self.center_ind + n_plot // 2
                ]

                # the indices of values in section that are above the level
                # threshold indicating the presence of an interferogram
                ind = (
                    abs(section - np.mean(section)) > self._level - np.mean(section)
                ).nonzero()[0]

                # if the number of indices is less than half the original value
                # (set when we know there was an interferogram in there)
                if 0 < len(ind) < self._N_ind * 0.25:
                    avg = np.mean(ind)

                    # if the interferogram is walking out right of the screen,
                    # the correction should be to move it left ( the plot
                    # window would move right)
                    if avg > n_plot // 2:
                        correction = n_plot

                    # otherwise it is walking out to the left of the screen,
                    # and the correction should be to move it right (the plot
                    # window would move left)
                    else:
                        correction = -n_plot

                    # an easy way to implement this correction is to move to
                    # the next interferogram in the buffer
                    self.center_ind += self.ppifg + correction

                    # when we reach the end of the buffer, wrap around
                    if self.center_ind > len(y):
                        self.center_ind -= len(y)

                # same thing as above, except if the number of indices is
                # already 0, use the previous set of indices
                elif len(ind) == 0 and self._ind_old is not None:
                    avg = np.mean(self._ind_old)
                    if avg > n_plot // 2:
                        correction = n_plot
                    else:
                        correction = -n_plot

                    self.center_ind += self.ppifg + correction
                    if self.center_ind > len(y):
                        self.center_ind -= len(y)

                # save the current set of indices in case this set passed
                # without correction, but the next set has no elements
                self._ind_old = ind

            self.curve.setData(
                x=x[self.center_ind - n_plot // 2 : self.center_ind + n_plot // 2],
                y=y[self.center_ind - n_plot // 2 : self.center_ind + n_plot // 2],
            )

        # self.update_progress_bar() <- we do not update progress bar and
        # display simultaneously anymore

    def change_stream_buffer_size(self, num, mB=False):
        if mB:
            self.streaming_buffer_size_to_set = int(num * 1e6)
        else:
            self.streaming_buffer_size_to_set = int(num)

    def update_acquire_post_trigger_npts_from_le(self):
        num = int(self.le_npts_post_trigger.text())
        if num < 20000:
            raise_error(self.ErrorWindow, "npts has to at least be 20,000")
            return
        self.acquire_npts = num

    def update_ppifg_from_le(self):
        print("I'm updating ppifg")
        num = int(self.le_ppifg.text())
        if num < 1000:
            raise_error(self.ErrorWindow, "surely ppifg is at least 1000")
            self.le_ppifg.setText(str(1000))
            num = 1000
        self.ppifg = num

        # this slows it down so much it basically freezes
        # if self.single_acquire_array is not None:
        #     self.apply_ppifg()

    def acquire(self):
        if self.card_stream_running.is_set():
            raise_error(self.ErrorWindow, "stop card stream first")
            return

        self.single_acquire_array = Acquire.acquire(
            self.acquire_npts, inifile=self.inifile_acquire
        )
        gc.collect()

        npts_int, npts_float, level = pf.find_npts(
            self.single_acquire_array, level_percent=self.plotchecklevel
        )
        self._level = level

        self.le_ppifg.setText(str(npts_float))
        self.ppifg = npts_int

    def plot_check_ppifg(self):
        arr = self.single_acquire_array[self.ppifg // 2 :]
        if len(arr) > self.ppifg * 10:
            arr = arr[: int(self.ppifg * 10)]
        x = np.arange(len(arr))
        self.curve.setData(x=x, y=normalize(arr))

    def apply_ppifg(self):
        if self.ppifg is None:
            raise_error(self.ErrorWindow, "no ppifg yet")
            return
        buffer_size_bytes = 2 * self.ppifg * 2

        # have the buffer be at least 13 MB
        if buffer_size_bytes < 13e6:
            N = int(np.ceil(13e6 / buffer_size_bytes))
            buffer_size_bytes *= N

        self.change_stream_buffer_size(buffer_size_bytes, mB=False)

        print(
            "applied points per interferogram, {N} interferograms per buffer".format(
                N=buffer_size_bytes / (2 * self.ppifg * 2)
            )
        )

        # given that it's only called by update_ppifg_from_le which already
        # does this check, this check may be redundant
        if self.single_acquire_array is not None:
            pass
            # self.plot_check_ppifg()

        self.adjusted_buffer_to_ppifg = True

        self.le_ppifg.setText(str(self.ppifg))
        self.center_ind = self.ppifg

        section = self.single_acquire_array[
            self.ppifg - self._nplot // 2 : self.ppifg + self._nplot // 2
        ]
        ind = (
            abs(section - np.mean(section)) > self._level - np.mean(section)
        ).nonzero()[0]
        self._ind_old = None
        self._N_ind = len(ind)

    def save(self):
        if self.data_storage_buffer is None:
            raise_error(self.ErrorWindow, "no data saved to storage buffer yet")
            return

        filename, _ = qt.QFileDialog.getSaveFileName(self, "Save Data")
        if filename == "":
            return

        N_ifgs = self.data_storage_buffer.size // (self.ppifg * 2)
        filename += "_{Nifgs}x{ppifg}".format(ppifg=int(self.ppifg), Nifgs=int(N_ifgs))
        filename += ".bin"

        self.data_storage_buffer.tofile(filename)


class UpdateDisplay(qtc.QThread):
    def __init__(self, parent, wait_time=100):
        qtc.QThread.__init__(self)
        parent: Gui
        self.parent = parent
        self.wait_time = wait_time

        self.dT = 1e-9
        N = len(self.parent.copyOfWorkBuffer1) // 2
        self.x = np.linspace(0, N * self.dT, N)

        self.signal = Signal()

        self._terminate = False

        self.y = np.zeros(self.x.shape)

        if self.parent.ppifg is not None:
            self.parent.center_ind = self.parent.ppifg

        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.timer_timeout)
        self.timer.moveToThread(self)

    def run(self):
        self.timer.start(self.wait_time)
        loop = qtc.QEventLoop()
        loop.exec()

    def timer_timeout(self):
        if self.parent._terminate:
            self.timer.stop()
        else:
            # print("hey there")
            self.y[:] = np.frombuffer(self.parent.plotCopyOfWorkBuffer, "h")
            self.signal.progress.emit([self.x, self.y])


# TODO push forward on this so we can look at the spectrum in real time
class SpectrumUpdate:
    def __init__(self, parent):
        parent: Gui
        self.parent = parent

        self.NPTS = len(self.parent.plotCopyOfWorkBuffer)
        self.ppifg = self.parent.ppifg
        self.N_ifgs = self.NPTS // self.ppifg

        self.array = np.reshape(
            self.parent.plotCopyOfWorkBuffer, (self.N_ifgs, self.ppifg)
        )
        self.r, self.c = np.ogrid[: self.array.shape[0], : self.array.shape[1]]
        self.ind_maxes = np.zeros(self.array.shape[0])
        self.shifts = np.zeros(self.array.shape[0])

        self.fft_input = pyfftw.empty_aligned(self.array.shape[1], dtype="complex128")
        self.fft_output = pyfftw.empty_aligned(self.array.shape[1], dtype="complex128")

        self.fft = pyfftw.FFTW(
            self.fft_input,
            self.fft_output,
            axes=[0],
            direction="FFTW_FORWARD",
            flags="FFTW_MEASURE",
        )

    def run(self):
        # fill the array
        self.array.resize(self.NPTS)
        self.array[:] = self.parent.plotCopyOfWorkBuffer[:]
        self.array.resize((self.N_ifgs, self.ppifg))

        # shift correction
        self.ind_maxes[:] = np.argmax(self.array, axis=1)
        self.shifts = self.ind_maxes - self.ind_maxes[0]
        self.array[:] = self.array[self.r, self.c - self.shifts[:, np.newaxis]]

        # compute the fft
        self.fft_input = np.mean(self.array, axis=0)
        self.fft()


# # not fast enough :(
# class UpdateDisplaySaveData:
#     def __init__(self, parent):
#         parent: Gui
#         self.parent = parent
#
#         self.dT = 1e-9
#         N = len(self.parent.copyOfWorkBuffer1)
#         self.x = np.linspace(0, N * self.dT, N)
#
#         self.signal = Signal()
#
#         self._terminate = False
#
#     def run(self):
#         while not self.parent._terminate:
#             buffer = self.parent.data_storage_buffer[self.parent.loop_count]
#             self.signal.progress.emit([self.x, buffer])
#             time.sleep(.05)

"""I now run Gui() inside RUN_DataStreamApplication.py """
# if __name__ == '__main__':
#     app = qt.QApplication([])
#     hey = Gui()
#     app.exec()
