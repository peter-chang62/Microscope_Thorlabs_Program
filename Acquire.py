from __future__ import print_function
import sys
import time
import ProcessingFunctions as pf

sys.path.append("include")
import matplotlib.pyplot as plt
from builtins import int
import sys
from datetime import datetime
import GageSupport as gs
import GageConstants as gc
import numpy as np
import PyGage3_64 as PyGage


def convert_adc_to_volts(x, stHeader, scale_factor, offset):
    return (((stHeader['SampleOffset'] - x) / stHeader['SampleRes']) * scale_factor) + offset


def normalize(vec):
    return vec / np.max(abs(vec))


def configure_system(handle, filename, segment_size=None):
    acq, sts = gs.LoadAcquisitionConfiguration(handle, filename)

    # added this
    if segment_size is not None:
        if isinstance(acq, dict):
            acq['Depth'] = segment_size
            acq['SegmentSize'] = segment_size

    if isinstance(acq, dict) and acq:
        status = PyGage.SetAcquisitionConfig(handle, acq)
        if status < 0:
            return status
    else:
        print("Using defaults for acquisition parameters")

    if sts == gs.INI_FILE_MISSING:
        print("Missing ini file, using defaults")
    elif sts == gs.PARAMETERS_MISSING:
        print("One or more acquisition parameters missing, using defaults for missing values")

    system_info = PyGage.GetSystemInfo(handle)
    acq = PyGage.GetAcquisitionConfig(
        handle)  # check for error - copy to GageAcquire.py

    channel_increment = gs.CalculateChannelIndexIncrement(acq['Mode'],
                                                          system_info[
                                                              'ChannelCount'],
                                                          system_info[
                                                              'BoardCount'])

    missing_parameters = False
    for i in range(1, system_info['ChannelCount'] + 1, channel_increment):
        chan, sts = gs.LoadChannelConfiguration(handle, i, filename)
        if isinstance(chan, dict) and chan:
            status = PyGage.SetChannelConfig(handle, i, chan)
            if status < 0:
                return status
        else:
            print("Using default parameters for channel ", i)

        if sts == gs.PARAMETERS_MISSING:
            missing_parameters = True

    if missing_parameters:
        print("One or more channel parameters missing, using defaults for missing values")

    missing_parameters = False
    # in this example we're only using 1 trigger source, if we use
    # system_info['TriggerMachineCount'] we'll get warnings about
    # using default values for the trigger engines that aren't in
    # the ini file
    trigger_count = 1
    for i in range(1, trigger_count + 1):
        trig, sts = gs.LoadTriggerConfiguration(handle, i, filename)
        if isinstance(trig, dict) and trig:
            status = PyGage.SetTriggerConfig(handle, i, trig)
            if status < 0:
                return status
        else:
            print("Using default parameters for trigger ", i)

        if sts == gs.PARAMETERS_MISSING:
            missing_parameters = True

    if missing_parameters:
        print("One or more trigger parameters missing, using defaults for missing values")

    status = PyGage.Commit(handle)
    return status


def initialize():
    status = PyGage.Initialize()
    if status < 0:
        return status
    else:
        handle = PyGage.GetSystem(0, 0, 0, 0)
        return handle


def get_data(handle, mode, app, system_info):
    status = PyGage.StartCapture(handle)
    if status < 0:
        return status

    status = PyGage.GetStatus(handle)
    while status != gc.ACQ_STATUS_READY:
        status = PyGage.GetStatus(handle)

    acq = PyGage.GetAcquisitionConfig(handle)

    # Validate the start address and the length. This is especially
    # necessary if trigger delay is being used.
    min_start_address = acq['TriggerDelay'] + acq['Depth'] - acq['SegmentSize']
    if app['StartPosition'] < min_start_address:
        print("\nInvalid Start Address was changed from {0} to {1}".format(app['StartPosition'], min_start_address))
        app['StartPosition'] = min_start_address

    max_length = acq['TriggerDelay'] + acq['Depth'] - min_start_address
    if app['TransferLength'] > max_length:
        print("\nInvalid Transfer Length was changed from {0} to {1}".format(app['TransferLength'], max_length))
        app['TransferLength'] = max_length

    stHeader = {}
    if acq['ExternalClock']:
        stHeader['SampleRate'] = acq['SampleRate'] / acq['ExtClockSampleSkip'] * 1000
    else:
        stHeader['SampleRate'] = acq['SampleRate']

    stHeader['Start'] = app['StartPosition']
    stHeader['Length'] = app['TransferLength']
    stHeader['SampleSize'] = acq['SampleSize']
    stHeader['SampleOffset'] = acq['SampleOffset']
    stHeader['SampleRes'] = acq['SampleResolution']
    stHeader['SegmentNumber'] = 1  # this example only does single capture
    stHeader['SampleBits'] = acq['SampleBits']

    if app['SaveFileFormat'] == gs.TYPE_SIG:
        stHeader['SegmentCount'] = 1
    else:
        stHeader['SegmentCount'] = acq['SegmentCount']

    # we are only streaming data from 1 channel:
    # so i = 1
    buffer = PyGage.TransferData(handle, 1, 0, 1, app['StartPosition'], app['TransferLength'])
    if isinstance(buffer, int):  # an error occurred
        print("Error transferring channel ", 1)
        return buffer

    # if call succeeded (buffer is not an integer) then
    # buffer[0] holds the actual data, buffer[1] holds
    # the actual start and buffer[2] holds the actual length

    chan = PyGage.GetChannelConfig(handle, 1)
    stHeader['InputRange'] = chan['InputRange']
    stHeader['DcOffset'] = chan['DcOffset']

    scale_factor = stHeader['InputRange'] / 2000
    offset = stHeader['DcOffset'] / 1000

    # buffer[0] is a numpy array
    # I don't know why in their code they converted the array to list and then used map,
    # it's a heck of a lot longer to do it that way.
    data = convert_adc_to_volts(buffer[0], stHeader, scale_factor, offset)

    return status, data


def acquire(segment_size, handle=None, inifile=None):
    try:
        # initialization common amongst all sample programs:
        # ____________________________________________________________________________________________________
        if inifile is None:
            inifile = 'include/Acquire.ini'

        # if handle is None, then get the handle for the first card available
        if handle is None:
            handle = initialize()
            if handle < 0:
                # get error string
                error_string = PyGage.GetErrorString(handle)
                print("Error: ", error_string)
                raise SystemExit

        # in case handle was supplied, make sure handle is an int here
        # if it was supplied and doesn't refer to a card, the error will be caught later
        assert isinstance(handle, int)

        system_info = PyGage.GetSystemInfo(handle)
        if not isinstance(system_info,
                          dict):  # if it's not a dict, it's an int indicating an error
            print("Error: ", PyGage.GetErrorString(system_info))
            PyGage.FreeSystem(handle)
            raise SystemExit

        print("\nBoard Name: ", system_info["BoardName"])

        status = configure_system(handle, inifile, segment_size)
        if status < 0:
            # get error string
            error_string = PyGage.GetErrorString(status)
            print("Error: ", error_string)
        else:
            acq_config = PyGage.GetAcquisitionConfig(handle)
            app, sts = gs.LoadApplicationConfiguration(inifile)

            if segment_size is not None:
                app['TransferLength'] = segment_size

            # we don't need to check for gs.INI_FILE_MISSING because if there's no ini file
            # we've already reported when calling configure_system
            if sts == gs.PARAMETERS_MISSING:
                print(
                    "One or more application parameters missing, using defaults for missing values")

            # _________________________________________________________________________________________________
            # initialization done

            status, data = get_data(handle, acq_config['Mode'], app,
                                    system_info)
            if isinstance(status, int):
                if status < 0:
                    error_string = PyGage.GetErrorString(status)
                    print("Error: ", error_string)

                # these error checks regard the saving of the data

                # comment out ____________________________________________________________________
            #     elif status == 0:  # could not open o write to the data file
            #         print("Error opening or writing ", filename)
            #     else:
            #         if app['SaveFileFormat'] == gs.TYPE_SIG:
            #             print("\nAcquisition completed.\nAll channels saved as "
            #                   "GageScope SIG files in the current directory\n")
            #         elif app['SaveFileFormat'] == gs.TYPE_BIN:
            #             print("\nAcquisition completed.\nAll channels saved "
            #                   "as binary files in the current directory\n")
            #         else:
            #             print("\nAcquisition completed.\nAll channels saved "
            #                   "as ASCII data files in the current directory\n")
            # else:  # not an int, we can't open or write the file so we returned the filename
            #     print("Error opening or writing ", status)
            # ________________________________________________________________________________________

            # free the handle and return the data data
            PyGage.FreeSystem(handle)
            return data
    except KeyboardInterrupt:
        print("Exiting program")

    PyGage.FreeSystem(handle)


# %% testing
n = 17860


def test(ppifg, plot=True, Nplot=10):
    d = acquire(30000000)
    N = 30000000 // ppifg
    d = d[:N * ppifg]
    d = d[ppifg // 2:-ppifg // 2]
    d = d.reshape((N - 1, ppifg))

    if plot:
        plt.figure()
        [plt.plot(d[i, ppifg // 2 - 100: ppifg // 2 + 100]) for i in range(Nplot)]

    return d
