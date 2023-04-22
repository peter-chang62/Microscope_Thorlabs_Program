from __future__ import print_function
from builtins import int
import platform
import sys
from datetime import datetime
import GageSupport as gs
import GageConstants as gc
import numpy as np
from configparser import ConfigParser


# Code used to determine if python is version 2.x or 3.x
# and if os is 32 bits or 64 bits.  If you know they
# python version and os you can skip all this and just
# import the appropriate version

# returns is_64bits for python
# (i.e. 32 bit python running on 64 bit windows should return false)

is_64_bits = sys.maxsize > 2**32

if is_64_bits:
    if sys.version_info >= (3, 0):
        import PyGage3_64 as PyGage
    else:
        import PyGage2_64 as PyGage
else:
    if sys.version_info > (3, 0):
        import PyGage3_32 as PyGage
    else:
        import PyGage2_32 as PyGage


def edit_inifile(inifile, segment_size):
    config = ConfigParser()
    config.read(inifile)
    config["Acquisition"]["Depth"] = str(segment_size)  # depth
    config["Acquisition"]["SegmentSize"] = str(segment_size)  # segmentsize
    config["Application"]["TransferLength"] = str(segment_size)  # transfer length
    with open(inifile, "w") as configfile:
        config.write(configfile)


def configure_system(handle, filename):
    acq, sts = gs.LoadAcquisitionConfiguration(handle, filename)

    if isinstance(acq, dict) and acq:
        status = PyGage.SetAcquisitionConfig(handle, acq)
        if status < 0:
            return status
    else:
        print("Using defaults for acquisition parameters")

    if sts == gs.INI_FILE_MISSING:
        print("Missing ini file, using defaults")
    elif sts == gs.PARAMETERS_MISSING:
        print(
            "One or more acquisition parameters missing, using defaults for missing values"
        )

    system_info = PyGage.GetSystemInfo(handle)
    acq = PyGage.GetAcquisitionConfig(
        handle
    )  # check for error - copy to GageAcquire.py

    channel_increment = gs.CalculateChannelIndexIncrement(
        acq["Mode"], system_info["ChannelCount"], system_info["BoardCount"]
    )

    missing_parameters = False
    for i in range(1, system_info["ChannelCount"] + 1, channel_increment):
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
        print(
            "One or more channel parameters missing, using defaults for missing values"
        )

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
        print(
            "One or more trigger parameters missing, using defaults for missing values"
        )

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

    capture_time = 0
    status = PyGage.GetStatus(handle)
    while status != gc.ACQ_STATUS_READY:
        status = PyGage.GetStatus(handle)
        # if we've triggered, get the time of day
        # this is just to demonstrate how to use the time stamp
        # in the SIG file header
        if status == gc.ACQ_STATUS_TRIGGERED:
            capture_time = datetime.now().time()

    # just in case we missed the trigger time, we'll use the capture time
    if capture_time == 0:
        capture_time = datetime.now().time()

    channel_increment = gs.CalculateChannelIndexIncrement(
        mode, system_info["ChannelCount"], system_info["BoardCount"]
    )

    acq = PyGage.GetAcquisitionConfig(handle)
    # These fields are common for all the channels

    # Validate the start address and the length. This is especially
    # necessary if trigger delay is being used.

    min_start_address = acq["TriggerDelay"] + acq["Depth"] - acq["SegmentSize"]
    if app["StartPosition"] < min_start_address:
        print(
            "\nInvalid Start Address was changed from {0} to {1}".format(
                app["StartPosition"], min_start_address
            )
        )
        app["StartPosition"] = min_start_address

    max_length = acq["TriggerDelay"] + acq["Depth"] - min_start_address
    if app["TransferLength"] > max_length:
        print(
            "\nInvalid Transfer Length was changed from {0} to {1}".format(
                app["TransferLength"], max_length
            )
        )
        app["TransferLength"] = max_length

    for i in range(1, system_info["ChannelCount"] + 1, channel_increment):
        buffer = PyGage.TransferData(
            handle, i, 0, 1, app["StartPosition"], app["TransferLength"]
        )
        if isinstance(buffer, int):  # an error occurred
            print("Error transferring channel ", i)
            return buffer

        # if call succeeded (buffer is not an integer) then
        # buffer[0] holds the actual data, buffer[1] holds
        # the actual start and buffer[2] holds the actual length

        # TransferData may change the actual length of the buffer
        # (i.e. if the requested transfer length was too large), so we can
        # change it in the header to be the length of the buffer or
        # we can use the actual length (buffer[2])

    return status, buffer[0]


def acquire(segment_size, handle=None, inifile=None):
    if inifile is None:
        inifile = "Acquire.ini"
    if handle is None:
        handle = initialize()
        if handle < 0:
            # get error string
            error_string = PyGage.GetErrorString(handle)
            print("Error: ", error_string)
            raise SystemExit
    assert isinstance(handle, int)
    assert isinstance(segment_size, int)

    edit_inifile(inifile, segment_size)

    system_info = PyGage.GetSystemInfo(handle)
    if not isinstance(
        system_info, dict
    ):  # if it's not a dict, it's an int indicating an error
        print("Error: ", PyGage.GetErrorString(system_info))
        PyGage.FreeSystem(handle)
        raise SystemExit

    print("\nBoard Name: ", system_info["BoardName"])

    status = configure_system(handle, inifile)
    if status < 0:
        # get error string
        error_string = PyGage.GetErrorString(status)
        print("Error: ", error_string)
    else:
        acq_config = PyGage.GetAcquisitionConfig(handle)
        app, sts = gs.LoadApplicationConfiguration(inifile)

        # we don't need to check for gs.INI_FILE_MISSING because if there's no ini file
        # we've already reported when calling configure_system
        if sts == gs.PARAMETERS_MISSING:
            print(
                "One or more application parameters missing, using defaults for missing values"
            )

        status, data = get_data(handle, acq_config["Mode"], app, system_info)
        if status < 0:
            error_string = PyGage.GetErrorString(status)
            print("Error: ", error_string)
        PyGage.FreeSystem(handle)
        data = np.frombuffer(data, "<h")
        return data


if __name__ == "__main__":
    data = acquire(400000000, handle=None, inifile=None)
