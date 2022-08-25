"""Same issue!"""

import serial
import struct
import time

port = 'COM6'
ser = serial.Serial()
ser.port = port
ser.baudrate = 115200
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.stopbits = serial.STOPBITS_ONE
ser.timeout = 1

ser.open()

dst = 0x50  # generic USB device
src = 0x01  # host

# %%____________________________________________________________________________________________________________________
# MGMSG_HW_STOP_UPDATEMSGS
write_buffer = struct.pack("<BBBBBB", 0x12, 0x00, 0x00, 0x00, dst, src)
ser.write(write_buffer)

# %%____________________________________________________________________________________________________________________
# get trigger parameters

# MGMSG_MOT_REQ_KCUBETRIGCONFIG 0x0524
write_buffer = struct.pack("<6B", 0x27, 0x05, 0x01, 0x00, dst, src)
ser.write(write_buffer)

# MGMSG_MOT_GET_KCUBETRIGCONFIG 0x0525
time.sleep(.1)
read_buffer = ser.read(40)
result = struct.unpack("<8l", read_buffer[8:])
time.sleep(.1)

# %%____________________________________________________________________________________________________________________
# change trigger io parameters

# channel = 0x01
# trig1_mode = 0x01
# trig1_polarity = 0x01
# trig2_mode = 0x0A
# trig2_polarity = 0x01
#
# header = struct.pack("<6B", 0x23, 0x05, 0x0C, 0x00, dst, src)
# packet = struct.pack("<11H", channel, trig1_mode, trig1_polarity, trig2_mode, trig2_polarity,
#                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
# write_buffer = header + packet
# ser.write(write_buffer)

# %%____________________________________________________________________________________________________________________
# get trigger io parameters

# time.sleep(.1)
# # MGMSG_MOT_REQ_KCUBETRIGCONFIG 0x0524
# write_buffer = struct.pack("<6B", 0x24, 0x05, 0x01, 0x00, dst, src)
# ser.write(write_buffer)
#
# # MGMSG_MOT_GET_KCUBETRIGCONFIG 0x0525
# time.sleep(.1)
# read_buffer = ser.read(28)
# result = struct.unpack("<6B11H", read_buffer)
# time.sleep(.1)

# %%____________________________________________________________________________________________________________________
# change trigger settings
# header = struct.pack("<6B", 0x26, 0x05, 0x22, 0x00, dst, src)
#
# channel = 0x01
# startposfwd = 0
# intervalfwd = 3430
# numpulsesfwd = 1
# startposrev = 34304 * 5
# intervalrev = 3430
# numpulsesrev = 1
# pulsewidth = 100000
# numcycles = 1
#
# packet = struct.pack("<H8l", channel, startposfwd, intervalfwd, numpulsesfwd, startposrev, intervalrev, numpulsesrev,
#                      pulsewidth, numcycles)
# write_buffer = header + packet
# ser.write(write_buffer)

# %%____________________________________________________________________________________________________________________
# get trigger parameters

# # MGMSG_MOT_REQ_KCUBETRIGCONFIG 0x0524
# write_buffer = struct.pack("<6B", 0x27, 0x05, 0x01, 0x00, dst, src)
# ser.write(write_buffer)
#
# # MGMSG_MOT_GET_KCUBETRIGCONFIG 0x0525
# time.sleep(.1)
# read_buffer = ser.read(40)
# result = struct.unpack("<8l", read_buffer[8:])
# time.sleep(.1)

# %%____________________________________________________________________________________________________________________
ser.close()
