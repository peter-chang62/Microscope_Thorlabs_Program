import struct
import time

from APT import _auto_connect
import numpy as np
import APT as apt
import gc


class AptMotor(apt.KDC101_PRM1Z8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ENC_CNT_MM = 34304.
        self.VEL_SCALING_FACTOR = 65536
        self.SAMPLING_INTERVAL = 2048 / 6e6
        self.ENC_CNT_MM_S = self.ENC_CNT_MM * self.SAMPLING_INTERVAL * self.VEL_SCALING_FACTOR
        self.ENC_CNT_MM_S2 = self.ENC_CNT_MM * self.SAMPLING_INTERVAL ** 2 * self.VEL_SCALING_FACTOR

        self.set_trigger_io_params(1, 10)
        self._trigger_on = False

        self.step_mm = 0.01
        self.pulse_width_ms = 1

    @property
    def trigger_on(self):
        return self._trigger_on

    @trigger_on.setter
    def trigger_on(self, flag):
        assert isinstance(flag, bool)

        # If turning trigger_on to false and trigger_on was originally true, turn off the trigger for the K-cube. If
        # the user is turning the trigger on, this will be caught whenever the position is changed.
        if (not flag) and self.trigger_on:
            self.set_trigger_io_params(1, 10)

        # update trigger_on
        self._trigger_on = flag

    @_auto_connect
    def get_position(self):
        # Get the current position
        # MGMSG_MOT_REQ_POSCOUNTER 0x0411
        write_buffer = struct.pack("<BBBBBB", 0x11, 0x04,
                                   0x01, 0x00,
                                   self.dst, self.src)
        self.write(write_buffer)
        # MGMSG_MOT_GET_POSCOUNTER 0x0412
        read_buffer = self.read(0x12, 0x04, req_buffer=write_buffer)
        result = struct.unpack("<BBBBBBHl", read_buffer)
        position = result[7] / self.ENC_CNT_MM
        return position

    @_auto_connect
    def set_position(self, value_mm):
        # set trigger parameters if relevant
        if self.trigger_on:
            pos = self.get_position()

            if pos < value_mm:  # going forward
                self.set_trigger_pos_params(pos, value_mm, self.step_mm, self.pulse_width_ms, 13)
            elif pos > value_mm:  # going backwards
                self.set_trigger_pos_params(value_mm, pos, self.step_mm, self.pulse_width_ms, 14)
            else:  # current position is already the target position
                pass  # do nothing

        # Calculate the encoder value
        enc_cnt = int(round(value_mm * self.ENC_CNT_MM))
        # MGMSG_MOT_MOVE_ABSOLUTE
        write_buffer = struct.pack('<BBBBBBHl', 0x53, 0x04,
                                   0x06, 0x00,
                                   self.dst | 0x80, self.src,
                                   0x0001, enc_cnt)
        self.write(write_buffer)

    @_auto_connect
    def move_relative(self, rel_position):
        enc_cnt = int(round(rel_position * self.ENC_CNT_MM))
        # MGMSG_MOT_MOVE_RELATIVE
        write_buffer = struct.pack("<BBBBBBHl", 0x48, 0x04,
                                   0x06, 0x00,
                                   self.dst | 0x80, self.src,
                                   0x0001, enc_cnt)
        self.write(write_buffer)

    @_auto_connect
    def stop(self):
        write_buffer = struct.pack("<6B", 0x65, 0x04, 0x01, 0x02,
                                   self.dst,
                                   self.src)
        self.write(write_buffer)

    @_auto_connect
    def get_velocity_params(self):
        # Request Velocity parameters
        # MGMSG_MOT_REQ_VELPARAMS 0x0414
        write_buffer = struct.pack("<6B", 0x14, 0x04, 0x01, 0x00,
                                   self.dst,
                                   self.src)
        self.write(write_buffer)

        # Get velocity parameters
        # MGMSG_MOT_GET_VELPARAMS 0x0415
        read_buffer = self.read(0x15, 0x04, req_buffer=write_buffer)
        result = struct.unpack("<6BHlll", read_buffer)

        [*header, start_vel, accel, max_vel] = result
        start_vel /= self.ENC_CNT_MM_S  # divide by ENC_CNT_MM_S
        max_vel /= self.ENC_CNT_MM_S  # divide by ENC_CNT_MM_S
        accel /= self.ENC_CNT_MM_S2  # divide by ENC_CNT_MM_S2
        return start_vel, max_vel, accel

    @_auto_connect
    def set_max_vel(self, max_vel_mm_s):
        _, _, accel = self.get_velocity_params()
        accel_enc_s2 = int(round(accel * self.ENC_CNT_MM_S2))  # multiply by ENC_CNT_MM_S2
        max_vel_enc_s = int(round(max_vel_mm_s * self.ENC_CNT_MM_S))  # multiply by ENC_CNT_MM_S

        # Set Velocity Parameters
        # MGMSG_MOT_SET_VELPARAMS 0x0413
        write_buffer = struct.pack("<6BHlll", 0x13, 0x04, 0x0E, 0x00,
                                   self.dst | 0x80,
                                   self.src,
                                   0x01,
                                   0, accel_enc_s2, max_vel_enc_s)  # passed in as min_vel, accel, max_vel
        self.write(write_buffer)

    @_auto_connect
    def get_trigger_io_params(self):
        # MGMSG_MOT_REQ_KCUBETRIGCONFIG 0x0524
        write_buffer = struct.pack("<6B", 0x24, 0x05, 0x01, 0x00,
                                   self.dst,
                                   self.src)
        self.write(write_buffer)

        # MGMSG_MOT_GET_KCUBETRIGCONFIG 0x0525

        # OUTPUT Trigger Modes
        # 10: General purpose logic output (set using the MOD_SET_DIGOUTPUTS message).
        #
        # 11: Trigger output active (level) when motor 'in motion'. The output trigger goes high (5V) or low (0V) (as
        # set in the lTrig1Polarity and lTrig2Polarity parameters) when the stage is in motion
        #
        # 12: Trigger output active (level) when motor at 'max velocity'.
        #
        # 13: Trigger output active (pulsed) at pre-defined positions moving forward (set using StartPosFwd,
        # IntervalFwd, NumPulsesFwd and PulseWidth parameters in the SetKCubePosTrigParams message). Only one Trigger
        # port at a time can be set to this mode.
        #
        # 14: Trigger output active (pulsed) at pre-defined positions moving backwards (set using StartPosRev,
        # IntervalRev, NumPulsesRev and PulseWidth parameters in the SetKCubePosTrigParams message). Only one Trigger
        # port at a time can be set to this mode
        #
        # 15: Trigger output active (pulsed) at pre-defined positions moving forwards and backward. Only one Trigger
        # port at a time can be set to this mode.
        read_buffer = self.read(0x25, 0x05, req_buffer=write_buffer)
        result = struct.unpack("<6B5H6H", read_buffer)
        [chan_ident, trig1_mode, trig1_pol, trig2_mode, trig2_pol, *reserved] = result[6:]
        return trig1_mode, trig1_pol, trig2_mode, trig2_pol

    @_auto_connect
    def set_trigger_io_params(self, trig1_mode=0x01, trig2_mode=0x0A):
        # MGMSG_MOT_SET_KCUBETRIGIOCONFIG 0x0523

        # OUTPUT Trigger Modes
        # 10: General purpose logic output (set using the MOD_SET_DIGOUTPUTS message).
        #
        # 11: Trigger output active (level) when motor 'in motion'. The output trigger goes high (5V) or low (0V) (as
        # set in the lTrig1Polarity and lTrig2Polarity parameters) when the stage is in motion
        #
        # 12: Trigger output active (level) when motor at 'max velocity'.
        #
        # 13: Trigger output active (pulsed) at pre-defined positions moving forward (set using StartPosFwd,
        # IntervalFwd, NumPulsesFwd and PulseWidth parameters in the SetKCubePosTrigParams message). Only one Trigger
        # port at a time can be set to this mode.
        #
        # 14: Trigger output active (pulsed) at pre-defined positions moving backwards (set using StartPosRev,
        # IntervalRev, NumPulsesRev and PulseWidth parameters in the SetKCubePosTrigParams message). Only one Trigger
        # port at a time can be set to this mode
        #
        # 15: Trigger output active (pulsed) at pre-defined positions moving forwards and backward. Only one Trigger
        # port at a time can be set to this mode.
        header = struct.pack("<6B", 0x23, 0x05, 0x16, 0x00, 0xd0, self.src)
        data = struct.pack("11H", 0x01, trig1_mode, 0x01, trig2_mode, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
        write_buffer = header + data
        self.write(write_buffer)

    @_auto_connect
    def get_trigger_pos_params(self):
        # MGMSG_MOT_REQ_KCUBEPOSTRIGPARAMS 0x0527
        write_buffer = struct.pack("<6B", 0x27, 0x05, 0x01, 0x00, self.dst, self.src)
        self.write(write_buffer)

        # MGMSG_MOT_GET_KCUBEPOSTRIGPARAMS 0x0528
        read_buffer = self.read(0x28, 0x05, req_buffer=write_buffer)
        result = struct.unpack("<8l", read_buffer[8:8 + 32])
        [start_fwd, interval_fwd, num_fwd, start_rev, interval_rev, num_rev, width_us, num_cycles] = result
        start_fwd /= self.ENC_CNT_MM
        interval_fwd /= self.ENC_CNT_MM
        start_rev /= self.ENC_CNT_MM
        interval_rev /= self.ENC_CNT_MM
        width_ms = width_us * 1e-3
        return start_fwd, interval_fwd, num_fwd, start_rev, interval_rev, num_rev, width_ms, num_cycles

    @_auto_connect
    def set_trigger_pos_params(self, start_mm, stop_mm, step_mm, width_ms,
                               trigger_mode=15):
        # MGMSG_MOT_SET_KCUBEPOSTRIGPARAMS 0x0526
        header = struct.pack("<6B", 0x26, 0x05, 0x2E, 0x00, 0xD0, self.src)

        start_enc = int(np.round(start_mm * self.ENC_CNT_MM))
        stop_enc = int(np.round(stop_mm * self.ENC_CNT_MM))
        step_enc = int(np.round(step_mm * self.ENC_CNT_MM))
        num_pulses = int(np.floor(abs(start_enc - stop_enc) / step_enc))
        width_us = int(np.round(width_ms * 1e3))
        num_cycle = 2147483647

        data = struct.pack("<H8l", 1, start_enc, step_enc, num_pulses, stop_enc,
                           step_enc, num_pulses, width_us, num_cycle)

        reserved = struct.pack("<3l", 0, 0, 0)
        write_buffer = header + data + reserved
        self.write(write_buffer)

        if trigger_mode == 13:
            self.set_trigger_io_params(1, 13)
        if trigger_mode == 14:
            self.set_trigger_io_params(1, 14)
        if trigger_mode == 15:
            self.set_trigger_io_params(1, 15)


class KDC101(AptMotor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_in_motion(self):
        status = self.status()["flags"]
        to_check = [
            status["moving forward"],
            status["moving reverse"],
            status["jogging forward"],
            status["jogging reverse"],
            status["homing"]
        ]
        return np.any(to_check)

    @property
    def position(self):
        return super().get_position()

    @position.setter
    def position(self, value_mm):
        # assuming it's in millimeters
        super().set_position(value_mm)

    # ___________________________ redundant functions just for naming convention _______________________________________
    def move_to(self, value_mm):
        self.position = value_mm

    def move_by(self, value_mm, blocking=False):
        self.move_relative(value_mm)

    def move_home(self, *args):
        self.home(True)

    def stop_profiled(self):
        self.stop()

    def get_stage_axis_info(self):
        return 0., 12., "mm", None


# %% ___________________________________________________________________________________________________________________
m = KDC101('com5')
m.trigger_on = True
m.step_mm = 0.01
m.pulse_width_ms = 1

m.set_max_vel(.1)
m.position = 0
m.stop_profiled()

# m.get_trigger_io_params()
# m.get_trigger_pos_params()
