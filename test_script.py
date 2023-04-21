# import MotorClassFromAptProtocolConnor as apt
# import time
#
# motor = apt.KDC101('com5')
#
# %% ------------------------------------------------------------------
# if homing is necessary
# motor.move_home()
# while motor.is_in_motion:
#     time.sleep(.1)
# print("done with homing move")
#
# %% --------------------------------------------------------------------------
# motor.set_max_vel(.1)  # 100 um /s
# motor.step_mm = .1  # every 100 um (every 1 s)
# motor.pulse_width_ms = 1  # pulse width = 1 ms
# motor.trigger_on = True  # turn on the trigger
#
# %% --------------------------------------------------------------------------
# repeatedly move forwards and backwards :)
# while True:
#     motor.position = 5
#     print("moving forward")
#     while motor.is_in_motion:
#         time.sleep(.1)
#     print("moving backwards")
#     motor.position = 0
#     while motor.is_in_motion:
#         time.sleep(.1)

# %% --------------------------------------------------------------------------
# good
class Aclass:
    def __init__(self):
        self._a = 0

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, val):
        self._a = val


class Bclass:
    def __init__(self, A):
        assert isinstance(A, Aclass)
        A: Aclass
        self.A = A

    @property
    def a(self):
        return self.A.a

    @a.setter
    def a(self, val):
        self.A.a = val
