import MotorClassFromAptProtocolConnor as apt
import time

motor = apt.KDC101('com5')

# %%____________________________________________________________________________________________________________________
# if homing is necessary
# motor.move_home()
# while motor.is_in_motion:
#     time.sleep(.1)
print("done with homing move")

# %%____________________________________________________________________________________________________________________
motor.set_max_vel(.1)  # 100 um /s
motor.step_mm = .1  # every 100 um (every 1 s)
motor.pulse_width_ms = 1  # pulse width = 1 ms
motor.trigger_on = True  # turn on the trigger

# %%____________________________________________________________________________________________________________________
# repeatedly move forwards and backwards :)
while True:
    motor.position = 5
    print("moving forward")
    while motor.is_in_motion:
        time.sleep(.1)
    print("moving backwards")
    motor.position = 0
    while motor.is_in_motion:
        time.sleep(.1)
