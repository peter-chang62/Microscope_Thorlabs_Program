"""This file should handle all the direct hardware interfacing"""


class Motor:
    """
    This class expects a motor instance. You can incorporate a motor and pass
    it to here by creating a motor class with the following attributes
    and methods:

    methods:

        1. get_stage_axis_info: returns min_pos, max_pos, units, pitch

        note: the important thing is the min_pos and max_pos, if you like you
        can return None for the other two

        2. move_by(value, blocking): moves the motor, blocking doesn't need
        to do anything

        3. move_home(blocking): homes the motor, blocking doesn't need to do
        anything

        4. stop_profiled(): stops the motor

    attributes:

        1.  position

            note: position needs to be a property, where the getter returns
            the current position, and the setter moves the motor to the new
            position

        2. is_in_motion
    """

    def __init__(self, motor):
        self.motor = motor

        self._min_pos, self._max_pos, self._units, self._pitch = \
            self.motor.get_stage_axis_info()

    @property
    def position_mm(self):
        # returns the motor position
        return self.motor.position

    @property
    def max_pos_mm(self):
        return self._max_pos

    @property
    def min_pos_mm(self):
        return self._min_pos

    @property
    def units(self):
        return self._units

    @position_mm.setter
    def position_mm(self, value):
        # setting the motor position tells the motor to move in absolute mode
        # and is non-blocking
        self.motor.position = value

    @property
    def is_in_motion(self):
        # is the motor currently in motion?
        return self.motor.is_in_motion

    def move_by(self, value, blocking=False):
        # move relative
        self.motor.move_by(value, blocking)

    def home_motor(self, blocking=False):
        # home the motor
        self.motor.move_home(blocking)

    def stop_motor(self):
        self.motor.stop_profiled()
