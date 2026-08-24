import traitlets
from .racecar import Racecar

try:
    from adafruit_servokit import ServoKit
    HAS_SERVOKIT = True
except ImportError:
    HAS_SERVOKIT = False


class NvidiaRacecar(Racecar):
    """Nvidia JetRacer Racecar driver for Adafruit PCA9685 ServoKit."""
    i2c_address = traitlets.Integer(default_value=0x40)
    steering_gain = traitlets.Float(default_value=-0.65)
    steering_offset = traitlets.Float(default_value=0)
    steering_channel = traitlets.Integer(default_value=0)
    throttle_gain = traitlets.Float(default_value=0.8)
    throttle_channel = traitlets.Integer(default_value=1)

    def __init__(self, *args, **kwargs):
        super(NvidiaRacecar, self).__init__(*args, **kwargs)
        if HAS_SERVOKIT:
            self.kit = ServoKit(channels=16, address=self.i2c_address)
            self.steering_motor = self.kit.continuous_servo[self.steering_channel]
            self.throttle_motor = self.kit.continuous_servo[self.throttle_channel]
        else:
            self.kit = None
            self.steering_motor = None
            self.throttle_motor = None

    @traitlets.observe('steering')
    def _on_steering(self, change):
        if self.steering_motor:
            self.steering_motor.throttle = change['new'] * self.steering_gain + self.steering_offset

    @traitlets.observe('throttle')
    def _on_throttle(self, change):
        if self.throttle_motor:
            self.throttle_motor.throttle = change['new'] * self.throttle_gain
