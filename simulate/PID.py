import time

class PIDController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.integral   = 0.0
        self.last_error = None
        self.prev_raw_x = None
        self.last_time  = time.time()
        self.smoothed_x = 0.0

    def update(self, raw_x, kp, ki, kd, alpha, bias):
        now = time.time()
        dt  = now - self.last_time
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05
        self.last_time = now

        # 1. Low-pass filter
        self.smoothed_x = alpha * raw_x + (1.0 - alpha) * self.smoothed_x
        error = self.smoothed_x

        # 2. Integral
        self.integral += error * dt
        max_int = 0.3 / (ki + 1e-9)
        self.integral = max(-max_int, min(max_int, self.integral))

        # 3. Derivative
        if self.prev_raw_x is None:
            derivative = 0.0
        else:
            derivative = (raw_x - self.prev_raw_x) / dt
        self.prev_raw_x = raw_x

        # 4. PID Output
        steering = kp * error + ki * self.integral + kd * derivative + bias
        return max(-1.0, min(1.0, steering))
