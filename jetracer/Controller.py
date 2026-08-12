import math
import time

class KalmanFilter1D:
    """
    1D Kalman Filter estimating position x and velocity v = dx/dt.
    Provides optimal noise filtering and accurate derivative estimation.
    """
    def __init__(self, q_process=0.01, r_measure=0.1):
        self.q_process = q_process  # Process noise (how fast target position can change)
        self.r_measure = r_measure  # Measurement noise (camera frame jitter / model noise)
        self.reset()

    def reset(self, initial_x=0.0):
        self.x = initial_x  # State estimate [position]
        self.v = 0.0        # State estimate [velocity]
        self.p00 = 1.0      # Covariance matrix P
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1.0
        self.initialized = False

    def update(self, z, dt):
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05

        if not self.initialized:
            self.x = z
            self.v = 0.0
            self.initialized = True
            return self.x, self.v

        # 1. Predict state extrapolation
        x_pred = self.x + self.v * dt
        v_pred = self.v

        # Predict covariance extrapolation
        p00_pred = self.p00 + dt * (self.p01 + self.p10) + dt * dt * self.p11 + self.q_process
        p01_pred = self.p01 + dt * self.p11
        p10_pred = self.p10 + dt * self.p11
        p11_pred = self.p11 + self.q_process

        # 2. Measurement Update
        y = z - x_pred  # Innovation
        s = p00_pred + self.r_measure  # Innovation covariance

        k0 = p00_pred / s  # Kalman gain for position
        k1 = p10_pred / s  # Kalman gain for velocity

        # Update state
        self.x = x_pred + k0 * y
        self.v = v_pred + k1 * y

        # Update covariance
        self.p00 = p00_pred - k0 * p00_pred
        self.p01 = p01_pred - k0 * p01_pred
        self.p10 = p10_pred - k1 * p00_pred
        self.p11 = p11_pred - k1 * p01_pred

        return self.x, self.v


class PIDController:
    def __init__(self, q_process=0.01, r_measure=0.1):
        self.kf = KalmanFilter1D(q_process=q_process, r_measure=r_measure)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_time = time.time()
        self.kf.reset()

    @property
    def smoothed_x(self):
        return self.kf.x

    def update(self, raw_x, kp, ki, kd, alpha=0.7, bias=0.0, r_measure=None, q_process=None):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05
        self.last_time = now

        # Dynamic measurement noise tuning if provided
        if r_measure is not None:
            self.kf.r_measure = r_measure
        if q_process is not None:
            self.kf.q_process = q_process
        elif alpha is not None and alpha > 0:
            # Map legacy alpha slider (0.1..1.0) to measurement noise R: high alpha -> low R noise
            self.kf.r_measure = max(0.001, (1.0 - alpha) * 0.2)

        # 1. Kalman Filter update -> optimal position x and velocity v
        filtered_x, estimated_v = self.kf.update(raw_x, dt)
        error = filtered_x

        # 2. Integral with anti-windup clamping
        self.integral += error * dt
        max_int = 0.3 / (ki + 1e-9)
        self.integral = max(-max_int, min(max_int, self.integral))

        # 3. Derivative from Kalman estimated velocity (noise-free)
        derivative = estimated_v

        # 4. PID Output calculation
        steering = kp * error + ki * self.integral + kd * derivative + bias
        return max(-1.0, min(1.0, steering))


class StanleyController:
    """
    Stanley Steering Controller using Kalman-filtered CTE (Cross-Track Error)
    and Kalman estimated lateral velocity for heading error estimation.
    
    Formula:
        delta = heading_error + arctan(k * CTE / (speed + epsilon)) + bias
    """
    def __init__(self, q_process=0.01, r_measure=0.1):
        self.kf = KalmanFilter1D(q_process=q_process, r_measure=r_measure)
        self.reset()

    def reset(self):
        self.last_time = time.time()
        self.kf.reset()

    @property
    def smoothed_x(self):
        return self.kf.x

    def update(self, raw_x, k=1.2, base_throttle=0.20, brake_gain=0.10, bias=0.0, alpha=0.7):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05
        self.last_time = now

        # Update measurement noise R from alpha slider
        if alpha is not None and alpha > 0:
            self.kf.r_measure = max(0.001, (1.0 - alpha) * 0.2)

        # 1. Kalman Filter update -> optimal position x (CTE) and velocity v (lateral drift rate)
        cte, vx = self.kf.update(raw_x, dt)

        # 2. Heading Error Estimate (psi): heading angle in radians from lateral drift rate
        heading_error = math.atan2(vx, 1.0)

        # 3. Stanley Steering Angle Formula
        speed = max(0.05, base_throttle)
        cte_correction = math.atan2(k * cte, speed + 0.05)
        
        steering = heading_error + cte_correction + bias
        steering = max(-1.0, min(1.0, steering))

        # 4. Adaptive Throttle
        dyn_throttle = base_throttle - brake_gain * abs(steering)
        dyn_throttle = max(0.05, min(0.8, dyn_throttle))


        return steering, dyn_throttle


