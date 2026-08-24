"""
1D Kalman Filter for position & velocity state estimation.
"""

class KalmanFilter1D:
    """
    1D Kalman Filter estimating position x and velocity v = dx/dt.
    Provides optimal noise filtering and accurate derivative estimation.
    """

    def __init__(self, q_process=0.01, r_measure=0.1):
        self.q_process = q_process
        self.r_measure = r_measure
        self.reset()

    def reset(self, initial_x=0.0):
        self.x = initial_x
        self.v = 0.0
        self.p00 = 1.0
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
        y = z - x_pred
        s = p00_pred + self.r_measure

        k0 = p00_pred / s
        k1 = p10_pred / s

        self.x = x_pred + k0 * y
        self.v = v_pred + k1 * y

        self.p00 = p00_pred - k0 * p00_pred
        self.p01 = p01_pred - k0 * p01_pred
        self.p10 = p10_pred - k1 * p00_pred
        self.p11 = p11_pred - k1 * p01_pred

        return self.x, self.v
