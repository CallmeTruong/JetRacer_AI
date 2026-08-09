import gymnasium as gym
import gym_donkeycar
import numpy as np
import logging
import cv2
import traitlets


def bgr8_to_jpeg(value, quality=85):
    return bytes(cv2.imencode('.jpg', value)[1])


class CarSimulator(traitlets.HasTraits):
    value = traitlets.Any()

    def __init__(self, env_name="donkey-warren-track-v0", verbose=True):
        super().__init__()
        self.verbose = verbose
        self.env = gym.make(env_name)
        self.obs = self.env.reset()[0]
        self.value = bgr8_to_jpeg(self.obs)
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )

    def run(self, steering, throttle):
        action = np.array([steering, throttle])
        results = self.env.step(action)
        if self.verbose:
            self.logger.info(
                "steering=%.2f | throttle=%.2f | reward=%.3f",
                steering,
                throttle,
                results[1]
            )

        self.obs = results[0]
        self.value = bgr8_to_jpeg(self.obs)
        
        return dict(
                    obs = results[0],
                    reward = results[1],
                    terminated = results[2],
                    truncated = results[3],
                    info = results[4]
                    )

    def reset(self):
        self.obs = self.env.reset()[0]
        self.value = bgr8_to_jpeg(self.obs)
        return self.obs

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


    