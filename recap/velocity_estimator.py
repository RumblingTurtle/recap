import numpy as np


class VelocityEstimator:
    """
    Estimates velocity using finite difference
    """

    def __init__(self, in_shape: int, window_size: int, sample_dt: float):
        """
        Initialize the VelocityEstimator object.

        Args:
            in_shape (int): Dimensionality of the input position.
            window_size (int): Size of the sliding window for averaging.
            sample_dt (float): Time interval between position samples.
        """
        self.sample_dt = sample_dt
        self.window_size = window_size
        self.in_shape = in_shape
        self.reset()

    def __call__(self, position: np.ndarray):
        """Return the average finite difference velocity of the last self.window_size position samples

        Args:
            position (np.ndarray): current position sample

        Returns:
            np.ndarray: velocity estimate
        """
        self.sample_count = min(self.sample_count + 1, self.window_size)
        self.samples[1:] = self.samples[:-1]
        self.samples[0] = position

        if self.sample_count <= 1:
            return np.zeros(3)

        return np.mean(np.diff(self.samples[: self.sample_count], axis=0) / self.sample_dt, axis=0)

    def reset(self):
        self.samples = np.zeros((self.window_size, self.in_shape))
        self.sample_count = 0
