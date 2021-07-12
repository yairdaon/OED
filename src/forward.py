import numpy as np

from multiplier import FourierMultiplier


class Heat(FourierMultiplier):
    """Run forward heat equation for some time"""

    def __init__(self, time, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.time = time
        self.multiplier = np.exp(-self.alpha * np.pi ** 2 * self.time * self.freqs ** 2)

    def __str__(self):
        return f'Heat operator alpha {self.alpha} time {self.time}'
