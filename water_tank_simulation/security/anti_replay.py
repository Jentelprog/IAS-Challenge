# security/anti_replay.py
from typing import Tuple


class ReplayDetector:
    def __init__(self, window: int = 5, round_digits: int = 3):
        """
        window: how many identical consecutive samples to treat as replay
        round_digits: rounding precision for floats when comparing
        """
        self.window = window
        self.round_digits = round_digits
        self.last_sample = None
        self.repeat_count = 0

    def _round_sample(self, sample_tuple: Tuple[float, ...]) -> Tuple:
        return tuple(round(float(x), self.round_digits) for x in sample_tuple)

    def check(self, sample_tuple: Tuple[float, ...]) -> bool:
        """
        Return True if this sample looks like a replay (repeated >= window).

        sample_tuple: (level, flow_in, flow_out, pressure, pump_current)
        """
        rounded = self._round_sample(sample_tuple)

        if self.last_sample is not None and rounded == self.last_sample:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
            self.last_sample = rounded

        return self.repeat_count >= self.window
