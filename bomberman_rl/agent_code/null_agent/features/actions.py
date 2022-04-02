from enum import Enum
import numpy as np


class Actions(Enum):
    """Author: Samuel Melm"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5
    NONE = 6

    def as_string(self):
        """Author: Samuel Melm"""
        self.name

    def as_one_hot(self):
        """Author: Samuel Melm"""
        return np.array([int(self.value == index) for index in range(6)])

    @classmethod
    def from_one_hot(cls, one_hot_vec: np.array):
        """Author: Samuel Melm"""
        if len(one_hot_vec) == len(Actions) - 1:
            one_hot_vec = np.concatenate((one_hot_vec, np.array([0])), axis=0)

        if not np.any(one_hot_vec == 1):
            return cls.NONE

        return cls(np.arange(len(cls))[one_hot_vec == 1])
