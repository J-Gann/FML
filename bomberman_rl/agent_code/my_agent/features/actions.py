from enum import Enum


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5
    NONE = 6

    def as_string(self):
        self.name

    def as_one_hot(self):
        return [int(self.value == index) for index in range(6)]