import numpy as np

def setup(self):
    # Initialize to force exploitation. This gets overwritten by the training setup to 1 in case of a training session
    self.exploration_probability = 0

def act(self, game_state: dict):
    # Exploit or explore according to the exploration probability
    if np.random.randint(0, 1) < self.exploration_probability: explore(self)
    else: exploit(self)


def explore(self):
    pass

def exploit(self):
    pass
