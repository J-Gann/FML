import numpy as np

def setup(self):
    # Initialize to force exploitation. This gets overwritten by the training setup to 1 in case of a training session
    self.exploration_probability = 0

def act(self, game_state: dict):
    # Exploit or explore according to the exploration probability
    if np.random.randint(0, 1) < self.exploration_probability: return explore(self, game_state)
    else: return exploit(self, game_state)


def explore(self, game_state):
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT'])

def exploit(self, game_state):

    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT']) #TODO: Use prediction

