import os
from joblib import dump, load
import random
from .train import state_to_features, Actions, allFit
import numpy as np


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """


# import os.path
# if os.path.isfile("models.joblib"):
#     print("loaded model")
#     self.trees = load("models.joblib")


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    if allFit(self):
        if game_state["round"] < 1000:
            rand = 1
        else:
            rand = 0
        if rand < 0.5:
            features = state_to_features(self, game_state)
            # print(features)
            # print("BOMB VALUE:" ,self.trees["BOMB"].predict(features[Actions["BOMB"].value].reshape(-1, 1)))
            # print("UP VALUE:" ,self.trees["UP"].predict(features[Actions["UP"].value].reshape(-1, 1)))
            # print("DOWN VALUE:" ,self.trees["DOWN"].predict(features[Actions["DOWN"].value].reshape(-1, 1)))
            # print("LEFT VALUE:" ,self.trees["LEFT"].predict(features[Actions["LEFT"].value].reshape(-1, 1)))
            # print("RIGHT VALUE:" ,self.trees["RIGHT"].predict(features[Actions["RIGHT"].value].reshape(-1, 1)))

            winner_index = np.argmax(
                [
                    self.trees[tree].predict(
                        features[Actions[tree].value].reshape(-1, 1)
                    )
                    for tree in self.trees
                ]
            )
            self.next_action = Actions(winner_index).name
        else:
            self.next_action = np.random.choice(
                ["RIGHT", "LEFT", "UP", "DOWN", "BOMB", "WAIT"]
            )

    else:
        self.next_action = np.random.choice(
            ["RIGHT", "LEFT", "UP", "DOWN", "BOMB", "WAIT"]
        )
    # print("Performed Action:", self.next_action)
    return self.next_action
