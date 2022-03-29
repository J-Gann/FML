from typing import List
from .learning_utilities import (
    setup_learning_features,
    train_q_model,
    update_action_value_data,
    update_action_value_last_step,
)


def setup_training(self):
    # Some initialization for the regression models and learning algorithm. This loads an existing model from the filesystem when possible.
    setup_learning_features(self)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    if old_game_state == None or self_action == None:
        return
    update_action_value_data(self, old_game_state,
                             self_action, new_game_state, events)   # Update the experience of the agent based on the transition


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Update the experience of the agent based on the transition
    update_action_value_last_step(self, last_game_state, last_action, events)
    # After each 5th round, train a new model for predicting the action values
    train_q_model(self, last_game_state, 5)
