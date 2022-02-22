from matplotlib.pyplot import new_figure_manager
from sklearn import tree
from .path_utilities import move_to_nearest_coin
import events as e
import pickle
from enum import Enum
from sklearn import tree
import numpy as np

DISCOUNT = 0.95
LEARNING_RATE = 0.5

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup_learning_features(self):
    if not hasattr(self, "old_features"): self.old_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    if not hasattr(self, "new_features"): self.new_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    if not hasattr(self, "rewards"): self.rewards = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    self.trees = {
        "UP": tree.DecisionTreeRegressor(),
        "DOWN": tree.DecisionTreeRegressor(),
        "LEFT": tree.DecisionTreeRegressor(),
        "RIGHT": tree.DecisionTreeRegressor(),
        "WAIT": tree.DecisionTreeRegressor(),
        "BOMB": tree.DecisionTreeRegressor()
    }
    for action_tree in self.trees: self.trees[action_tree].fit([[0]], [0])

def update_transitions(self, old_game_state, self_action, new_game_state, events):
    old_features = features_from_game_state(self, old_game_state, self_action)
    new_features = features_from_game_state(self, new_game_state, self_action)
    rewards = _rewards_from_events(events)
    self.old_features[self_action].append(old_features)
    self.new_features[self_action].append(new_features)
    self.rewards[self_action].append(rewards)

def train_q_model(self):
    self.trees = _train_q_model(self.new_features, self.old_features, self.rewards, self.trees)

def _train_q_model(new_features, old_features, rewards, trees):
    new_trees = {}
    for action in Actions:
        action = action.name
        old_features_action = np.array(old_features[action])
        new_features_action = np.array(new_features[action])
        rewards_action = np.array(rewards[action])
        if old_features_action.shape[0] == 0 or new_features_action.shape[0] == 0 or rewards_action.shape[0] == 0:
            new_trees[action] = trees[action]

        else:
            current_guess = trees[action].predict(old_features_action.reshape(-1, 1))
            predicted_future_reward = np.max([trees[action_tree].predict(new_features_action.reshape(-1, 1)) for action_tree in trees])
            predicted_action_value = trees[action].predict(old_features_action.reshape(-1, 1))

            q_values = current_guess + LEARNING_RATE * ( rewards_action + DISCOUNT * predicted_future_reward - predicted_action_value)

            new_tree = tree.DecisionTreeRegressor()
            new_tree.fit(old_features_action, q_values)
            new_trees[action] = new_tree
        
    return new_trees

def features_from_game_state(self, game_state, self_action):
    move = move_to_nearest_coin(self, game_state["self"][3], game_state["coins"])
    if move == self_action: return np.array([1])
    else: return np.array([0])

def _rewards_from_events(events):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
