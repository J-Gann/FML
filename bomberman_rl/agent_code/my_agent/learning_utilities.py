from matplotlib.pyplot import new_figure_manager
from sklearn import tree
from .path_utilities import move_to_nearest_coin
import events as e
import pickle
from enum import Enum
from sklearn import tree
import numpy as np
from joblib import dump, load
import os

DISCOUNT = 0.95
LEARNING_RATE = 0.5

# Learning model:
# ---------------
# 1) Initially fit action_value_models to 0
# 2) For each episode (f.e. 100 rounds):
#   3) In each step: update action_value_data for the state and action pair and store it in a dictionary throughout all episodes (use existing predictions of action_value_models)
#   4) After each episode: Fit action_value_models using (updated) action_value_data

# Notes:
# ------
# Datastructure action_value_data:
action_value_data = { "UP": {tuple(np.array([0,0,1])): 100}, "DOWN": {}, "LEFT": {}, "RIGHT": {}, "WAIT": {}, "BOMB": {} }


# ########################################################################################

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup_learning_features(self, load_model=False):
    # self.old_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    # self.new_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    # self.rewards = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    self.action_value_data = { "UP": {}, "DOWN": {}, "LEFT": {}, "RIGHT": {}, "WAIT": {}, "BOMB": {} }

    if load_model and os.path.isfile("models.joblib"): self.trees = load("models.joblib")
    elif load_model:
        print("[Warn] Cannot load model from the filesystem! Initializing untrained model.")
        self.trees = {
            "UP": tree.DecisionTreeRegressor(),
            "DOWN": tree.DecisionTreeRegressor(),
            "LEFT": tree.DecisionTreeRegressor(),
            "RIGHT": tree.DecisionTreeRegressor(),
            "WAIT": tree.DecisionTreeRegressor(),
            "BOMB": tree.DecisionTreeRegressor()
        }
        for action_tree in self.trees: self.trees[action_tree].fit([[0]], [0])
    else:
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
    # Store transitions (not the optimal solution, TODO: delete this later)
    # old_features = features_from_game_state(self, old_game_state, self_action)
    # new_features = features_from_game_state(self, new_game_state, self_action)
    # rewards = _rewards_from_events(events)
    # self.old_features[self_action].append(old_features)
    # self.new_features[self_action].append(new_features)
    # self.rewards[self_action].append(rewards)

    # Calculate new action_value for the old_game_state
    update_action_value_data(self, old_game_state, self_action, new_game_state, events)

def _action_value_data(trees, old_features, self_action, new_features, rewards):
    current_guess = trees[self_action].predict(old_features.reshape(-1, 1))
    predicted_future_reward = np.max([trees[action_tree].predict(new_features.reshape(-1, 1)) for action_tree in trees])
    predicted_action_value = trees[self_action].predict(old_features.reshape(-1, 1))
    q_value = current_guess + LEARNING_RATE * ( rewards + DISCOUNT * predicted_future_reward - predicted_action_value)
    return q_value

def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    old_features = np.array(features_from_game_state(self, old_game_state, self_action))
    new_features = np.array(features_from_game_state(self, new_game_state, self_action))
    rewards = _rewards_from_events(events)
    q_value = _action_value_data(self.trees, old_features, self_action, new_features, rewards)
    self.action_value_data[self_action][tuple(old_features)] = q_value

def train_q_model(self, saveModel=False):
    self.trees = _train_q_model(self.action_value_data)
    if saveModel: dump(self.trees, "models.joblib")

def _train_q_model(action_value_data):
    new_trees = {}
    for action in Actions:
        action = action.name

        action_value_data_action = action_value_data[action]

        features = []
        values = []

        for key in action_value_data_action:
            feature = np.array(key)
            value = action_value_data_action[key]
            features.append(feature)
            values.append(value)
            
        new_tree = tree.DecisionTreeRegressor()
        new_tree.fit(np.array(features).reshape(-1, 1), np.array(values))
        new_trees[action] = new_tree        

    return new_trees

def features_from_game_state(self, game_state, self_action):
    # Additional features for exercise 02:
    # - AgentCanPlaceBomb (1/0) => indicator whether the agent can place its bomb
    # - BoxInTheWay (1/0) => indicator wether the move of the tree would place the agent on a box
    # - AgentInBlastZone (1/0) => indicator whether the agent is in the blast zone of a placed bomb
    # - FastestMoveOutOfBlastzone (1/0) => indicator whether the move of the tree is the fastest move away from the blast zone
    # - BlastInTheWay (1/0) => indicator whether the move of the tree would place the agent in an active blast
    # - BlastInTheWayTTL (1..n) => number of steps the blast in the way of the tree is still active

    # Additional thoughts:
    # - Add features to give the agent a sense of direction (append all features of the last step, append the last n moves of the agent, append the last n moves of enemies)
    # - Add the state of enemy agents to for example predict their actions (can this be learned during the actual game?) or predict when they are in a vulnerable state for example, they are in a corner or do not have a bomb
    # - BlastZoneInTheWay (0/1) => indicator whether the action of the tree would place the agent in the blast zone of an enemy
    # - AgentPosition (x,y coordinates or node label) => indicator where the agent is. May be useful for the agent to get spacial awareness (is it in the center or one of the corners)

    if len(game_state["coins"]) == 0: return [0]
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
