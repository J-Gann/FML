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

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup_learning_features(self, load_model=False):
    self.old_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    self.new_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    self.rewards = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }

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
    old_features = features_from_game_state(self, old_game_state, self_action)
    new_features = features_from_game_state(self, new_game_state, self_action)
    rewards = _rewards_from_events(events)
    self.old_features[self_action].append(old_features)
    self.new_features[self_action].append(new_features)
    self.rewards[self_action].append(rewards)

def train_q_model(self, saveModel=False):
    self.trees = _train_q_model(self.new_features, self.old_features, self.rewards, self.trees)
    if saveModel: dump(self.trees, "models.joblib")

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
