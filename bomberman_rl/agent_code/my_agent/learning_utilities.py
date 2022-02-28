
from sklearn import tree
from .path_utilities import FeatureExtraction
import events as e
from enum import Enum
from sklearn import tree
import numpy as np
from sklearn.tree import export_graphviz
import os
from joblib import dump, load

DISCOUNT = 0.95
LEARNING_RATE = 0.1
EPSILON = 1
MODEL_PATH = "model.joblib"
feature_names = ["move_to_coin_up", "move_to_coin_right","move_to_coin_down","move_to_coin_left","move_to_coin_wait","move_to_coin_bomb"] + ["move_to_safety_up", "move_to_safety_right","move_to_safety_down","move_to_safety_left","move_to_safety_wait","move_to_safety_bomb"] + ["move_to_box_up", "move_to_box_right","move_to_box_down","move_to_box_left","move_to_box_wait","move_to_box_bomb"] + ["in_blast_zone"] + ["move_up_allowed", "move_right_allowed","move_down_allowed","move_left_allowed","move_wait_allowed","move_bomb_allowed"] 

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup_learning_features(self, load_model=True):
    self.EPSILON = EPSILON
    self.episode_coins = 0
    self.action_value_data = { "UP": {}, "DOWN": {}, "LEFT": {}, "RIGHT": {}, "WAIT": {}, "BOMB": {} }

    if load_model and os.path.isfile(MODEL_PATH): self.trees = load(MODEL_PATH)
    else:
        if load_model: print("[WARN] Unable to load model from filesystem. Reinitializing model!")
        self.trees = {
            "UP": tree.DecisionTreeRegressor(),
            "DOWN": tree.DecisionTreeRegressor(),
            "LEFT": tree.DecisionTreeRegressor(),
            "RIGHT": tree.DecisionTreeRegressor(),
            "WAIT": tree.DecisionTreeRegressor(),
            "BOMB": tree.DecisionTreeRegressor()
            }
        for action_tree in self.trees: self.trees[action_tree].fit(np.array(np.zeros(25)).reshape(1, -1) , [0])

def _action_value_data(trees, old_features, self_action, new_features, rewards):
    current_guess = trees[self_action].predict(old_features.reshape(1, -1) )
    predicted_future_reward = np.max([trees[action_tree].predict(new_features.reshape(1, -1) ) for action_tree in trees])
    predicted_action_value = trees[self_action].predict(old_features.reshape(1, -1) )
    q_value = current_guess + LEARNING_RATE * ( rewards + DISCOUNT * predicted_future_reward - predicted_action_value)
    return q_value

def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    if e.COIN_COLLECTED in events: self.episode_coins += 1
    old_features = np.array(features_from_game_state(self, old_game_state, self_action))
    new_features = np.array(features_from_game_state(self, new_game_state, self_action))
    rewards = _rewards_from_events(events)
    q_value = _action_value_data(self.trees, old_features, self_action, new_features, rewards)
    self.action_value_data[self_action][tuple(old_features)] = q_value

def train_q_model(self, game_state, episode_rounds, save_model=True):
    round = game_state["round"]
    if round % episode_rounds == 0:
        print("Average coins:", self.episode_coins / episode_rounds, "Epsilon:", self.EPSILON)
        self.episode_coins = 0
        self.EPSILON -= 0.01
        self.trees = _train_q_model(self.action_value_data)
        if save_model: dump(self.trees, MODEL_PATH)
        for action in Actions:
            export_graphviz(
                self.trees[action.name],
                out_file="./trees/"+action.name+".dot",
                feature_names=feature_names)

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
        new_tree = tree.DecisionTreeRegressor(max_depth=20)
        new_tree.fit(np.array(features), np.array(values))
        new_trees[action] = new_tree        
    return new_trees

def features_from_game_state(self, game_state, self_action):
    feature_extraction = FeatureExtraction(game_state)
    features = []
    move = feature_extraction.FEATURE_move_to_nearest_coin().as_one_hot()
    features += move
    move = feature_extraction.FEATURE_move_out_of_blast_zone().as_one_hot()
    features += move
    move = feature_extraction.FEATURE_move_to_nearest_box().as_one_hot()
    features += move
    in_blast = feature_extraction.FEATURE_in_blast_zone()
    features += in_blast
    move = feature_extraction.FEATURE_action_possible()
    features += move
    return np.array(features)

def _rewards_from_events(events):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -5,
        e.WAITED: -1,
        e.INVALID_ACTION: -1,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
