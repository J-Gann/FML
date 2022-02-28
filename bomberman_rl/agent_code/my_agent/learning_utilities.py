
from sklearn import tree
from .path_utilities import FeatureExtraction, Actions
import events as e
from enum import Enum
from sklearn import tree
import numpy as np
from sklearn.tree import export_graphviz
import os
from joblib import dump, load


DISCOUNT = 0.95
LEARNING_RATE = 0.05
EPSILON = 0.1
EPSILON_DECREASE_RATE = 1
MODEL_PATH = "model.joblib"

feature_names = ["move_to__nearest_coin_up",
"move_to__nearest_coin_right",
"move_to__nearest_coin_down",
"move_to__nearest_coin_left",
"move_to__nearest_coin_wait",
"move_to__nearest_coin_bomb",
"move_to_safety_up",
"move_to_safety_right",
"move_to_safety_down",
"move_to_safety_left",
"move_to_safety_wait",
"move_to_safety_bomb",
"move_to_nearest_box_up",
"move_to_nearest_box_right",
"move_to_nearest_box_down",
"move_to_nearest_box_left",
"move_to_nearest_box_wait",
"move_to_nearest_box_bomb",
 "in_blast_zone", "boxes_in_range"]


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
        for action_tree in self.trees: self.trees[action_tree].fit(np.array(np.zeros(20)).reshape(1, -1) , [0])

def _action_value_data(trees, old_features, self_action, new_features, rewards):
    current_guess = trees[self_action].predict(old_features.reshape(1, -1) )
    predicted_future_reward = np.max([trees[action_tree].predict(new_features.reshape(1, -1) ) for action_tree in trees])
    predicted_action_value = trees[self_action].predict(old_features.reshape(1, -1) )
    q_value_update = rewards + DISCOUNT * predicted_future_reward - predicted_action_value
    q_value = current_guess + LEARNING_RATE * q_value_update
    return q_value

def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    if e.COIN_COLLECTED in events: self.episode_coins += 1
    old_features = np.array(features_from_game_state(self, old_game_state)[0])
    new_features, distance_to_bomb = features_from_game_state(self, new_game_state) 
    new_features = np.array(new_features)
    rewards = _rewards_from_events(self, events, distance_to_bomb)
    q_value = _action_value_data(self.trees, old_features, self_action, new_features, rewards)
    self.action_value_data[self_action][tuple(old_features)] = q_value

def train_q_model(self, game_state, episode_rounds, save_model=True):
    round = game_state["round"]
    if round % episode_rounds == 0:
        print("Average coins:", self.episode_coins / episode_rounds, "Epsilon:", self.EPSILON)
        self.episode_coins = 0
        self.EPSILON *= EPSILON_DECREASE_RATE
        self.trees = _train_q_model(self.action_value_data)
        if save_model: dump(self.trees, MODEL_PATH)
        for action in Actions:
            if action == Actions.NONE: continue # Do not train a model for the "noop"
            export_graphviz(
                self.trees[action.name],
                out_file="./trees/"+action.name+".dot",
                feature_names=feature_names)

def _train_q_model(action_value_data):
    new_trees = {}
    for action in Actions:
        if action == Actions.NONE: continue # Do not train a model for the "noop"
        action = action.name
        action_value_data_action = action_value_data[action]
        features = []
        values = []
        for key in action_value_data_action:
            feature = np.array(key)
            value = action_value_data_action[key]
            features.append(feature)
            values.append(value)
        new_tree = tree.DecisionTreeRegressor(max_depth=4)
        new_tree.fit(np.array(features), np.array(values))
        new_trees[action] = new_tree        
    return new_trees

def features_from_game_state(self, game_state):
    feature_extraction = FeatureExtraction(game_state)
    features = []
    move = feature_extraction.FEATURE_move_to_nearest_coin().as_one_hot()
    features += move
    move = feature_extraction.FEATURE_move_out_of_blast_zone().as_one_hot()
    features += move
    move = feature_extraction.FEATURE_move_next_to_nearest_box().as_one_hot()
    features += move
    in_blast = feature_extraction.FEATURE_in_blast_zone()
    features += in_blast
    #move = feature_extraction.FEATURE_action_possible(self_action)
    #features += move
    blast_boxes = feature_extraction.FEATURE_boxes_in_agent_blast_range()
    features += blast_boxes
    #print(features)
    
    return np.array(features), feature_extraction._distance_to_nearest_bomb()

SUCCESSFUL_BOMB = "SUCCESSFUL_BOMB"
def _rewards_from_events(self, events, distance_to_bomb):
    distance_to_bomb = distance_to_bomb[0]
    game_rewards = {
        e.COIN_COLLECTED: 0,
        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.WAITED: -5,
        e.KILLED_SELF: -10,
        e.INVALID_ACTION: 0,
        e.BOMB_DROPPED: 0,
        SUCCESSFUL_BOMB: 0
    }

    reward_sum = -1 if distance_to_bomb == np.inf else distance_to_bomb

    print(reward_sum, end=", ")

    if self.bomb_timer == 0:
        print("SUCCESSFUL BOMB")
        events += [SUCCESSFUL_BOMB]

    self.bomb_timer = -1

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

        if event == e.BOMB_DROPPED:
            self.bomb_timer = 6
   
    return reward_sum
    