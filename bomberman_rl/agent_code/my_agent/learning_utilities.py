
from .path_utilities import FeatureExtraction, Actions
import events as e
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import copy

DISCOUNT = 0.95
LEARNING_RATE = 0.01
EPSILON = 1
EPSILON_MIN = 0.05
EPSILON_DECREASE_RATE = 0.95
MODEL_PATH = "model.joblib"
N_STEP = 6

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
 "in_blast_zone", 
 "move_up_possible",
"move_right_possible",
"move_down_possible",
"move_left_possible",
"move_wait_possible",
"move_bomb_possible",
 "boxes_in_range",
 "bomb_good"]

def setup_learning_features(self, load_model=True):
    self.EPSILON = EPSILON
    self.action_value_data = { "UP": {}, "DOWN": {}, "LEFT": {}, "RIGHT": {}, "WAIT": {}, "BOMB": {} }
    self.n_states_old = []
    self.n_states_new = []
    self.n_actions = []
    self.n_rewards = []

    self.past_moves = []
    self.last_bomb_position = []

    self.rewards_round = 0
    self.q_updates = 0
    self.q_updates_sum = 0

    if load_model and os.path.isfile(MODEL_PATH): self.trees = load(MODEL_PATH)
    else:
        if load_model: print("[WARN] Unable to load model from filesystem. Reinitializing model!")
        self.trees = {
            "UP": RandomForestRegressor(max_depth=4, bootstrap=False),
            "DOWN": RandomForestRegressor(max_depth=4, bootstrap=False),
            "LEFT": RandomForestRegressor(max_depth=4, bootstrap=False),
            "RIGHT": RandomForestRegressor(max_depth=4, bootstrap=False),
            "WAIT": RandomForestRegressor(max_depth=4, bootstrap=False),
            "BOMB": RandomForestRegressor(max_depth=4, bootstrap=False)
            }
        for action_tree in self.trees: self.trees[action_tree].fit(np.array(np.zeros(27)).reshape(1, -1) , [0])

def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    self.past_moves.append(self_action)
    if Actions[self_action] == Actions.BOMB: self.last_bomb_position.append(old_game_state["self"][3])

    score_diff = new_game_state["self"][1] - old_game_state["self"][1]

    feature_extration_old = FeatureExtraction(old_game_state)
    feature_extration_new = FeatureExtraction(new_game_state)

    old_features = np.array(features_from_game_state(self, feature_extration_old))
    new_features = np.array(features_from_game_state(self, feature_extration_new))
    rewards = _rewards_from_events(self, feature_extration_old, events, self_action, score_diff)

    q_value_old = self.trees[self_action].predict(old_features.reshape(1, -1))
    q_value_new = rewards + DISCOUNT * np.max([self.trees[action_tree].predict(new_features.reshape(1, -1) ) for action_tree in self.trees])
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[self_action][tuple(old_features)] = q_value_old + q_value_update

    self.q_updates_sum += q_value_old + q_value_new
    self.q_updates += 1

    """
    self.rewards_round += rewards

    self.n_states_old.append(old_features)
    self.n_states_new.append(new_features)
    self.n_actions.append(self_action)
    self.n_rewards.append(rewards)

    if len(self.n_states_old) == N_STEP + 1:
        state_old = self.n_states_old.pop(0)    # get first state in the list
        action = self.n_actions.pop(0)    # get first action in the list
        reward = self.n_rewards.pop(0)    # get first reward in the list

        q_value_update = reward
        for i in range(N_STEP):
            q_value_update += DISCOUNT**(i+1) * self.n_rewards[i]

        state_new = self.n_states_new[-1]
        q_value_update += DISCOUNT**(N_STEP+1) * np.max([self.trees[action_tree].predict(state_new.reshape(1, -1) ) for action_tree in self.trees])
        current_guess = self.trees[action].predict(state_old.reshape(1, -1))
        q_value = current_guess + LEARNING_RATE * (q_value_update - current_guess)
        self.action_value_data[action][tuple(state_old)] = q_value

        self.q_updates_sum += q_value_update - current_guess
        self.q_updates += 1
    """

def update_action_value_last_step(self, last_game_state, last_action, events):
    print("Score:", last_game_state["self"][1])
    self.past_moves.append(last_action)
    if Actions[last_action] == Actions.BOMB: self.last_bomb_position.append(last_game_state["self"][3])
    feature_extration_old = FeatureExtraction(last_game_state)
    old_features = np.array(features_from_game_state(self, feature_extration_old))
    rewards = _rewards_from_events(self, feature_extration_old, events, last_action, 0)
    self.rewards_round += rewards
    self.rewards_round = 0
    self.past_moves = []
    self.last_bomb_position = []

    q_value_old = self.trees[last_action].predict(old_features.reshape(1, -1))
    q_value_new = rewards + 0
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[last_action][tuple(old_features)] = q_value_old + q_value_update

    self.q_updates_sum += q_value_old + q_value_new
    self.q_updates += 1

"""
    self.n_states_old.append(old_features)
    self.n_actions.append(last_action)
    self.n_rewards.append(rewards)


    while len(self.n_states_old) > 0:
        state_old = self.n_states_old.pop(0)    # get first state in the list
        action = self.n_actions.pop(0)    # get first action in the list
        reward = self.n_rewards.pop(0)    # get first reward in the list

        q_value_update = reward
        for i in range(len(self.n_rewards)):
            q_value_update += DISCOUNT**(i+1) * self.n_rewards[i]

        if len(self.n_states_new) > 0:
            state_new = self.n_states_new[-1]
            q_value_update += DISCOUNT**(N_STEP+1) * np.max([self.trees[action_tree].predict(state_new.reshape(1, -1) ) for action_tree in self.trees])
            current_guess = self.trees[action].predict(state_old.reshape(1, -1))
            q_value = current_guess + LEARNING_RATE * (q_value_update - current_guess)
            self.action_value_data[action][tuple(state_old)] = q_value

            self.q_updates_sum += q_value_update - current_guess
            self.q_updates += 1
"""

def train_q_model(self, game_state, episode_rounds, save_model=True):
    round = game_state["round"]
    if round % episode_rounds == 0:
        self.EPSILON *= EPSILON_DECREASE_RATE
        self.EPSILON = max(EPSILON_MIN, self.EPSILON)
        self.trees = _train_q_model(self, self.action_value_data)
        if save_model: dump(self.trees, MODEL_PATH)

def _train_q_model(self, action_value_data):
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
        new_tree = RandomForestRegressor(max_depth=4, bootstrap=False)
        features = np.array(features)
        values = np.ravel(np.array(values))
        X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=0.33)
        new_tree.fit(X_train, y_train)
        print("Epsilon:", self.EPSILON)
        print("Tree score test",action, new_tree.score(X_test, y_test))
        print("Tree score train",action, new_tree.score(X_train, y_train))

        new_trees[action] = new_tree 
    
    print("Average q_value updates:", (self.q_updates_sum / self.q_updates)[0])
    self.q_updates_sum = 0
    self.q_updates = 0     
    return new_trees

def features_from_game_state(self, feature_extraction):
    features = []
    move = feature_extraction.FEATURE_move_to_nearest_coin().as_one_hot()
    features += move
    move = feature_extraction.FEATURE_move_out_of_blast_zone().as_one_hot()
    features += move
    move = feature_extraction.FEATURE_move_next_to_nearest_box().as_one_hot()
    features += move
    in_blast = feature_extraction.FEATURE_in_blast_zone()
    features += in_blast
    move = feature_extraction.FEATURE_actions_possible()
    features += move
    blast_boxes = feature_extraction.FEATURE_boxes_in_agent_blast_range()
    features += blast_boxes
    bomb_good = feature_extraction.FEATURE_could_escape_own_bomb()
    features += bomb_good
    return np.array(features)

def _rewards_from_events(self, feature_extraction, events, action, score_diff):
    action = Actions[action]
    rewards = 0

    action_to_coin = feature_extraction.FEATURE_move_to_nearest_coin()
    action_to_safety = feature_extraction.FEATURE_move_out_of_blast_zone()
    action_to_box = feature_extraction.FEATURE_move_next_to_nearest_box()
    in_blast = feature_extraction.FEATURE_in_blast_zone()[0]
    blast_boxes = feature_extraction.FEATURE_boxes_in_agent_blast_range()[0]
    bomb_good = feature_extraction.FEATURE_could_escape_own_bomb()[0]

    can_place_bomb = feature_extraction._action_possible(Actions.BOMB)

    #print("safety", action_to_safety)
    #print("box", action_to_box)
    #print("blast boxes", blast_boxes)
    #print("bomb good", bomb_good)
    if action_to_safety != Actions.NONE:
        if action == action_to_safety: rewards += 1
        else: rewards -= 1
    elif action_to_coin != Actions.NONE:
        if action == action_to_coin: rewards += 1
        else: rewards -= 1
    elif can_place_bomb and bomb_good and blast_boxes > 0:
        if action == Actions.BOMB: rewards += 1
        else: rewards -= 1
    elif action_to_box != Actions.NONE:
        if action == action_to_box: rewards += 1
        else: rewards -= 1
    else:
        rewards -= 1 # Do something

    print(rewards)
    return rewards + 10 * score_diff