
from argparse import Action
from tkinter.messagebox import NO
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
EPSILON_DECREASE_RATE = 0.96
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

    # Go to a box if no path to a coin exists
    if action_to_coin == Actions.NONE and action == action_to_box: rewards += 1
    elif action == action_to_box: rewards -= 1
    # Go to a coin if one exists (whether or not a path to a box exists)
    if action_to_coin != Actions.NONE and action == action_to_coin: rewards += 1
    elif action == action_to_coin: rewards -= 1
    # If no path to a coin exists and there are boxes in the blast zone and a bomb can be placed and the bomb wont kill the agent, place a bomb
    if action_to_coin == Actions.NONE and blast_boxes > 0 and can_place_bomb and bomb_good and action == Actions.BOMB: rewards += 1#blast_boxes
    elif action == Actions.BOMB: rewards -= 1
    # If no path to neither a coin nor a box exists, wait
    if action_to_coin == Actions.NONE and action_to_box == Actions.NONE and action == Actions.WAIT: rewards += 1
    elif action == Actions.WAIT: rewards -= 1
    # If the agent is in a blast zone, move to safety
    if in_blast and action == action_to_safety: rewards += 1
    if in_blast and action != action_to_safety: rewards -= 1
    if not feature_extraction._action_possible(action): rewards -= 1



    #print(action_to_coin, action_to_box, blast_boxes, in_blast, action_to_safety, bomb_good)
    #print(rewards)
    print(rewards)
    if rewards == 0: return -1 # Do some shit
    return rewards # + 10 * score_diff
"""

    # GENERAL MOVEMENT
    general_movement_reward = 0
    if feature_extraction._action_possible(action) and action != Actions.WAIT and action != Actions.BOMB: pass# general_movement_reward += 1   # Did an allowed move
    else: general_movement_reward -= 1  # Did a not allowed move
    if action == Actions.WAIT and (action_to_coin != Actions.NONE or action_to_box != Actions.NONE) and not in_blast[0]: general_movement_reward -= 1  # Did wait although there was no need to
    if action == Actions.WAIT and action_to_coin != Actions.NONE  and not in_blast[0] and feature_extraction._action_possible(Actions.BOMB): general_movement_reward -= 1  # Did wait although we could have placed a bomb next to a crate
    if not feature_extraction._action_possible(action): general_movement_reward -= 1

    # TODO: Detect circular actions in general
    if len(self.past_moves) > 2:
        if action == Actions.LEFT and Actions[self.past_moves[-2]] == Actions.RIGHT: general_movement_reward -= 2   # Do not reward switching between two nodes
        if action == Actions.RIGHT and Actions[self.past_moves[-2]] == Actions.LEFT: general_movement_reward -= 2   # Do not reward switching between two nodes
        if action == Actions.UP and Actions[self.past_moves[-2]] == Actions.DOWN: general_movement_reward -= 2   # Do not reward switching between two nodes
        if action == Actions.DOWN and Actions[self.past_moves[-2]] == Actions.UP: general_movement_reward -= 2   # Do not reward switching between two nodes

    general_movement_reward = max(-1, general_movement_reward) # Do not penalize below -1

    # BOMBS
    bomb_reward = 0
    if e.BOMB_DROPPED in events: pass#bomb_reward += 1   # Generally give a bonus for dropping bombs
    if e.KILLED_SELF in events: bomb_reward -= 1   # Strike bomb bonus if bomb killed agent
    if action == Actions.BOMB and not feature_extraction._action_possible(action): bomb_reward -= 1
    if action == Actions.BOMB and not bomb_good[0]: bomb_reward -= 1   # Strike bomb bonus if bomb will eventually kill agent
    if action == Actions.BOMB and bomb_good[0] and blast_boxes[0] > 0: bomb_reward += 1 # Bomb will have an effect
    else: bomb_reward -= 1 # Bomb will have no effect
    if len(self.last_bomb_position) > 2:
        if action == Actions.BOMB and self.last_bomb_position[-2] == feature_extraction.agent_index: bomb_reward -= 1 # Placed a bomb at the same place as previously

    bomb_reward = max(0, bomb_reward) # Do not penalize below -1

    # SAFETY
    safety_reward = 0
    if in_blast[0] and action == action_to_safety: safety_reward += 1  # Did move towards safety when necessary
    else: safety_reward -= 1   # Did not move towards safety when necessary
    if action == Actions.WAIT and not in_blast[0] and action_to_coin == Actions.NONE and action_to_box == Actions.NONE: safety_reward += 1  # If there there are no paths to either boxes or coins and the agent is not in a plast, he is probably cornered by a blast
    safety_reward = max(0, safety_reward) # Do not penalize below -1

    # BOXES
    boxes_reward = 0
    if not in_blast[0] and action == action_to_box: boxes_reward += 1  # Did move towards box when not in danger
    else: boxes_reward -= 1   # Did not move towards safety when necessary
    boxes_reward = max(0, boxes_reward) # Do not penalize below -1

    # COINS
    coins_reward = 0
    if not in_blast[0] and action == action_to_coin: coins_reward += 1  # Did move towards coin when not in danger
    else: coins_reward -= 1   # Did not move towards safety when necessary
    coins_reward = max(0, coins_reward) # Do not penalize below -1

    movement_importance = 1
    bomb_importance = 4 # can place bomb only every 4 rounds
    safety_importance = 4
    box_importance = 1
    coin_importance = 1

    other_rewards = 0

    if e.SURVIVED_ROUND in events:
        #other_rewards += 100
        pass
    if e.COIN_COLLECTED in events:
        other_rewards += 1
    return movement_importance * general_movement_reward + bomb_importance * bomb_reward + safety_importance * safety_reward + box_importance * boxes_reward + coin_importance * coins_reward + other_rewards
"""