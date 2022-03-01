from .path_utilities import FeatureExtraction, Actions
import events as e
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

DISCOUNT = 0.95
LEARNING_RATE = 0.01
EPSILON = 1
EPSILON_MIN = 0.05
EPSILON_DECREASE_RATE = 0.9  # 0.96
MODEL_PATH = "model.joblib"
N_STEP = 5

feature_names = [
    "move_to__nearest_coin_up",
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
    "move_up_to_death",
    "move_right_to_death",
    "move_down_to_death",
    "move_left_to_death",
    "move_wait_to_death",
    "move_bomb_to_death",
    "bomb_good",
]


def setup_learning_features(self, load_model=True):
    self.EPSILON = EPSILON
    self.action_value_data = {
        "UP": {},
        "DOWN": {},
        "LEFT": {},
        "RIGHT": {},
        "WAIT": {},
        "BOMB": {},
    }
    self.n_states_old = []
    self.n_states_new = []
    self.n_actions = []
    self.n_rewards = []

    self.past_moves = []

    if load_model and os.path.isfile(MODEL_PATH):
        self.trees = load(MODEL_PATH)
    else:
        if load_model:
            print("[WARN] Unable to load model from filesystem. Reinitializing model!")
        self.trees = {
            "UP": RandomForestRegressor(n_estimators=100, max_depth=3),
            "DOWN": RandomForestRegressor(n_estimators=100, max_depth=3),
            "LEFT": RandomForestRegressor(n_estimators=100, max_depth=3),
            "RIGHT": RandomForestRegressor(n_estimators=100, max_depth=3),
            "WAIT": RandomForestRegressor(n_estimators=100, max_depth=3),
            "BOMB": RandomForestRegressor(n_estimators=100, max_depth=3),
        }
        for action_tree in self.trees:
            self.trees[action_tree].fit(np.array(np.zeros(33)).reshape(1, -1), [0])


def _action_value_data(trees, old_features, self_action, new_features, rewards):
    current_guess = trees[self_action].predict(old_features.reshape(1, -1))
    predicted_future_reward = np.max(
        [
            trees[action_tree].predict(new_features.reshape(1, -1))
            for action_tree in trees
        ]
    )
    predicted_action_value = trees[self_action].predict(old_features.reshape(1, -1))
    q_value_update = (
        rewards + DISCOUNT * predicted_future_reward - predicted_action_value
    )
    q_value = current_guess + LEARNING_RATE * q_value_update
    return q_value


def _action_value_data_last_step(trees, old_features, self_action, rewards):
    current_guess = trees[self_action].predict(old_features.reshape(1, -1))
    predicted_future_reward = 0
    predicted_action_value = trees[self_action].predict(old_features.reshape(1, -1))
    q_value_update = (
        rewards + DISCOUNT * predicted_future_reward - predicted_action_value
    )
    q_value = current_guess + LEARNING_RATE * q_value_update
    return q_value


def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    self.past_moves.append(self_action)

    feature_extration_old = FeatureExtraction(old_game_state)
    feature_extration_new = FeatureExtraction(new_game_state)

    old_features = np.array(features_from_game_state(self, feature_extration_old))
    new_features = np.array(features_from_game_state(self, feature_extration_new))
    rewards = _rewards_from_events(self, feature_extration_old, events, self_action)
    self.n_states_old.append(old_features)
    self.n_states_new.append(new_features)
    self.n_actions.append(self_action)
    self.n_rewards.append(rewards)
    # print("############################")
    # print("rewards", self.n_rewards)
    # print("actions", self.n_actions)

    if len(self.n_states_old) == N_STEP + 1:
        state_old = self.n_states_old.pop(0)  # get first state in the list
        action = self.n_actions.pop(0)  # get first action in the list
        reward = self.n_rewards.pop(0)  # get first reward in the list

        q_value_update = reward
        for i in range(N_STEP):
            # print("add reward", self.n_rewards[i])
            q_value_update += DISCOUNT ** (i + 1) * self.n_rewards[i]

        state_new = self.n_states_new[-1]
        q_value_update += DISCOUNT ** (N_STEP + 1) * np.max(
            [
                self.trees[action_tree].predict(state_new.reshape(1, -1))
                for action_tree in self.trees
            ]
        )
        # print("q_value", q_value)

        current_guess = self.trees[action].predict(state_old.reshape(1, -1))

        q_value = current_guess + LEARNING_RATE * q_value_update

        self.action_value_data[action][tuple(state_old)] = q_value

    # q_value = _action_value_data(self.trees, old_features, self_action, new_features, rewards)


def update_action_value_last_step(self, last_game_state, last_action, events):
    self.past_moves.append(last_action)

    feature_extration_old = FeatureExtraction(last_game_state)

    old_features = np.array(features_from_game_state(self, feature_extration_old))

    rewards = _rewards_from_events(self, feature_extration_old, events, last_action)

    self.past_moves = []

    self.n_states_old.append(old_features)
    self.n_actions.append(last_action)
    self.n_rewards.append(rewards)

    while len(self.n_states_old) > 0:

        # print("############################")
        # print("rewards", self.n_rewards)
        # print("actions", self.n_actions)

        state_old = self.n_states_old.pop(0)  # get first state in the list
        action = self.n_actions.pop(0)  # get first action in the list
        reward = self.n_rewards.pop(0)  # get first reward in the list

        q_value_update = reward
        for i in range(len(self.n_rewards)):
            # print("add reward", self.n_rewards[i])
            q_value_update += DISCOUNT ** (i + 1) * self.n_rewards[i]

        if len(self.n_states_new) > 0:
            state_new = self.n_states_new[-1]
            q_value_update += DISCOUNT ** (N_STEP + 1) * np.max(
                [
                    self.trees[action_tree].predict(state_new.reshape(1, -1))
                    for action_tree in self.trees
                ]
            )
            # print("q_value", q_value)

            current_guess = self.trees[action].predict(state_old.reshape(1, -1))

            q_value = current_guess + LEARNING_RATE * q_value_update

            self.action_value_data[action][tuple(state_old)] = q_value


def train_q_model(self, game_state, episode_rounds, save_model=True):
    round = game_state["round"]
    if round % episode_rounds == 0:
        self.EPSILON *= EPSILON_DECREASE_RATE
        self.EPSILON = max(EPSILON_MIN, self.EPSILON)
        self.trees = _train_q_model(self.action_value_data)
        if save_model:
            dump(self.trees, MODEL_PATH)


def _train_q_model(action_value_data):
    new_trees = {}
    for action in Actions:
        if action == Actions.NONE:
            continue  # Do not train a model for the "noop"
        action = action.name
        action_value_data_action = action_value_data[action]
        features = []
        values = []
        for key in action_value_data_action:
            feature = np.array(key)
            value = action_value_data_action[key]
            features.append(feature)
            values.append(value)
        new_tree = RandomForestRegressor(n_estimators=100, max_depth=3)
        features = np.array(features)
        values = np.ravel(np.array(values))
        # X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=0.33)
        new_tree.fit(features, values)
        # print("Tree score test",action, new_tree.score(X_test, y_test))
        # print("Tree score train",action, new_tree.score(X_train, y_train))

        new_trees[action] = new_tree
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
    move = feature_extraction.FEATURE_move_into_death()
    features += move
    bomb_good = feature_extraction.FEATURE_could_escape_own_bomb()
    features += bomb_good
    return np.array(features)


def _rewards_from_events(self, feature_extraction, events, action):
    action = Actions[action]
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.WAITED: -1,
        e.KILLED_SELF: -10,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 5,
    }
    reward_sum = 0
    action_to_coin = feature_extraction.FEATURE_move_to_nearest_coin()
    action_to_safety = feature_extraction.FEATURE_move_out_of_blast_zone()
    action_to_box = feature_extraction.FEATURE_move_next_to_nearest_box()
    in_blast = feature_extraction.FEATURE_in_blast_zone()
    blast_boxes = feature_extraction.FEATURE_boxes_in_agent_blast_range()
    action_to_death = feature_extraction.FEATURE_move_into_death()
    bomb_good = feature_extraction.FEATURE_could_escape_own_bomb()

    # print("Action possible", feature_extraction._action_possible(action))
    # print("action_to_coin", action_to_coin)
    # print("action_to_box", action_to_box)
    # print("in_blast", in_blast[0])
    # print("bomb_good", bomb_good[0])
    # print("blast_boxes", blast_boxes[0])
    # print("action_to_safety", action_to_safety)
    # print("action_to_death", action_to_death)

    # GENERAL MOVEMENT
    general_movement_reward = 0
    if (
        feature_extraction._action_possible(action)
        and action != Actions.WAIT
        and action != Actions.BOMB
    ):
        pass  # general_movement_reward += 1   # Did an allowed move
    else:
        general_movement_reward -= 1  # Did a not allowed move
    if (
        action == Actions.WAIT
        and (action_to_coin != Actions.NONE or action_to_box != Actions.NONE)
        and not in_blast[0]
    ):
        general_movement_reward -= 1  # Did wait although there was no need to
    if not feature_extraction._action_possible(action):
        general_movement_reward -= 1

    if len(self.past_moves) > 2:
        if action == Actions.LEFT and Actions[self.past_moves[-2]] == Actions.RIGHT:
            general_movement_reward -= 1  # Do not reward switching between two nodes
        if action == Actions.RIGHT and Actions[self.past_moves[-2]] == Actions.LEFT:
            general_movement_reward -= 1  # Do not reward switching between two nodes
        if action == Actions.UP and Actions[self.past_moves[-2]] == Actions.DOWN:
            general_movement_reward -= 1  # Do not reward switching between two nodes
        if action == Actions.DOWN and Actions[self.past_moves[-2]] == Actions.UP:
            general_movement_reward -= 1  # Do not reward switching between two nodes

    general_movement_reward = max(
        0, general_movement_reward
    )  # Do not penalize below -1

    # BOMBS
    bomb_reward = 0
    if e.BOMB_DROPPED in events:
        bomb_reward += 1  # Generally give a bonus for dropping bombs
    if e.KILLED_SELF in events:
        bomb_reward -= 1  # Strike bomb bonus if bomb killed agent
    if action == Actions.BOMB and not feature_extraction._action_possible(action):
        bomb_reward -= 1
    if action == Actions.BOMB and not bomb_good[0]:
        bomb_reward -= 1  # Strike bomb bonus if bomb will eventually kill agent
    if action == Actions.BOMB and bomb_good[0] and blast_boxes[0] > 0:
        bomb_reward += 1  # Bomb will have an effect
    else:
        bomb_reward -= 1  # Bomb will have no effect
    # bomb_reward = max(0, bomb_reward) # Do not penalize below -1

    # SAFETY
    safety_reward = 0
    if in_blast[0] and action == action_to_safety:
        safety_reward += 1  # Did move towards safety when necessary
    else:
        safety_reward -= 1  # Did not move towards safety when necessary
    if action_to_death == action:
        safety_reward -= 1  # Did move into certain death
    else:
        safety_reward += 1  # Did not move into certain death
    if (
        action == Actions.WAIT
        and not in_blast[0]
        and action_to_coin == Actions.NONE
        and action_to_box == Actions.NONE
    ):
        safety_reward += 1  # If there there are no paths to either boxes or coins and the agent is not in a plast, he is probably cornered by a blast
    # safety_reward = max(0, safety_reward) # Do not penalize below -1

    # BOXES
    boxes_reward = 0
    if not in_blast[0] and action == action_to_box:
        boxes_reward += 1  # Did move towards box when not in danger
    else:
        boxes_reward -= 1  # Did not move towards safety when necessary
    # boxes_reward = max(0, boxes_reward) # Do not penalize below -1

    # COINS
    coins_reward = 0
    if not in_blast[0] and action == action_to_coin:
        coins_reward += 1  # Did move towards coin when not in danger
    else:
        coins_reward -= 1  # Did not move towards safety when necessary
    # coins_reward = max(0, coins_reward) # Do not penalize below -1

    # print("movement reward", general_movement_reward)
    # print("bomb reward", bomb_reward)
    # print("safety reward", safety_reward)
    # print("boxes reward", boxes_reward)
    # print("coins reward", coins_reward)
    # print("######")

    movement_importance = 1
    bomb_importance = 1.2
    safety_importance = 1.3
    box_importance = 1.2
    coin_importance = 1

    return (
        movement_importance * general_movement_reward
        + bomb_importance * bomb_reward
        + safety_importance * safety_reward
        + box_importance * boxes_reward
        + coin_importance * coins_reward
    )