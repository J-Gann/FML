from agent_code.my_agent.features.feature import (
    AgentInBlastZone,
    BoxesInBlastRange,
    CouldEscapeOwnBomb,
    EnemiesInBlastRange,
    FeatureCollector,
    MoveNextToNearestBox,
    MoveOutOfBlastZone,
    MoveToNearestCoin,
    MoveToNearestEnemy,
    PastMoves,
    PossibleActions,
)
from agent_code.my_agent.features.movement_graph import MovementGraph
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
LEARNING_RATE = 0.1
EPSILON = 1
EPSILON_MIN = 0.05
EPSILON_DECREASE_RATE = 0.95
MODEL_PATH = "model.joblib"
N_STEP = 6


def setup_learning_features(self, load_model=True):
    self.EPSILON = EPSILON
    self.action_value_data = {"UP": {}, "DOWN": {}, "LEFT": {}, "RIGHT": {}, "WAIT": {}, "BOMB": {}}
    self.n_states_old = []
    self.n_states_new = []
    self.n_actions = []
    self.n_rewards = []

    self.last_bomb_position = []

    self.rewards_round = 0
    self.q_updates = 0
    self.q_updates_sum = 0

    self.feature_collector = FeatureCollector(
        MoveToNearestCoin(),
        MoveOutOfBlastZone(),
        MoveNextToNearestBox(),
        MoveToNearestEnemy(),
        EnemiesInBlastRange(),
        PastMoves(),
        BoxesInBlastRange(),
        AgentInBlastZone(),
        PossibleActions(),
        CouldEscapeOwnBomb(),
    )

    if load_model and os.path.isfile(MODEL_PATH):
        self.trees = load(MODEL_PATH)
    else:
        if load_model:
            print("[WARN] Unable to load model from filesystem. Reinitializing model!")
        self.trees = {
            "UP": RandomForestRegressor(max_depth=5, bootstrap=False),
            "DOWN": RandomForestRegressor(max_depth=5, bootstrap=False),
            "LEFT": RandomForestRegressor(max_depth=5, bootstrap=False),
            "RIGHT": RandomForestRegressor(max_depth=5, bootstrap=False),
            "WAIT": RandomForestRegressor(max_depth=5, bootstrap=False),
            "BOMB": RandomForestRegressor(max_depth=5, bootstrap=False),
        }
        for action_tree in self.trees.items():
            action_tree.fit(np.array(np.zeros(self.feature_collector.dim())).reshape(1, -1), [0])


def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    if Actions[self_action] == Actions.BOMB:
        self.last_bomb_position.append(old_game_state["self"][3])

    score_diff = new_game_state["self"][1] - old_game_state["self"][1]

    old_features = self.feature_collector.compute_feature(self, old_game_state, MovementGraph(old_game_state))
    new_features = self.feature_collector.compute_feature(self, new_game_state, MovementGraph(new_game_state))

    rewards = _rewards_from_events(self, feature_extration_old, events, self_action, score_diff)

    q_value_old = self.trees[self_action].predict(old_features.reshape(1, -1))
    q_value_new = rewards + DISCOUNT * np.max(
        [self.trees[action_tree].predict(new_features.reshape(1, -1)) for action_tree in self.trees]
    )
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[self_action][tuple(old_features)] = q_value_old + q_value_update

    self.q_updates_sum += q_value_old + q_value_new
    self.q_updates += 1


def update_action_value_last_step(self, last_game_state, last_action, events):
    print("Score:", last_game_state["self"][1])
    if Actions[last_action] == Actions.BOMB:
        self.last_bomb_position.append(last_game_state["self"][3])
    feature_extration_old = FeatureExtraction(last_game_state, self.past_moves)
    old_features = np.array(features_from_game_state(self, feature_extration_old))
    rewards = _rewards_from_events(self, feature_extration_old, events, last_action, 0)
    self.rewards_round += rewards
    self.rewards_round = 0
    self.last_bomb_position = []

    q_value_old = self.trees[last_action].predict(old_features.reshape(1, -1))
    q_value_new = rewards + 0
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[last_action][tuple(old_features)] = q_value_old + q_value_update

    self.q_updates_sum += q_value_old + q_value_new
    self.q_updates += 1


def train_q_model(self, game_state, episode_rounds, save_model=True):
    round = game_state["round"]
    if round % episode_rounds == 0:
        self.EPSILON *= EPSILON_DECREASE_RATE
        self.EPSILON = max(EPSILON_MIN, self.EPSILON)
        self.trees = _train_q_model(self, self.action_value_data)
        if save_model:
            dump(self.trees, MODEL_PATH)


def _train_q_model(self, action_value_data):
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
        new_tree = RandomForestRegressor(max_depth=5, bootstrap=False)
        features = np.array(features)
        values = np.ravel(np.array(values))
        X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=0.33)
        new_tree.fit(X_train, y_train)
        print("Epsilon:", self.EPSILON)
        print("Tree score test", action, new_tree.score(X_test, y_test))
        print("Tree score train", action, new_tree.score(X_train, y_train))

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
    move = feature_extraction.FEATURE_move_next_to_nearest_enemy().as_one_hot()
    features += move
    moves = feature_extraction.FEATURE_past_moves()
    features += moves
    blast_enemies = feature_extraction.FEATURE_enemies_in_agent_blast_range()
    features += blast_enemies
    return np.array(features)


def _rewards_from_events(self, feature_extraction, events, action, score_diff):
    action = Actions[action]
    rewards = 0

    action_to_coin = feature_extraction.FEATURE_move_to_nearest_coin()
    action_to_safety = feature_extraction.FEATURE_move_out_of_blast_zone()
    action_to_box = feature_extraction.FEATURE_move_next_to_nearest_box()
    action_to_enemy = feature_extraction.FEATURE_move_next_to_nearest_enemy()
    in_blast = feature_extraction.FEATURE_in_blast_zone()[0]
    blast_boxes = feature_extraction.FEATURE_boxes_in_agent_blast_range()[0]
    bomb_good = feature_extraction.FEATURE_could_escape_own_bomb()[0]
    blast_enemies = feature_extraction.FEATURE_enemies_in_agent_blast_range()[0]

    can_place_bomb = feature_extraction._action_possible(Actions.BOMB)

    if action_to_safety != Actions.NONE:
        if action == action_to_safety:
            rewards += 1
        else:
            rewards -= 1
    elif action_to_coin != Actions.NONE:
        if action == action_to_coin:
            rewards += 1
        else:
            rewards -= 1
    elif can_place_bomb and bomb_good and (blast_boxes > 0 or blast_enemies > 0):
        if action == Actions.BOMB:
            rewards += 1
        else:
            rewards -= 1
    elif action_to_box != Actions.NONE:
        if action == action_to_box:
            rewards += 1
        else:
            rewards -= 1
    elif action_to_enemy != Actions.NONE:
        if action == action_to_coin:
            rewards += 1
        else:
            rewards -= 1
    else:
        # rewards -= 1 # Do something
        pass

    print(rewards)
    return rewards + 10 * score_diff
