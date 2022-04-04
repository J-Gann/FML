import events as e
import numpy as np
import os
from joblib import dump, load
from sklearn.linear import Ridge
from sklearn.model_selection import train_test_split
from agent_code.null_agent.features.feature import (
    BoxesInBlastRange,
    CouldEscapeOwnBomb,
    EnemiesInBlastRange,
    FeatureCollector,
    MoveNextToNearestBox,
    MoveOutOfBlastZone,
    MoveToNearestCoin,
    MoveToNearestEnemy,
    PossibleActions,
    NearestEnemyPossibleMoves,
    EnemyDistance,
    CoinDistance
)
from .features.actions import Actions

DISCOUNT = 0.95
LEARNING_RATE = 0.2
EPSILON = 0.01
EPSILON_MIN = 0.01
EPSILON_DECREASE_RATE = 0.95
MODEL_PATH = "model.joblib"
ACTION_VALUE_DATA_PATH = "action_values.joblib"


def setup_learning_features(self, load_model=True):
    """Author: Jonas Gann"""
    self.EPSILON = EPSILON
    # Object where the experience of the agent is stored.
    # State-reward pairs are inserted into the action-object which resulted in the rewards.
    self.action_value_data = {"UP": {}, "DOWN": {},
                              "LEFT": {}, "RIGHT": {}, "WAIT": {}, "BOMB": {}}
    self.scores_sum = 0  # Keep track of the toal scores a model achieved

    # Load an existing model if possible
    if load_model and os.path.isfile(MODEL_PATH) and os.path.isfile(ACTION_VALUE_DATA_PATH):
        # We used git-lfs to store models. Therefore in the model files, only a reference to
        # a file store is included. To pull the actual file you have to install git-lfs:
        # https://git-lfs.github.com/
        self.trees = load(MODEL_PATH)
        self.action_value_data = load(ACTION_VALUE_DATA_PATH)
    else:
        if load_model:
            print("[WARN] Unable to load model from filesystem. Reinitializing model!")
        # Initialize random forest regressors with 0 if no model could be loaded
        self.trees = {
            "UP": Ridge(),
            "DOWN": Ridge(),
            "LEFT": Ridge(),
            "RIGHT": Ridge(),
            "WAIT": Ridge(),
            "BOMB": Ridge(),
        }
        for action_tree in self.trees:
            self.trees[action_tree].fit(
                np.array(np.zeros(self.feature_collector.dim())).reshape(1, -1), [0])


def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    """Author: Jonas Gann"""
    """Use a transition to update the experience of the agent using q-learning."""
    # Compute features out of the old game state
    old_features = self.feature_collector.compute_feature(old_game_state, self)
    # Compute features out of the new game state
    new_features = self.feature_collector.compute_feature(new_game_state, self)
    # Compute rewards from the events and the performed action to utilize reward shaping
    rewards = _rewards_from_events(self, old_features, events, self_action)
    # Predict the expected q-value using the current model
    q_value_old = self.trees[self_action].predict(old_features.reshape(1, -1))
    # Compute the expected q-value knowing the reward of the current action and the expected future rewards
    q_value_new = rewards + DISCOUNT * np.max(
        [self.trees[action_tree].predict(
            new_features.reshape(1, -1)) for action_tree in self.trees]
    )
    # Update the experience
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[self_action][tuple(
        old_features)] = q_value_old + q_value_update


def update_action_value_last_step(self, last_game_state, last_action, events):
    """Author: Jonas Gann"""
    """Use a transition to update the experience of the agent using q-learning."""
    old_features = self.feature_collector.compute_feature(
        last_game_state, self)
    rewards = _rewards_from_events(self, old_features, events, last_action)
    q_value_old = self.trees[last_action].predict(old_features.reshape(1, -1))
    q_value_new = rewards + 0
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[last_action][tuple(
        old_features)] = q_value_old + q_value_update
    self.scores_sum += last_game_state["self"][1]


def train_q_model(self, game_state, episode_rounds, save_model=True):
    """Author: Jonas Gann"""
    round = game_state["round"]
    if round % episode_rounds == 0:
        # Reduce epsilon by epsilon decrease rate
        self.EPSILON *= EPSILON_DECREASE_RATE
        # Do no go below minimum epsilon
        self.EPSILON = max(EPSILON_MIN, self.EPSILON)
        if save_model:  # Save model along with the scores it achieved
            dump(self.trees, MODEL_PATH + "_" + str(self.scores_sum))
            dump(self.action_value_data, ACTION_VALUE_DATA_PATH +
                 "_" + str(self.scores_sum))
        self.scores_sum = 0  # Reset scores to 0 for next model
        # Train new model using all experience
        self.trees = _train_q_model(self, self.action_value_data)
        if save_model:
            # Save currently used model
            dump(self.trees, MODEL_PATH)
            dump(self.action_value_data, ACTION_VALUE_DATA_PATH)


def _train_q_model(self, action_value_data):
    """Author: Jonas Gann"""
    new_trees = {}
    for action in Actions:  # Train a new regression forest for each action
        if action == Actions.NONE:
            continue  # Do not train a model for the "noop"
        action = action.name
        # Retrieve experience for the current action
        action_value_data_action = action_value_data[action]
        # Some format transformations
        features = []
        values = []
        for key in action_value_data_action:
            feature = np.array(key)
            value = action_value_data_action[key]
            features.append(feature)
            values.append(value)
        # Create new regression forest
        new_tree = Ridge()
        features = np.array(features)
        values = np.ravel(np.array(values))
        # Create training and test set
        X_train, X_test, y_train, y_test = train_test_split(
            features, values, test_size=0.33)
        # Fit new forest
        new_tree.fit(X_train, y_train)
        # Log performance of the model
        print("Epsilon:", self.EPSILON)
        print("Tree score test", action, new_tree.score(X_test, y_test))
        print("Tree score train", action, new_tree.score(X_train, y_train))
        new_trees[action] = new_tree
    return new_trees


def _rewards_from_events(self, feature_vector, events, action):
    """Author: Jonas Gann"""
    action = Actions[action]
    feature_collector: FeatureCollector = self.feature_collector
    action_to_coin = feature_collector.single_feature_from_vector(
        feature_vector, MoveToNearestCoin)
    action_to_coin = Actions.from_one_hot(action_to_coin)
    action_to_safety = feature_collector.single_feature_from_vector(
        feature_vector, MoveOutOfBlastZone)
    action_to_safety = Actions.from_one_hot(action_to_safety)
    action_to_box = feature_collector.single_feature_from_vector(
        feature_vector, MoveNextToNearestBox)
    action_to_box = Actions.from_one_hot(action_to_box)
    action_to_enemy = feature_collector.single_feature_from_vector(
        feature_vector, MoveToNearestEnemy)
    action_to_enemy = Actions.from_one_hot(action_to_enemy)
    blast_boxes = feature_collector.single_feature_from_vector(
        feature_vector, BoxesInBlastRange)[0]
    bomb_good = feature_collector.single_feature_from_vector(
        feature_vector, CouldEscapeOwnBomb)[0]
    blast_enemies = feature_collector.single_feature_from_vector(
        feature_vector, EnemiesInBlastRange)[0]
    possible_actions = feature_collector.single_feature_from_vector(
        feature_vector, PossibleActions)
    can_place_bomb = possible_actions[Actions.BOMB.value] == 1
    nearest_enemy_possible_moves = feature_collector.single_feature_from_vector(
        feature_vector, NearestEnemyPossibleMoves)
    enemy_distance = feature_collector.single_feature_from_vector(
        feature_vector, EnemyDistance)[0]
    coin_distance = feature_collector.single_feature_from_vector(
        feature_vector, CoinDistance)[0]

    local_rewards = 0
    global_rewards = 0

    if action_to_safety != Actions.NONE:
        if action == action_to_safety:
            local_rewards += 7
    if action_to_coin != Actions.NONE:
        if action == action_to_coin:
            local_rewards += 3
    if can_place_bomb and bomb_good and (blast_boxes > 0 or blast_enemies > 0):
        if action == Actions.BOMB:
            local_rewards += 1
        if action != Actions.BOMB and action_to_safety == Actions.NONE:
            local_rewards -= 1
    if can_place_bomb and bomb_good and not (blast_boxes > 0 or blast_enemies > 0):
        if action == Actions.BOMB:
            local_rewards -= 8
    if action_to_box != Actions.NONE:
        if action == action_to_box:
            local_rewards += 1
    if action_to_enemy != Actions.NONE:
        if action == action_to_enemy:
            local_rewards += 1

    if e.COIN_COLLECTED in events:
        global_rewards += 40
    if e.CRATE_DESTROYED in events:
        global_rewards += 10
    if e.KILLED_OPPONENT in events:
        global_rewards += 100
    if e.GOT_KILLED in events:
        global_rewards -= 100
    if e.WAITED in events:
        global_rewards -= 0.5

    if action_to_box != Actions.NONE and action_to_enemy == Actions.NONE and action_to_coin == Actions.NONE:
        if action == action_to_box:
            global_rewards += 10
    if action_to_coin != Actions.NONE and coin_distance < 5:
        if action == action_to_coin:
            global_rewards += 10
    if nearest_enemy_possible_moves <= 1 and enemy_distance <= 3 and can_place_bomb and bomb_good and blast_enemies > 0:
        if action == Actions.BOMB:
            local_rewards += 10
    rewards = local_rewards + global_rewards

    return rewards
