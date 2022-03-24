from argparse import Action
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
    AgentFieldNeighbors,
    AgentExplosionNeighbors,
    NearestEnemyPossibleMoves,
    EnemyDistance
)
from .features.actions import Actions

from agent_code.my_agent.features.movement_graph import MovementGraph
import events as e
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DISCOUNT = 0.9
LEARNING_RATE = 0.3
EPSILON = 0#1
EPSILON_MIN = 0.05
EPSILON_DECREASE_RATE = 0.9
MODEL_PATH = "model.joblib"
ACTION_VALUE_DATA_PATH = "action_values.joblib"


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

    if load_model and os.path.isfile(MODEL_PATH) and os.path.isfile(ACTION_VALUE_DATA_PATH):
        self.trees = load(MODEL_PATH)
        self.action_value_data = load(ACTION_VALUE_DATA_PATH)
    else:
        if load_model:
            print("[WARN] Unable to load model from filesystem. Reinitializing model!")
        self.trees = {
            "UP": RandomForestRegressor(),
            "DOWN": RandomForestRegressor(),
            "LEFT": RandomForestRegressor(),
            "RIGHT": RandomForestRegressor(),
            "WAIT": RandomForestRegressor(),
            "BOMB": RandomForestRegressor(),
        }
        for action_tree in self.trees:
            self.trees[action_tree].fit(np.array(np.zeros(self.feature_collector.dim())).reshape(1, -1), [0])


def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
    if self_action == Actions.BOMB:
        self.last_bomb_position.append(old_game_state["self"][3])

    score_diff = new_game_state["self"][1] - old_game_state["self"][1]

    old_features = self.feature_collector.compute_feature(old_game_state, self)
    new_features = self.feature_collector.compute_feature(new_game_state, self)

    rewards = _rewards_from_events(self, old_features, events, self_action, score_diff)

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
    if last_action == Actions.BOMB:
        self.last_bomb_position.append(last_game_state["self"][3])

    old_features = self.feature_collector.compute_feature(last_game_state, self)
    rewards = _rewards_from_events(self, old_features, events, last_action, 0)

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
            dump(self.action_value_data, ACTION_VALUE_DATA_PATH)


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
        new_tree = RandomForestRegressor()
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


def _rewards_from_events(self, feature_vector, events, action, score_diff):
    action = Actions[action]
    rewards = 0

    feature_collector: FeatureCollector = self.feature_collector

    action_to_coin = feature_collector.single_feature_from_vector(feature_vector, MoveToNearestCoin)
    action_to_coin = Actions.from_one_hot(action_to_coin)

    action_to_safety = feature_collector.single_feature_from_vector(feature_vector, MoveOutOfBlastZone)
    action_to_safety = Actions.from_one_hot(action_to_safety)

    action_to_box = feature_collector.single_feature_from_vector(feature_vector, MoveNextToNearestBox)
    action_to_box = Actions.from_one_hot(action_to_box)

    action_to_enemy = feature_collector.single_feature_from_vector(feature_vector, MoveToNearestEnemy)
    action_to_enemy = Actions.from_one_hot(action_to_enemy)

    blast_boxes = feature_collector.single_feature_from_vector(feature_vector, BoxesInBlastRange)[0]
    bomb_good = feature_collector.single_feature_from_vector(feature_vector, CouldEscapeOwnBomb)[0]
    blast_enemies = feature_collector.single_feature_from_vector(feature_vector, EnemiesInBlastRange)[0]

    possible_actions = feature_collector.single_feature_from_vector(feature_vector, PossibleActions)
    can_place_bomb = possible_actions[Actions.BOMB.value] == 1

    nearest_enemy_possible_moves = feature_collector.single_feature_from_vector(feature_vector, NearestEnemyPossibleMoves)

    enemy_distance = feature_collector.single_feature_from_vector(feature_vector, EnemyDistance)[0]

    local_rewards = 0
    global_rewards = 0

    if action_to_safety != Actions.NONE:
        if action == action_to_safety:  
           local_rewards += 5       # Agent should really escape a bomb when necessary (penalty of death is not incentivizing escape enough)
    if action_to_coin != Actions.NONE:
        if action == action_to_coin:
            local_rewards += 3      # Collecting a coin is more important than placing a bomb or destroying a crate or moving to an enemy
    if can_place_bomb and bomb_good and (blast_boxes > 0 or blast_enemies > 0):
        if action == Actions.BOMB:
            local_rewards += blast_boxes + blast_enemies
        if action != Actions.BOMB and action_to_safety == Actions.NONE:  # Agent should place bombs as much as possible
            local_rewards -= blast_boxes + blast_enemies
    if can_place_bomb and bomb_good and not (blast_boxes > 0 or blast_enemies > 0):
        if action == Actions.BOMB:
            local_rewards -= 25     # Prevent Agent from rewarding itself by escaping its own bomb repeatedly
    if action_to_box != Actions.NONE:
        if action == action_to_box:
            local_rewards += 1
    if action_to_enemy != Actions.NONE:
        if action == action_to_enemy:
            local_rewards += 1

    # War tactics

    if nearest_enemy_possible_moves <= 2 and action_to_enemy != Actions.NONE:
        print("BOMB1")
        if action == action_to_enemy:  
            local_rewards += 3      # If nearest enemy has few options to move, move towards it to attack 

    if nearest_enemy_possible_moves <= 1 and action_to_enemy != Actions.NONE and enemy_distance <= 2:
        print("BOMB2")
        if action == action_to_enemy:
            local_rewards += 5     # If nearest enemy has very few options to move and agent is near, try to block enemy

    if nearest_enemy_possible_moves <= 1 and enemy_distance <= 3 and can_place_bomb and bomb_good and blast_enemies > 0:
        print("BOMB3")
        if action == Actions.BOMB:
            local_rewards += 8           # Agent places bomb for cornered enemy

    if e.COIN_COLLECTED in events: global_rewards += 10
    if e.CRATE_DESTROYED in events: global_rewards += 5
    if e.KILLED_OPPONENT in events: global_rewards += 50
    if e.GOT_KILLED in events: global_rewards -= 50
    if e.WAITED in events: global_rewards -= 0.5

    rewards = local_rewards + global_rewards

    print(rewards)

    return rewards
