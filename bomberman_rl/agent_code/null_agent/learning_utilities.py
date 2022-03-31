import pathlib
import sys
import pandas as pd
import events as e
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
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
    CoinDistance,
)
from .features.actions import Actions

DISCOUNT = 0.95
LEARNING_RATE = 0.2
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECREASE_RATE = 0.95
MODEL_PATH = "model.joblib"
ACTION_VALUE_DATA_PATH = "action_values.joblib"


def setup_learning_features(self, load_model=True):
    self.EPSILON = EPSILON
    # Object where the experience of the agent is stored.
    # State-reward pairs are inserted into the action-object which resulted in the rewards.
    self.action_value_data = {"UP": {}, "DOWN": {}, "LEFT": {}, "RIGHT": {}, "WAIT": {}, "BOMB": {}}
    self.scores_sum = 0  # Keep track of the toal scores a model achieved

    # Load an existing model if possible
    if load_model and os.path.isfile(MODEL_PATH) and os.path.isfile(ACTION_VALUE_DATA_PATH):
        self.trees = load(MODEL_PATH)
        self.action_value_data = load(ACTION_VALUE_DATA_PATH)
    else:
        if load_model:
            print("[WARN] Unable to load model from filesystem. Reinitializing model!")
        # Initialize random forest regressors with 0 if no model could be loaded
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

    self.training_results = []
    self.best_5_models = []
    self.reward_history = []
    self.last_round_with_improvement = 0
    self.q_value_history = []
    self.q_update_history = []


def update_action_value_data(self, old_game_state, self_action, new_game_state, events):
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
        [self.trees[action_tree].predict(new_features.reshape(1, -1)) for action_tree in self.trees]
    )
    # Update the experience
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[self_action][tuple(old_features)] = q_value_old + q_value_update


def update_action_value_last_step(self, last_game_state, last_action, events):
    """Use a transition to update the experience of the agent using q-learning."""
    old_features = self.feature_collector.compute_feature(last_game_state, self)
    rewards = _rewards_from_events(self, old_features, events, last_action)
    q_value_old = self.trees[last_action].predict(old_features.reshape(1, -1))
    q_value_new = rewards + 0
    q_value_update = LEARNING_RATE * (q_value_new - q_value_old)
    self.action_value_data[last_action][tuple(old_features)] = q_value_old + q_value_update
    self.scores_sum += last_game_state["self"][1]

    self.q_value_history.append(q_value_new)
    self.q_update_history.append(q_value_update)


def train_q_model(self, game_state, rounds_per_episode, save_model=True):
    model_hash = abs(hash(str({k: hash(v) for k, v in self.trees.items()})))
    score_avg = self.scores_sum / (game_state["round"] % rounds_per_episode + 1)

    training_result = {
        "round": game_state["round"],
        "scores_sum": self.scores_sum,
        "score_avg": score_avg,
        "hash": model_hash,
        "epsilon": self.EPSILON,
        "steps": game_state["step"],
    }

    # only update every rounds_per_episode round
    if game_state["round"] % rounds_per_episode != 0:
        # only add training results without scores since no traiing is done
        self.training_results.append(training_result)

        return

    # Reduce epsilon by epsilon decrease rate, do no go below minimum epsilon
    self.EPSILON = max(EPSILON_MIN, self.EPSILON * EPSILON_DECREASE_RATE)

    # Train new model using all experience
    self.trees, scores = _train_q_model(self, self.action_value_data)
    print("test/train score:", sum(scores.values()) / len(scores))

    self.training_results.append({**training_result, **scores})

    best_5_model_hashes = update_best_5_models(self, game_state, score_avg, training_result)

    # make sure model dir exists
    regressor_name = self.trees["UP"].__class__.__name__
    data_dir = f"data/{regressor_name}"
    model_dir = f"{data_dir}/models"
    action_value_dir = f"{data_dir}/action_value"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(action_value_dir).mkdir(parents=True, exist_ok=True)

    # only save the current model if it is among the best five models
    if save_model and model_hash in best_5_model_hashes:
        # Save model along with the scores it achieved
        dump(self.trees, f"{model_dir}/{model_hash}_score_{self.scores_sum}.joblib")
        dump(self.action_value_data, f"{action_value_dir}/{model_hash}_score_{self.scores_sum}.joblib")

        open(f"{data_dir}/best_model_hash.txt", "w").write(str(best_5_model_hashes[0]))

    # delete old models
    for model_file in os.listdir(model_dir):
        model_hash = int(model_file.split("_")[0])
        if model_hash not in best_5_model_hashes:
            os.remove(f"{model_dir}/{model_file}")

    self.scores_sum = 0  # Reset scores to 0 for next model

    if game_state["round"] % 20 == 0:
        pd.DataFrame(self.training_results).to_csv(f"{data_dir}/training_results.csv")
        pd.DataFrame(self.reward_history).to_csv(f"{data_dir}/reward_histroy.csv")
        pd.DataFrame(self.q_value_history).to_csv(f"{data_dir}/q_value_history.csv")
        pd.DataFrame(self.q_update_history).to_csv(f"{data_dir}/q_update_history.csv")


def update_best_5_models(self, game_state, score_avg, training_result):
    # if current model is better than one of the 5 best, add it and remove the worst
    print("best five model scores", *[m["score_avg"] for m in self.best_5_models])

    worst_score = min(*[m["score_avg"] for m in self.best_5_models], 0, 0)
    if score_avg > worst_score:
        self.best_5_models.append(training_result)
        self.best_5_models.sort(key=lambda x: x["score_avg"], reverse=True)
        self.best_5_models = self.best_5_models[:5]
        self.last_round_with_improvement = game_state["round"]

    best_5_model_hashes = [m["hash"] for m in self.best_5_models]
    return best_5_model_hashes


def _train_q_model(self, action_value_data):
    new_trees = dict()
    scores = dict()
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
        new_tree = RandomForestRegressor()
        features = np.array(features)
        values = np.ravel(np.array(values))
        # Create training and test set
        X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=0.33)
        # Fit new forest
        new_tree.fit(X_train, y_train)
        # Log performance of the model
        train_score = new_tree.score(X_train, y_train)
        test_score = new_tree.score(X_test, y_test)
        scores[f"train_{action}"] = train_score
        scores[f"test_{action}"] = test_score

        # print("Epsilon:", self.EPSILON)
        # print("Tree score test", action, test_score)
        # print("Tree score train", action, train_score)
        new_trees[action] = new_tree
    return new_trees, scores


def _rewards_from_events(self, feature_vector, events, action):
    action = Actions[action]
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
    nearest_enemy_possible_moves = feature_collector.single_feature_from_vector(
        feature_vector, NearestEnemyPossibleMoves
    )
    enemy_distance = feature_collector.single_feature_from_vector(feature_vector, EnemyDistance)[0]
    coin_distance = feature_collector.single_feature_from_vector(feature_vector, CoinDistance)[0]

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

    self.reward_history.append(rewards)
    return rewards
