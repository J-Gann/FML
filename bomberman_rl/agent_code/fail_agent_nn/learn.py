import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
from os import listdir

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor, BayesianRidge, ARDRegression
from sklearn.neural_network import MLPRegressor

from agent_code.my_agent.features.actions import Actions


FEATURE_VEC_DIR = "data/feature_vecs"
FEATURES_TO_DROP = ["agent_position", "crate_direction", "bomb_direction", "past_moves"]


def load_feature_vecs():
    files = [f for f in listdir(FEATURE_VEC_DIR) if f.startswith("agent_0")]
    files.sort(key=lambda x: int(x.split("_")[2]))

    feature_vecs = pd.concat([pd.read_pickle(f"{FEATURE_VEC_DIR}/{f}") for f in tqdm(files)])

    feature_vecs = feature_vecs.drop(
        labels=[c for c in feature_vecs.columns if any([c.startswith(x) for x in FEATURES_TO_DROP])],
        axis=1,
    )

    return feature_vecs


def load_rewards():
    returns = pd.read_pickle("data/returns.pkl").reset_index()
    returns = returns[returns.rounds != 2800]["agent_0_return_gamma_0_95"]
    return returns


def load_data():
    X = load_feature_vecs()
    y = load_rewards()
    assert len(X) == len(y), f"{len(X)} != {len(y)}"

    return X, y


def split_data_into_actions(X, y):
    X_by_action = dict()
    y_by_action = dict()

    actions = pd.read_pickle("data/actions.pkl")[: len(y)]["agent_0_action"]

    for action in Actions:
        if action == Actions.NONE:
            continue

        X_by_action[action] = X[actions == action]
        y_by_action[action] = y[actions == action]

    return X_by_action, y_by_action


def train(X_by_action, y_by_action, method):
    models = dict()
    scores = dict()
    for action in Actions:
        if action == Actions.NONE:
            continue

        X = X_by_action[action]
        y = y_by_action[action]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        models[action] = method()

        print(f"training model for {action.name}...")
        print(f"setup: training size {len(X_train)} test size {len(X_test)}")

        models[action] = models[action].fit(X_train, y_train)

        print(f"evaluating model for {action.name}...")
        scores[action] = models[action].score(X_test, y_test)

    return models, scores


X, y = load_data()

x_sample = X.sample(10000)
pca = PCA(2)
x_trans = pca.fit_transform(x_sample)

print(X.shape)
4 / 0
X_by_action, y_by_action = split_data_into_actions(X, y)

shapes = [60, 55, 44, 33, 22]
scores = {s: dict() for s in shapes}

for action in Actions:
    if action == Actions.NONE:
        continue

    X = X_by_action[action]
    y = y_by_action[action]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for shape in shapes:
        model = MLPRegressor(activation="logistic", hidden_layer_sizes=(shape,), verbose=True)
        model = model.fit(X_train, y_train)

        scores[shape][action] = model.score(X_test, y_test)

# X = X_by_action[Actions.UP]
# y = y_by_action[Actions.UP]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = MLPRegressor(activation="logistic", hidden_layer_sizes=(shape,), verbose=True)
# model = model.fit(X_train, y_train)
