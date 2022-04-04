"""Author: Samuel Melm"""

# this file contains the parallel computation of the feature vectors
# this worked until pandarallel broke on my machine

import pandas as pd

from timeit import timeit

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=7)

import numpy as np
from os import listdir
from tqdm.auto import tqdm

tqdm.pandas()

from random import choice

from agent_code.my_agent.features.movement_graph import MovementGraph
from agent_code.my_agent.features.feature import FeatureCollector
from agent_code.my_agent.features.actions import Actions

AGENTS = [f"agent_{i}" for i in range(4)]


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_batches():

    batch_files = [f for f in listdir("batches") if f.startswith("batch")]
    batches = len(batch_files) * [None]

    for batch_file in tqdm(batch_files):
        batch_number = int(batch_file.split("_")[1].split(".")[0])

        batch = pd.read_pickle(f"batches/{batch_file}")
        batch["batch_number"] = batch_number
        # already sort by batch number
        batches[batch_number - 1] = batch

    assert not any(b is None for b in batches)

    last_round = 0

    for i, batch in tqdm(enumerate(batches)):
        batch = batch.rename({"round": "rounds"}, axis=1)
        batch.rounds += last_round
        last_round = batch.rounds.max()

        batches[i] = batch

    return batches


def prepare_data(batches):
    batches = load_batches()

    data = pd.concat(batches).reset_index(drop=True)
    # zero indexed, does not exists
    data = data.drop(labels=["agent_4_state", "agent_4_action"], axis=1)

    # replace action = None with Actions.NONE
    data = data.fillna("NONE")

    # convert action strings to actrion enum values
    for agent in AGENTS:
        col_name = f"{agent}_action"
        data[col_name] = data[col_name].apply(lambda x: Actions[x])

    return data


def shift_past_moves(data):
    for shift_by in tqdm(range(1, 5)):
        shifted = data.groupby(["rounds"]).shift(shift_by).fillna(Actions.NONE)
        for agent in range(4):
            data[f"agent_{agent}_move_{shift_by}_steps_ago"] = shifted[f"agent_{agent}_action"]
    return data


fc = FeatureCollector.create_with_all_features()


def compute_feature_vecs(d):
    for agent in AGENTS:
        data["self"] = data[f"agent_{agent}_state"]
        data["others"] = pd.Series(data[[f"agent_{i}_state" for i in range(0, 4) if i != agent]].values.tolist())
        self_obj = Dotdict({"past_moves": [d[f"agent_{agent}_move_{steps}_steps_ago"] for steps in range(1, 4)]})

        fc.compute_feature_as_series(d, self_obj)


data = prepare_data()
data = shift_past_moves(data)

data[:1].apply(compute_feature_vecs, axis=1)
