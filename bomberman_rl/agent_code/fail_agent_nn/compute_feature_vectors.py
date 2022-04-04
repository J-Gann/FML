"""Author: Samuel Melm"""

# this is the sequential version of the feature computation after pandarallel broke

import pandas as pd

import numpy as np
from os import listdir
from tqdm.auto import tqdm

tqdm.pandas()

from agent_code.my_agent.features.feature import FeatureCollector
from agent_code.my_agent.features.actions import Actions

AGENTS = [f"agent_{i}" for i in range(4)]


def load_batches():
    print("loading batches...")
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


def prepare_data():
    print("preparing data...")
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
    print("computing past moves...")
    for agent in AGENTS:
        shifted = pd.DataFrame(index=data.index)

        for shift_by in tqdm(range(1, 5)):
            shifted[shift_by] = data.groupby(["rounds"])[f"{agent}_action"].shift(shift_by).fillna(Actions.NONE)
        data[f"{agent}_past_moves"] = shifted.values.tolist()
    return data


fc = FeatureCollector.create_with_all_features()

def compute_feature_vec(d):
    return fc.compute_feature_as_series(d, d)


def compute_feature_vecs(data, agent):
    data["self"] = data[f"{agent}_state"]
    data["others"] = pd.Series(data[[f"{other}_state" for other in AGENTS if other != agent]].values.tolist())
    data["past_moves"] = data[f"{agent}_past_moves"]

    return data.progress_apply(compute_feature_vec, axis=1)


data = prepare_data()
data = shift_past_moves(data)
# data.to_pickle("data.pkl")

section_count = 700
section_size = 2800 // section_count

for agent in tqdm(AGENTS[2:]):
    data["self"] = data[f"{agent}_state"]
    data["others"] = pd.Series(data[[f"{other}_state" for other in AGENTS if other != agent]].values.tolist())
    data["past_moves"] = data[f"{agent}_past_moves"]

    for sec_start in tqdm(range(0, 2800, section_size)):
        sec_end = sec_start + section_size
        cond = (sec_start <= data.rounds) & (data.rounds < sec_end)

        result = data[cond].apply(compute_feature_vec, axis=1)
        result.to_pickle(f"data/feature_vecs/{agent}_{sec_start}_to_{sec_end}.pkl")


# from importlib import reload
# import agent_code.my_agent.features.feature

# fc = reload(agent_code.my_agent.features.feature).FeatureCollector.create_with_all_features()
