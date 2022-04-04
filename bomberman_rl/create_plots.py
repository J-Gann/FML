"""Samuel Melm"""

import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sn

names = [
    "Ridge",
    "SVR",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "mlp20",
    "mlp40",
    "mlp60",
]

titles = {
    "Ridge": "Ridge",
    "SVR": "Support Vector Regression",
    "DecisionTreeRegressor": "Decision Tree Regression",
    "RandomForestRegressor": "Random Forest Regression",
    "mlp20": "Neural Network, hidden layer=20",
    "mlp40": "Neural Network, hidden layer=40",
    "mlp60": "Neural Network, hidden layer=60",
}

dfs = []

for name in names:
    d = pd.read_csv(f"data/run_02_04/{name}/training_results.csv")
    d["name"] = name
    dfs.append(d)

data = pd.concat(dfs)


def avg_scores(data):
    actions = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
    avg_scores = pd.DataFrame()

    for a in actions:
        avg_scores[a] = (data[f"train_{a}"] + data[f"test_{a}"]) / 2
    return avg_scores


def compute_highest_step_count(data):
    df = pd.DataFrame()

    df["steps"] = data.steps
    df["highest step count"] = data.steps.cummax()

    return df


def plot_for_all(data, labels, legend=None, title=None, file_name=None):
    fig, axs = plt.subplots(ncols=2, nrows=4, sharex=True, sharey=True, figsize=(10, 10))
    fig.delaxes(axs.flat[-1])

    for d, label, ax in zip(data, labels, axs.flat):
        ax.plot(d)
        if label in titles:
            ax.set_title(titles[label])
        else:
            ax.set_title(label)

    if legend:
        fig.legend(legend, loc="upper left", bbox_to_anchor=[0.58, 0.25])

    if title:
        fig.suptitle(title)

    if file_name:
        fig.savefig(file_name)
    fig.show()


plot_for_all(
    [avg_scores(d).dropna()[7:] for d in dfs],
    labels=names,
    legend=[c.replace("train_", "").lower() for c in data.columns if c.startswith("train")],
    title="Scores after round 30",
    file_name="model_scores_after_round_30.png"
)

plot_for_all([compute_highest_step_count(d) for d in dfs], labels=names, title0"Step Count", file_name="step_count.png")
