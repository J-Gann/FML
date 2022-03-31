import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("agent_code/null_agent/training_results.csv")

data["highest_avg_score"] = data.score_avg.cummax() / data.score_avg.max()

fig, axs = plt.subplots(ncols=2)

for prefix, ax in zip(["train", "test"], axs):
    columns = [c for c in data.columns if c.startswith(prefix + "_")]
    training_scores = (
        data[columns + ["highest_avg_score"]].dropna().rename({c: c.replace(prefix + "_", "") for c in columns}, axis=1)
    )
    training_scores.plot(title=prefix + " scores", ax=ax)

plt.show()
