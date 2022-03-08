from main import main
from random import choices

AGENTS = ["rule_based_agent", "coin_collector_agent", "peaceful_agent", "random_agent"]


def collect_data():
    main(["collect_data.py", "play", "--n-rounds", "5", "--agents"] + choices(AGENTS, k=4))


if __name__ == "__main__":
    collect_data()
