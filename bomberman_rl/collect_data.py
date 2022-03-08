from main import main
from itertools import combinations_with_replacement

AGENTS = ["rule_based_agent", "coin_collector_agent", "peaceful_agent", "random_agent"]


def collect_data():
    agent_combinations = [list(a) for a in combinations_with_replacement(AGENTS, r=4)]
    for i, agents in enumerate(agent_combinations):
        print("\n")
        print(f"batch: {i}, progress: {100 * i / len(agent_combinations):.2f}%, {' vs '.join(agents)}")
        main(["collect_data.py", "play", "--n-rounds", "400", "--no-gui", "--agents"] + agents)


if __name__ == "__main__":
    collect_data()
