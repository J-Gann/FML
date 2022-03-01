from joblib import dump, load
from typing import List

import events as e
import numpy as np
import settings as s
from sklearn import tree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import math
from enum import Enum

from sklearn.tree import export_graphviz


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5


DISCOUNT = 0.95
LEARNING_RATE = 0.5


def setup_training(self):
    # Create regression trees predicting the value of their move in a given state
    if not hasattr(self, "trees"):
        self.trees = {
            "UP": tree.DecisionTreeRegressor(),
            "DOWN": tree.DecisionTreeRegressor(),
            "LEFT": tree.DecisionTreeRegressor(),
            "RIGHT": tree.DecisionTreeRegressor(),
            "WAIT": tree.DecisionTreeRegressor(),
            "BOMB": tree.DecisionTreeRegressor(),
        }

    self.fit = {
        "UP": False,
        "DOWN": False,
        "LEFT": False,
        "RIGHT": False,
        "WAIT": False,
        "BOMB": False,
    }

    # Array with columns: state, action, next_state, reward for every game step
    self.old_features = []
    self.new_features = []
    self.actions = []
    self.rewards = []
    self.round_rewards = 0


def allFit(self):
    fit = True
    for key in self.fit:
        if not self.fit[key]:
            fit = False
    return fit


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    # Initialize graph from field once
    if not hasattr(self, "graph"):
        self.graph = graph(new_game_state["field"])

    # Calculate features from game states
    self.old_features.append(state_to_features(self, old_game_state))
    self.new_features.append(state_to_features(self, new_game_state))
    # Calculate rewards from events
    self.rewards.append(reward_from_events(self, events))
    self.actions.append(Actions[self_action].value)

    self.round_rewards += reward_from_events(self, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    actions_all = np.array(self.actions)
    old_features_all = np.array(self.old_features)
    new_features_all = np.array(self.new_features)
    rewards_all = np.array(self.rewards)

    for action in Actions:
        action_indices = actions_all == action.value
        if np.sum(action_indices) > 0:
            old_features = old_features_all[action_indices][:, action.value]
            new_features = new_features_all[action_indices][:, action.value]
            rewards = rewards_all[action_indices]
            q_values = rewards
            # current guess + learning rate * (    (actual reward + predicted future reward)     -    predicted action value  )
            if allFit(self):
                current_guess = self.trees[action.name].predict(old_features)
                actual_rewards = rewards
                predicted_future_reward = np.max(
                    [self.trees[tree].predict(new_features) for tree in self.trees]
                )
                predicted_action_value = self.trees[action.name].predict(old_features)

                q_values = current_guess + LEARNING_RATE * (
                    actual_rewards
                    + DISCOUNT * predicted_future_reward
                    - predicted_action_value
                )

            # train new action tree with updated values and overwrite old tree (pseudocode)
            newTree = tree.DecisionTreeRegressor()
            newTree.fit(old_features, q_values)
            self.trees[action.name] = newTree
            self.fit[action.name] = True

            if action.name == "UP" and last_game_state["round"] == 1100:
                export_graphviz(
                    newTree,
                    out_file="uptree.dot",
                    feature_names=["actionToNearestCoin"],
                )
                dump(self.trees, "models.joblib")
            if action.name == "BOMB" and last_game_state["round"] == 1100:
                export_graphviz(
                    newTree, out_file="bomb.dot", feature_names=["actionToNearestCoin"]
                )
                dump(self.trees, "models.joblib")

    # if (last_game_state["round"] % 1) == 0: print("rewards:", self.round_rewards)

    self.round_rewards = 0


def state_to_features(self, game_state):
    if game_state is None:
        return np.array([[0], [0], [0], [0], [1], [0]])

    distanceToNearestCoin, actionToNearestCoin = getPathFeatures(self, game_state)
    # print("ACTION TO NEAREST COIN", actionToNearestCoin)
    features = np.array(
        [
            [int(actionToNearestCoin == "UP")],
            [int(actionToNearestCoin == "RIGHT")],
            [int(actionToNearestCoin == "DOWN")],
            [int(actionToNearestCoin == "LEFT")],
            [int(actionToNearestCoin == "WAIT")],
            [int(actionToNearestCoin == "BOMB")],
        ]
    )

    return features


def getPathFeatures(self, game_state):
    coins = [indexToNode(coin[0], coin[1]) for coin in game_state["coins"]]
    if len(coins):
        x, y = game_state["self"][3]
        agentNode = indexToNode(x, y)
        dist_matrix, predecessors = dijkstra(
            csgraph=self.graph,
            directed=False,
            indices=agentNode,
            return_predecessors=True,
        )

        coinDistances = dist_matrix[coins]
        distanceToNearestCoin = np.min(coinDistances)

        nearestCoin = np.argmin(coinDistances)

        def getPath(node):
            if node == agentNode:
                return []
            pred = predecessors[node]
            if pred == agentNode:
                return []
            path = getPath(pred)
            path.append(pred)
            return path

        path = getPath(nearestCoin)

        nextNode = None
        if len(path) == 0:
            nextNode = agentNode
        else:
            nextNode = path[0]

        nextIndices = NodeToIndex(nextNode)

        xdiff = nextIndices[0] - x
        ydiff = nextIndices[1] - y

        # print("------------")
        # print(path)
        # print("Agent:", agentNode)
        # print("Next Node:", nextNode)
        # print("Coin Node:", nearestCoin)

        actionToNearestCoin = None

        if xdiff > 0:
            actionToNearestCoin = "RIGHT"
        if xdiff < 0:
            actionToNearestCoin = "LEFT"
        if ydiff > 0:
            actionToNearestCoin = "DOWN"
        if ydiff < 0:
            actionToNearestCoin = "UP"
        if ydiff == 0 and xdiff == 0:
            actionToNearestCoin = "WAIT"

        # print("Action to Coin:",actionToNearestCoin)

        return distanceToNearestCoin, actionToNearestCoin
    else:
        return math.inf, "None"


def reward_from_events(self, events: List[str]):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum


def indexToNode(x, y):
    return y * s.COLS + x


def NodeToIndex(node):
    x = node % s.COLS
    y = math.floor(node / s.COLS)
    return x, y


def graph(field):
    adjMatrix = np.zeros((s.COLS * s.ROWS, s.COLS * s.ROWS))
    for x in range(s.COLS):
        for y in range(s.ROWS):
            node = indexToNode(x, y)
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                neighbor = indexToNode(nx, ny)
                if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                    if field[nx, ny] >= 0 and field[x, y] >= 0:
                        adjMatrix[node, neighbor] = 1
                    else:
                        adjMatrix[node, neighbor] = math.inf
    printField(field)
    return csr_matrix(adjMatrix)


def printField(field):
    print(" ", end="")
    for x in range(s.COLS):
        print(" ___ ", end="")
    print("")
    for x in range(s.COLS):
        print("|", end="")
        for y in range(s.ROWS):
            if field[x, y] == -1:
                print(" XXX ", end="")
            else:
                node = indexToNode(x, y)
                numb = ""
                if node < 10:
                    numb = "00" + str(node)
                elif node < 100:
                    numb = "0" + str(node)
                elif node >= 100:
                    numb = "" + str(node)
                print(" " + numb + " ", end="")
        print("|")
    print(" ", end="")
    for x in range(s.COLS):
        print(" ___ ", end="")
    print("")
