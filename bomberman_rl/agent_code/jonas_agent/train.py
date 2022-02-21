from collections import namedtuple, deque
from operator import mod
import pickle
from typing import List

from cv2 import undistort
import events as e
from sklearn import linear_model
import numpy as np
import settings as s
from sklearn import tree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import math

LEARNING_RATE = 0.5
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DISCOUNT = 0.95

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Create regression trees predicting the value of their move in a given state
    self.trees = {
        "UP": tree.DecisionTreeRegressor(),
        "DOWN": tree.DecisionTreeRegressor(),
        "LEFT": tree.DecisionTreeRegressor(),
        "RIGHT": tree.DecisionTreeRegressor(),
        "WAIT": tree.DecisionTreeRegressor(),
        "BOMB": tree.DecisionTreeRegressor()
    }

    # Array with columns: state, action, next_state, reward for every game step
    self.transitions = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if not hasattr(self, "graph"): self.graph = graph(new_game_state["field"])

    self.transitions.append([state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events)])

    # Initialize game field as graph (can not be done in setup, as game_state is not available)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    #batch = filter(lambda transition: transition)

    # 1) get transitions where action a was performed: self.transitions[self.transitions[:,2] == action]
    # 2) for each of the selected transitions:
    # 4)    calculate state_transposed * ( actual_sum_of_rewards_after_action - state * action_beta)
    # 5) sum up the results
    # 6) normalize
    # 7) add to previous action_beta

    for action in ACTIONS:
        # get bathc of transitions with specified action
      #  batch = filter(lambda transition: transition[1] == action, self.transitions)

        # calculate q_value_update (pseudocode)
        # current guess + learning rate * (    (actual reward + predicted future reward)     -    predicted action value  )
       # q_update = self.trees[action].predict(batch.old_game_state) + LEARNING_RATE * (batch.rewards_from_event + DISCOUNT * max([for tree in self.trees: tree.predict(batch.new_game_state)]) - self.trees[action].predict(batch.old_game_state))

        # train new action tree with updated values and overwrite old tree (pseudocode)
       # newTree = tree.DecisionTreeRegressor()
       # newTree.fit(batch.old_game_states, q_update)
       # self.trees[action] = newTree
       pass


def state_to_features(self, game_state):
    """"
    Modify game state to expressive features which can be used by the learning algorithm
    """
    if(game_state is None): return None
    # use something like one encoding for actions:
    # [possibleActions, actionTowardsNearestCoin, actionsToSafety] = [0,0,0,1,0,  0,1,0,0,0    0,1,1,0,0]

    #better?
    # create feature vector for each actionTree:
    # [actionPossible(1/0), goesTowardCoinOfSmallestDistance(1/0), ] = [1,   1,   0]
    
    # [upTreeFeatures, downTreeFeatures, ..., BombTreeFeatures]

    # additionalFeatures
    # [WeInEnemyBlastZone(1/0), WeMovesOutOfEnemyBlastZone(1..n), EnemiesInOurBlastZone(1..n), EnemiesMovesOutOfOurBlastZoneSum(1..n), DistanceToNearestBlastZone(1..n),
    # ActionTowardsNearestBlastZone(1/0), TTLNearestBlastZone(1..n), ]
    # ...

    distanceToNearestCoin, actionToNearestCoin = getPathFeatures(self, game_state)

    features = {
        "UP": [distanceToNearestCoin, int(actionToNearestCoin == "UP")],
        "DOWN": [distanceToNearestCoin, int(actionToNearestCoin == "DOWN")],
        "LEFT": [distanceToNearestCoin, int(actionToNearestCoin == "LEFT")],
        "RIGHT": [distanceToNearestCoin, int(actionToNearestCoin == "RIGHT")],
        "WAIT": [distanceToNearestCoin, int(actionToNearestCoin == "WAIT")],
        "BOMB": [distanceToNearestCoin, int(actionToNearestCoin == "BOMB")]
    }

    return features
# get distanceToNearestCoin and actionToNearestCoin
def getPathFeatures(self, game_state):
    coins = [indexToNode(coin[0], coin[1]) for coin in game_state["coins"]]
    if(len(coins)):
        x, y = game_state["self"][3]
        agentNode = indexToNode(x, y)
        dist_matrix, predecessors = dijkstra(csgraph=self.graph, directed=False, indices=agentNode, return_predecessors=True)
        print(dist_matrix)
        coinDistances = dist_matrix[coins]
        distanceToNearestCoin = np.min(coinDistances)

        nearestCoin = np.argmin(coinDistances)

        def getPath(node):
            pred = predecessors[node]
            if pred == agentNode:
                return []
            path = getPath(pred)
            path.append(pred)
            return path

        path = getPath(nearestCoin)

        nextNode = None
        if len(path) == 0:
            nextNode =  agentNode
        else:
            nextNode = path[0]

        nextIndices = NodeToIndex(nextNode)

        xdiff = nextIndices[0] - x
        ydiff = nextIndices[1] - y


        actionToNearestCoin = None

        if(xdiff > 0): actionToNearestCoin = "RIGHT"
        if(xdiff < 0): actionToNearestCoin = "LEFT"
        if(ydiff > 0): actionToNearestCoin = "DOWN"
        if(ydiff < 0): actionToNearestCoin = "UP"
        if(ydiff == 0 and xdiff == 0): actionToNearestCoin = "Wait"

        return distanceToNearestCoin, actionToNearestCoin
    else:
        return math.inf, "None"

def reward_from_events(self, events: List[str]):
    """
    Modify the rewards your agent get so as to en/discourage certain behavior.
    """
    return 1

def indexToNode(x, y): return y * s.COLS + x

def NodeToIndex(node):
    x = node % s.COLS 
    y = math.floor(node / s.COLS)
    return x, y

def graph(field): 
    adjMatrix = np.zeros((s.COLS * s.ROWS, s.COLS * s.ROWS))
    for x in range(s.COLS):
        for y in range(s.ROWS):
            node = indexToNode(x, y)
            neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            for nx, ny in neighbors:
                neighbor = indexToNode(nx, ny)
                if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                    if field[nx, ny] >= 0 and field [x, y] >= 0: adjMatrix[node, neighbor] = 1
                    else: adjMatrix[node, neighbor] = math.inf
    return csr_matrix(adjMatrix)
