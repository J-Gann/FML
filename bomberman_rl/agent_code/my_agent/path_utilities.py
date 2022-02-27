import settings as s
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from enum import Enum
import os

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5


def print_field(field):
    print(" ", end="")
    for x in range(s.COLS):
        print(" ___ ", end="")
    print("")
    # Notice: x and y loops are reversed, to transpose the matrix and present the field as seen on the gui
    for y in range(s.COLS):
        print("|",end="")
        for x in range(s.ROWS):
            if field[x,y] == -1: print(" XXX ",end="")
            else: 
                node = _index_to_node((x,y))
                numb = ""
                if node < 10: numb = "00" + str(node)
                elif node < 100: numb = "0" + str(node)
                elif node >= 100: numb = "" + str(node)
                print(" " + numb + " ",end="")
        print("|")
    print(" ", end="")
    for x in range(s.COLS):
        print(" ___ ", end="")
    print("")

def move_out_of_blast_zone():
    # filter nodes which are out of blast zone of placed bomb
    # _next_step_to_nearest_node
    # return move and distance out of blast zone
    pass

def box_in_the_way():
    # for each movement (left, right, up, down) return 1 if it would place the agent on a box
    pass

def move_to_nearest_box():
    # compute next move to nearest box
    pass

def blast_in_the_way():
    # for each movement (left, right, up, down) return a number if it would place the agent
    # on a blast zone. the number is the steps until the blast
    pass

def in_blast_zone():
    # return a number if the agent is in a blast zone. the number is time till explosion
    pass

def move_possible():
    # for each movement (left, right, up, down, wait, bomb) return 1 if the movement is possible
    pass

def move_to_nearest_coin(game_state):
    coins = game_state["coins"]
    return _next_step_to_nearest_node(coins, game_state)

def _next_step_to_nearest_node(nodes, game_state):
    if len(nodes) == 0: return Actions.WAIT.name, -9999
    nodes = [_index_to_node(node) for node in nodes]
    nodes = list(filter(lambda node: _is_not_blocked_node(game_state["field"], game_state["explosion_map"], node), nodes)) # remove all coins which are at a blocked position (inside an explosion)
    if len(nodes) == 0: return Actions.WAIT.name, -9999
    agent_node = _index_to_node(game_state["self"][3])
    if not _is_not_blocked_node(game_state["field"], game_state["explosion_map"], agent_node): return Actions.WAIT.name, -9999

    movement_graph = _create_graph(game_state["field"], game_state["explosion_map"])
    distances, predecessors, sources = dijkstra(csgraph=movement_graph, directed=True, indices=nodes, return_predecessors=True, unweighted=True, min_only=True)
    source = sources[agent_node]
    action = Actions.WAIT.name
    distance = -9999
    if source != -9999: # A path to one of the nodes exists
        distance = distances[agent_node]
        next_node = predecessors[agent_node]
        cx, cy = _node_to_index(next_node)
        ax, ay = _node_to_index(agent_node)
        if cx - ax > 0: action = Actions.RIGHT.name
        elif cx - ax < 0: action = Actions.LEFT.name
        elif cy - ay > 0: action = Actions.DOWN.name
        elif cy - ay < 0: action = Actions.UP.name
    return action_to_one_hot(action), [distance]

def action_to_one_hot(action): return [int(action == "UP"), int(action == "RIGHT"), int(action == "DOWN"), int(action == "LEFT"), int(action == "WAIT"), int(action == "BOMB")]

def _is_not_blocked_node(field, explosion_map, node):
    x,y = _node_to_index(node)
    return field[x, y] == 0 and explosion_map[x, y] == 0

def _create_graph(field, explosion_map):    
    # Fields are connected to at most 4 other fields, therefore the matrix is sparse.
    # Use an adjacency list to limit resource consumtion.
    row = []
    col = []
    data = []
    for x in range(s.COLS):
        for y in range(s.ROWS):
            # Transform inices to node label
            neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            for nx, ny in neighbors:
                # Test if the neighbor node exists
                if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                    node = _index_to_node((x, y))
                    neighbor = _index_to_node((nx, ny))
                    # Insert an edge between node and neighbor if neither node nor neighbor is a wall and neighbor is no explosion
                    if _is_not_blocked_node(field, explosion_map, node) and _is_not_blocked_node(field, explosion_map, neighbor):
                        row.append(node)
                        col.append(neighbor)
                        data.append(1)
    # Create a sparse adjacency matrix
    return csr_matrix((data, (row, col)))

def _index_to_node(index):
    return index[1] * s.COLS + index[0]

def _node_to_index(node):
    x = node % s.COLS 
    y = math.floor(node / s.COLS)
    return (x, y)

