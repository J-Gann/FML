import settings as s
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import pickle
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

def setup_graph_features(self, field, load=True, save=False):
    if load and os.path.isfile("./data/bomberman_graph") and os.path.isfile("./data/bomberman_dist_matrix") and os.path.isfile("./data/bomberman_predecessors_matrix"):
        bomberman_graph_file = open('./data/bomberman_graph', 'rb')
        self.graph = pickle.load(bomberman_graph_file)
        bomberman_graph_file.close()

        bomberman_dist_matrix_file = open('./data/bomberman_dist_matrix', 'rb')
        self.dist_matrix = pickle.load(bomberman_dist_matrix_file)
        bomberman_dist_matrix_file.close()

        bomberman_predecessors_matrix_file = open('./data/bomberman_predecessors_matrix', 'rb')
        self.predecessors_matrix = pickle.load(bomberman_predecessors_matrix_file)
        bomberman_predecessors_matrix_file.close()
    elif load:
        print("[Warn] Cannot load precomputed path data. Recomputing ...")
        self.graph = _create_graph(field)
        self.dist_matrix, self.predecessors_matrix = _shortest_paths(self.graph)
    else:
        self.graph = _create_graph(field)
        self.dist_matrix, self.predecessors_matrix = _shortest_paths(self.graph)


    if save:
        bomberman_graph_file = open('./data/bomberman_graph', 'wb')
        pickle.dump(self.graph, bomberman_graph_file)
        bomberman_graph_file.close()

        bomberman_dist_matrix_file = open('./data/bomberman_dist_matrix', 'wb')
        pickle.dump(self.dist_matrix, bomberman_dist_matrix_file)
        bomberman_dist_matrix_file.close()
        
        bomberman_predecessors_matrix_file = open('./data/bomberman_predecessors_matrix', 'wb')
        pickle.dump(self.predecessors_matrix, bomberman_predecessors_matrix_file)
        bomberman_predecessors_matrix_file.close()

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


def move_to_nearest_bomb(self, agent, bombs):
    bombs = [(bomb[0][0], bomb[0][1]) for bomb in bombs]
    bomb_nodes = [_index_to_node(bomb) for bomb in bombs]
    agent_node = _index_to_node(agent)
    return _move_to_nearest_bomb(self.dist_matrix, self.predecessors_matrix, agent_node, bomb_nodes)
    

def _move_to_nearest_bomb(dist_matrix, predecessors_matrix, agent_node, bomb_nodes):
    distances = [dist_matrix[agent_node, bomb] for bomb in bomb_nodes]
    nearest_bomb = bomb_nodes[np.argmin(distances)]
    shortest_path = _traverse_shortest_path(predecessors_matrix, agent_node, nearest_bomb)
    if len(shortest_path) == 0: return Actions.WAIT.name

    next_node = shortest_path[1]

    cx, cy = _node_to_index(next_node)
    ax, ay = _node_to_index(agent_node)

    if cx - ax > 0: return Actions.RIGHT.name
    elif cx - ax < 0: return Actions.LEFT.name
    elif cy - ay > 0: return Actions.DOWN.name
    elif cy - ay < 0: return Actions.UP.name


def box_in_the_way(game_state):
    pass

def in_blast_zone():
    pass



def move_to_nearest_coin(self, agent, coins):
    coin_nodes = [_index_to_node(coin) for coin in coins]
    agent_node = _index_to_node(agent)
    return _move_to_nearest_coin(self.dist_matrix, self.predecessors_matrix, agent_node, coin_nodes)

def _move_to_nearest_coin(dist_matrix, predecessors_matrix, agent_node, coin_nodes):
    distances = [dist_matrix[agent_node, coin] for coin in coin_nodes]
    nearest_coin = coin_nodes[np.argmin(distances)]
    shortest_path = _traverse_shortest_path(predecessors_matrix, agent_node, nearest_coin)
    if len(shortest_path) == 0: return Actions.WAIT.name

    next_node = shortest_path[1]

    cx, cy = _node_to_index(next_node)
    ax, ay = _node_to_index(agent_node)

    if cx - ax > 0: return Actions.RIGHT.name
    elif cx - ax < 0: return Actions.LEFT.name
    elif cy - ay > 0: return Actions.DOWN.name
    elif cy - ay < 0: return Actions.UP.name

def _create_graph(field):    
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
                    row.append(node)
                    col.append(neighbor)
                    # Insert an edge between node and neighbor if neither node nor neighbor is a wall
                    if field[nx, ny] >= 0 and field[x, y] >= 0: data.append(1)
                    # If either node or neighbor is a wall, set edge weight to infinity => shortest path algorithms wont use that
                    # edge, e.g wont move to the field with the wall 
                    else: data.append(math.inf)
    # Create a sparse adjacency matrix
    return csr_matrix((data, (row, col)))

def _shortest_paths(graph):
    dist_matrices, predecessors_matrix = shortest_path(csgraph=graph, directed=False, return_predecessors=True)
    return dist_matrices, predecessors_matrix

def _traverse_shortest_path(predecessors_matrix, source, target):
    if source == target: return []
    predecessors = predecessors_matrix[source]
    def get_path(node):
        pred = predecessors[node]
        if pred == source: return [source]
        path = get_path(pred)
        path.append(pred)
        return path
    path = get_path(target)
    path.append(target)
    return path

def _index_to_node(index):
    return index[1] * s.COLS + index[0]

def _node_to_index(node):
    x = node % s.COLS 
    y = math.floor(node / s.COLS)
    return (x, y)

