import settings as s
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from enum import Enum

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

    def as_string(self): self.name

    def as_one_hot(self): return [int(self.value == index) for index in range(6)]


class FeatureExtraction():
    def __init__(self, game_state):
        self.game_state = game_state
        self.field = self.game_state["field"]
        self.explosion_map = self.game_state["explosion_map"]
        self.movement_graph = self._create_movement_graph()
        self.agent_node = self._to_node(self.game_state["self"][3])
        self.agent_index = self.game_state["self"][3]
        self.coins = game_state["coins"]
        self.bombs = game_state["bombs"]
        self.bomb_indices = [(bomb[0][0], bomb[0][1]) for bomb in self.bombs]

    def _create_movement_graph(self):    
        row = []
        col = []
        data = []
        for x in range(s.COLS):
            for y in range(s.ROWS):
                neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
                for nx, ny in neighbors:
                    if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                        node = self._to_node((x, y))
                        neighbor = self._to_node((nx, ny))
                        if not self._node_obstructed(node) and not self._node_obstructed(neighbor):
                            row.append(node)
                            col.append(neighbor)
                            data.append(1)
        return csr_matrix((data, (row, col)))

    def _index_obstructed(self, index):
        # Check if the field is obstructed at the index by wither a wall, a boy, an explosion or an out of range error
        x, y = index
        in_range = 0 <= x < s.COLS and 0 <= y < s.ROWS 
        is_wall = self.field[x, y] == -1
        is_box = self.field[x, y] == 1
        is_explosion = self.explosion_map[x, y] != 0
        return is_wall or is_box or is_explosion or not in_range

    def _node_in_movement_range(self, node):
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        return node < self.movement_graph.shape[0]

    def _index_in_movement_range(self, index):
        node = self._to_node(index)
        return self._node_in_movement_range(node)

    def _node_obstructed(self, node):
        # Check if the field is obstructed at the node by wither a wall, a boy or an explosion
        x, y = self._to_index(node)
        return self._index_obstructed((x, y))

    def _to_node(self, index): return index[1] * s.COLS + index[0]

    def _to_index(self, node): return (node % s.COLS, math.floor(node / s.COLS))

    def _remove_obstructed_nodes(self, nodes):
        free_nodes = []
        for node in nodes:
            if not self._node_obstructed(node): free_nodes.append(node)
        return free_nodes

    def _remove_nodes_out_of_range(self, nodes):
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        free_nodes = []
        for node in nodes:
            if self._node_in_movement_range(node): free_nodes.append(node)
        return free_nodes

    def _next_step_to_nearest_index(self, indices):
        nodes = [self._to_node(index) for index in indices]
        return self._next_step_to_nearest_node(nodes)

    def _next_step_to_nearest_node(self, nodes):
        # Find the nearest reachable node in from the nodes array originating from the agent position and return the next move along the shortest path
        nodes = self._remove_obstructed_nodes(nodes)
        nodes = self._remove_nodes_out_of_range(nodes)
        if len(nodes) == 0: return Actions.WAIT
        distances, predecessors, sources = dijkstra(csgraph=self.movement_graph, directed=True, indices=nodes, return_predecessors=True, unweighted=True, min_only=True)
        if not self._node_in_movement_range(self.agent_node):
            # Agent node is currently seperated from all other nodes and is therefore not contained in the movement_graph
            return Actions.WAIT 
        source = sources[self.agent_node]
        if source != -9999: # A path to one of the nodes exists
            distance = distances[self.agent_node]
            next_node = predecessors[self.agent_node]
            cx, cy = self._to_index(next_node)
            ax, ay = self.agent_index
            if cx - ax > 0: return Actions.RIGHT
            elif cx - ax < 0: return Actions.LEFT
            elif cy - ay > 0: return Actions.DOWN
            elif cy - ay < 0: return Actions.UP
        else:
            return Actions.WAIT

    def _blast_indices(self):
        blast_indices = []
        for x, y in self.bomb_indices:
            for i in range(3):
                if 0 < x+i < s.COLS: blast_indices.append((x+i, y))
                if s.COLS > x-i > 0: blast_indices.append((x-i, y))
                if 0 < y+i < s.ROWS: blast_indices.append((x, y+i))
                if s.ROWS > y-i > 0: blast_indices.append((x, y-i))
        return blast_indices

    def _blast_nodes(self):
        blast_indices = self._blast_indices()
        return [self._to_node(index) for index in blast_indices]

    def FEATURE_move_to_nearest_coin(self): return self._next_step_to_nearest_index(self.coins)

    def FEATURE_move_out_of_blast_zone(self):
        free_indices = []
        blast_indices = self._blast_indices()
        for x in range(s.COLS):
            for y in range(s.ROWS):
                index = (x, y)
                obstructed = self._index_obstructed(index)
                in_blast_zone = index in blast_indices
                if not obstructed and not in_blast_zone: free_indices.append(index)
        if self.agent_index in blast_indices: return self._next_step_to_nearest_index(free_indices)
        else: return Actions.WAIT

    def FEATURE_move_to_nearest_box(self):
        indices = np.argwhere(self.field == 1)
        return self._next_step_to_nearest_index(indices)

    def FEATURE_in_blast_zone(self):
        blast_nodes = self._blast_nodes()
        if self.agent_node in blast_nodes: return [1]
        else: return [0]

    def FEATURE_action_possible(self):
        neighbor_up = (self.agent_index[0], self.agent_index[1]+1)
        neighbor_down = (self.agent_index[0], self.agent_index[1]-1)
        neighbor_left = (self.agent_index[0]+1, self.agent_index[1])
        neighbor_right = (self.agent_index[0]-1, self.agent_index[1])

        up_allowed = not self._index_obstructed(neighbor_up) and self._index_in_movement_range(neighbor_up)
        down_allowed  = not self._index_obstructed(neighbor_down) and self._index_in_movement_range(neighbor_down)
        left_allowed  = not self._index_obstructed(neighbor_left) and self._index_in_movement_range(neighbor_left)
        right_allowed  = not self._index_obstructed(neighbor_right) and self._index_in_movement_range(neighbor_right)

        wait_allowed = True

        bomb_allowed = self.game_state["self"][2]

        return [int(up_allowed ), int(right_allowed ), int(down_allowed ), int(left_allowed ), int(wait_allowed), int(bomb_allowed)]



'''
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


def box_in_the_way(game_state):
    agent = game_state["self"][3]
    agent_x = agent[0]
    agent_y = agent[1]

    neighbor_up = (agent_x, agent_y+1)
    neighbor_down = (agent_x, agent_y-1)
    neighbor_left = (agent_x+1, agent_y)
    neighbor_right = (agent_x-1, agent_y)

    up_box = game_state["field"][neighbor_up[0]][neighbor_up[1]] == 1
    down_box = game_state["field"][neighbor_down[0]][neighbor_down[1]] == 1
    left_box = game_state["field"][neighbor_left[0]][neighbor_left[1]] == 1
    right_box = game_state["field"][neighbor_right[0]][neighbor_right[1]] == 1
    return [int(up_box), int(right_box), int(down_box), int(left_box)]
    

def blast_in_the_way():
    # for each movement (left, right, up, down) return a number if it would place the agent
    # on a blast zone. the number is the steps until the blast
    pass
'''