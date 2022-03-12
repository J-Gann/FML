import settings as s
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from enum import Enum
import copy

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5
    NONE = 6

    def as_string(self): self.name

    def as_one_hot(self): return [int(self.value == index) for index in range(6)]


class FeatureExtraction():
    def __init__(self, game_state, past_moves):
        self.game_state = game_state
        self.field = self.game_state["field"]
        self.explosion_map = self.game_state["explosion_map"]
        self.agent_node = self._to_node(self.game_state["self"][3])
        self.agent_index = self.game_state["self"][3]
        self.coins = game_state["coins"]
        self.bombs = game_state["bombs"]
        self.bomb_indices = [(bomb[0][0], bomb[0][1]) for bomb in self.bombs]
        self.movement_graph = self._create_movement_graph()
        self.enemies = game_state["others"]
        self.enemy_indices = [(enemy[3][0], enemy[3][1]) for enemy in self.enemies]
        self.past_moves = [Actions[move].value for move in past_moves]

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
        # Check if the field is obstructed at the index by either a wall, a box, an explosion or an out of range error
        x, y = index
        in_range = 0 <= x < s.COLS and 0 <= y < s.ROWS 
        is_wall = self.field[x, y] == -1
        is_box = self.field[x, y] == 1
        is_explosion = self.explosion_map[x, y] != 0
        is_bomb = (x, y) in self.bomb_indices and self.agent_index != (x, y)
        is_explosion_in_next_step = False
        for bomb in self.game_state["bombs"]:
            (xb, yb) = bomb[0]
            time_till_explosion = bomb[1]
            blast_indices = [(xb, yb)]
            for i in range(4):
                if self.field[xb+i, yb] == -1: break
                if 0 < xb+i < s.COLS: blast_indices.append((xb+i, yb))
            for i in range(4):
                if self.field[xb-i, yb] == -1: break
                if s.COLS > xb-i > 0: blast_indices.append((xb-i, yb))
            for i in range(4):
                if self.field[xb, yb+i] == -1: break
                if 0 < yb+i < s.ROWS: blast_indices.append((xb, yb+i))
            for i in range(4):
                if self.field[xb, yb-i] == -1: break
                if s.ROWS > yb-i > 0: blast_indices.append((xb, yb-i))
            if (x, y) in blast_indices and time_till_explosion == 0: is_explosion_in_next_step = True
        return is_wall or is_box or is_explosion or not in_range or is_bomb or is_explosion_in_next_step

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
        if len(nodes) == 0: return Actions.NONE
        distances, predecessors, sources = dijkstra(csgraph=self.movement_graph, directed=True, indices=nodes, return_predecessors=True, unweighted=True, min_only=True)
        if not self._node_in_movement_range(self.agent_node):
            # Agent node is currently seperated from all other nodes and is therefore not contained in the movement_graph
            return Actions.NONE 
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
            return Actions.NONE

    def _blast_indices(self):
        blast_indices = []
        for x, y in self.bomb_indices:
            blast_indices.append((x, y))
            for i in range(4):
                if self.field[x+i, y] == -1: break
                if 0 < x+i < s.COLS: blast_indices.append((x+i, y))
            for i in range(4):
                if self.field[x-i, y] == -1: break
                if s.COLS > x-i > 0: blast_indices.append((x-i, y))
            for i in range(4):
                if self.field[x, y+i] == -1: break
                if 0 < y+i < s.ROWS : blast_indices.append((x, y+i))
            for i in range(4):
                if self.field[x, y-i] == -1: break
                if s.ROWS > y-i > 0: blast_indices.append((x, y-i))
        #print("blast_indices", blast_indices)
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
        if self.agent_index in blast_indices:
            res = self._next_step_to_nearest_index(free_indices)
            #print("move_out_of_blast", res)
            return res
        else:
            #print("move_out_of_blast", Actions.NONE)
            return Actions.NONE

    def FEATURE_move_next_to_nearest_box(self):
        box_neighbors = []
        indices = np.argwhere(self.field == 1)
        tuple_indices = [(index[0], index[1]) for index in indices]
        # find all neighbors of boxes the agent can move to (the box itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            for nx, ny in neighbors:
                if not self._index_obstructed((nx,ny)):
                    box_neighbors.append((nx, ny))
        if self.agent_index in box_neighbors:
            #print("move_to_nearest_box", Actions.WAIT)
            return Actions.WAIT
        step = self._next_step_to_nearest_index(box_neighbors)
        #print("move_to_nearest_box", step)
        #print(step)
        return step

    def FEATURE_boxes_in_agent_blast_range(self):
        sum = 0
        x, y = self.agent_index
        for i in range(4):
            if self.field[x+i, y] == -1: break
            if 0 < x+i < s.COLS and self.field[x+i, y] == 1: sum+= 1
        for i in range(4):
            if self.field[x-i, y] == -1: break
            if s.COLS > x-i > 0 and self.field[x-i, y] == 1: sum+= 1
        for i in range(4):
            if self.field[x, y+i] == -1: break
            if 0 < y+i < s.ROWS and self.field[x, y+i] == 1: sum+= 1
        for i in range(4):
            if self.field[x, y-i] == -1: break
            if s.ROWS > y-i > 0 and self.field[x, y-i] == 1: sum+= 1
        #print("boxes_in_blast",sum)
        return [sum]

    def FEATURE_in_blast_zone(self):
        blast_nodes = self._blast_nodes()
        if self.agent_node in blast_nodes:
            #print("in_blast", [1])
            return [1]
        else:
            #print("in_blast", [0])
            return [0]

    def _action_new_index(self, action):
        # return the node an action puts the agent on
        if action == Actions.UP: return (self.agent_index[0], self.agent_index[1]-1)
        elif action == Actions.DOWN: return (self.agent_index[0], self.agent_index[1]+1)
        elif action == Actions.LEFT: return (self.agent_index[0]-1, self.agent_index[1])
        elif action == Actions.RIGHT: return (self.agent_index[0]+1, self.agent_index[1])
        else: return self.agent_index

    def _action_new_node(self, action):
        index = self._action_new_index(action)
        return self._to_node(index)

    def FEATURE_actions_possible(self):
        res = []
        for action in Actions:
            if action == Actions.NONE: pass
            elif action == Actions.BOMB: res.append(int(self.game_state["self"][2]))
            else: 
                new_index = self._action_new_index(action)
                obstructed = self._index_obstructed(new_index)
                res.append(int(not obstructed))
        #print("moves_possible", res)
        #print("agent_index", self.agent_index)
        return res

    def _action_possible(self, action):
        if action == Actions.NONE: pass
        elif action == Actions.BOMB: return self.game_state["self"][2]
        else: 
            new_index = self._action_new_index(action)
            #print("agent_index", self.agent_index)
            #print("new_index", new_index)
            obstructed = self._index_obstructed(new_index)
            return not obstructed

    def FEATURE_could_escape_own_bomb(self):
        old_bomb_indices = copy.deepcopy(self.bomb_indices)
        self.bomb_indices.append(self.agent_index)
        res = self.FEATURE_move_out_of_blast_zone() != Actions.NONE
        self.bomb_indices = old_bomb_indices
        #print("bomb_good", res)
        return [res]

    def FEATURE_move_next_to_nearest_enemy(self):
        enemy_neighbors = []
        tuple_indices = [(index[0], index[1]) for index in self.enemy_indices]
        # find all neighbors of enemyes the agent can move to (the enemy itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            for nx, ny in neighbors:
                if not self._index_obstructed((nx,ny)):
                    enemy_neighbors.append((nx, ny))
        if self.agent_index in enemy_neighbors:
            #print("move_to_nearest_enemy", Actions.WAIT)
            return Actions.WAIT
        step = self._next_step_to_nearest_index(enemy_neighbors)
        #print("move_to_nearest_enemy", step)
        #print(step)
        return step

    def FEATURE_enemies_in_agent_blast_range(self):
        sum = 0
        x, y = self.agent_index
        for i in range(4):
            if self.field[x+i, y] == -1: break
            if 0 < x+i < s.COLS and (x+i, y) in self. enemy_indices: sum+= 1
        for i in range(4):
            if self.field[x-i, y] == -1: break
            if s.COLS > x-i > 0 and (x-i, y) in self. enemy_indices: sum+= 1
        for i in range(4):
            if self.field[x, y+i] == -1: break
            if 0 < y+i < s.ROWS and (x, y+i) in self. enemy_indices: sum+= 1
        for i in range(4):
            if self.field[x, y-i] == -1: break
            if s.ROWS > y-i > 0 and (x, y-i) in self. enemy_indices: sum+= 1
        #print("boxes_in_blast",sum)
        return [sum]

    def FEATURE_past_moves(self, n=4):
        if len(self.past_moves) < n: return [-1 for i in range(n)]
        moves = self.past_moves[-n:]
        return moves