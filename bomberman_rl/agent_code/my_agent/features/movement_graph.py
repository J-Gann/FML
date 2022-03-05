from typing import List, Tuple
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .actions import Actions

COLS, ROWS = 17, 17


def bomb_indices(game_state: dict) -> List[Tuple]:
    return [(bomb[0][0], bomb[0][1]) for bomb in game_state["bombs"]]


def to_node(index):
    return index[1] * COLS + index[0]


def to_index(node):
    return (node % COLS, math.floor(node / COLS))


class MovementGraph:
    def __init__(self, game_state: dict) -> None:
        self.field = game_state["field"]
        self.explosion_map = game_state["explosion_map"]
        self.bomb_indices = bomb_indices(game_state)
        self.agent_index = game_state["self"][3]
        self.agent_node = to_node(self.agent_index)

        row = []
        col = []
        data = []
        for x in range(COLS):
            for y in range(ROWS):
                neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
                for nx, ny in neighbors:
                    if self.is_within_field(nx, ny):
                        node = to_node((x, y))
                        neighbor = to_node((nx, ny))
                        if not self.node_obstructed(node) and not self.node_obstructed(neighbor):
                            row.append(node)
                            col.append(neighbor)
                            data.append(1)
        self.matrix = csr_matrix((data, (row, col)))

    def is_within_field(self, x: int, y: int) -> bool:
        return 0 <= x < COLS and 0 <= y < ROWS

    def node_obstructed(self, node):
        # Check if the field is obstructed at the node by wither a wall, a boy or an explosion
        x, y = to_index(node)
        return self.index_obstructed((x, y))

    def next_step_to_nearest_node(self, nodes):
        # Find the nearest reachable node in from the nodes array originating from the agent position and return the next move along the shortest path
        nodes = self.remove_obstructed_nodes(nodes)
        nodes = self.remove_nodes_out_of_range(nodes)
        if len(nodes) == 0:
            return Actions.NONE
        distances, predecessors, sources = dijkstra(
            csgraph=self.matrix,
            directed=True,
            indices=nodes,
            return_predecessors=True,
            unweighted=True,
            min_only=True,
        )
        if not self._node_in_movement_range(self.agent_node):
            # Agent node is currently seperated from all other nodes and is therefore not contained in the movement_graph
            return Actions.NONE
        source = sources[self.agent_node]
        if source != -9999:  # A path to one of the nodes exists
            distance = distances[self.agent_node]
            next_node = predecessors[self.agent_node]
            cx, cy = to_index(next_node)
            ax, ay = self.agent_index
            if cx - ax > 0:
                return Actions.RIGHT
            elif cx - ax < 0:
                return Actions.LEFT
            elif cy - ay > 0:
                return Actions.DOWN
            elif cy - ay < 0:
                return Actions.UP
        else:
            return Actions.NONE

    def index_obstructed(self, index):
        # Check if the field is obstructed at the index by wither a wall, a boy, an explosion or an out of range error
        x, y = index
        in_range = 0 <= x < COLS and 0 <= y < ROWS
        is_wall = self.field[x, y] == -1
        is_box = self.field[x, y] == 1
        is_explosion = self.explosion_map[x, y] != 0
        is_bomb = (x, y) in self.bomb_indices and self.agent_index != (x, y)
        return is_wall or is_box or is_explosion or not in_range or is_bomb

    def _node_in_movement_range(self, node):
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        return node < self.matrix.shape[0]

    def _index_in_movement_range(self, index):
        node = to_node(index)
        return self._node_in_movement_range(node)

    def remove_obstructed_nodes(self, nodes):
        free_nodes = []
        for node in nodes:
            if not self.node_obstructed(node):
                free_nodes.append(node)
        return free_nodes

    def remove_nodes_out_of_range(self, nodes):
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        free_nodes = []
        for node in nodes:
            if self._node_in_movement_range(node):
                free_nodes.append(node)
        return free_nodes

    def next_step_to_nearest_index(self, indices):
        nodes = [to_node(index) for index in indices]
        return self.next_step_to_nearest_node(nodes)

    def blast_indices(self):
        blast_indices = []
        for x, y in self.bomb_indices:
            for i in range(4):
                if 0 < x + i < COLS:
                    blast_indices.append((x + i, y))
                if COLS > x - i > 0:
                    blast_indices.append((x - i, y))
                if 0 < y + i < ROWS:
                    blast_indices.append((x, y + i))
                if ROWS > y - i > 0:
                    blast_indices.append((x, y - i))
        return blast_indices

    def blast_nodes(self):
        blast_indices = self.blast_indices()
        return [to_node(index) for index in blast_indices]
