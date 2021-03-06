from typing import List, Tuple
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .actions import Actions
import settings as s

COLS, ROWS = 17, 17


def bomb_indices(game_state: dict) -> List[Tuple]:
    """Author: Jonas Gann"""
    return [(bomb[0][0], bomb[0][1]) for bomb in game_state["bombs"]]


def enemy_indices(game_state: dict) -> List[Tuple]:
    """Author: Jonas Gann"""
    return [(other[3][0], other[3][1]) for other in game_state["others"]]


def to_node(index):
    """Author: Jonas Gann"""
    return index[1] * COLS + index[0]


def to_index(node):
    """Author: Jonas Gann"""
    return (node % COLS, math.floor(node / COLS))


class MovementGraph:
    def __init__(self, game_state: dict) -> None:
        """Author: Jonas Gann"""
        # Initialize some variables for easy access
        self.field = game_state["field"]
        self.explosion_map = game_state["explosion_map"]
        self.bomb_indices = bomb_indices(game_state)
        self.agent_index = game_state["self"][3]
        self.agent_node = to_node(self.agent_index)
        self.bombs = game_state["bombs"]
        self.enemies = game_state["others"]
        self.enemy_indices = enemy_indices(game_state)
        self.obstructed = {}
        # Create an adjacency matrix of the game field. Each field is interpreted as a node. A path between two nodes
        # exist if the corresponding fields lie next to each other and neither of the nodes is obstructed by e.g. a wall.
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
                        node_obstructed = self._node_obstructed(
                            node)   # Check if node is obstructed
                        self.obstructed[node] = node_obstructed
                        neighbor_obstructed = self._node_obstructed(
                            neighbor)   # Check if node is obstructed
                        if not node_obstructed and not neighbor_obstructed:  # Only add edge if neither nodes are obstructed
                            row.append(node)
                            col.append(neighbor)
                            data.append(1)
        self.matrix = csr_matrix((data, (row, col)))

    def is_within_field(self, x: int, y: int) -> bool:
        """Author: Jonas Gann"""
        """Check if a coordinate is within the size of the field"""
        return 0 <= x < COLS and 0 <= y < ROWS

    def _node_obstructed(self, node):
        """Author: Jonas Gann"""
        """Helper function to index_obstructed"""
        # Check if the field is obstructed at the node by wither a wall, a boy or an explosion
        x, y = to_index(node)
        return self._index_obstructed((x, y))

    def nearest_index(self, indices):
        """Author: Jonas Gann"""
        """Helper function for nearest_node"""
        nodes = [to_node(index) for index in indices]
        nearest_node = self.nearest_node(nodes)
        if nearest_node != None:
            return to_index(nearest_node)

    def nearest_node(self, nodes):
        """Author: Jonas Gann"""
        """Compute which of the passed nodes is the one nearest to the current agent position"""
        nodes = self.remove_obstructed_nodes(nodes)
        nodes = self.remove_nodes_out_of_range(nodes)
        if len(nodes) == 0:
            return None
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
            return None
        source = sources[self.agent_node]
        if source != -9999:
            return source
        else:
            return None

    def nearest_distance_index(self, index):
        """Author: Jonas Gann"""
        """Helper function for nearest_distance_node"""
        node = to_node(index)
        return self.nearest_distance_node(node)

    def nearest_distance_node(self, node):
        """Author: Jonas Gann"""
        """Computes the distance to the nearest of the passed nodes"""
        if node == None:
            return None
        distances, predecessors, sources = dijkstra(
            csgraph=self.matrix,
            directed=True,
            indices=node,
            return_predecessors=True,
            unweighted=True,
            min_only=True,
        )
        if not self._node_in_movement_range(self.agent_node):
            # Agent node is currently seperated from all other nodes and is therefore not contained in the movement_graph
            return None
        source = sources[self.agent_node]
        if source != -9999:  # A path to one of the nodes exists
            distance = distances[self.agent_node]
            return distance
        else:
            return None

    def next_step_to_nearest_node(self, nodes):
        """Author: Jonas Gann"""
        """Compute the next step which would move the agent towards the nearest of the passed nodes"""
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
        """Author: Jonas Gann"""
        """Check if the index is obstructed"""
        node = to_node(index)
        if node in self.obstructed:
            return self.obstructed[node]
        else:
            return True

    def node_obstructed(self, node):
        """Author: Jonas Gann"""
        """Check if the node is obstructed"""
        if node in self.obstructed:
            return self.obstructed[node]
        else:
            return True

    def _index_obstructed(self, index):
        """Author: Jonas Gann"""
        """Compute for a given game state which nodes are obstructed by e.g a wall"""
        # Check if the field is obstructed at the index by either a wall, a box, an explosion or an out of range error
        x, y = index
        in_range = 0 <= x < s.COLS and 0 <= y < s.ROWS
        is_wall = self.field[x, y] == -1
        is_box = self.field[x, y] == 1
        is_explosion = self.explosion_map[x, y] != 0
        is_bomb = (x, y) in self.bomb_indices and self.agent_index != (x, y)
        is_explosion_in_next_step = False
        is_enemy = (x, y) in self.enemy_indices
        for bomb in self.bombs:
            (xb, yb) = bomb[0]
            time_till_explosion = bomb[1]
            blast_indices = [(xb, yb)]
            for i in range(4):
                if self.field[xb + i, yb] == -1:
                    break
                if 0 < xb + i < s.COLS:
                    blast_indices.append((xb + i, yb))
            for i in range(4):
                if self.field[xb - i, yb] == -1:
                    break
                if s.COLS > xb - i > 0:
                    blast_indices.append((xb - i, yb))
            for i in range(4):
                if self.field[xb, yb + i] == -1:
                    break
                if 0 < yb + i < s.ROWS:
                    blast_indices.append((xb, yb + i))
            for i in range(4):
                if self.field[xb, yb - i] == -1:
                    break
                if s.ROWS > yb - i > 0:
                    blast_indices.append((xb, yb - i))
            if (x, y) in blast_indices and time_till_explosion == 0:
                is_explosion_in_next_step = True
        return is_wall or is_box or is_explosion or not in_range or is_bomb or is_explosion_in_next_step or is_enemy

    def _node_in_movement_range(self, node):
        """Author: Jonas Gann"""
        """Validate if a node can be reached by any other node"""
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        return node < self.matrix.shape[0]

    def _index_in_movement_range(self, index):
        """Author: Jonas Gann"""
        """Helper function for _node_in_movement_range"""
        node = to_node(index)
        return self._node_in_movement_range(node)

    def remove_obstructed_nodes(self, nodes):
        """Author: Jonas Gann"""
        """Remove nodes which are obstructed"""
        free_nodes = []
        for node in nodes:
            if not self.node_obstructed(node):
                free_nodes.append(node)
        return free_nodes

    def remove_nodes_out_of_range(self, nodes):
        """Author: Jonas Gann"""
        """Remove nodes which can not be reached by any other node"""
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        free_nodes = []
        for node in nodes:
            if self._node_in_movement_range(node):
                free_nodes.append(node)
        return free_nodes

    def next_step_to_nearest_index(self, indices):
        """Author: Jonas Gann"""
        """Helper function for next_step_to_nearest_node"""
        nodes = [to_node(index) for index in indices]
        return self.next_step_to_nearest_node(nodes)

    def blast_indices(self):
        """Author: Jonas Gann"""
        """Compute which indices are currently under blast"""
        blast_indices = []
        for x, y in self.bomb_indices:
            blast_indices.append((x, y))
            for i in range(4):
                if self.field[x + i, y] == -1:
                    break
                if 0 < x + i < s.COLS:
                    blast_indices.append((x + i, y))
            for i in range(4):
                if self.field[x - i, y] == -1:
                    break
                if s.COLS > x - i > 0:
                    blast_indices.append((x - i, y))
            for i in range(4):
                if self.field[x, y + i] == -1:
                    break
                if 0 < y + i < s.ROWS:
                    blast_indices.append((x, y + i))
            for i in range(4):
                if self.field[x, y - i] == -1:
                    break
                if s.ROWS > y - i > 0:
                    blast_indices.append((x, y - i))
        return blast_indices

    def blast_nodes(self):
        """Author: Jonas Gann"""
        """Helper function for blast_indices"""
        blast_indices = self.blast_indices()
        return [to_node(index) for index in blast_indices]
