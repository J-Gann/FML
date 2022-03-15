import numbers
import numpy as np
from .actions import Actions
import re

# TODO: import from settings
COLS = 17
ROWS = 17
positions = np.empty((ROWS, COLS, 2))

for i in range(ROWS):
    for j in range(COLS):
        positions[i][j][0] = i
        positions[i][j][1] = j


def format_position(v: np.array) -> str:
    "Formats a 2d vector"

    assert len(v.shape) == 1 and v.shape[0] == 2, "must be 2D"

    return f"{int(v[0]):2d}, {int(v[1]):2d}"


def format_boolean(v: np.array) -> str:
    """takes bool or int, returns yes or no"""
    assert isinstance(v, numbers.Number) or (len(v.shape) == 1 and v.shape[0] == 1), "must be scalar"

    if int(v) == 1:
        return "yes"
    else:
        return "no"


def k_closest(position, objects, k=5, pad=True):
    """If pad=True returns size k x 2 (pads with zeros). Uses manhattan distance"""
    if objects.shape[0] == 0:
        return np.zeros((k, 2))

    if pad:
        objects = pad_matrix(objects, k)

    distances = manhattan_distance(position, objects)

    return objects[np.argsort(distances)][:k, :]


def crate_positions(field):
    return positions[field == 1]


def manhattan_distance(position, objects):
    """objects with shape D x 2"""
    return np.abs(objects - position).sum(axis=1)


def pad_matrix(mat, size):
    """Pads a N x 2 matrix to size x 2"""
    N = mat.shape[0]
    zero_rows_to_add = max(0, size - N)
    return np.concatenate([mat, np.zeros((zero_rows_to_add, 2))])


def action_new_index(agent_position, action):
    x, y = agent_position
    # return the node an action puts the agent on
    if action == Actions.UP:
        return (x, y - 1)
    elif action == Actions.DOWN:
        return (x, y + 1)
    elif action == Actions.LEFT:
        return (x - 1, y)
    elif action == Actions.RIGHT:
        return (x + 1, y)
    else:
        return agent_position


def get_agent_position(game_state: dict) -> np.array:
    return game_state["self"][-1]


def get_enemy_positions(game_state: dict) -> np.array:
    return [(enemy[3][0], enemy[3][1]) for enemy in game_state["others"]]


def camel_to_snake_case(s):
    return re.sub("(?!^)([A-Z]+)", r"_\1", s).lower()


def snake_to_camel_case(s):
    return re.sub("(^|_)(.)", lambda x: x.group(2).upper(), s)
