import numpy as np

positions = np.empty((17, 17, 2))

for i in range(17):
    for j in range(17):
        positions[i][j][0] = i
        positions[i][j][1] = j


def format_position(v: np.array) -> str:
    "Formats a 2d vector"

    assert len(v.shape) == 1 and v.shape[0] == 2, "must be 2D"

    return f"{int(v[0]):2d}, {int(v[1]):2d}"


def format_boolean(v: np.array) -> str:
    """takes bool or int, returns yes or no"""
    assert len(v.shape) == 1 and v.shape[0] == 1, "must be scalar"

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