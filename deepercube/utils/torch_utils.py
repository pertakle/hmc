# import numpy as np
import torch

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_torch_cube_device(device: torch.device) -> None:
    _DEVICE = device

def get_torch_cube_device() -> torch.device:
    return _DEVICE

SURROUNDINGS = torch.tensor(
    [
        [
            [4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
        ],
        [
            [0, 0, 0, 2, 2, 2, 5, 5, 5, 4, 4, 4],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
        ],
        [
            [0, 0, 0, 3, 3, 3, 5, 5, 5, 1, 1, 1],
            [2, 2, 2, 0, 1, 2, 0, 0, 0, 2, 1, 0],
            [0, 1, 2, 0, 0, 0, 2, 1, 0, 2, 2, 2],
        ],
        [
            [5, 5, 5, 2, 2, 2, 0, 0, 0, 4, 4, 4],
            [2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
        ],
        [
            [3, 3, 3, 0, 0, 0, 1, 1, 1, 5, 5, 5],
            [2, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2],
            [2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 1, 2],
        ],
        [
            [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        ],
    ],
    dtype=torch.long,
    device=_DEVICE
)

SURROUNDINGS_SHIFT = torch.stack([torch.roll(SURROUNDINGS, 3, 2), torch.roll(SURROUNDINGS, -3, 2)])
