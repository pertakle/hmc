import numpy as np
import torch
from deepercube.utils import torch_utils as tut
from typing import Iterable

TColorT = torch.long
TCube = torch.Tensor


def new_cube() -> TCube:
    cube = torch.zeros((6, 3, 3), dtype=TColorT, device=tut._DEVICE)
    for colour in range(6):
        cube[colour] = colour  # + 1
    return cube


_SOLVED_CUBE = new_cube()


def print_cube(cube: TCube) -> None:
    cube = cube.cpu().numpy()
    for line in cube[0]:
        print(" " * 7, line, sep="")
    for line_index in range(3):
        for face in cube[1:5]:
            print(face[line_index], end="")
        print()
    for line in cube[5]:
        print(" " * 7, line, sep="")


def is_same(cube1: TCube, cube2: TCube) -> bool:
    return torch.equal(cube1, cube2)


def is_solved(cube: TCube) -> bool:
    return is_same(cube, _SOLVED_CUBE)


def make_move(cube: TCube, move: int) -> None:
    """
    Moves are 1 ... 6, clockwise.
    Negative moves ~ counterclockwise.
    """
    abs_move = np.abs(move) - 1
    direction_01 = (1 - np.sign(move)) // 2
    direction_11 = -np.sign(move)

    surroundings = tut.SURROUNDINGS[abs_move]

    surr_shift = tut.SURROUNDINGS_SHIFT[direction_01, abs_move]

    cube[abs_move] = torch.rot90(cube[abs_move], direction_11)
    cube[surroundings[0], surroundings[1], surroundings[2]] = cube[
        surr_shift[0], surr_shift[1], surr_shift[2]
    ]  # for pythonu 3.10 which cannot use *surroundings


def make_moves(cube: TCube, moves: Iterable[int]) -> None:
    for move in moves:
        make_move(cube, move)


def generate_random_moves(num_moves: int = 1) -> torch.Tensor:
    assert num_moves > 0, "The number of moves must be at least one!"
    shape = (num_moves,)
    return torch.randint(1, 6, shape) * (
        torch.randint(0, 2, shape) * 2 - 1
    )  # choice([-1, 1])
    # TODO: dtype of moves


def scramble(cube: TCube, num_moves: int) -> None:
    make_moves(cube, generate_random_moves(num_moves))  # type: ignore


if __name__ == "__main__":
    import deepercube.kostka.kostka as ko

    cube = ko.nova_kostka()
    tcube = new_cube()
    tcube2 = new_cube()

    print("Running test torch_cube...")
    for i in range(10_000):
        move = ko.vygeneruj_nahodny_tah()
        move2 = np.sign(move) * ((abs(move) % 6) + 1)
        assert type(move) == int
        ko.tahni_tah(cube, move)
        make_move(tcube, move)
        make_move(tcube2, move2)
        if not is_same(
            torch.as_tensor(cube, dtype=tcube.dtype, device=tcube.device), tcube
        ) or not ko.je_stejna(cube, tcube.cpu().numpy()
        ) or ko.je_stejna(cube, tcube2.cpu().numpy()
        ) or is_same(tcube, tcube2):
            print()
            print("Difference detected at move", i)
            print("Numpy:")
            ko.print_kostku(cube)
            print()
            print("PyTorch:")
            print_cube(tcube)
            print()
            print_cube(tcube2)
            exit()
    print("Passed!")
