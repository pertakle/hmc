import torch

from deepercube.utils import torch_utils as tut
from deepercube.kostka import torch_cube as tcu


TCubeVec = torch.Tensor


def new_cube_vec(num_cubes: int) -> TCubeVec:
    return torch.stack([tcu._SOLVED_CUBE] * num_cubes)


def is_same_vec(cube1: TCubeVec, cube2: TCubeVec | tcu.TCube) -> torch.Tensor:
    if len(cube2.shape) + 1 == len(cube1.shape):
        cube2 = cube2[None]
    return torch.all(cube1 == cube2, dim=(1, 2, 3))


def is_solved_vec(cube: TCubeVec) -> torch.Tensor:
    return is_same_vec(cube, tcu._SOLVED_CUBE)


def print_cube_vec(cube: TCubeVec) -> None:
    for cubie in cube:
        tcu.print_cube(cubie)
        print()


def make_move_vec(cube: TCubeVec, move: torch.Tensor) -> None:
    """
    Makes one particular move with each cube.
    ```python
    cube.shape = (C, ...)
    move.shape = (C,)
    ```
    """
    assert torch.all(1 <= move.abs())
    assert torch.all(move.abs() <= 12)

    abs_move = move.abs() - 1
    direction_01 = (1 - move.sign()) // 2
    direction_cl = direction_01 > 0
    direction_ccl = ~direction_cl

    # TODO: remove the need for transposition
    surroundings = tut.SURROUNDINGS[abs_move].permute([1, 0, 2])
    surroundings_shift = tut.SURROUNDINGS_SHIFT[direction_01, abs_move].permute([1, 0, 2])

    dir_cl_idx = direction_cl.nonzero().flatten()
    dir_ccl_idx = direction_ccl.nonzero().flatten()

    cube[dir_cl_idx, abs_move[dir_cl_idx]] = torch.rot90(
        cube[dir_cl_idx, abs_move[dir_cl_idx]], 1, (1, 2)
    )
    cube[dir_ccl_idx, abs_move[dir_ccl_idx]] = torch.rot90(
        cube[dir_ccl_idx, abs_move[dir_ccl_idx]], -1, (1, 2)
    )

    ar = torch.arange(len(cube), dtype=torch.long, device=tut._DEVICE).reshape([-1, 1])
    cube[ar, surroundings[0], surroundings[1], surroundings[2]] = cube[
        ar, surroundings_shift[0], surroundings_shift[1], surroundings_shift[2]
    ]  # no * for python 3.10


def make_moves_vec(cube: TCubeVec, moves: torch.Tensor) -> None:
    """
    Makes sequence of moves with the vectorized cube.
    Sequences are along dim 0.
    ```python
    cube.shape = (K, ...)
    cube.shape = (T, K)
    ```
    """
    for move in moves:
        make_move_vec(cube, move)


def generate_random_moves_vec(shape: tuple[int, ...] | list[int] | torch.Size) -> torch.Tensor:
    return torch.randint(1, 6, shape) * (torch.randint(0, 2, shape) * 2 - 1)
    # TODO: dtype of moves


def scramble_vec(cube: TCubeVec, num_moves: int) -> None:
    make_moves_vec(cube, generate_random_moves_vec([num_moves, cube.shape[0]]))


_ALL_MOVES = torch.tensor([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], device=tut._DEVICE)


def make_all_moves(cube: tcu.TCube) -> TCubeVec:
    """
    Makes all possible moves with `cube`
    and returns the results as a `CubeVec`.
    """
    cubes = torch.stack([cube] * 12)
    make_move_vec(cubes, _ALL_MOVES)
    return cubes


def make_all_moves_vec(cube: TCubeVec) -> TCubeVec:
    """
    Makes all possible moves with each cube in `cube`
    and return the results as a single `CubeVec`.

    The resulting cubes are ordered same as their parents.
    Each 12 cubes next to eachother corresponds to the same parent.

    'ci' ~ i-th cube in `cube`
    'cimj' ~ i-th cube in `cube` moved with j-th move

    `cube` = (c1, c2, c3)
    result = (c1m1, c1m2, ..., c1m12, c2m1, c2m2, ..., c3m12)
    """
    cubes = cube.repeat(0, 12)
    moves = torch.hstack([_ALL_MOVES] * len(cube))
    make_move_vec(cubes, moves)
    return cubes

if __name__ == "__main__":
    import deepercube.kostka.kostka_vek as kv

    n = 7
    cube = kv.nova_kostka_vek(n)
    tcube = new_cube_vec(n)
    tcube2 = new_cube_vec(n)
    scramble_vec(tcube2, 20)

    print("Running test torch_cube_vec...")
    for i in range(10_000):
        moves = generate_random_moves_vec((10, n))
        kv.tahni_tahy_vek(cube, moves.cpu().numpy())
        make_moves_vec(tcube, moves)
        make_moves_vec(tcube2, moves)
        if not is_same_vec(
            torch.as_tensor(cube, dtype=tcube.dtype, device=tcube.device), tcube
        ).all() or not kv.je_stejna(cube, tcube.cpu().numpy()
        ).all() or kv.je_stejna(cube, tcube2.cpu().numpy()
        ).any() or is_same_vec(tcube, tcube2).any():
            print()
            print("Difference detected at move", i)
            print("Numpy:")
            kv.print_kostku_vek(cube)
            print()
            print("PyTorch:")
            print_cube_vec(tcube)
            print()
            print_cube_vec(tcube2)
            exit()
    print("Passed!")
