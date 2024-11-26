import numpy as np
from deepercube.utils import utils as ut
from typing import Iterable, Tuple

ColorT = np.uint8
Kostka = np.ndarray[Tuple[6, 3, 3], np.dtype[ColorT]]


def nova_kostka() -> Kostka:
    kostka = np.zeros((6, 3, 3), dtype=ColorT)
    for barva in range(6):
        kostka[barva] = barva  # + 1
    return kostka


SLOZENA_KOSTKA = nova_kostka()


def print_kostku(kostka: Kostka) -> None:
    for l in kostka[0]:
        print(" " * 7, l, sep="")
    for i in range(3):
        for s in kostka[1:5]:
            print(s[i], end="")
        print()
    for l in kostka[5]:
        print(" " * 7, l, sep="")


def je_stejna(kostka1: Kostka, kostka2: Kostka) -> bool:
    return np.array_equal(kostka1, kostka2)


def je_slozena(kostka: Kostka) -> bool:
    return je_stejna(kostka, SLOZENA_KOSTKA)


def tahni_tah(kostka: Kostka, tah: int) -> None:
    """
    Tahy jsou 1 ... 6, po smeru hodinovych rucicek.
    Zaporne znamenaji otacet proti smeru
    hodinovych rucicek.
    """
    abs_tah = np.abs(tah) - 1
    smer_01 = (1 - np.sign(tah)) // 2
    smer_11 = -np.sign(tah)

    okoli = ut.OKOLI[abs_tah]

    okoli_posun = ut.OKOLI_POSUN[smer_01, abs_tah]

    kostka[abs_tah] = np.rot90(kostka[abs_tah], smer_11)
    # kostka[*okoli] = kostka[*okoli_posun]
    kostka[okoli[0], okoli[1], okoli[2]] = kostka[
        okoli_posun[0], okoli_posun[1], okoli_posun[2]
    ]  # kvůli pythonu 3.10


def tahni_tahy(kostka: Kostka, tahy: Iterable[int]) -> None:
    for tah in tahy:
        tahni_tah(kostka, tah)


def vygeneruj_nahodny_tah(shape=None) -> int | np.ndarray:
    if shape is None:
        return np.random.randint(1, 6) * np.random.choice([-1, 1])
    return np.random.randint(1, 6, shape) * np.random.choice([-1, 1], shape)


def zamichej(kostka: Kostka, pocet_tahu: int) -> None:
    tahni_tahy(kostka, vygeneruj_nahodny_tah(pocet_tahu))  # type: ignore
