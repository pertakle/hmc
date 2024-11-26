from deepercube.utils import utils as ut
from deepercube.kostka import kostka as ko
import numpy as np
from typing import Any, Tuple


KostkaVek = np.ndarray[Tuple[-1, 6, 3, 3], np.dtype[ko.ColorT]]


def nova_kostka_vek(n: int) -> KostkaVek:
    return np.stack([ko.SLOZENA_KOSTKA] * n, dtype=ko.SLOZENA_KOSTKA.dtype)


def je_stejna(kostka1: KostkaVek, kostka2: KostkaVek | ko.Kostka) -> np.ndarray:
    return np.all(kostka1 == kostka2, axis=(1, 2, 3))


def je_slozena(kostka: KostkaVek) -> np.ndarray:
    return je_stejna(kostka, ko.SLOZENA_KOSTKA)


def print_kostku_vek(kostka: KostkaVek) -> None:
    for k in kostka:
        ko.print_kostku(k)
        print()


def tahni_tah_vek(kostka: KostkaVek, tah: np.ndarray) -> None:
    """
    Provede jeden patřičný tah na na každé kostce.

    ```python
    kostka.shape = (K, ...)
    tah.shape = (K,)
    ```
    """
    assert np.all(1 <= np.abs(tah))
    assert np.all(np.abs(tah) <= 12)

    abs_tah = np.abs(tah) - 1
    smer_01 = (1 - np.sign(tah)) // 2
    # smer_11 = -np.sign(tah)
    smer_po = smer_01 > 0
    smer_proti = np.logical_not(smer_po)

    # TODO: odstranit potrebu transpozice
    okoli = ut.OKOLI[abs_tah].transpose([1, 0, 2])
    okoli_posun = ut.OKOLI_POSUN[smer_01, abs_tah].transpose([1, 0, 2])

    smer_po_idx = np.flatnonzero(smer_po)
    smer_proti_idx = np.flatnonzero(smer_proti)

    kostka[smer_po_idx, abs_tah[smer_po_idx]] = np.rot90(
        kostka[smer_po_idx, abs_tah[smer_po_idx]], 1, (1, 2)
    )
    kostka[smer_proti_idx, abs_tah[smer_proti_idx]] = np.rot90(
        kostka[smer_proti_idx, abs_tah[smer_proti_idx]], -1, (1, 2)
    )

    ar = np.arange(len(kostka)).reshape([-1, 1])
    # kostka[ar, *okoli] = kostka[ar, *okoli_posun]
    kostka[ar, okoli[0], okoli[1], okoli[2]] = kostka[
        ar, okoli_posun[0], okoli_posun[1], okoli_posun[2]
    ]  # kvůli pythonu 3.10


def tahni_tahy_vek(kostka: KostkaVek, tahy: np.ndarray) -> None:
    """
    Provede postupně sekvenci tahů na vektorové kostce.
    Tahy jsou v řádcích vektorové tahy sekvence.
    ```python
    kostka.shape = (K, ...)
    tahy.shape = (T, K)
    ```
    """
    for tah in tahy:
        tahni_tah_vek(kostka, tah)


def vygeneruj_nahodny_tah_vek(shape: Any) -> np.ndarray:
    return np.random.randint(1, 6, shape) * np.random.choice([-1, 1], shape)


def zamichej_nahodnymi_tahy_vek(kostka: KostkaVek, tahu: int) -> None:
    tahni_tahy_vek(kostka, vygeneruj_nahodny_tah_vek([tahu, kostka.shape[0]]))


VSECHNY_TAHY = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], dtype=int)


def tahni_vsechny_tahy(kostka: ko.Kostka) -> KostkaVek:
    kostky = np.stack([kostka] * 12)
    tahni_tah_vek(kostky, VSECHNY_TAHY)
    return kostky


def tahni_vsechny_tahy_vek(kostka: KostkaVek) -> KostkaVek:
    kostky = np.repeat(kostka, 12, axis=0)
    tahy = np.hstack([VSECHNY_TAHY] * len(kostka))
    tahni_tah_vek(kostky, tahy)
    return kostky
