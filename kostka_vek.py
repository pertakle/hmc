import utils as ut
import kostka as ko
import numpy as np


KostkaVek = np.ndarray

def nova_kostka_vek(n: int) -> KostkaVek:
    return np.stack([ko.SLOZENA_KOSTKA]*n)

def je_slozena(kostka: KostkaVek) -> bool:
    return np.all(kostka == ko.SLOZENA_KOSTKA, axis=(1,2,3))


def print_kostku_vek(kostka: KostkaVek) -> None:
    for k in kostka:
        for l in k[0]:
            print(" "*7, l, sep="")
        for i in range(3):
            for s in k[1:5]:
                print(s[i], end="")
            print()
        for l in k[5]:
            print(" "*7, l, sep="")
        print()

def tahni_tah_vek(kostka: KostkaVek, tah: np.ndarray) -> None:
    abs_tah = np.abs(tah) - 1
    smer_01 = (1-np.sign(tah))//2
    # smer_11 = -np.sign(tah)
    smer_po = smer_01 > 0
    smer_proti = np.logical_not(smer_po)

    # TODO: odstranit potrebu transpozice
    okoli = ut.OKOLI[abs_tah].transpose([1,0,2]) 
    okoli_posun = ut.OKOLI_POSUN[smer_01, abs_tah].transpose([1,0,2])


    smer_po_idx = np.flatnonzero(smer_po)
    smer_proti_idx = np.flatnonzero(smer_proti)

    kostka[smer_po_idx, abs_tah[smer_po_idx]] = np.rot90(kostka[smer_po_idx, abs_tah[smer_po_idx]], 1, (1,2))
    kostka[smer_proti_idx, abs_tah[smer_proti_idx]] = np.rot90(kostka[smer_proti_idx, abs_tah[smer_proti_idx]], -1, (1,2))
    
    ar = np.arange(len(kostka)).reshape([-1,1])
    #kostka[ar, *okoli] = kostka[ar, *okoli_posun]
    kostka[ar, okoli[0], okoli[1], okoli[2]] = kostka[ar, okoli_posun[0], okoli_posun[1], okoli_posun[2]] # kvůli pythonu 3.10


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

def vygeneruj_nahodny_tah_vek(kostek: int) -> np.ndarray:
    return np.random.randint(1, 6, kostek) * np.random.choice([-1, 1], kostek)
