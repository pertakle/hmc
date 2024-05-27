import numpy as np
import kostka as ko
import kostka_vek as kv


# def maint():
#     B = 4096
#     T = 28
#     batch = np.empty([B, 6*9], dtype=np.uint8)
#     for b in range(B):
#         k = ko.nova_kostka()
#         ko.tahni_tahy(k, np.random.randint(1, 6, T, dtype=np.uint8) * np.random.choice([-1,1], T))
#         batch[b] = k.reshape([-1])
#     return batch
# 
# if __name__ == "__main__":
#     maint()
#     exit()

def main():
    k = ko.nova_kostka()
    while True:
        print()
        ko.print_kostku(k)
        print(ko.je_slozena(k))

        inp = input()
        if len(inp) == 0:
            break
        try:
            tah = int(inp)
        except:
            print("Chyba")
            continue
        ko.tahni_tah(k, tah)

def main():
    N = 5
    k = kv.nova_kostka_vek(N)
    k[:,0,0,0] = 9
    kv.print_kostku_vek(k)
    print("---")
    tah_vek = np.ones(N, dtype=int)
    tah_vek *= (np.arange(N)%2)+1
    kv.tahni_tah_vek(k, tah_vek)
    kv.print_kostku_vek(k)
    # kv.tahni_tah_vek(k, 2*tah_vek)
    # kv.print_kostku_vek(k)

if __name__ == "__main__":
    main()
