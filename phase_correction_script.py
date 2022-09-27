import os
import phase_correction as pc
import digital_phase_correction as dpc
import matplotlib.pyplot as plt
import numpy as np


def reshape_single_acquire(x, ppifg):
    center = ppifg // 2
    x = x[np.argmax(x[:ppifg]):][center:]
    N = len(x) // ppifg
    x = x[:N * ppifg]
    x.resize((N, ppifg))

    return x


def phase_correct_single_acquire(x, ppifg, ll, ul, N_apod):
    x = reshape_single_acquire(x, ppifg)
    pdiff = dpc.get_pdiff(x, ll, ul, N_apod)
    dpc.apply_t0_and_phi0_shift(pdiff, x)
    return x


def key(s):
    return float(s.split(".npy")[0])


path_spectra = r'D:\Microscope\databackup/spectra/'
names = [i.name for i in os.scandir(path_spectra)]
names = sorted(names, key=key)
names = [path_spectra + i for i in names]
get_data = lambda n: np.load(names[n])

ppifg = 74180
ll, ul = .0858, 0.2093
X = np.zeros((len(names), ppifg))

for n in range(len(names)):
    x = get_data(n)
    x = reshape_single_acquire(x, ppifg)
    pdiff = dpc.get_pdiff(x, ll, ul, 200)
    dpc.apply_t0_and_phi0_shift(pdiff, x)

    x = np.mean(x, 0)
    X[n] = x
    print(len(names) - n)

np.save(r'D:\Microscope\09-27-2022_Data/spectra.npy', X)
