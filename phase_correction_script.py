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


def phase_correct_one(x, ppifg, ll, ul, N_apod):
    x = reshape_single_acquire(x, ppifg)
    pdiff = dpc.get_pdiff(x, ll, ul, N_apod)
    dpc.apply_t0_and_phi0_shift(pdiff, x)
    return x


def phase_correct_all(X2D, ppifg, ll, ul, N_apod):
    for x in X2D:
        phase_correct_one(x, ppifg, ll, ul, N_apod)


ll, ul = .0068, 0.1484
