import scipy.constants as sc
from scipy.integrate import simps
import os
import phase_correction as pc
import digital_phase_correction as dpc
import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()


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


# %% ___________________________________________________________________________________________________________________
path_spectra = r'D:\Microscope\databackup/spectra/'
names = [i.name for i in os.scandir(path_spectra)]
names = sorted(names, key=key)
names = [path_spectra + i for i in names]
get_data = lambda n: np.load(names[n])
ppifg = 74180
center = ppifg // 2
ll, ul = .0858, 0.2093

# %% ___________________________________________________________________________________________________________________
# X = np.zeros((len(names), ppifg))
#
# for n in range(len(names)):
#     x = get_data(n)
#     x = reshape_single_acquire(x, ppifg)
#     pdiff = dpc.get_pdiff(x, ll, ul, 200)
#     dpc.apply_t0_and_phi0_shift(pdiff, x)
#
#     x = np.mean(x, 0)
#     X[n] = x
#     print(len(names) - n)
#
# np.save(r'D:\Microscope\09-27-2022_Data/spectra.npy', X)

# %% ___________________________________________________________________________________________________________________
X = np.load(r'D:\Microscope\09-27-2022_Data/spectra.npy')
FT = pc.fft(X, 1)
FT = FT[:, center:].__abs__()  # third nyquist window is +ve frequencies
integral = simps(FT, axis=1)
integral /= integral.max()
pos = np.load(r'D:\Microscope\09-27-2022_Data/stage_positions_um.npy')
pos -= pos[0]

Nyq_Freq = center * (1010e6 - 10012996)
Nyquist_Window = 3
translation = (Nyquist_Window - 1) * Nyq_Freq
nu = np.linspace(0, Nyq_Freq, center) + translation
wl = np.where(nu > 0, sc.c * 1e6 / nu, np.nan)

fig, ax = plt.subplots(1, 2, figsize=np.array([8.42, 4.8]))
save = True
for n, i in enumerate(FT):
    [i.clear() for i in ax]
    ax[0].plot(wl, i)
    ax[1].plot(pos[:n], integral[:n])
    ax[1].set_xlim(*pos[[0, -1]])
    ax[1].set_ylim(integral.min(), integral.max())
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("wavelength ($\mathrm{\mu m}$)")
    ax[1].set_xlabel("stage position ($\mathrm{\mu m}$)")
    ax[0].set_ylabel("power (arb. units)")
    ax[1].set_ylabel("integrated power (arb. units)")
    if save:
        plt.savefig(f"fig/{n}.png")
    else:
        plt.pause(.1)

    print(len(FT) - n)
