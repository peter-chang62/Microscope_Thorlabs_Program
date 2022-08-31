import numpy as np
import matplotlib.pyplot as plt

"""This find the points per interferogram method works so long as you don't have only a few points per interferogram. 
the returned points per interferogram is the average of the distance between interferogram maxes """


def adjust_data_and_reshape(data, ppifg):
    """
    :param data:
    :param ppifg:

    :return: data truncated (both start and end) to an integer number of interferograms and reshaped.
    because it might be important to know, I also return the number of points I truncated
    from the start of the data
    """

    # skip to the max of the first interferogram, and then NEGATIVE or POSITIVE ppifg // 2 after that
    start = data[:ppifg]
    ind = np.argmax(abs(start))
    if ind > ppifg // 2:
        ind -= ppifg // 2
    elif ind < ppifg // 2:
        ind += ppifg // 2

    data = data[ind:]
    N = len(data) // ppifg
    data = data[:N * ppifg]
    N = len(data) // ppifg
    data = data.reshape(N, ppifg)
    return data, ind


def plot_section(arr, npts, npts_plot):
    for i in range(1, len(arr) // npts):
        plt.plot(arr[npts // 2:][npts * (i - 1):npts * i][npts // 2 - npts_plot:npts // 2 + npts_plot])


def find_npts(dat, level_percent=40):
    level = np.max(abs(dat)) * level_percent * .01
    ind = (dat > level).nonzero()[0]
    diff = np.diff(ind)
    ind_diff = (diff > 1000).nonzero()[0]

    h = 0
    trial = []
    for i in (ind_diff + 1):
        trial.append(ind[h:i])
        h = i

    ind_maxes = []
    for i in trial:
        ind_maxes.append(i[np.argmax(dat[i])])

    mean = np.mean(np.diff(ind_maxes))

    return int(np.round(mean)), mean, level
