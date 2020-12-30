import numpy as np
import pickle

def convert_counts_to_distribution(count_dict, normed=True, x_range=None, fractional_w=False, min_size=1E-8):
    '''
    Converts count dict into complementary cumulative distribution.

    Parameters
    __________

    count_dict : list of samples; each samples can have arbitrary number of entries
    '''
    s_average = average_clone_sizes(count_dict, min_size=min_size)
    if normed == True:
        s_max = (max(count_dict.keys()) + 1) / s_average
        s_min = 1 / s_average
        ds = 1 / s_average
    else:
        s_max = max(count_dict.keys()) + 1
        s_min = 1
        ds = 1

    if fractional_w == False:
        x = np.arange(s_min, s_max, ds)
    else:
        s_max = max(count_dict.keys()) / s_average
        s_min = max(min(count_dict.keys()) / s_average, min_size / s_average)
        x = np.geomspace(s_min, s_max, 1000)

    distr = np.zeros(len(x))
    clone_sizes = np.array(list(count_dict.keys()))
    for i, s_c in enumerate(x):
        for size in clone_sizes[np.where(clone_sizes >= s_c*s_average)[0]]:
            distr[i] += count_dict[size]
    return x, distr / distr[0]

def average_clone_sizes(size_dict, min_size=0):
    s_mean = 0
    total_counts = 0
    for s in size_dict.keys():
        if s > min_size:
            s_mean += s * size_dict[s]
            total_counts += size_dict[s]
    return s_mean / total_counts

def reverse_cumulative(counts, normed=True, x_range=None):
    '''
    Returns the reverse cumulative distribution by sorting entries in counts.

    Parameters
    __________

    counts : list of samples; each samples can have arbitrary number of entries
    '''
    y_all = []
    means = []
    for sample in counts:
        y_all += list(sample)
        means.append(np.mean(sample))
    y_mean = np.mean(means)
    if normed == True:
        y = np.array(y_all) / y_mean
    else:
        y = np.array(y_all)

    x = np.sort(y)
    reverse_hist = np.array([(len(x) - i) / len(x) for i in range(len(x))])

    return x, reverse_hist


def reverse_cumulative_hist(counts, normed=True, log_scale_x=True, x_range=None):
    '''
    Returns the reverse cumulative distribution of counts by integrating the histogram.

    Parameters
    __________

    counts : list of samples; each samples can have arbitrary number of entries
    '''
    y_all = []
    means = []
    for sample in counts:
        y_all += list(sample)
        means.append(np.mean(sample))
    y_mean = np.mean(means)
    if normed == True:
        y = np.array(y_all) / y_mean
    else:
        y = np.array(y_all)

    if log_scale_x == True:
        logx_range = [np.log(x_range[0]), np.log(x_range[1])]
        hist, bin_edges = np.histogram(np.log(y), range=logx_range, bins=1000)
        logx = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
        x = np.exp(logx)
    else:
        hist, bin_edges = np.histogram(y, range=x_range, bins=1000)
        x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    return x, 1 - np.cumsum(hist) / np.sum(hist)


