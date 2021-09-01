import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import analysis_tools as tools
from scipy import stats

input_path = 'data/'
output_plots = 'plots/'
output_files = 'data/interim/'

# Set up colors dict
#colors_dict = {'lightsalmon':(1., 0.627, 0.478), 'lightskyblue':(0.529, 0.808, 0.98), 'lightgreen':(0.565, 0.933, 0.565)}
colors = [(1., 0.627, 0.478), (0.565, 0.933, 0.565), (0.529, 0.808, 0.98)]
colors_dict = {'pulled':(1., 0.627, 0.478), 'semi-pushed':(0.565, 0.933, 0.565), 'fully-pushed':(0.529, 0.808, 0.98)}
wave_dict = { 0:'pulled', 1:'semi-pushed', 2:'fully-pushed' }

def cm2inch(x, y):
    return (x/2.54, y/2.54)


def strip(s): #separates numbers from letters in string
    head = s.strip('-.0123456789')
    tail = s[len(head):]
    return head, tail


def get_variables(name):
    name_root = name.split('/')[-1].split('.')#Get file name
    if 'txt' in name_root:
        name_root.remove('txt') #remove ending
    elif 'csv' in name_root:
        name_root.remove('csv') #remove ending

    name_root = '.'.join(name_root) #reform name
    aux = [strip(s) for s in name_root.split('_')]
    #default values if none found
    r0 = 0.01
    m0 = 0.01
    A = 0.0
    B = 0.0
    N = 10000
    for s in aux:
        if s[0] == 'm':
            m0 = float(s[1])
        elif s[0] == 'A':
            A = float(s[1])
        elif s[0] == 'r':
            r0 = float(s[1])
        elif s[0] == 'B':
            B = float(s[1])
        elif s[0] == 'N':
            N = int(s[1])

    return m0, A, r0, B, N


def generate_fname(head, N, A, B, tail='.csv', rm=None):
    if rm == None:
        return head + 'N' +str(N) + '_A' + str(A) + '_B' + str(B) + tail
    else:
        return head + 'N' +str(N) + '_r' + str(rm[0]) + '_m' + str(rm[1]) + '_A' + str(A) + '_B' + str(B) + tail


def KS_test(A, B):
    expansion_label = tools.linear_cooperativity(A, B)
    if expansion_label == 0:
        return 'pulled'
    elif expansion_label == 1:
        return 'semi-pushed'
    elif expansion_label == 2:
        return 'uniform'
    else:
        return None


def find_timepoint(fraction_array, time_index, p_target):
    i_target = 0
    dp_target = 1.0
    runs = fraction_array.shape[1]
    time_points = []
    for i in time_index:
        f_survived =  fraction_array[i][np.where((fraction_array[i] != 0.0)*(fraction_array[i] != 1.0))[0]]
        p_survived = len(f_survived)/float(runs)
        dp = abs(p_survived - p_target)

        if dp < dp_target:
            i_target = i
            dp_target = dp

    return i_target


def overlap_fractions(f_array):
    return np.array([f if f > 0.5 else 1 - f for f in f_array])


def theory_cdf_numeric(bin_centers, alpha=0, normalize=True):
    f_min = bin_centers[0]
    cdf = np.zeros(len(bin_centers))
    cdf[0] = 1 / (f_min * (1 - f_min))
    for i, f in enumerate(bin_centers[:-1]):
        if f > 0.5:
            nu = f
        else:
            nu = 1 - f
        if alpha == 0:
            cdf[i + 1] = cdf[i] + 1 / (nu * (1 - nu))
        else:
            cdf[i + 1] = cdf[i] + 1 / ((nu * (1 - nu))**alpha)

    if normalize == True:
        cdf /= cdf[-1]

    return cdf


def logit(f_array):
    return np.log(f_array/(1 - f_array))


def get_distribution(fraction_array, p_target, epsilon=1E-5):
    (time_samples, runs) = fraction_array.shape
    time_point = find_timepoint(fraction_array, range(time_samples), p_target)

    f_survived =  fraction_array[time_point][np.where((fraction_array[time_point] > epsilon) * (fraction_array[time_point] < 1.0 - epsilon))[0]]
    p_survived = len(f_survived) / float(runs)

    return time_point, p_survived, f_survived


