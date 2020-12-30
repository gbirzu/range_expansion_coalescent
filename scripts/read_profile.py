import numpy as np
import matplotlib.pyplot as plt
import time


def read_profile(fname, density=False):
    lines = []
    with open(fname, 'r') as fprof:
        lines = fprof.readlines()
    prof_array = parse_lines(lines)
    if density == True:
        prof_array = np.array([sum(deme != 0) for deme in prof_array])
    return prof_array


def parse_lines(lines):
    prof_array = []
    for line in lines:
        individuals = line.split(',')
        individuals = [int(indiv) for indiv in individuals]
        prof_array.append(individuals)
    prof_array = np.array(prof_array)
    return prof_array


def plot_density(prof_array, title='', fig_name=''):
    population_density = [sum(deme != 0) for deme in prof_array]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('position, x')
    ax.set_ylabel('population size, n')

    ax.plot(population_density)
    if fig_name != '':
        plt.savefig(fig_name)


def plot_subtype_sizes(prof_array, title='', fig_name=''):
    unique = find_unique(prof_array)
    subtype_sizes = clone_sizes(prof_array)
    plot_subtypes(subtype_sizes, title='', fig_name='')


def find_unique(prof_array):
    unique = np.unique(prof_array)
    return sorted(unique)

def clone_sizes(prof_array):
    individuals = prof_array.flatten()
    individuals = individuals[np.where(individuals != 0)[0]]
    unique, counts = np.unique(individuals, return_counts=True)
    return counts

def plot_subtypes(subtype_sizes, title='', fig_name=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('position, x')
    ax.set_ylabel('clone size, $n_i$')

    for subtype in subtype_sizes:
        ax.plot(subtype)

    if fig_name != '':
        plt.savefig(fig_name)

def test_functions():
    data_path = '../data/N200_r0.05_m0.25_fback10/'
    profile_fpath = data_path + 'profile_tau1_N200_r0.05_m0.25_fback10_relabel10_run1.txt'
    prof_array = read_profile(profile_fpath)

    # Number of distinct families
    start = time.time()
    unique = find_unique(prof_array)
    end = time.time()
    print("find_unique time:", end - start)

    # Family sizes
    start = time.time()
    clone_size_counts = clone_sizes(prof_array)
    end = time.time()
    print("clone_sizes time:", end - start)


if __name__ == '__main__':
    data_path = '../data/N200_r0.05_m0.25_fback10/'

    test_functions()

    '''
    tau = 1
    profile_fpath = data_path + 'profile_tau' + str(tau) + '_N200_r0.05_m0.25_fback10_relabel10_run1.txt'
    prof_array = read_profile(profile_fpath)
    plot_subtype_sizes(prof_array, title='tau = ' + str(tau), fig_name='subtype_density_tau' + str(tau) + '.pdf')
    plt.show()
    '''
