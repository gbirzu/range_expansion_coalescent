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


def theory_cdf(x):
    #cdf = (1/x[0] - 1/x)
    #cdf /= cdf[-1]
    #return cdf
    return (2/12) * (np.log(x/(1 - x)) - np.log(x[0]))


def theory_cdf_numeric(bin_centers, normalize=True):
    f_min = bin_centers[0]
    cdf = np.zeros(len(bin_centers))
    #cdf[0] = (2 * f_min - 1) / (f_min * (1 - f_min) * (np.log(f_min) - np.log(1 - f_min)))
    cdf[0] = 1 / (f_min * (1 - f_min))
    for i, f in enumerate(bin_centers[:-1]):
        if f > 0.5:
            nu = f
        else:
            nu = 1 - f
        #cdf[i + 1] = cdf[i] + (2 * nu - 1) / (nu * (1 - nu) * (np.log(nu) - np.log(1 - nu)))
        cdf[i + 1] = cdf[i] + 1 / (nu * (1 - nu))

    if normalize == True:
        cdf /= cdf[-1]

    return cdf


def theory_cdf_corrections_numeric(bin_centers, normalize=True):
    f_min = bin_centers[0]
    cdf = np.zeros(len(bin_centers))
    cdf[0] = (2 * f_min - 1) / (f_min * (1 - f_min) * (np.log(f_min) - np.log(1 - f_min)))
    for i, f in enumerate(bin_centers[:-1]):
        if f > 0.5:
            nu = f
        else:
            nu = 1 - f
        cdf[i + 1] = cdf[i] + (2 * nu - 1) / (nu * (1 - nu) * (np.log(nu) - np.log(1 - nu)))

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


def plot_pdf2ax(ax, f_survived, title, bins=20, overlap=True, data_label=None):
    if overlap == True:
        f_array = overlap_fractions(f_survived)
    else:
        f_array = f_survived
    counts, bin_edges = np.histogram(f_array, bins=bins, density=True)

    ax.set_title(title)
    ax.set_xlabel('label 1 fraction, f')
    ax.set_ylabel('probability density, P(f)')
    #ax.bar(bin_edges, counts)
    ax.hist(f_array, bins=bins, density=True)


def plot_cdf2ax(ax, f_survived, title, KS_test, bins=50, overlap=True, data_label=None):
    if overlap == True:
        f_overlap = overlap_fractions(f_survived)
    else:
        f_overlap = f_survived

    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()

    # KS test
    if KS_test == 'uniform':
        print(title)
        print('KS test for uniform: ', stats.kstest(cdf, stats.uniform.cdf))
        print('KS test for BSC distribution: ', stats.ks_2samp(cdf, theory_cdf(bin_centers)))
        #ax.plot(bin_centers, stats.uniform.cdf(bin_centers, scale=0.5), linestyle='--', lw=1.0, c='k', label='uniform')
        ax.plot(bin_centers, stats.uniform.cdf(bin_centers, loc=0.5, scale=0.5), linestyle='--', lw=1.0, c='k')
    elif KS_test == 'pulled':
        print(title)
        print('KS test for uniform: ', stats.kstest(cdf, stats.uniform.cdf))
        print('KS test for BSC distribution: ', stats.ks_2samp(cdf, theory_cdf(bin_centers)))
        #ax.plot(bin_centers, theory_cdf(bin_centers), ls='--', lw=1, c='k')
        ax.plot(bin_centers, theory_cdf_numeric(bin_centers), ls='--', lw=1, c='k')
    elif KS_test == 'semi-pushed':
        print(title)
        print('KS test for uniform: ', stats.kstest(cdf, stats.uniform.cdf))
        print('KS test for BSC distribution: ', stats.ks_2samp(cdf, theory_cdf(bin_centers)))
        #ax.plot(bin_centers, stats.uniform.cdf(bin_centers, scale=0.5), linestyle='--', lw=1.0, label='uniform', c='k')
        ax.plot(bin_centers, stats.uniform.cdf(bin_centers, loc=0.5, scale=0.5), linestyle='--', lw=1.0, c='k')
        #ax.plot(bin_centers, theory_cdf(bin_centers), ls='--', lw=1.0, c='k')
        ax.plot(bin_centers, theory_cdf_numeric(bin_centers), ls='--', lw=1.0, c='k')

    # Set plot labels
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('majority label fraction, f')
    ax.set_ylabel('CDF')

    # Plot data
    #ax.plot(bin_centers, cdf, linestyle='-', lw=1.5, c='r', label=data_label)
    ax.plot(bin_centers, cdf, linestyle='-', lw=1.5, label=data_label)


def plot_logit2ax(ax, f_survived, title, bins=20, data_label=None):
    f_logit = logit(f_survived)
    counts, bin_edges = np.histogram(f_logit, bins=bins, density=True)

    ax.set_title(title)
    ax.set_xlabel('$Logit(f)$')
    ax.set_ylabel('probability density, P(f)')
    #ax.bar(bin_edges, counts)
    ax.hist(f_logit, bins=bins, density=True)


def plot_logitcdf2ax(ax, f_survived, title, bins=20, data_label=None):
    f_logit = logit(f_survived)
    counts, bin_edges = np.histogram(f_logit, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf_logit = np.cumsum(counts) / counts.sum()

    ax.set_title(title)
    ax.set_xlabel('$Logit(f)$')
    ax.set_ylabel('CDF')
    #ax.bar(bin_edges, counts)
    #ax.hist(f_logit, bins=bins, density=True)
    ax.plot(bin_centers, cdf_logit, linestyle='-', lw=1.5, label=data_label)


def plot_parameters(data_dir, N, A=0.0, B=1.0, KS_test=None, p_target=None, bins=30, extension='', save_plots=True):
    # Load file
    fname = data_dir + generate_fname('fraction_samples_', N, A, B, '_points=auto' + extension + '.csv')
    try:
        fraction_array = np.loadtxt(fname, delimiter=',')
    except:
        print(fname, ' file not found. Exiting without ploting.')
        return -1

    time_point, p_survived, f_survived = get_distribution(fraction_array, p_target)

    # Define labels
    wave_type = wave_dict[tools.linear_cooperativity(A, B)]
    #title = 'N = ' + str(N) + ', ' + wave_type + ', $p_{surv} = $' + str(p_survived)
    title = wave_type + extension

    fig = plt.figure(figsize=cm2inch(17.8, 8))
    ax1 = fig.add_subplot(121)
    plot_pdf2ax(ax1, f_survived, title)

    ax2 = fig.add_subplot(122)
    plot_cdf2ax(ax2, f_survived, title, KS_test=KS_test)

    plt.tight_layout()

    if save_plots == True:
        tail = '_p' + str(p_survived) + '_bins' + str(bins) + extension + '.pdf'
        plot_name = generate_fname('fraction_', N, A, B, tail)
        plt.savefig(output_plots + plot_name)


def plot_stationarity(data_dir, ax, N, A=0.0, B=1.0, p_target=None, bins=30, save_plots=True):
    wave_type = wave_dict[tools.linear_cooperativity(A, B)]
    file_list = sorted(glob.glob(data_dir + generate_fname('fraction_*_', N, A, B, tail='_points=auto_stochastic.csv')))
    #sampling_times_fname = data_dir + '/sampling_times_N' + str(N) + '_A' + str(A) + '_B' + str(B) + '_points=auto.csv'
    #sampling_times = np.loadtxt(sampling_times_fname, delimiter=',')

    if len(file_list) == 0:
        return 0

    fname = file_list[0]
    identifier = 'N' + str(N) + '_A' + str(A) + '_B' + str(B)

    '''
    if (wave_type == 'pulled') or (wave_type == 'semi-pushed'):
        ax.set_yscale('log')
        hist = np.histogram(f_survived, bins=100, density=True)
        p_density = hist[0]
        c = p_density[int(2*len(p_density)/5):int(3*len(p_density)/5)].mean()

        if c == 0.0:
            c = min(p_density[np.where(p_density != 0.0)[0]])

        f_plot = np.linspace(1E-2, 1.0 - 1E-2, 1000)
        p_plot = (c/4)/(f_plot*(1.0 - f_plot))
        ax.plot(f_plot, p_plot, lw=2, c='r', label='$\propto \\frac{1}{f(1-f)}$')

        if wave_type == 'semi-pushed':
            alpha=0.5
            p_sqroot_plot = (c/4**alpha)/((f_plot*(1.0 - f_plot))**alpha)
            ax.plot(f_plot, p_sqroot_plot, lw=2, c='g', ls=':', label='$\propto \\frac{1}{ [f(1-f)]^{0.5}}$')

            alpha=0.8
            p_sqroot_plot = (c/4**alpha)/((f_plot*(1.0 - f_plot))**alpha)
            ax.plot(f_plot, p_sqroot_plot, lw=2, c='g', ls='--', label='$\propto \\frac{1}{ [f(1-f)]^{0.8}}$')

        #if wave_type == 'pulled':
        #    p_corrected_plot = (c/4)/(-f_plot*np.log(1.0 - f_plot)*(f_plot - 1.0)*np.log(1.0 - f_plot))
        #    ax.plot(f_plot, p_plot, lw=2, ls='--', c='purple', label='$\propto \\frac{1}{f(1 - f) \log f\log(1 - f)}$')

        ax.legend(loc='upper center')

    '''

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.4)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()

    ax.set_title('N = ' + str(N) + ', ' + wave_type + ', $p_{surv} = $' + str(p_survived))
    ax.set_xlabel('allele fraction, f')
    ax.set_ylabel('CDF')
    #ax.plot(bin_centers, cdf, linestyle='-.', label='p_surv = 0.4')
    ax.plot(bin_centers, stats.uniform.cdf(bin_centers, scale=0.5), linestyle='-', lw=1)

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.3)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='--', label='p_surv = 0.3')

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.2)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='-', label='p_surv = 0.2')

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.1)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle=':', label='p_surv = 0.1')

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.9)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='-.', label='p_surv = 0.9')

    ax.legend(loc='best')


def fig_stationarity(data_dir, N, A=0.0, B=1.0, p_target=None, bins=30, save_plots=True):
    wave_type = wave_dict[tools.linear_cooperativity(A, B)]
    file_list = sorted(glob.glob(data_dir + '/fraction_*_N' +str(N) + '_A' + str(A) + '_B' + str(B) + '_points=auto*.csv'))
    #sampling_times_fname = data_dir + '/sampling_times_N' + str(N) + '_A' + str(A) + '_B' + str(B) + '_points=auto.csv'
    #sampling_times = np.loadtxt(sampling_times_fname, delimiter=',')

    fname = file_list[0]
    identifier = 'N' + str(N) + '_A' + str(A) + '_B' + str(B)

    fig = plt.figure()
    ax = fig.add_subplot(131)

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.4)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()

    ax.set_title('N = ' + str(N) + ', ' + wave_type + ', $p_{surv} = $' + str(p_survived))
    ax.set_xlabel('allele fraction, f')
    ax.set_ylabel('CDF')
    ax.plot(bin_centers, cdf, linestyle='-.', label='p_surv = 0.4')
    ax.plot(bin_centers, stats.uniform.cdf(bin_centers, scale=0.5), linestyle='-', lw=1)

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.3)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='--', label='p_surv = 0.3')

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.2)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='-', label='p_surv = 0.2')

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.1)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='-', label='p_surv = 0.1')

    ax.legend(loc='best')


    ax = fig.add_subplot(132)

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.4)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()

    ax.set_title('N = ' + str(N) + ', ' + wave_type + ', $p_{surv} = $' + str(p_survived))
    ax.set_xlabel('allele fraction, f')
    ax.set_ylabel('CDF')
    ax.plot(bin_centers, cdf, linestyle='-.', label='p_surv = 0.4')
    ax.plot(bin_centers, stats.uniform.cdf(bin_centers, scale=0.5), linestyle='-', lw=1)

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.3)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='--', label='p_surv = 0.3')

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.2)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='-', label='p_surv = 0.2')

    fraction_array = np.loadtxt(fname, delimiter=',')
    time_point, p_survived, f_survived = get_distribution(fraction_array, 0.2)
    f_overlap = overlap_fractions(f_survived)
    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()
    ax.plot(bin_centers, cdf, linestyle='-', label='p_surv = 0.2')

    ax.legend(loc='best')


def plot_fractions(N, B, runs, front_type='stochastic', A=0.0):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/fraction_paths/'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/fraction_paths/'
    else:
        print("Unknown front type. Quitting.")
        return -1

    file_list = sorted(glob.glob(input_path + data_dir + generate_fname('hetero_', N, 0.0, B, rm=[0.001, 0.25], tail='_*.txt')))
    if len(file_list) <= runs:
        print('Using all available runs for plot: ', len(file_list))
        run_max = len(file_list) - 1
    else:
        run_max = runs
    file_list = file_list[:run_max]

    fig = plt.figure(figsize=cm2inch(11.4, 9.4))
    ax = fig.add_subplot(111)
    ax.set_xlabel('time, t')
    ax.set_ylabel('allele fraction, f')

    for i, fname in enumerate(file_list):
        hetero_array = np.loadtxt(fname, delimiter=',')
        if i == 0:
            time_cutoff = len(hetero_array) // 10

        time_array = hetero_array[:time_cutoff, 0]
        fraction_array = hetero_array[:time_cutoff, 2]

        ax.plot(time_array, fraction_array, lw=1, c='k')


def fig1_workflow(N, B, runs, front_type='stochastic', A=0.0, save=True):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/'
        extension = '_points=auto_stochastic.csv'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/'
        extension = '_points=auto_deterministic.csv'
    else:
        print("Unknown front type. Quitting.")
        return -1

    file_list = sorted(glob.glob(input_path + data_dir + 'fraction_paths/' + generate_fname('hetero_', N, 0.0, B, rm=[0.001, 0.25], tail='_*.txt')))
    if len(file_list) <= runs:
        print('Using all available runs for plot: ', len(file_list))
        run_max = len(file_list) - 1
    else:
        run_max = runs
    file_list = file_list[:run_max]

    fig = plt.figure(figsize=cm2inch(17.8, 5.8))

    ax = fig.add_subplot(131)
    ax.set_xlabel('time, t')
    ax.set_ylabel('allele fraction, f')

    for i, fname in enumerate(file_list):
        hetero_array = np.loadtxt(fname, delimiter=',')
        if i == 0:
            time_cutoff = len(hetero_array) // 10
        time_array = hetero_array[:time_cutoff, 0]
        fraction_array = hetero_array[:time_cutoff, 2]

        ax.plot(time_array, fraction_array, lw=1, c='k')

    # Load fraction histogram
    fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
    try:
        fraction_array = np.loadtxt(fname, delimiter=',')
    except:
        print(fname, ' file not found. Exiting without ploting.')
        return -1
    time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=0.2)

    ax = fig.add_subplot(132)
    plot_pdf2ax(ax, f_survived, '')

    ax = fig.add_subplot(133)
    plot_cdf2ax(ax, f_survived, '', KS_test='uniform', data_label='simulations')

    plt.tight_layout()
    if save == True:
        plt.savefig(output_plots + 'fig1_workflow.pdf')


def fig2_cdfs(N, B_list, front_type='stochastic', p_target=0.2, A=0.0, save=True):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/'
        extension = '_points=auto_stochastic.csv'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/'
        extension = '_points=auto_deterministic.csv'
    else:
        print("Unknown front type. Quitting.")
        return -1

    fig = plt.figure(figsize=cm2inch(17.8, 5.8))

    for i, B in enumerate(B_list):
        ax = fig.add_subplot(1, 3, i + 1)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        wave_type = wave_dict[tools.linear_cooperativity(A, B)]
        # Load fraction histogram
        fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
        try:
            fraction_array = np.loadtxt(fname, delimiter=',')
        except:
            print(fname, ' file not found. Exiting without ploting.')
            return -1
        time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=p_target)
        if front_type == 'stochastic':
            plot_cdf2ax(ax, f_survived, wave_type + '; ' + front_type, KS_test=KS_test(A, B), data_label='simulations')
        else:
            plot_cdf2ax(ax, f_survived, wave_type + '; ' + front_type, KS_test='uniform', data_label='simulations')

    plt.tight_layout()
    if save == True:
        plt.savefig(output_plots + 'fig2_cdfs_' + front_type + '.pdf')


def fig3_cdfs_nooverlap(N, B_list, front_type='stochastic', p_target=0.2, A=0.0, save=True):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/'
        extension = '_points=auto_stochastic.csv'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/'
        extension = '_points=auto_deterministic.csv'
    else:
        print("Unknown front type. Quitting.")
        return -1

    fig = plt.figure(figsize=cm2inch(17.8, 5.8))

    for i, B in enumerate(B_list):
        ax = fig.add_subplot(1, 3, i + 1)
        wave_type = wave_dict[tools.linear_cooperativity(A, B)]
        # Load fraction histogram
        fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
        try:
            fraction_array = np.loadtxt(fname, delimiter=',')
        except:
            print(fname, ' file not found. Exiting without ploting.')
            return -1
        time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=p_target)
        if front_type == 'stochastic':
            plot_cdf2ax(ax, f_survived, wave_type + '; ' + front_type, KS_test=KS_test(A, B), overlap=False, data_label='simulations')
        else:
            plot_cdf2ax(ax, f_survived, wave_type + '; ' + front_type, KS_test='uniform', overlap=False, data_label='simulations')

    plt.tight_layout()
    if save == True:
        plt.savefig(output_plots + 'fig2_cdfs_' + front_type + '.pdf')


def fig4_stationary(N, B_list, p_list, front_type='stochastic', A=0.0, save=True):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/'
        extension = '_points=auto_stochastic.csv'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/'
        extension = '_points=auto_deterministic.csv'
    else:
        print("Unknown front type. Quitting.")
        return -1

    fig = plt.figure(figsize=cm2inch(17.8, 5.8))

    for i, B in enumerate(B_list):
        ax = fig.add_subplot(1, 3, i + 1)
        wave_type = wave_dict[tools.linear_cooperativity(A, B)]
        # Load fraction histogram
        fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
        try:
            fraction_array = np.loadtxt(fname, delimiter=',')
        except:
            print(fname, ' file not found. Exiting without ploting.')
            return -1

        for p in p_list:
            time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=p)
            if front_type == 'stochastic':
                plot_cdf2ax(ax, f_survived, wave_type + '; ' + front_type, KS_test=KS_test(A, B), data_label='p = ' + str(p))
            else:
                plot_cdf2ax(ax, f_survived, wave_type + '; ' + front_type, KS_test='uniform', data_label='p = ' + str(p))
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    if save == True:
        plt.savefig(output_plots + 'fig4_stationary_' + front_type + '.pdf')


def fig5_logit(N, B_list, front_type='stochastic', p_target=0.2, epsilon=1E-5, A=0.0, save=True):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/'
        extension = '_points=auto_stochastic.csv'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/'
        extension = '_points=auto_deterministic.csv'
    else:
        print("Unknown front type. Quitting.")
        return -1

    #fig = plt.figure(figsize=cm2inch(17.8, 5.8))
    fig = plt.figure(figsize=cm2inch(17.8, 11.6))

    for i, B in enumerate(B_list):
        ax = fig.add_subplot(2, 3, i + 4)
        wave_type = wave_dict[tools.linear_cooperativity(A, B)]
        # Load fraction histogram
        fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
        try:
            fraction_array = np.loadtxt(fname, delimiter=',')
        except:
            print(fname, ' file not found. Exiting without ploting.')
            return -1
        time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=p_target, epsilon=epsilon)

        # Load sampling times
        fname = input_path + data_dir + generate_fname('sampling_times_', N, A, B, extension)
        delta_index = fraction_array.shape[1]
        sampling_times = np.loadtxt(fname)
        sampling_time = sampling_times[time_point + 1001 - delta_index]

        print(B, sampling_time, delta_index)

        if front_type == 'stochastic':
            plot_logit2ax(ax, f_survived, wave_type + '; ' + front_type, data_label='simulations')
        else:
            plot_logit2ax(ax, f_survived, wave_type + '; ' + front_type, data_label='simulations')

        psi = np.linspace(1E-5, 10, 100)
        pdf_psi = np.exp(psi) / ((1+np.exp(psi)) * np.log(1+np.exp(psi)))
        pdf_psi /= pdf_psi.sum() * (psi[1] - psi[0])

        psi_jackpot = np.linspace(-10, 10, 100)
        pdf_jackpot = np.tanh(psi_jackpot/2) / psi_jackpot
        pdf_jackpot /= pdf_jackpot.sum() * (psi_jackpot[1] - psi_jackpot[0])

        i_center = np.argmin(abs(psi_jackpot))
        psi_0 = np.zeros(len(psi_jackpot))
        psi_0[i_center] = len(psi_jackpot)
        #psi_0[i_center-10:i_center+10] = len(psi_jackpot)/20

        #ax.plot(psi, pdf_psi/2, lw=1, linestyle='--', c='k')
        #ax.plot(-psi, pdf_psi/2, lw=1, linestyle='--', c='k')
        ax.plot(psi_jackpot, pdf_jackpot, lw=2, linestyle='-', c='k')

        for t in [ 1 ]:
            pdf_propagator = tools.bs_propagator(psi, 0, t, 1)
            ax.plot(psi, pdf_propagator, lw=1, ls='--', c='r', label='$t/N_e=$' + str(t))
            ax.plot(-psi, pdf_propagator, lw=1, ls='--', c='r')

    for i, B in enumerate(B_list):
        ax = fig.add_subplot(2, 3, i + 1)
        wave_type = wave_dict[tools.linear_cooperativity(A, B)]
        # Load fraction histogram
        fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
        try:
            fraction_array = np.loadtxt(fname, delimiter=',')
        except:
            print(fname, ' file not found. Exiting without ploting.')
            return -1
        time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=p_target, epsilon=epsilon)
        if front_type == 'stochastic':
            plot_pdf2ax(ax, f_survived, wave_type + '; ' + front_type, overlap=False, data_label='simulations')
        else:
            plot_pdf2ax(ax, f_survived, wave_type + '; ' + front_type, overlap=False, data_label='simulations')


    plt.tight_layout()
    if save == True:
        plt.savefig(output_plots + 'fig5_logitpdf_' + front_type + '.pdf')


def fig6_logit_cdf(N, B_list, front_type='stochastic', p_target=0.2, A=0.0, save=True):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/'
        extension = '_points=auto_stochastic.csv'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/'
        extension = '_points=auto_deterministic.csv'
    else:
        print("Unknown front type. Quitting.")
        return -1

    fig = plt.figure(figsize=cm2inch(17.8, 5.8))

    for i, B in enumerate(B_list):
        ax = fig.add_subplot(1, 3, i + 1)
        wave_type = wave_dict[tools.linear_cooperativity(A, B)]
        # Load fraction histogram
        fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
        try:
            fraction_array = np.loadtxt(fname, delimiter=',')
        except:
            print(fname, ' file not found. Exiting without ploting.')
            return -1
        time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=p_target)
        if front_type == 'stochastic':
            plot_logitcdf2ax(ax, f_survived, wave_type + '; ' + front_type, data_label='simulations')
        else:
            plot_logitcdf2ax(ax, f_survived, wave_type + '; ' + front_type, data_label='simulations')

    plt.tight_layout()
    if save == True:
        plt.savefig(output_plots + 'fig5_logitcdf_' + front_type + '.pdf')


def fig7_pdfs(N, B_list, front_type='stochastic', p_target=0.2, A=0.0, save=True):
    if front_type == 'stochastic':
        data_dir = 'stochastic_fronts/'
        extension = '_points=auto_stochastic.csv'
    elif front_type == 'deterministic':
        data_dir = 'deterministic_fronts/'
        extension = '_points=auto_deterministic.csv'
    else:
        print("Unknown front type. Quitting.")
        return -1

    fig = plt.figure(figsize=cm2inch(17.8, 5.8))

    for i, B in enumerate(B_list):
        ax = fig.add_subplot(1, 3, i + 1)
        wave_type = wave_dict[tools.linear_cooperativity(A, B)]
        # Load fraction histogram
        fname = input_path + data_dir + generate_fname('fraction_samples_', N, A, B, extension)
        try:
            fraction_array = np.loadtxt(fname, delimiter=',')
        except:
            print(fname, ' file not found. Exiting without ploting.')
            return -1
        time_point, p_survived, f_survived = get_distribution(fraction_array, p_target=p_target)
        if front_type == 'stochastic':
            plot_pdf2ax(ax, f_survived, wave_type + '; ' + front_type, overlap=False, data_label='simulations')
        else:
            plot_pdf2ax(ax, f_survived, wave_type + '; ' + front_type, overlap=False, data_label='simulations')

    plt.tight_layout()
    if save == True:
        plt.savefig(output_plots + 'fig2_cdfs_' + front_type + '.pdf')


def plot_all():
    A = 0.0
    B_stochastic = [0.0, 1.0, 3.0, 3.3, 3.5, 8.0, 10.0]

    for B in B_stochastic:
        expansion_label = tools.linear_cooperativity(A, B)
        if expansion_label == 0:
            KS_test = 'pulled'
        elif expansion_label == 1:
            KS_test = 'semi-pushed'
        elif expansion_label == 2:
            KS_test = 'uniform'
        else:
            KS_test = None

        plot_parameters(input_path + 'stochastic_fronts/', N=1000000, A=A, B=B, KS_test=KS_test, p_target=0.2, bins=100, extension='_stochastic')

    print('\n')

    B_deterministic = [0.0, 3.33, 10.0]
    for B in B_deterministic:
        plot_parameters(input_path + 'deterministic_fronts/', N=1000000, A=A, B=B, KS_test='uniform', p_target=0.2, bins=100, extension='_deterministic')

    B_list = [0.0, 1.0, 3.0, 3.33, 3.5, 8.0, 10.0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_stationarity(input_path + 'stochastic_fronts/', ax, N=1000000, B=10.0)
    plt.savefig(output_plots + 'stationary_N1000000_B10.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_stationarity(input_path + 'stochastic_fronts/', ax, N=1000000, B=3.0)
    plt.savefig(output_plots + 'stationary_N1000000_B3.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_stationarity(input_path + 'stochastic_fronts/', ax, N=1000000, B=1.0)
    plt.savefig(output_plots + 'stationary_N1000000_B1.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_stationarity(input_path + 'stochastic_fronts/', ax, N=1000000, B=0.0)
    plt.savefig(output_plots + 'stationary_N1000000_B0.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_stationarity(input_path + 'deterministic_fronts/', ax, N=1000000, B=10.0)
    plt.savefig(output_plots + 'stationary_N1000000_B10_deterministic.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_stationarity(input_path + 'deterministic_fronts/', ax, N=1000000, B=3.33)
    plt.savefig(output_plots + 'stationary_N1000000_B3.33_deterministic.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_stationarity(input_path + 'deterministic_fronts/', ax, N=1000000, B=0.0)
    plt.savefig(output_plots + 'stationary_N1000000_B0_deterministic.pdf')


if __name__=='__main__':
    #fig1_workflow(1000000, 10.0, 30, save=False)
    fig2_cdfs(1000000, [0.0, 3.33, 10.0], front_type='stochastic', p_target=0.9, save=True)
    fig3_cdfs_nooverlap(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.9, save=False)
    fig3_cdfs_nooverlap(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.5, save=False)
    fig3_cdfs_nooverlap(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.2, save=False)
    fig4_stationary(1000000, [0.0, 3.5, 10.0], [0.2, 0.5, 0.90], front_type='stochastic')
    fig5_logit(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.9, epsilon=1E-5, save=True)
    fig5_logit(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.5, epsilon=1E-5, save=True)
    fig5_logit(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.2, epsilon=1E-5, save=True)
    #fig6_logit_cdf(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.9, save=False)
    #fig7_pdfs(1000000, [0.0, 3.5, 10.0], front_type='stochastic', p_target=0.9, save=False)
    plt.show()
