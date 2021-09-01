import pickle
import theory
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import coalescentmoments as moments
import branching_processes as bp
import forward_in_time_stats as forward
import data_table as dt
import analysis_tools as tools
import frequency_distribution as ft
import analyze_distributions as ad
from fpmi import fpmi
from mpl_toolkits.axes_grid1 import AxesGrid


main_figures_dir = '../figures/'
figure_panels_dir = '../figures/'

# Configure matplotlib environment
helvetica_scale_factor = 0.92 # rescale Helvetica to other fonts of same size
mpl.rcParams['font.size'] = 10 * helvetica_scale_factor
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['axes.titlesize'] = 12 * helvetica_scale_factor
#mpl.rcParams['text.usetex'] = True

single_col_width = 3.43 # = 8.7 cm
double_col_width = 7.01 # = 17.8 cm


def plot_mixing_time(t_list=['1', '50', '100'], verbose=False):
    if verbose:
        print('Making mixing time panels (Fig. 2)...')

    fig = plt.figure(figsize=(single_col_width, 1.6 * single_col_width))

    # Initial conditions
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(-25, 50)
    ax1.set_xticks([-25, 0, 25, 50])
    ax1.set_ylim(0, 0.4)
    ax1.set_ylabel(r'ancestor distribution', fontsize=14)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax = ax1.twinx()
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([])

    backward_results = '../results/backward_averaged.dat'
    ancestors_data = dt.load_table(backward_results)
    backward_table = ancestors_data.table
    profile = backward_table['profiles'].values[0]['f']
    profile_density = profile / np.max(profile)
    ancestors = backward_table['ancestor distributions'].values[0]
    distr_z1 = tools.normalize_distribution(ancestors['z_lower'][0])
    distr_z2 = tools.normalize_distribution(ancestors['z_upper'][0])
    zeta = np.arange(len(distr_z1)) - len(distr_z1) // 2

    ax1.plot(zeta, distr_z1, ls='-', lw=1, drawstyle='steps-post', label=f'')
    ax1.plot(zeta, distr_z2, ls='-', lw=1, drawstyle='steps-post', label=f'')
    ax.plot(zeta, profile_density, ls='-', lw=2, c='gray')

    # Ancestor distribution
    ax3 = fig.add_subplot(212)
    ax3.set_xlabel(r'position, $\zeta$', fontsize=14)
    ax3.set_xlim(-25, 50)
    ax3.set_xticks([-25, 0, 25, 50])
    ax3.set_ylabel(r'ancestor distribution', fontsize=14)
    ax3.set_ylim(0.0, 0.15)
    ax3.set_yticks([0, 0.05, 0.1, 0.15])
    ax = ax3.twinx()
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.1)

    ancestor_avg = backward_table['ancestor distributions'].values[0]
    for t in [5]:
        distr_timeseries = ancestor_avg['z_lower']
        distr = tools.normalize_distribution(distr_timeseries[t])
        zeta = np.arange(len(distr)) - len(distr) // 2
        ax3.plot(zeta, distr, ls='-', lw=1, drawstyle='steps-post', label=f't = {t}')

        distr_timeseries = ancestor_avg['z_upper']
        distr = tools.normalize_distribution(distr_timeseries[t])
        zeta = np.arange(len(distr)) - len(distr) // 2
        ax3.plot(zeta, distr, ls='-', lw=1, drawstyle='steps-post', label=f't = {t}')
    ax.plot(zeta, profile_density, ls='-', lw=2, c='gray')

    plt.tight_layout()
    plt.savefig(main_figures_dir + f'mixing_time.pdf')

    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(single_col_width, single_col_width))

    ax = fig.add_subplot(111)
    ax.set_xlabel('')
    ax.set_xlim(-20, 20)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.1)

    num_profiles = 30
    backward_runs = dt.load_table('../results/backward_ancestors.dat')
    for i in range(num_profiles):
        ancestor_data = backward_runs.table.loc[i, 'ancestor distributions']
        profile = ancestor_data['profile'] / np.max(ancestor_data['profile'])
        ax.plot(zeta, profile, lw=0.5, c='gray', zorder=1)
        i_lower = ancestor_data['z_lower']['z']
        ax.scatter([zeta[i_lower]], [profile[i_lower]], facecolor=cmap(0), s=10, zorder=2)
        i_upper = ancestor_data['z_upper']['z']
        ax.scatter([zeta[i_upper]], [profile[i_upper]], facecolor=cmap(1), s=10, zorder=2)

    plt.savefig(f'{main_figures_dir}mixing_time_inset.pdf')


def plot_sfs(w=15, n=40, data_scale=100, verbose=False):
    if verbose:
        print('Making SFS figure (Fig. 4)...')

    y_min = 1E-3
    xticks = [1, 5, 10, 15, 19]
    i = np.arange(1, n)
    xi_bs = moments.sfs_moments(n, alpha=1.0, m2=False)
    xi_bs /= np.sum(xi_bs)
    xi_beta = moments.sfs_moments(n, alpha=1.5, m2=False)
    xi_beta /= np.sum(xi_beta)
    xi_kingman = moments.sfs_moments(n, alpha=2.0, m2=False)
    xi_kingman /= np.sum(xi_kingman)
    xi_min = min(np.min(xi_bs), np.min(xi_beta), np.min(xi_kingman))
    xi_max = max(np.max(xi_bs), np.max(xi_beta), np.max(xi_kingman))

    fp_sfs_dt = dt.load_table(f'../results/trees_tmrca2000_fullypushed_n{n}_s10_w{w}_sfs_avg.dat')
    fp_sfs = fp_sfs_dt.table.loc[0, 'SFS']
    x_fp = np.arange(1, len(fp_sfs) + 1)
    sp_sfs_dt = dt.load_table(f'../results/trees_tmrca2000_semipushed_n{n}_s10_w{w}_sfs_avg.dat')
    sp_sfs = sp_sfs_dt.table.loc[0, 'SFS']
    x_sp = np.arange(1, len(sp_sfs) + 1)
    p_sfs_dt = dt.load_table(f'../results/trees_tmrca2000_pulled_n{n}_s10_w{w}_sfs_avg.dat')
    p_sfs = p_sfs_dt.table.loc[0, 'SFS']
    x_p = np.arange(1, len(p_sfs) + 1)

    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    ax = fig.add_subplot(131)
    ax.set_title('fully pushed')
    ax.set_xlabel(r'derived allele count, $i$')
    ax.set_ylabel(r'relative frequency, $\langle \xi_i \rangle$')
    ax.set_yscale('log')
    ax.set_xlim([0.5, i[-1] + 0.5])
    ax.set_xticks(xticks)
    ax.set_ylim([y_min, 2 * xi_max])

    ax.plot(i, xi_kingman, c='b', lw=1, ls='-', label='Kingman')
    ax.scatter(x_fp, fp_sfs, c='b', s=20)
    ax.legend(loc='lower right', fontsize=9)

    ax = fig.add_subplot(132)
    ax.set_title('semi-pushed')
    ax.set_xlabel(r'derived allele count, $i$')
    ax.set_yscale('log')
    ax.set_xlim([0.5, i[-1] + 0.5])
    ax.set_xticks(xticks)
    ax.set_ylim([y_min, 2 * xi_max])

    ax.plot(i, xi_beta, c='g', lw=1, ls='-', label=r'Beta, $\alpha=1.5$')
    ax.scatter(x_sp, sp_sfs, c='g', s=20)
    ax.legend(loc='lower right', fontsize=9)

    ax = fig.add_subplot(133)
    ax.set_title('pulled')
    ax.set_xlabel(r'derived allele count, $i$')
    ax.set_yscale('log')
    ax.set_xlim([0.5, i[-1] + 0.5])
    ax.set_xticks(xticks)
    ax.set_ylim([y_min, 2 * xi_max])

    ax.plot(i, xi_bs, c='r', lw=1, ls='-', label='Bolthausen-Sznitman')
    ax.scatter(x_p, p_sfs, c='r', s=20)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'sfs.pdf')


def plot_2sfs(w=15, n=20, verbose=False):
    if verbose:
        print('Making 2-SFS figure (Fig. S1)...')

    ticks = np.array([0.5, 4.5, 9.5, 14.5, 18.5])
    tick_labels = [1, 5, 10, 15, 19]

    fig = plt.figure(figsize=(double_col_width, 2 * double_col_width / 3))

    grid = AxesGrid(fig, 211, nrows_ncols=(1, 3), axes_pad=0.075, cbar_mode='edge', cbar_location='right', cbar_pad=0.1)
    for ax in grid:
        ax.set_ylabel(r'derived allele count, $j$', fontsize=9)
        ax.set_xticks(ticks - 0.5)
        ax.set_xticklabels([])
        ax.set_yticks(ticks - 0.5)
        ax.tick_params(length=2.5)
    grid[0].set_title(r'Kingman')
    grid[0].set_yticklabels(tick_labels)
    grid[1].set_title(r'Beta, $\alpha=1.5$')
    grid[2].set_title(r'Bolthausen-Sznitman')

    #-------------------------------------------------------------------------#
    # Coalescent predictions (first row)
    #-------------------------------------------------------------------------#

    allele_count = np.arange(1, n)
    bs_sfs, bs_sfs2 = moments.sfs_moments(n, alpha=1.0, m2=True)
    bs_var = np.zeros(bs_sfs2.shape)
    bs_fpmi = fpmi(bs_sfs, bs_sfs2)
    for i in range(bs_var.shape[0]):
        for j in range(i + 1):
            bs_var[i][j] = bs_sfs2[i][j] - bs_sfs[i] * bs_sfs[j]
            bs_var[j][i] = bs_var[i][j]

    beta_sfs, beta_sfs2 = moments.sfs_moments(n, alpha=1.5, m2=True)
    beta_var = np.zeros(beta_sfs2.shape)
    beta_fpmi = fpmi(beta_sfs, beta_sfs2)
    for i in range(beta_var.shape[0]):
        for j in range(i + 1):
            beta_var[i][j] = beta_sfs2[i][j] - beta_sfs[i] * beta_sfs[j]
            beta_var[j][i] = beta_var[i][j]

    kingman_sfs, kingman_sfs2 = moments.sfs_moments(n, alpha=2.0, m2=True)
    kingman_var = np.zeros(kingman_sfs2.shape)
    kingman_fpmi = fpmi(kingman_sfs, kingman_sfs2)
    for i in range(kingman_var.shape[0]):
        for j in range(i + 1):
            kingman_var[i][j] = kingman_sfs2[i][j] - kingman_sfs[i] * kingman_sfs[j]
            kingman_var[j][i] = kingman_var[i][j]

    max_sfs2 = 3.0
    cmap='PuOr_r'
    grid[0].imshow(kingman_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2, origin='lower')
    grid[1].imshow(beta_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2, origin='lower')
    im = grid[2].imshow(bs_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2, origin='lower')

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_ylabel(r'$\langle \xi_i \xi_j \rangle - \langle \xi_i \rangle \langle \xi_j \rangle$')

    #-------------------------------------------------------------------------#
    # Expansion data (second row)
    #-------------------------------------------------------------------------#

    grid = AxesGrid(fig, 212, nrows_ncols=(1, 3), axes_pad=0.075, cbar_mode='edge', cbar_location='right', cbar_pad=0.1)
    for ax in grid:
        ax.set_xlabel(r'derived allele count, $i$', fontsize=9)
        ax.set_ylabel(r'derived allele count, $j$', fontsize=9)
        ax.set_xticks(ticks - 0.5)
        ax.set_xticklabels([1, 5, 10, 15, 19])
        ax.set_yticks(ticks - 0.5)
        ax.tick_params(length=2.5)
    grid[0].set_title(r'fully pushed')
    grid[0].set_yticklabels(tick_labels)
    grid[1].set_title(r'semi-pushed')
    grid[2].set_title(r'pulled')

    allele_count = np.arange(1, n + 1)
    max_fpmi = 3.0

    for i, wave_type in enumerate(['fullypushed', 'semipushed', 'pulled']):
        sfs_dt = dt.load_table(f'../results/trees_tmrca2000_{wave_type}_n{n}_s10_w{w}_sfs_avg.dat')
        sfs_m1 = sfs_dt.table.loc[0, 'SFS']
        sfs_m2 = sfs_dt.table.loc[0, '2-SFS']
        data_fpmi = fpmi(sfs_m1, sfs_m2)
        im = grid[i].imshow(data_fpmi[:-1, :-1], cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi, origin='lower')

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_ylabel(r'$\langle \xi_i \xi_j \rangle - \langle \xi_i \rangle \langle \xi_j \rangle$')
    plt.savefig(main_figures_dir + '2sfs.pdf')


def plot_offspring_distribution_si(y_min=1E-3, y_max=2, verbose=False):
    if verbose:
        print('Making offspring distribution figure (Fig. S4)...')

    label_fontsize = 12
    markers = ['o', '^', 's']
    y = np.logspace(-3,3)
    y_range = [min(y), max(y)]

    offspring_data = dt.load_table(f'../results/offspring_n9600_w25_atfront_avg.dat')

    Times = [0, 10, 25]
    alpha = 0.9
    CC_st = bp.exact_complementary_cumulative(y, alpha='short')

    t = 2
    fullypushed_table = offspring_data.table.loc[1, :]
    clone_sizes = fullypushed_table['clone_sizes']
    x, compl_cumulative = forward.convert_counts_to_distribution(clone_sizes[t])
    cutoff = 150 * fullypushed_table['n'] / forward.average_clone_sizes(clone_sizes[t])

    pulled_table = offspring_data.table.loc[0, :]
    clone_sizes = pulled_table['clone_sizes']
    x, compl_cumulative = forward.convert_counts_to_distribution(clone_sizes[t])

    fig = plt.figure(figsize=(single_col_width, single_col_width))

    #-------------------------------------------------------------------------#
    # Complementary cumulative
    #-------------------------------------------------------------------------#

    ax = fig.add_subplot(111)
    ax.set_xlabel('clone sizes, $s$', fontsize=label_fontsize)
    ax.set_xlim([min(x), 1E4])
    ax.set_ylabel('complementary cumulative, $1 - F(s)$', fontsize=label_fontsize)
    ax.set_ylim([y_min, y_max])

    alpha = 0.9
    t = 2
    CC_st = bp.exact_complementary_cumulative(y, alpha='short')
    pulled_table = offspring_data.table.loc[0, :]
    clone_sizes = pulled_table['clone_sizes']
    x_p, compl_cumulative_p = forward.convert_counts_to_distribution(clone_sizes[t])
    fullypushed_table = offspring_data.table.loc[1, :]
    clone_sizes = fullypushed_table['clone_sizes']
    x_fp, compl_cumulative_fp = forward.convert_counts_to_distribution(clone_sizes[t])

    ax.loglog(x_p, compl_cumulative_p, c='r', lw=1, label=f'pulled')
    ax.loglog(x_fp, compl_cumulative_fp, c='b', lw=1, label=f'fully pushed')
    ax.loglog(y, CC_st, '-', lw=1, color='gray', alpha=0.5, label='short tail')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'offspring_distribution_comparison_si.pdf')


def plot_stochastic_pdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs, b_semipushed=3.33):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    # Fully-pushed waves
    bins_fullypushed=20
    ax = fig.add_subplot(131)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 3.4])
    ax.set_yticks([0, 1, 2.0, 3.0])
    time_index = 200
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)
    plot_histogram(freqs, ax, 'stochastic fully pushed', y_label=True, bins=bins_fullypushed)

    f_theory = np.linspace(0, 1, 100)
    y_theory = np.ones(len(f_theory))
    ax.plot(f_theory, y_theory, lw=2, c='b')

    # Semi-pushed waves
    bins_semipushed = 30
    ax = fig.add_subplot(132)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 4.0])
    ax.set_yticks([0, 1, 2, 3, 4])
    time_index = 700
    freqs = semipushed_freqs.get_nonfixed_f(time_index, logit=False)
    tau = semipushed_freqs.get_t(time_index, units='ne')
    plot_histogram(freqs, ax, 'stochastic semi-pushed', y_label=False, bins=bins_semipushed)

    # Approximate fit of theoretical prediction
    counts, bin_edges = np.histogram(freqs, bins=bins_semipushed)
    delta_x = bin_edges[1] - bin_edges[0]
    epsilon = np.sqrt(np.prod(bin_edges[:2]))
    alpha = tools.alpha_coop(b_semipushed, 0.01, 0.4)
    f_theory = np.linspace(bin_edges[1]/5, 1 - bin_edges[1]/5, 100)
    y_theory = (f_theory * (1 - f_theory))**(-alpha)
    y_theory /= np.sum(y_theory) * delta_x / 2
    ax.plot(f_theory, y_theory, lw=2, c='g')

    # Pulled waves
    bins_pulled = 30
    ax = fig.add_subplot(133)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    time_index=38
    freqs = pulled_freqs.get_nonfixed_f(time_index, logit=False)
    plot_histogram(freqs, ax, 'stochastic pulled', y_label=False, bins=bins_pulled)

    counts, bin_edges = np.histogram(freqs, bins=bins_pulled)
    delta_x = bin_edges[1] - bin_edges[0]
    f_theory = np.linspace(bin_edges[1] / 2, 1 - (1 - bin_edges[-2]) / 2, 100)
    y_theory = 1/(f_theory * (1 - f_theory))
    y_theory /= np.sum(y_theory) * delta_x / 2
    ax.plot(f_theory, y_theory, lw=2, c='r')

    plt.tight_layout()
    plt.savefig(figure_panels_dir + 'stochastic_waves_pdf.pdf')


def plot_histogram(frequencies, ax, title, y_label=True, bins=30):
    ax.set_title(title)
    ax.set_xlabel('allele frequency, $f$', fontsize=12)
    if y_label == True:
        ax.set_ylabel('probability density, $P(f)$', fontsize=12)
    ax.hist(frequencies, bins=bins, density=True, color='slategray')


def plot_deterministic_pdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs):
    f_theory = np.linspace(0, 1, 100)
    y_theory = np.ones(len(f_theory))

    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    # Fully-pushed waves
    print("\t...fully pushed deterministic fronts AFD eigenvector")
    bins_fullypushed=20
    time_index = 250
    time_in_neff = pulled_freqs.get_t(time_index)
    time_string = '{:.2f}'.format(time_in_neff)
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)

    ax = fig.add_subplot(131)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 1.55])
    ax.set_yticks([0, 0.5, 1, 1.5])

    plot_histogram(freqs, ax, 'deterministic fully pushed', y_label=True, bins=bins_fullypushed)
    ax.plot(f_theory, y_theory, lw=2, c='b', ls='--')

    # Semi-pushed waves
    print("\t...semi-pushed deterministic fronts AFD eigenvector")
    bins_semipushed = 20
    time_index = 70
    time_in_neff = pulled_freqs.get_t(time_index)
    time_string = '{:.2f}'.format(time_in_neff)
    freqs = semipushed_freqs.get_nonfixed_f(time_index, logit=False)

    ax = fig.add_subplot(132)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 1.55])
    ax.set_yticks([0, 0.5, 1, 1.5])

    plot_histogram(freqs, ax, 'deterministic semi-pushed', y_label=False, bins=bins_semipushed)
    ax.plot(f_theory, y_theory, lw=2, c='g', ls='--')

    # Pulled waves
    print("\t...pulled deterministic fronts AFD eigenvector")
    bins_pulled = 20
    time_index=200
    time_in_neff = pulled_freqs.get_t(time_index)
    time_string = '{:.2f}'.format(time_in_neff)
    freqs = pulled_freqs.get_nonfixed_f(time_index, logit=False)

    ax = fig.add_subplot(133)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 1.55])
    ax.set_yticks([0, 0.5, 1, 1.5])

    plot_histogram(freqs, ax, 'deterministic pulled', y_label=False, bins=bins_pulled)
    ax.plot(f_theory, y_theory, lw=2, c='r', ls='--')

    plt.tight_layout()
    plt.savefig(figure_panels_dir + 'deterministic_waves_pdf.pdf')


def plot_stochastic_cdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    # Fully-pushed waves
    bins_fullypushed=50
    time_index = 200
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)

    ax = fig.add_subplot(131)
    plot_cdf2ax(ax, freqs, '', 'b', ls='-', uniform_theory=True, pulled_theory=False, y_label=True, bins=bins_fullypushed)

    # Semi-pushed waves
    bins_semipushed = 50
    time_index = 700
    freqs = semipushed_freqs.get_nonfixed_f(time_index, logit=False)
    tau = semipushed_freqs.get_t(time_index, units='ne')

    ax = fig.add_subplot(132)
    #plot_cdf2ax(ax, freqs, '', 'b', ls='-', uniform_theory=True, pulled_theory=True, y_label=False, bins=bins_semipushed)
    plot_cdf2ax(ax, freqs, '', 'b', ls='-', uniform_theory=False, semipushed_theory=True, pulled_theory=False, y_label=False, bins=bins_semipushed)

    # Pulled waves
    bins_pulled = 50
    time_index=100
    freqs = pulled_freqs.get_nonfixed_f(time_index, logit=False)

    ax = fig.add_subplot(133)
    plot_cdf2ax(ax, freqs, '', 'r', uniform_theory=False, pulled_theory=True, y_label=False, bins=bins_pulled)

    plt.tight_layout()
    plt.savefig(figure_panels_dir + 'stochastic_waves_cdf.pdf')


def plot_cdf2ax(ax, f_survived, title, theory_color, ls='--', KS_test='uniform', uniform_theory=True, semipushed_theory=False, pulled_theory=True, b_semipushed=3.33, y_label=True, bins=50, overlap=True, data_label=None):
    if overlap == True:
        f_overlap = ad.overlap_fractions(f_survived)
    else:
        f_overlap = f_survived

    counts, bin_edges = np.histogram(f_overlap, bins=bins, density=True)
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(bins)])
    cdf = np.cumsum(counts) / counts.sum()

    if uniform_theory == True:
        if ls == '--':
            ax.plot(bin_centers, stats.uniform.cdf(bin_centers, loc=0.5, scale=0.5), linestyle=ls, lw=1.25, c=theory_color)
        elif ls == '-':
            ax.plot(bin_centers, stats.uniform.cdf(bin_centers, loc=0.5, scale=0.5), linestyle=ls, lw=1.0, c=theory_color)

    if semipushed_theory == True:
        alpha = tools.alpha_coop(b_semipushed, 0.01, 0.4)
        ax.plot(bin_centers, ad.theory_cdf_numeric(bin_centers, alpha=alpha), ls='-', lw=1.0, c='g')

    if pulled_theory == True:
        ax.plot(bin_centers, ad.theory_cdf_numeric(bin_centers), ls='-', lw=1.0, c='r')

    # Set plot labels
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('major allele frequency, $f$', fontsize=12)
    ax.set_xticks([0.5, 0.75, 1.0])
    if y_label == True:
        ax.set_ylabel('cumulative probability', fontsize=12)
    ax.set_yticks([0, 0.5, 1.0])

    # Plot data
    ax.plot(bin_centers, cdf, linestyle=':', c='k', lw=2.5, label=data_label)


def plot_deterministic_cdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    # Fully-pushed waves
    bins_fullypushed=50
    time_index = 250
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)

    ax = fig.add_subplot(131)
    plot_cdf2ax(ax, freqs, '', 'b', uniform_theory=True, pulled_theory=False, y_label=True, bins=bins_fullypushed)

    # Semi-pushed waves
    bins_semipushed = 50
    time_index = 70
    freqs = semipushed_freqs.get_nonfixed_f(time_index, logit=False)
    tau = semipushed_freqs.get_t(time_index, units='ne')

    ax = fig.add_subplot(132)
    plot_cdf2ax(ax, freqs, '', 'g', uniform_theory=True, pulled_theory=False, y_label=False, bins=bins_semipushed)

    # Pulled waves
    bins_pulled = 50
    time_index=200
    freqs = pulled_freqs.get_nonfixed_f(time_index, logit=False)

    ax = fig.add_subplot(133)
    plot_cdf2ax(ax, freqs, '', 'r', uniform_theory=True, pulled_theory=False, y_label=False, bins=bins_pulled)

    plt.tight_layout()
    plt.savefig(figure_panels_dir + 'deterministic_waves_cdf.pdf')


def plot_eigenvector_panels(data_dir='../data/', verbose=False):
    if verbose:
        print('Making allele frequency distribution eigenvector panels (Figs. 5 and S3)...')

    pulled_freqs = ft.frequency_distribution(
            data_dir + 'stochastic_fronts/fraction_samples_N1000000_A0.0_B0.0_points=auto_stochastic.csv'
            )
    semipushed_freqs = ft.frequency_distribution(
            data_dir + 'stochastic_fronts/fraction_samples_N1000000_A0.0_B3.33_points=auto_stochastic.csv'
            )
    fullypushed_freqs = ft.frequency_distribution(
            data_dir + 'stochastic_fronts/fraction_samples_N1000000_A0.0_B10.0_points=auto_stochastic.csv'
            )
    plot_stochastic_pdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs)
    plot_stochastic_cdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs)

    pulled_det = ft.frequency_distribution(
            data_dir + 'deterministic_fronts/fraction_samples_N1000000_A0.0_B0.0_points=auto_deterministic.csv'
            )
    semipushed_det = ft.frequency_distribution(
            data_dir + 'deterministic_fronts/fraction_samples_N1000000_A0.0_B3.33_points=auto_deterministic.csv'
            )
    fullypushed_det = ft.frequency_distribution(
            data_dir + 'deterministic_fronts/fraction_samples_N1000000_A0.0_B10.0_points=auto_deterministic.csv'
            )
    plot_deterministic_pdfs(pulled_det, semipushed_det, fullypushed_det)
    plot_deterministic_cdfs(pulled_det, semipushed_det, fullypushed_det)


def plot_2sfs_summary_stats(sfs_dt_fname='../results/trees_tmrca2000_sfs_avg.dat', n=20, verbose=False):
    if verbose:
        print('Making 2-SFS summary stats figure (Fig. S2)...')

    bsc_sfs, bsc_sfs2 = moments.sfs_moments(n, alpha=1.0, m2=True)
    bsc_var = np.zeros(bsc_sfs2.shape)
    bsc_fpmi = fpmi(bsc_sfs, bsc_sfs2)
    for i in range(bsc_var.shape[0]):
        for j in range(bsc_var.shape[1]):
            bsc_var[i][j] = bsc_sfs2[i][j] - bsc_sfs[i] * bsc_sfs[j]
            bsc_var[j][i] = bsc_var[i][j]

    beta_sfs, beta_sfs2 = moments.sfs_moments(n, alpha=1.5, m2=True)
    beta_var = np.zeros(beta_sfs2.shape)
    beta_fpmi = fpmi(beta_sfs, beta_sfs2)
    for i in range(beta_var.shape[0]):
        for j in range(beta_var.shape[1]):
            beta_var[i][j] = beta_sfs2[i][j] - beta_sfs[i] * beta_sfs[j]
            beta_var[j][i] = beta_var[i][j]

    kingman_sfs, kingman_sfs2 = moments.sfs_moments(n, alpha=2.0, m2=True)
    kingman_var = np.zeros(kingman_sfs2.shape)
    kingman_fpmi = fpmi(kingman_sfs, kingman_sfs2)
    for i in range(kingman_var.shape[0]):
        for j in range(kingman_var.shape[1]):
            kingman_var[i][j] = kingman_sfs2[i][j] - kingman_sfs[i] * kingman_sfs[j]
            kingman_var[j][i] = kingman_var[i][j]

    sfs_dt = dt.load_table(sfs_dt_fname)
    sfs_df = sfs_dt.table
    data = sfs_df

    fp_sfs = data.loc[data['wave type'] == 'fullypushed', 'SFS'].values[0]
    fp_2sfs = data.loc[data['wave type'] == 'fullypushed', '2-SFS'].values[0]
    fp_fpmi = fpmi(fp_sfs, fp_2sfs)

    sp_sfs = data.loc[data['wave type'] == 'semipushed', 'SFS'].values[0]
    sp_2sfs = data.loc[data['wave type'] == 'semipushed', '2-SFS'].values[0]
    sp_fpmi = fpmi(sp_sfs, sp_2sfs)

    p_sfs = data.loc[data['wave type'] == 'pulled', 'SFS'].values[0]
    p_2sfs = data.loc[data['wave type'] == 'pulled', '2-SFS'].values[0]
    p_fpmi = fpmi(p_sfs, p_2sfs)

    fig = plt.figure(figsize=(single_col_width, 1.6 * single_col_width))
    ax = fig.add_subplot(211)
    ax.set_xticks([])
    ax.set_ylim(-1.5, 3.5)
    ax.set_ylabel('2-SFS summary statistic', fontsize=10)
    add_summary_stats_axis(kingman_fpmi, beta_fpmi, bsc_fpmi, ax, mk='D', labels=['Kingman', 'Beta', 'Bolthausen-Sznitman'])

    ax = fig.add_subplot(212)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_xticklabels(['upper diagonal', 'off diagonal', 'lower triangle', 'upper triangle'], fontsize=9, rotation=30)
    ax.set_ylim(-1.5, 3.5)
    ax.set_ylabel('2-SFS summary statistic', fontsize=10)
    add_summary_stats_axis(fp_fpmi, sp_fpmi, p_fpmi, ax)

    plt.tight_layout()
    plt.savefig(figure_panels_dir + '2sfs_summary_stats.pdf')
    plt.close()


def add_summary_stats_axis(kingman_fpmi, beta_fpmi, bsc_fpmi, ax, mk='o', labels=['fully pushed', 'semi-pushed', 'pulled'], n=20):
    x0 = 0
    x_feature = 2.0
    x_k = 0
    x_beta = 1
    x_bsc = 2
    epsilon = 0.3
    i_diag = n//2
    ms = 5

    kingman_off = []
    beta_off = []
    bsc_off = []
    for i in range(n - 1):
        if i != n - 2 - i:
            kingman_off.append(kingman_fpmi[i, n - 2 - i])
            beta_off.append(beta_fpmi[i, n - 2 - i])
            bsc_off.append(bsc_fpmi[i, n - 2 - i])

    kingman_lower = []
    kingman_upper = []
    beta_lower = []
    beta_upper = []
    bsc_lower = []
    bsc_upper = []
    for i in range(n - 1):
        for j in range(n - 1):
            if i != j:
                if i + j < n - 2:
                    kingman_lower.append(kingman_fpmi[j][i])
                    beta_lower.append(beta_fpmi[j][i])
                    bsc_lower.append(bsc_fpmi[i][j])
                if i + j > n - 2:
                    kingman_upper.append(kingman_fpmi[i][j])
                    beta_upper.append(beta_fpmi[i][j])
                    bsc_upper.append(bsc_fpmi[i][j])

    # Kingman coalescent
    ax.errorbar([x0 - epsilon], [np.mean(kingman_fpmi.diagonal()[i_diag:])], yerr=[np.std(kingman_fpmi.diagonal()[i_diag:])], c='b', fmt=mk, ms=ms, elinewidth=1, label=labels[0])
    ax.errorbar([x0 + x_feature - epsilon], [np.mean(kingman_off)], yerr=[np.std(kingman_off)], c='b', fmt=mk, ms=ms, elinewidth=1)
    ax.errorbar([x0 + 2 * x_feature - epsilon], [np.mean(kingman_lower)], yerr=[np.std(kingman_lower)], c='b', fmt=mk, ms=ms, elinewidth=1)
    ax.errorbar([x0 + 3 * x_feature - epsilon], [np.mean(kingman_upper)], yerr=[np.std(kingman_upper)], c='b', fmt=mk, ms=ms, elinewidth=1)

    # Beta coalescent
    ax.errorbar([x0], [np.mean(beta_fpmi.diagonal()[i_diag:])], yerr=[np.std(beta_fpmi.diagonal()[i_diag:])], c='g', fmt=mk, ms=ms, elinewidth=1, label=labels[1])
    ax.errorbar([x0 + x_feature], [np.mean(beta_off)], yerr=[np.std(beta_off)], c='g', fmt=mk, ms=ms, elinewidth=1)
    ax.errorbar([x0 + 2 * x_feature], [np.mean(beta_lower)], yerr=[np.std(beta_lower)], c='g', fmt=mk, ms=ms, elinewidth=1)
    ax.errorbar([x0 + 3 * x_feature], [np.mean(beta_upper)], yerr=[np.std(beta_upper)], c='g', fmt=mk, ms=ms, elinewidth=1)

    # Bolthausen-Sznitman coalescent
    ax.errorbar([x0 + epsilon], [np.mean(bsc_fpmi.diagonal()[1:])], yerr=[np.std(bsc_fpmi.diagonal()[i_diag:])], c='r', fmt=mk, ms=ms, elinewidth=1, label=labels[2])
    ax.errorbar([x0 + x_feature + epsilon], [np.mean(bsc_off)], yerr=[np.std(bsc_off)], c='r', fmt=mk, ms=ms, elinewidth=1)
    ax.errorbar([x0 + 2 * x_feature + epsilon], [np.mean(bsc_lower)], yerr=[np.std(bsc_lower)], c='r', fmt=mk, ms=ms, elinewidth=1)
    ax.errorbar([x0 + 3 * x_feature + epsilon], [np.mean(bsc_upper)], yerr=[np.std(bsc_upper)], c='r', fmt=mk, ms=ms, elinewidth=1)

    ax.legend(loc='upper right', fontsize=9)

if __name__ == '__main__':
    plot_mixing_time(verbose=True)
    plot_sfs(w=15, n=20, data_scale=500, verbose=True)
    plot_eigenvector_panels(verbose=True)
    plot_2sfs(n=20, verbose=True)
    plot_2sfs_summary_stats(n=20, verbose=True)
    plot_offspring_distribution_si(verbose=True)
