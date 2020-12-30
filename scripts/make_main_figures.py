import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import pickle
import scipy.stats as stats
import theory
import coalescentmoments as moments
import branching_processes as bp
import forward_in_time_stats as forward
import data_table as dt
import analysis_tools as tools
import frequency_distribution as ft
from fpmi import fpmi
from analyze_distributions import overlap_fractions
from analyze_distributions import theory_cdf_numeric
from analyze_distributions import theory_cdf_corrections_numeric


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


def plot_sfs(w=15, n=40, data_scale=100):
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
    #ax.set_xlim([0, 1])
    #ax.set_ylim([xi_min / 5, 2 * xi_max])
    ax.set_ylim([y_min, 2 * xi_max])
    #ax.plot(i / n, xi_kingman, c='b', lw=1, ls='-')
    ax.plot(i, xi_kingman, c='b', lw=1, ls='-', label='Kingman')

    '''
    if 'fullypushed' in data['wave type'].values:
        print('Plotting semipushed SFS...')
        fp_sfs = data.loc[data['wave type'] == 'fullypushed', 'SFS'].values[0]
        x_data = np.arange(1, len(fp_sfs) + 1)
        ax.scatter(x_data, fp_sfs, c='b', s=20)
    '''
    ax.scatter(x_fp, fp_sfs, c='b', s=20)

    ax.legend(loc='lower right', fontsize=9)

    ax = fig.add_subplot(132)
    ax.set_title('semi-pushed')
    ax.set_xlabel(r'derived allele count, $i$')
    ax.set_yscale('log')
    ax.set_xlim([0.5, i[-1] + 0.5])
    ax.set_xticks(xticks)
    #ax.set_xlim([0, 1])
    #ax.set_ylim([xi_min / 5, 2 * xi_max])
    ax.set_ylim([y_min, 2 * xi_max])
    #ax.plot(i / n, xi_beta, c='g', lw=1, ls='-')
    ax.plot(i, xi_beta, c='g', lw=1, ls='-', label=r'Beta, $\alpha=1.5$')

    '''
    if 'semipushed' in data['wave type'].values:
        print('Plotting semipushed SFS...')
        sp_sfs = data.loc[data['wave type'] == 'semipushed', 'SFS'].values[0]
        x_data = np.arange(1, len(sp_sfs) + 1)
        ax.scatter(x_data, sp_sfs, c='g', s=20)
    '''
    ax.scatter(x_sp, sp_sfs, c='g', s=20)

    ax.legend(loc='lower right', fontsize=9)

    ax = fig.add_subplot(133)
    ax.set_title('pulled')
    ax.set_xlabel(r'derived allele count, $i$')
    ax.set_yscale('log')
    ax.set_xlim([0.5, i[-1] + 0.5])
    ax.set_xticks(xticks)
    #ax.set_xlim([0, 1])
    #ax.set_ylim([xi_min / 5, 2 * xi_max])
    ax.set_ylim([y_min, 2 * xi_max])
    #ax.plot(i / n, xi_bs, c='r', lw=1, ls='-')
    ax.plot(i, xi_bs, c='r', lw=1, ls='-', label='Bolthausen-Sznitman')

    '''
    if 'pulled' in data['wave type'].values:
        print('Plotting pulled SFS...')
        p_sfs = data.loc[data['wave type'] == 'pulled', 'SFS'].values[0]
        x_data = np.arange(1, len(p_sfs) + 1)
        ax.scatter(x_data, p_sfs, c='r', s=20)
    '''
    ax.scatter(x_p, p_sfs, c='r', s=20)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'sfs.pdf')


def plot_2sfs(data, w=15, n=20, save=True):
    ticks = np.array([0.5, 4.5, 9.5, 14.5, 18.5])
    tick_labels = [1, 5, 10, 15, 19]

    fig = plt.figure(figsize=(double_col_width, 2 * double_col_width / 3))

    grid = AxesGrid(fig, 211, nrows_ncols=(1, 3), axes_pad=0.075, cbar_mode='edge', cbar_location='right', cbar_pad=0.1)
    for ax in grid:
        #ax.set_xlabel(r'derived allele count, $i$')
        ax.set_ylabel(r'derived allele count, $j$', fontsize=9)
        ax.set_xticks(ticks - 0.5)
        ax.set_xticklabels([])
        ax.set_yticks(ticks - 0.5)
        ax.tick_params(length=2.5)
        #ax.set_axis_off()
    grid[0].set_title(r'Kingman')
    grid[0].set_yticklabels(tick_labels)
    grid[1].set_title(r'Beta, $\alpha=1.5$')
    grid[2].set_title(r'Bolthausen-Sznitman')

    allele_count = np.arange(1, n)
    #allele_count = np.arange(1, n + 1)
    bs_sfs, bs_sfs2 = moments.sfs_moments(n, alpha=1.0, m2=True)
    #bs_sfs, bs_sfs2 = moments.sfs_moments(n + 1, alpha=1.0, m2=True)
    bs_var = np.zeros(bs_sfs2.shape)
    bs_fpmi = fpmi(bs_sfs, bs_sfs2)
    for i in range(bs_var.shape[0]):
        for j in range(i + 1):
            bs_var[i][j] = bs_sfs2[i][j] - bs_sfs[i] * bs_sfs[j]
            bs_var[j][i] = bs_var[i][j]
    print(bs_fpmi.shape, allele_count.shape)

    beta_sfs, beta_sfs2 = moments.sfs_moments(n, alpha=1.5, m2=True)
    #beta_sfs, beta_sfs2 = moments.sfs_moments(n + 1, alpha=1.5, m2=True)
    beta_var = np.zeros(beta_sfs2.shape)
    beta_fpmi = fpmi(beta_sfs, beta_sfs2)
    for i in range(beta_var.shape[0]):
        for j in range(i + 1):
            beta_var[i][j] = beta_sfs2[i][j] - beta_sfs[i] * beta_sfs[j]
            beta_var[j][i] = beta_var[i][j]

    kingman_sfs, kingman_sfs2 = moments.sfs_moments(n, alpha=2.0, m2=True)
    #kingman_sfs, kingman_sfs2 = moments.sfs_moments(n + 1, alpha=2.0, m2=True)
    kingman_var = np.zeros(kingman_sfs2.shape)
    kingman_fpmi = fpmi(kingman_sfs, kingman_sfs2)
    for i in range(kingman_var.shape[0]):
        for j in range(i + 1):
            kingman_var[i][j] = kingman_sfs2[i][j] - kingman_sfs[i] * kingman_sfs[j]
            kingman_var[j][i] = kingman_var[i][j]

    #max_sfs2 = max(np.max(abs(bs_fpmi)), np.max(abs(beta_fpmi)), np.max(abs(kingman_fpmi)))
    max_sfs2 = 3.0
    #cmap='Greys'
    cmap='PuOr_r'
    # TODO: pcolormesh displays axis wrongly; offdiagonal is shifted upwards.
    #ax1.pcolormesh(allele_count, allele_count, bs_sfs2, cmap=cmap, vmin=min_sfs2, vmax=max_sfs2)
    #ax1.pcolormesh(allele_count, allele_count, bs_var, cmap=cmap, vmin=min_sfs2, vmax=max_sfs2)
    #ax1.pcolormesh(allele_count, allele_count, kingman_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2)
    #grid[0].pcolormesh(allele_count, allele_count, kingman_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2)
    grid[0].imshow(kingman_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2, origin='lower')
    #grid[0].imshow(kingman_fpmi, origin='lower', cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2)
    #ax2.pcolormesh(allele_count, allele_count, beta_sfs2, cmap=cmap, vmin=min_sfs2, vmax=max_sfs2)
    #ax2.pcolormesh(allele_count, allele_count, beta_var, cmap=cmap, vmin=min_sfs2, vmax=max_sfs2)
    #ax2.pcolormesh(allele_count, allele_count, beta_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2)
    #grid[1].pcolormesh(allele_count, allele_count, beta_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2)
    grid[1].imshow(beta_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2, origin='lower')
    #im = ax3.pcolormesh(allele_count, allele_count, kingman_sfs2, cmap=cmap, vmin=min_sfs2, vmax=max_sfs2)
    #im = ax3.pcolormesh(allele_count, allele_count, kingman_var, cmap=cmap, vmin=min_sfs2, vmax=max_sfs2)
    #im = ax3.pcolormesh(allele_count, allele_count, bs_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2)
    #im = grid[2].pcolormesh(allele_count, allele_count, bs_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2)
    im = grid[2].imshow(bs_fpmi, cmap=cmap, vmin=-max_sfs2, vmax=max_sfs2, origin='lower')

    cbar = grid.cbar_axes[0].colorbar(im)
    #clb = fig.colorbar(im, ax=ax3)
    cbar.ax.set_ylabel(r'$\langle \xi_i \xi_j \rangle - \langle \xi_i \rangle \langle \xi_j \rangle$')

    #-------------------------------------------------------------------------#
    # Expansion data; second row
    #-------------------------------------------------------------------------#

    grid = AxesGrid(fig, 212, nrows_ncols=(1, 3), axes_pad=0.075, cbar_mode='edge', cbar_location='right', cbar_pad=0.1)

    for ax in grid:
        ax.set_xlabel(r'derived allele count, $i$', fontsize=9)
        ax.set_ylabel(r'derived allele count, $j$', fontsize=9)
        ax.set_xticks(ticks - 0.5)
        ax.set_xticklabels([1, 5, 10, 15, 19])
        ax.set_yticks(ticks - 0.5)
        ax.tick_params(length=2.5)
        #ax.set_axis_off()
    grid[0].set_title(r'fully pushed')
    grid[0].set_yticklabels(tick_labels)
    grid[1].set_title(r'semi-pushed')
    grid[2].set_title(r'pulled')

    #max_fpmi = max(np.max(abs(p_fpmi)), np.max(abs(sp_fpmi)), np.max(abs(fp_fpmi)))
    #allele_count = np.arange(1, n)
    allele_count = np.arange(1, n + 1)
    max_fpmi = 3.0

    for i, wave_type in enumerate(['fullypushed', 'semipushed', 'pulled']):
        sfs_dt = dt.load_table(f'../results/trees_tmrca2000_{wave_type}_n{n}_s10_w{w}_sfs_avg.dat')
        #sfs_m1 = sfs_dt.table.loc['SFS'].values[0]
        sfs_m1 = sfs_dt.table.loc[0, 'SFS']
        #sfs_m2 = fp_sfs_dt.table.loc['2-SFS'].values[0]
        sfs_m2 = sfs_dt.table.loc[0, '2-SFS']
        data_fpmi = fpmi(sfs_m1, sfs_m2)
        im = grid[i].imshow(data_fpmi[:-1, :-1], cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi, origin='lower')
        #im = grid[i].pcolormesh(allele_count, allele_count, data_fpmi, cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)
        #im = grid[i].pcolormesh(allele_count, allele_count, data_fpmi[:-1, :-1], cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)

    '''
    if 'fullypushed' in data['wave type'].values:
        fp_sfs = data.loc[data['wave type'] == 'fullypushed', 'SFS'].values[0]
        fp_2sfs = data.loc[data['wave type'] == 'fullypushed', '2-SFS'].values[0]
        fp_fpmi = fpmi(fp_sfs, fp_2sfs)
        #max_fpmi = np.max(abs(fp_fpmi))

        #ax1.pcolormesh(allele_count, allele_count, fp_fpmi, cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)
        im = grid[0].pcolormesh(allele_count, allele_count, fp_fpmi, cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)

    if 'semipushed' in data['wave type'].values:
        sp_sfs = data.loc[data['wave type'] == 'semipushed', 'SFS'].values[0]
        sp_2sfs = data.loc[data['wave type'] == 'semipushed', '2-SFS'].values[0]
        sp_fpmi = fpmi(sp_sfs, sp_2sfs)
        #max_fpmi = np.max(abs(sp_fpmi))

        #ax2.pcolormesh(allele_count, allele_count, sp_fpmi, cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)
        im = grid[1].pcolormesh(allele_count, allele_count, sp_fpmi, cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)

    if 'pulled' in data['wave type'].values:
        p_sfs = data.loc[data['wave type'] == 'pulled', 'SFS'].values[0]
        p_2sfs = data.loc[data['wave type'] == 'pulled', '2-SFS'].values[0]
        p_fpmi = fpmi(p_sfs, p_2sfs)
        #max_fpmi = np.max(abs(p_fpmi))

        #im = ax3.pcolormesh(allele_count, allele_count, p_fpmi, cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)
        im = grid[2].pcolormesh(allele_count, allele_count, p_fpmi, cmap=cmap, vmin=-max_fpmi, vmax=max_fpmi)
    '''

    cbar = grid.cbar_axes[0].colorbar(im)
    #clb = fig.colorbar(im, ax=ax3)
    cbar.ax.set_ylabel(r'$\langle \xi_i \xi_j \rangle - \langle \xi_i \rangle \langle \xi_j \rangle$')

    #plt.tight_layout()
    plt.savefig(main_figures_dir + '2sfs.pdf')


def plot_total_branch_length(data, n_max=10000, normalize=True):
    print(data)
    #scale_tree_length(data)
    #n = np.array(list(range(2, 100)) + list(np.geomspace(100, n_max, 100)))
    n = np.array(list(range(2, 100)))
    n_bs = np.array(list(range(2, 20)) + [100])
    ln_pulled = get_tree_length(data['pulled'], normalized=normalize)
    ln_bs = np.array([theory.total_branch_length(j, 1, normalized=normalize) for j in n_bs])
    ln_semipushed = get_tree_length(data['semipushed'], normalized=normalize)
    ln_beta = theory.total_branch_length(n, 1.5, normalized=normalize)
    ln_fullypushed = get_tree_length(data['fullypushed'], normalized=normalize)
    ln_kingman = np.array([theory.total_branch_length(j, 2.0, normalized=normalize) for j in n])
    #var_ln_bs = np.array([theory.var_ln_BS(k) for k in n])
    #var_ln_kingman = np.array([theory.var_ln_kingman(k) for k in n])
    #ln_max = max(np.max(ln_bs), np.max(ln_beta), np.max(ln_kingman))
    ln_max = max(np.max(ln_pulled[:, 1]), np.max(ln_semipushed[:, 1]), np.max(ln_fullypushed[:, 1]))


    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))
    ax = fig.add_subplot(131)
    #ax.set_title('pulled')
    ax.set_xlabel(r'sample size, $n$', fontsize=12)
    ax.set_ylabel(r'segregating sites, $L_n$', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([2, 1.5 * np.max(n)])
    ax.set_ylim([1, 1.5 * ln_max])
    ax.plot(n_bs, ln_bs, c='r', lw=1, ls='-')
    #ax.fill_between(n, ln_bs - 2 * np.sqrt(var_ln_bs), ln_bs + 2 * np.sqrt(var_ln_bs), color="lightsalmon", alpha=0.5)
    #ax.scatter(ln_pulled[:, 0], ln_pulled[:, 1], c='r', s=20)
    ax.errorbar(ln_pulled[:, 0], y=ln_pulled[:, 1], yerr=ln_pulled[:, 2], c='r', fmt='o', ms=3, elinewidth=1)

    ax = fig.add_subplot(132)
    #ax.set_title(r'semi-pushed')
    ax.set_xlabel(r'sample size, $n$', fontsize=12)
    #ax.set_ylabel(r'segregating sites, $L_n$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([2, 1.5 * np.max(n)])
    ax.set_ylim([1, 1.5 * ln_max])
    ax.plot(n, ln_beta, c='g', lw=1, ls='-')
    #ax.fill_between(n, ln_beta - 2 * np.sqrt(ln_beta), ln_beta + 2 * np.sqrt(ln_beta), color="lightgreen", alpha=0.5)
    #ax.scatter(ln_semipushed[:, 0], ln_semipushed[:, 1], c='g', s=20)
    ax.errorbar(ln_semipushed[:, 0], y=ln_semipushed[:, 1], yerr=ln_semipushed[:, 2], c='g', fmt='o', ms=3, elinewidth=1)

    ax = fig.add_subplot(133)
    #ax.set_title('fully pushed')
    ax.set_xlabel(r'sample size, $n$', fontsize=12)
    #ax.set_ylabel(r'segregating sites, $L_n$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([2, 1.5 * np.max(n)])
    ax.set_ylim([1, 1.5 * ln_max])
    ax.plot(n, ln_kingman, c='b', lw=1, ls='-')
    #ax.fill_between(n, ln_kingman - 2 * np.sqrt(var_ln_kingman), ln_kingman + 2 * np.sqrt(var_ln_kingman), color="lightskyblue", alpha=0.5)
    #ax.scatter(ln_fullypushed[:, 0], ln_fullypushed[:, 1], c='b', s=20)
    ax.errorbar(ln_fullypushed[:, 0], y=ln_fullypushed[:, 1], yerr=ln_fullypushed[:, 2], c='b', fmt='o', ms=3, elinewidth=1)

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'total_branch_length.pdf')

def scale_tree_length(data_table):
    data_table['L_n'] = None
    for i, row in data_table.iterrows():
        ln_scaling_data = row['L_n scaling']
        tk_data = row['T_k mean']
        t2 = tk_data['T2']
        ln_scaling_data[:, 1] /= t2
        data_table.at[i, 'L_n'] = ln_scaling_data
    return data_table

def get_tree_length(data, wave_type=None, normalized=False):
    if wave_type is None:
        # Data dictionary loaded directly from pickle
        n_array = np.array(list(data.keys()))
        ln_ensemble = np.array(list(data.values()))
        ln_array = np.array([n_array, ln_ensemble.mean(axis=(1, 2)), ln_ensemble.std(axis=(1, 2))]).T
        if normalized == True:
            t2 = ln_array[0, 1]
            print(t2)
            ln_array[:, 1] = ln_array[:, 1] / t2
            ln_array[:, 2] = ln_array[:, 2] / t2
    else:
        ln_array = data_table.loc[data_table['wave type'] == wave_type, 'L_n'].values[0]
        if normalized == True:
            tk_mean = data_table.loc[data_table['wave type'] == wave_type, 'T_k mean'].values[0]

            # Normalize tree lengths by T2
            ln_array[:, 1] = ln_array[:, 1] / tk_mean['T2']
    return ln_array


def plot_merger_times(data, export=False):
    k_list = [3, 4, 5]
    tk_kingman = [theory.tk_ratio_kingman(k) for k in k_list]
    tk_bs = [theory.tk_ratio_bs(k) for k in k_list]

    p_tk, p_var = get_tk_ratios(data, 'pulled')
    sp_tk, sp_var = get_tk_ratios(data, 'semipushed')
    fp_tk, fp_var = get_tk_ratios(data, 'fullypushed')

    if export == True:
        df_tk = pd.DataFrame(columns=["Wave type", "T3/T2", "T4/T2", "T5/T2"])
        p_dict = {"Wave type":"pulled", "T3/T2":p_tk[0], "T4/T2":p_tk[1], "T5/T2":p_tk[2]}
        df_tk = df_tk.append(p_dict, ignore_index=True)
        sp_dict = {"Wave type":"semipushed", "T3/T2":sp_tk[0], "T4/T2":sp_tk[1], "T5/T2":sp_tk[2]}
        df_tk = df_tk.append(sp_dict, ignore_index=True)
        fp_dict = {"Wave type":"fullypushed", "T3/T2":fp_tk[0], "T4/T2":fp_tk[1], "T5/T2":fp_tk[2]}
        df_tk = df_tk.append(fp_dict, ignore_index=True)
        df_tk.to_csv("../results/merger_times.tsv")

    fig = plt.figure(figsize=(single_col_width, single_col_width))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'number of mergers, $k$')
    ax.set_ylabel(r'merger time ratio, $T_k/T_2$')
    ax.set_xticks([3, 4, 5])
    ax.set_yticks([1, 1.25, 1.5, 1.75])
    ax.set_ylim([1, 1.75])
    ax.plot(k_list, tk_bs, lw=1, c='r')
    ax.plot(k_list, tk_kingman, lw=1, c='b')
    ax.scatter(k_list, p_tk, s=10, facecolor='none', edgecolor='r')
    ax.scatter(k_list, sp_tk, s=10, facecolor='none', edgecolor='g')
    ax.scatter(k_list, fp_tk, s=10, facecolor='none', edgecolor='b')

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'merger_times.pdf')


def get_tk_ratios(data, wave_type):
    tk_dict = data.loc[data['wave type'] == wave_type, 'T_k mean'].values[0]
    tk_var_dict = data.loc[data['wave type'] == wave_type, 'T_k variance'].values[0]
    t2 = tk_dict['T2']
    t3 = tk_dict['T3']
    t4 = tk_dict['T4']
    t5 = tk_dict['T5']
    t2_var = tk_var_dict['T2']
    t3_var = tk_var_dict['T3']
    t4_var = tk_var_dict['T4']
    t5_var = tk_var_dict['T5']

    tk_means = [t3 / t2, t4 / t2, t5 / t2]
    tk_vars = np.array([(t3**2 / t2**2) * (t3_var / t3**2 + t2_var / t2**2),\
            (t4**2 / t2**2) * (t4_var / t4**2 + t2_var / t2**2),\
            (t5**2 / t2**2) * (t5_var / t5**2 + t2_var / t2**2)])
    return tk_means, tk_vars


def plot_beta_comparisons(n=50, alpha=1.5):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    n_max = 10000
    n_array = np.geomspace(2, n_max, 20)
    ln_beta = theory.total_branch_length(n_array, 1.5)
    ln_max = np.max(ln_beta)

    ax = fig.add_subplot(131)
    ax.set_xlabel(r'sample size, $n$')
    ax.set_ylabel(r'total branch length, $L_n$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([2, 1.5 * n_max])
    ax.set_ylim([1, 1.5 * ln_max])
    ax.plot(n_array, ln_beta, c='g', lw=1, marker='o', markeredgecolor='None', markerfacecolor='g')


    t = np.geomspace(1E-3, 1E1, 20)

    ax = fig.add_subplot(132)
    ax.set_xlabel(r'external branch length, $T_n^{ext}$')
    ax.set_ylabel(r'probability density, $P(T_n^{ext})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(t, theory.external_branch_distribution(t, 1.5), c='g', lw=1, marker='o', markeredgecolor='None', markerfacecolor='g')


    i = np.arange(1, n)
    xi_beta = moments.sfs_moments(n, alpha=1.5, m2=False)
    xi_min = np.min(xi_beta)
    xi_max = np.max(xi_beta)

    ax = fig.add_subplot(133)
    ax.set_xlabel(r'derived allele count, $i$')
    ax.set_ylabel(r'relative frequency, $\langle \xi_i \rangle$')
    ax.set_yscale('log')
    ax.set_xlim([0, i[-1] + 1])
    ax.set_ylim([xi_min / 5, 2 * xi_max])
    ax.plot(i, xi_beta, c='g', lw=1, marker='o', markeredgecolor='None', markerfacecolor='g')

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'beta_comparison.pdf')


def plot_offspring_distribution_si(y_min=1E-3, y_max=2):
    label_fontsize = 12

    offspring_data = dt.load_table(f'../results/offspring_n9600_w25_atfront_avg.dat')

    markers = ['o', '^', 's']
    y = np.logspace(-3,3)
    y_range = [min(y), max(y)]

    fig = plt.figure(figsize=(double_col_width, double_col_width / 2))

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


    # Same panel figs
    fig = plt.figure(figsize=(double_col_width, double_col_width / 2))

    #-------------------------------------------------------------------------#
    ### Complementary cumulative
    #-------------------------------------------------------------------------#
    ax = fig.add_subplot(121)
    ax.set_xlabel('clone sizes, $s$', fontsize=label_fontsize)
    ax.set_ylabel('complementary cumulative, $1 - F(s)$', fontsize=label_fontsize)
    ax.set_ylim([y_min, y_max])

    alpha = 0.9
    CC_st = bp.exact_complementary_cumulative(y, alpha='short')
    t = 2

    pulled_table = offspring_data.table.loc[0, :]
    clone_sizes = pulled_table['clone_sizes']
    x_p, compl_cumulative_p = forward.convert_counts_to_distribution(clone_sizes[t])

    fullypushed_table = offspring_data.table.loc[1, :]
    clone_sizes = fullypushed_table['clone_sizes']
    x_fp, compl_cumulative_fp = forward.convert_counts_to_distribution(clone_sizes[t])

    ax.loglog(x_p, compl_cumulative_p, c='r', lw=1, label=f'pulled')
    ax.loglog(x_fp, compl_cumulative_fp, c='b', lw=1, label=f'fully pushed')
    ax.loglog(y, CC_st, '-', lw=1, color='gray', alpha=0.5, label='short tail')

    ax.set_xlim([min(x), 1E4])
    ax.legend(fontsize=10)

    #-------------------------------------------------------------------------#
    ### Cumulative
    #-------------------------------------------------------------------------#
    ax = fig.add_subplot(122)
    ax.set_xlabel('clone sizes, $s$', fontsize=label_fontsize)
    ax.set_ylabel('cumulative, $F(s)$', fontsize=label_fontsize)
    ax.set_ylim([y_min, y_max])

    ax.loglog(x_p, 1 - compl_cumulative_p, c='r', lw=1, label=f'pulled')
    ax.loglog(x_fp, 1 - compl_cumulative_fp, c='b', lw=1, label=f'fully pushed')
    ax.loglog(y, 1 - CC_st, '-', lw=1, color='gray', alpha=0.5, label='short tail')

    ax.set_xlim([min(x_fp), 1E4])
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'offspring_distribution_comparison_si.pdf')

def plot_external_branch_distribution():
    t = np.linspace(0, 5, 100)
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    ax = fig.add_subplot(131)
    ax.set_xlabel(r'external branch length, $T_n^{ext}$')
    ax.set_ylabel(r'probability density, $P(T_n^{ext})$')
    ax.plot(t, theory.external_branch_distribution(t, 1.0), c='r', lw=1)
    ax.plot(t, theory.external_branch_distribution(t, 1.5), c='g', lw=1)
    ax.plot(t, theory.external_branch_distribution(t, 2.0), c='b', lw=1)

    ax = fig.add_subplot(132)
    ax.set_xlabel(r'external branch length, $T_n^{ext}$')
    ax.set_ylabel(r'probability density, $P(T_n^{ext})$')
    ax.set_yscale('log')
    ax.plot(t, theory.external_branch_distribution(t, 1.0), c='r', lw=1)
    ax.plot(t, theory.external_branch_distribution(t, 1.5), c='g', lw=1)
    ax.plot(t, theory.external_branch_distribution(t, 2.0), c='b', lw=1)

    ax = fig.add_subplot(133)
    ax.set_xlabel(r'external branch length, $T_n^{ext}$')
    ax.set_ylabel(r'probability density, $P(T_n^{ext})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(t, theory.external_branch_distribution(t, 1.0), c='r', lw=1)
    ax.plot(t, theory.external_branch_distribution(t, 1.5), c='g', lw=1)
    ax.plot(t, theory.external_branch_distribution(t, 2.0), c='b', lw=1)

    plt.tight_layout()
    plt.savefig(main_figures_dir + 'external_branch_distribution.pdf')

def plot_mixing_time(t_list=['1', '50', '100'], save=True):

    # Initial conditions
    #fig = plt.figure(figsize=(double_col_width, double_col_width / 3))
    fig = plt.figure(figsize=(single_col_width, 1.6 * single_col_width))
    #ax1 = fig.add_subplot(131)
    ax1 = fig.add_subplot(211)
    #ax1.set_xlabel(r'position, $\zeta$')
    ax1.set_xlim(-25, 50)
    ax1.set_xticks([-25, 0, 25, 50])
    ax1.set_ylim(0, 0.4)
    ax1.set_ylabel(r'ancestor distribution', fontsize=14)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])

    backward_results = '../results/backward_averaged.dat'
    ancestors_data = dt.load_table(backward_results)
    backward_table = ancestors_data.table
    print(backward_table)
    profile = backward_table['profiles'].values[0]['f']
    profile_density = profile / np.max(profile)

    ancestors = backward_table['ancestor distributions'].values[0]
    distr_z1 = tools.normalize_distribution(ancestors['z_lower'][0])
    distr_z2 = tools.normalize_distribution(ancestors['z_upper'][0])
    zeta = np.arange(len(distr_z1)) - len(distr_z1) // 2
    ax1.plot(zeta, distr_z1, ls='-', lw=1, drawstyle='steps-post', label=f'')
    ax1.plot(zeta, distr_z2, ls='-', lw=1, drawstyle='steps-post', label=f'')
    #ax1.legend(loc='best')

    ax = ax1.twinx()
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([])
    ax.plot(zeta, profile_density, ls='-', lw=2, c='gray')


    # Ancestor distribution
    #ax3 = fig.add_subplot(133)
    ax3 = fig.add_subplot(212)
    ax3.set_xlabel(r'position, $\zeta$', fontsize=14)
    ax3.set_xlim(-25, 50)
    ax3.set_xticks([-25, 0, 25, 50])
    ax3.set_ylabel(r'ancestor distribution', fontsize=14)
    ax3.set_ylim(0.0, 0.15)
    ax3.set_yticks([0, 0.05, 0.1, 0.15])

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

    ax = ax3.twinx()
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.1)
    ax.plot(zeta, profile_density, ls='-', lw=2, c='gray')
    #ax3.legend(loc='best', fontsize=8)

    plt.tight_layout()

    if save == True:
        plt.savefig(main_figures_dir + f'mixing_time.pdf')

    # Make inset with sample locations

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

    if save == True:
        plt.savefig(f'{main_figures_dir}mixing_time_inset.pdf')


def plot_histogram(frequencies, ax, title, y_label=True, bins=30):
    ax.set_title(title)
    #ax.set_xlabel('allele 1 frequency, $f$', fontsize=12)
    ax.set_xlabel('allele frequency, $f$', fontsize=12)
    if y_label == True:
        ax.set_ylabel('probability density, $P(f)$', fontsize=12)
    ax.hist(frequencies, bins=bins, density=True, color='slategray')


def plot_stochastic_pdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    # Fully-pushed waves
    bins_fullypushed=20
    ax = fig.add_subplot(131)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 3.4])
    ax.set_yticks([0, 1, 2.0, 3.0])
    time_index = 200
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)
    print(fullypushed_freqs.get_t(time_index, units='ne'))
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


def plot_deterministic_pdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    f_theory = np.linspace(0, 1, 100)
    y_theory = np.ones(len(f_theory))

    # Fully-pushed waves
    print("Analyzing fully-pushed deterministic fronts...")
    bins_fullypushed=20
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax = fig.add_subplot(131)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 1.55])
    ax.set_yticks([0, 0.5, 1, 1.5])
    time_index = 250
    time_in_neff = pulled_freqs.get_t(time_index)
    time_string = '{:.2f}'.format(time_in_neff)
    print("tau = " + time_string)
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)
    plot_histogram(freqs, ax, 'deterministic fully pushed', y_label=True, bins=bins_fullypushed)
    ax.plot(f_theory, y_theory, lw=2, c='b', ls='--')

    #counts, bin_edges = np.histogram(freqs, bins=bins_pulled)
    #delta_x = bin_edges[1] - bin_edges[0]
    #f_theory = np.linspace(bin_edges[1] / 2, 1 - (1 - bin_edges[-2]) / 2, 100)
    #y_theory = 1 / (f_theory * (1 - f_theory))
    #y_theory /= np.sum(y_theory) * delta_x / 2
    #ax.plot(f_theory, y_theory, lw=2, c='r')

    # Semi-pushed waves
    print("Analyzing semi-pushed deterministic fronts...")
    bins_semipushed = 20
    ax = fig.add_subplot(132)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 1.55])
    ax.set_yticks([0, 0.5, 1, 1.5])
    time_index = 70
    time_in_neff = pulled_freqs.get_t(time_index)
    time_string = '{:.2f}'.format(time_in_neff)
    print("tau = " + time_string)
    freqs = semipushed_freqs.get_nonfixed_f(time_index, logit=False)
    plot_histogram(freqs, ax, 'deterministic semi-pushed', y_label=False, bins=bins_semipushed)
    ax.plot(f_theory, y_theory, lw=2, c='g', ls='--')

    # Pulled waves
    bins_pulled = 20
    print("Analyzing pulled deterministic fronts...")
    ax = fig.add_subplot(133)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim([0, 1.55])
    ax.set_yticks([0, 0.5, 1, 1.5])
    time_index=200

    time_in_neff = pulled_freqs.get_t(time_index)
    time_string = '{:.2f}'.format(time_in_neff)
    print("tau = " + time_string)
    freqs = pulled_freqs.get_nonfixed_f(time_index, logit=False)
    plot_histogram(freqs, ax, 'deterministic pulled', y_label=False, bins=bins_pulled)
    ax.plot(f_theory, y_theory, lw=2, c='r', ls='--')

    plt.tight_layout()
    plt.savefig(figure_panels_dir + 'deterministic_waves_pdf.pdf')


def plot_stochastic_cdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs, save=True):
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    # Fully-pushed waves
    bins_fullypushed=50
    ax = fig.add_subplot(131)
    #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    #ax.set_ylim([0, 1.4])
    #ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25])
    #time_index = 200
    time_index = 200
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)
    print(fullypushed_freqs.get_t(time_index, units='ne'))
    #plot_cdf2ax(ax, freqs, 'stochastic fully-pushed', 'b', ls='-', uniform_theory=True, pulled_theory=False, bins=bins_fullypushed)
    plot_cdf2ax(ax, freqs, '', 'b', ls='-', uniform_theory=True, pulled_theory=False, y_label=True, bins=bins_fullypushed)

    # Semi-pushed waves
    bins_semipushed = 50
    ax = fig.add_subplot(132)
    #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    #ax.set_ylim([0, 3.5])
    #ax.set_yticks([0, 1, 2, 3])
    #time_index = 700
    time_index = 700
    freqs = semipushed_freqs.get_nonfixed_f(time_index, logit=False)
    tau = semipushed_freqs.get_t(time_index, units='ne')
    #plot_cdf2ax(ax, freqs, 'stochastic semi-pushed', 'b', ls='-', uniform_theory=True, pulled_theory=True, bins=bins_semipushed)
    plot_cdf2ax(ax, freqs, '', 'b', ls='-', uniform_theory=True, pulled_theory=True, y_label=False, bins=bins_semipushed)

    # Pulled waves
    bins_pulled = 50
    ax = fig.add_subplot(133)
    #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    #ax.set_ylim([0, 4])
    #ax.set_yticks([0, 1, 2, 3, 4])
    #time_index=38
    time_index=100
    freqs = pulled_freqs.get_nonfixed_f(time_index, logit=False)
    #plot_cdf2ax(ax, freqs, 'stochastic pulled', 'r', uniform_theory=False, pulled_theory=True, bins=bins_pulled)
    plot_cdf2ax(ax, freqs, '', 'r', uniform_theory=False, pulled_theory=True, y_label=False, bins=bins_pulled)

    plt.tight_layout()
    if save == True:
        plt.savefig(figure_panels_dir + 'stochastic_waves_cdf.pdf')


def plot_cdf2ax(ax, f_survived, title, theory_color, ls='--', KS_test='uniform', uniform_theory=True, pulled_theory=True, y_label=True, bins=50, overlap=True, data_label=None):
    if overlap == True:
        f_overlap = overlap_fractions(f_survived)
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

    if pulled_theory == True:
        ax.plot(bin_centers, theory_cdf_numeric(bin_centers), ls='-', lw=1.0, c='r')
        #ax.plot(bin_centers, theory_cdf_corrections_numeric(bin_centers), ls='-', lw=1.0, c='r')

    # Set plot labels
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('major allele frequency, $f$', fontsize=12)
    ax.set_xticks([0.5, 0.75, 1.0])
    if y_label == True:
        ax.set_ylabel('cumulative probability', fontsize=12)
    #ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.5, 1.0])

    # Plot data
    #ax.plot(bin_centers, cdf, linestyle='-', lw=1.5, c='r', label=data_label)
    ax.plot(bin_centers, cdf, linestyle=':', c='k', lw=2.5, label=data_label)
    #ax.scatter(bin_centers, cdf, s=5, marker='o', edgecolor='none', facecolor='k', label=data_label)


def plot_deterministic_cdfs(pulled_freqs, semipushed_freqs, fullypushed_freqs):
    #fig = plt.figure(figsize=(double_col_width, double_col_width / 3))
    fig = plt.figure(figsize=(double_col_width, double_col_width / 3))

    # Fully-pushed waves
    bins_fullypushed=50
    ax = fig.add_subplot(131)
    #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    #ax.set_ylim([0, 1.4])
    #ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25])
    time_index = 250
    freqs = fullypushed_freqs.get_nonfixed_f(time_index, logit=False)
    print(fullypushed_freqs.get_t(time_index, units='ne'))
    #plot_cdf2ax(ax, freqs, 'deterministic fully-pushed', 'b', uniform_theory=True, pulled_theory=False, bins=bins_fullypushed)
    plot_cdf2ax(ax, freqs, '', 'b', uniform_theory=True, pulled_theory=False, y_label=True, bins=bins_fullypushed)

    # Semi-pushed waves
    bins_semipushed = 50
    ax = fig.add_subplot(132)
    #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    #ax.set_ylim([0, 3.5])
    #ax.set_yticks([0, 1, 2, 3])
    time_index = 70
    freqs = semipushed_freqs.get_nonfixed_f(time_index, logit=False)
    tau = semipushed_freqs.get_t(time_index, units='ne')
    #plot_cdf2ax(ax, freqs, 'deterministic semi-pushed', 'g', uniform_theory=True, pulled_theory=False, y_label=False, bins=bins_semipushed)
    plot_cdf2ax(ax, freqs, '', 'g', uniform_theory=True, pulled_theory=False, y_label=False, bins=bins_semipushed)

    # Pulled waves
    bins_pulled = 50
    ax = fig.add_subplot(133)
    #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    #ax.set_ylim([0, 4])
    #ax.set_yticks([0, 1, 2, 3, 4])
    time_index=200
    freqs = pulled_freqs.get_nonfixed_f(time_index, logit=False)
    #plot_cdf2ax(ax, freqs, 'deterministic pulled', 'r', uniform_theory=True, pulled_theory=False, y_label=False, bins=bins_pulled)
    plot_cdf2ax(ax, freqs, '', 'r', uniform_theory=True, pulled_theory=False, y_label=False, bins=bins_pulled)

    plt.tight_layout()
    plt.savefig(figure_panels_dir + 'deterministic_waves_cdf.pdf')


def plot_eigenvector_panels(data_dir='../data/'):
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



if __name__ == '__main__':
    plot_sfs(w=15, n=20, data_scale=500)

    sfs_dt = dt.load_table('../results/trees_tmrca2000_sfs_avg.dat')
    sfs_df = sfs_dt.table
    plot_2sfs(sfs_df, n=20)

    plot_offspring_distribution_si()
    plot_mixing_time()
    plot_eigenvector_panels()
