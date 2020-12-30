import numpy as np
import read_profile as rp
import glob
import pickle
import scipy.stats as stats
import analysis_tools as tools
from pathlib import Path
from os import path
import os


class SimulationResults:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = kwargs.get('n', 0)
        self.r0 = kwargs.get('r0', 0)
        self.m0 = kwargs.get('m0', 0)
        self.a = kwargs.get('a', 0)
        self.b = kwargs.get('b', 0)
        self.dir = kwargs.get('dir', None)
        self.delta_t = kwargs.get('delta_t', 0)
        self.results = {}
        self.run_mapping = []
        self.parameters = {'n':self.n, 'r0':self.r0, 'm0':self.m0, 'a':self.a, 'b':self.b, 'delta_t':self.delta_t}

    def __repr__(self):
        return f'Simulation parameters:\n \tdir:{self.dir}\n\tN={self.n}\n\tr0={self.r0}\n\tm0={self.m0}\n\tA={self.a}\n\tB={self.b}\n\tDelta t={self.delta_t}'

    def compute_clone_size_counts(self, data_dir=None):
        '''
        Computes count for unique alleles over independent simulations in data_dir
        Stores dictionary of clone sizes in results member
        '''
        if data_dir is None:
            if self.dir is not None:
                data_dir = self.dir
            else:
                print("Error: no dir provided to SimulationResults:compute_clone_size_counts().")
                return 0

        tau_list = self.get_tau_list(data_dir)
        clone_size = {}
        run_mapping = []
        for tau in tau_list:
            file_list = sorted(glob.glob(data_dir + 'profile_tau' + str(tau) + '_*.txt'))
            clone_size_counts = []
            run_mapping.append([])
            for fname in file_list:
                counts = self.read_clone_sizes(fname)
                clone_size_counts.append(counts)

                # Record run number
                var_dict = tools.get_variables(fname)
                run_mapping[-1].append(int(var_dict['run']))
            clone_size[tau] = np.array(clone_size_counts)
        self.results['clone_size'] = clone_size
        self.run_mapping = run_mapping

    def compute_clone_size_xwindow(self, dx, track_at_front=False, n_cutoff=10, data_dir=None):
        '''
        Computes count for unique alleles over independent simulations in data_dir
        Stores dictionary of clone sizes in results member
        '''
        if data_dir is None:
            if self.dir is not None:
                data_dir = self.dir
            else:
                print("Error: no dir provided to SimulationResults:compute_clone_size_counts().")
                return 0

        if 'profile' not in self.results:
            self.load_profiles()

        tracked_clones = self.get_clones_at_front(t='i', dx=dx, n_cutoff=n_cutoff)
        tau_list = self.get_tau_list(data_dir)
        clone_size = self.initialize_clone_sizes(tau_list)
        run_mapping = [[] for tau in tau_list]

        if track_at_front == True:
            counts_at_front = {}
            for tau in tau_list:
                clone_dict = self.get_clones_at_front(t=f'{tau}', dx=dx, n_cutoff=n_cutoff)
                tau_dict = {}
                for run in clone_dict:
                    #print(clone_dict[run])
                    #counts = np.array([np.sum(clone_dict[run] == clone) for clone in tracked_clones])
                    #tau_dict[run] = counts[counts > 0]
                    labels, counts = np.unique(clone_dict[run], return_counts=True)
                    tau_dict[run] = counts
                counts_at_front[tau] = tau_dict

        for run in tracked_clones:
            profile_fnames = self.get_profile_files(run)
            clones = tracked_clones[run]
            run_taus = []
            for i, tau in enumerate(tau_list):
                if f'{tau}' in profile_fnames:
                    if track_at_front == False:
                        counts = self.read_clone_sizes(profile_fnames[f'{tau}'], tracked_clones=clones)
                    elif run in counts_at_front[tau]:
                        counts = counts_at_front[tau][run]
                    else:
                        counts = []
                    clone_size[tau].append(counts)
                    run_mapping[i].append(run)

        self.results['clone_size'] = clone_size
        self.run_mapping = run_mapping

    def get_clones_at_front(self, t, dx, n_cutoff=10):
        clones = {}
        runs = self.run_list
        for run in runs:
            if t in self.results['profiles'][run]:
                front_profile = self.results['profiles'][run][t]
                profile_n = np.sum(front_profile > 0, axis=1)
                x_c = np.arange(len(profile_n))[profile_n > n_cutoff][-1]
                front_section = front_profile[x_c - dx - 1:x_c]
                section_labels = np.concatenate(front_section)
                clones[run] = section_labels[section_labels > 0]
        return clones

    def initialize_clone_sizes(self, tau_list):
        clone_sizes = {}
        for tau in tau_list:
            clone_sizes[tau] = []
        return clone_sizes

    def read_clone_sizes(self, profile_fname, tracked_clones=None):
        '''
        Returns sizes of clones from input profile

        Parameters:
        ___________
        profile_fname : path to wave profile
        '''

        profile_array = rp.read_profile(profile_fname)
        if tracked_clones is None:
            unique, counts = np.unique(profile_array[np.nonzero(profile_array)], return_counts=True)
        else:
            labels = profile_array.flatten()
            counts = np.array([np.sum(labels == clone) for clone in tracked_clones])
            counts = counts[counts > 0]
        return counts

    def compute_clone_size_expweighted(self, cutoff=1E-5, data_dir=None):
        '''
        Computes count for unique clones using u_i = \sum_x c_i(x) * e^{v * z / D} / (\sum_x c(x) * e^{v * z / D}).
        Stores dictionary of clone sizes in results member
        '''
        if data_dir is None:
            if self.dir is not None:
                data_dir = self.dir
            else:
                print("Error: no dir provided to SimulationResults:compute_clone_size_counts().")
                return 0

        if 'front_positions' not in self.results:
            self.load_front_positions()
            v = self.calculate_average_velocity()

        if 'profile' not in self.results:
            self.load_profiles()

        clone_size_dict = self.calculate_expweighted_clones(v, cutoff=cutoff)
        self.results['clone_size'] = clone_size_dict

    def calculate_average_velocity(self):
        v_avg = 0
        runs = 0
        for run, tx in self.results['front_positions'].items():
            fit_region = -min(100, len(tx) // 2)
            v, _, _, _, _ = stats.linregress(tx[fit_region:, 0], tx[fit_region:, 1])
            v_avg += v
            runs += 1
        return v_avg / runs

    def calculate_expweighted_clones(self, v, cutoff=1E-5):
        # Calculate cutoff distance z_c
        N = self.n
        D = self.m0 / 2
        z_c = D * np.log(cutoff / N) / v

        clone_size_dict = {}
        for run, front_dict in self.results['profiles'].items():
            time_dict = {}
            for i_t, front in front_dict.items():
                n = np.sum(front > 0, axis=1)
                z, z_index = self.construct_front_position(n, z_c)
                unique_clones = self.get_nonzero_clones(front[z_index])
                clone_sizes = []
                for clone in unique_clones:
                    n_i = np.sum(front[z_index] == clone, axis=1)
                    u_i = np.sum(n_i * np.exp(v * z / D)) / np.sum(n[z_index] * np.exp(v * z / D))
                    clone_sizes.append(u_i)
                time_dict[i_t] = clone_sizes
            clone_size_dict[run] = time_dict
        return clone_size_dict

    def construct_front_position(self, n, z_c):
        index = np.arange(len(n))
        x_mid = self.calculate_front_midpoint(n)
        x_max = index[n > 0][-1]
        z = index + 1 - x_mid
        z_index = index[z > z_c]
        z_index = z_index[z_index <= x_max]
        z = z[z_index]
        return z, z_index

    def calculate_front_midpoint(self, n):
        return np.sum(n / max(n))

    def get_nonzero_clones(self, front):
        unique_clones = np.unique(front)
        return unique_clones[unique_clones > 0]

    def fit_neff(self, n_fit=10):
        (t, h) = self.results['odd-even H'].T
        slope, intercept, r, p_value, stderr = stats.linregress(t[:n_fit], np.log(h[:n_fit]))
        self.results['odd-even Ne'] = -self.delta_t / slope

    def add_heterozygosity_timeseries(self, data_dir):
        tau_list = self.get_tau_list(data_dir)
        heterozygosity_data = []
        for tau in tau_list:
            even_allele_frequencies = self.aggregate_even_frequencies(data_dir, tau)
            h = self.heterozygosity(even_allele_frequencies)
            heterozygosity_data.append([tau, h])
        self.results['odd-even H'] = np.array(heterozygosity_data)

    def aggregate_even_frequencies(self, data_dir, tau):
        file_list = sorted(glob.glob(data_dir + 'profile_tau' + str(tau) + '_*.txt'))
        even_frequencies = []
        for fname in file_list:
            profile_array = rp.read_profile(fname)
            even_frequency = self.compute_even_frequency(profile_array)
            even_frequencies.append(even_frequency)
        return np.array(even_frequencies)

    def compute_even_frequency(self, profile_array):
        num_individuals = sum(profile_array[np.nonzero(profile_array)] > 0)
        f = sum(profile_array[np.nonzero(profile_array)] % 2 == 0) / num_individuals
        return np.array(f)

    def load_trees(self, trim_level=0):
        data_dir = self.dir
        tree_dict = {}
        run_list = self.read_run_list(file_ext='nwk')

        if self.is_tree_files() == False:
            for run in run_list:
                tree_dict.update({run:None})
            self.results['tree'] = tree_dict
        else:
            for run in run_list:
                fname = self.get_tree_file(run)
                if trim_level == 0:
                    tree = tools.read_tree(data_dir + fname)
                else:
                    tree_raw = tools.read_custom_tree(data_dir + fname)
                    tree_trimmed = tree_raw.trim_tree(trim_level)
                    tree_trimmed.save('trimmed_tree.nwk')
                    tree = tools.read_tree('trimmed_tree.nwk')
                tree_dict[run] = tree
            self.results['tree'] = tree_dict

    def read_run_list(self, file_ext=None):
        if file_ext is None:
            file_list = sorted(glob.glob(self.dir + 'position_' + self.parameter_str() + '_*.txt'))
        else:
            file_list = sorted(glob.glob(self.dir + '*' + self.parameter_str() + '_*.' + file_ext))
        run_list = []
        for fname in file_list:
            var_dict = tools.get_variables(fname)
            if 'run' in var_dict.keys():
                run_list.append(int(var_dict['run']))
        return list(np.unique(run_list))

    def is_tree_files(self):
        tree_files = glob.glob(self.dir + 'tree_' + self.parameter_str() + '_*.nwk')
        if len(tree_files) > 0:
            return True
        else:
            return False

    def get_tree_file(self, run, path=""):
        return path + "tree_" + self.parameter_str() + "_run" + str(run) + ".nwk"

    def parameter_str(self, delta_t_label="relabel"):
        return "N" + str(self.n) + "_r" + str(self.r0) + "_m" + \
                str(self.m0) + "_fback" + np.format_float_positional(self.b, trim='-') + \
                "_" + delta_t_label + str(self.delta_t)

    def table_format(self, table_columns, weighted_offspring=False):
        table_entries = []
        parameter_dict = self.map_parameters_to_columns(table_columns)
        #run_list = self.read_run_list()
        #for run in run_list:
        for run in self.run_list:
            table_row = parameter_dict.copy()
            table_row['run'] = run
            if 'clone_size' in self.results.keys():
                if weighted_offspring == False:
                    table_row['clone_size'] = self.get_clone_size_by_run(run)
                else:
                    table_row['clone_size'] = self.results['clone_size'][run]
            if 'profiles' in self.results.keys():
                table_row['profiles'] = self.results['profiles'][run]
            if 'front_positions' in self.results.keys() and run in self.results['front_positions']:
                table_row['front_positions'] = self.results['front_positions'][run]
            if 'tree' in self.results.keys():
                table_row['tree'] = self.results['tree'][run]
            table_entries.append(table_row)
        return table_entries

    def map_parameters_to_columns(self, column_list):
        par_dict = {}
        parameters = self.parameters
        for column in column_list:
            if column in parameters.keys():
                par_dict[column] = parameters[column]
        return par_dict

    def get_clone_size_by_run(self, run):
        full_counts = self.results['clone_size']
        run_counts = {}

        # Loop through values of tau in clone size counts while run is in run_mapping
        tau_list = self.get_tau_list(self.dir)
        tau_index = 0
        while (run in self.run_mapping[tau_index]):
            runs_at_tau = self.run_mapping[tau_index]
            run_index = runs_at_tau.index(run)
            tau = tau_list[tau_index]
            run_counts[tau] = full_counts[tau][run_index]
            tau_index += 1

            # Check if end of run_mapping
            if tau_index == len(self.run_mapping):
                break

        return run_counts

    def load_profiles(self):
        profile_dict = {}
        run_list = self.read_run_list()
        self.run_list = []
        for run in run_list:
            tau_dict = {}
            fname_dict = self.get_profile_files(run)
            for tau, fname in fname_dict.items():
                profile = rp.read_profile(fname)
                if len(profile) > 0:
                    tau_dict.update({tau:profile})
            if len(tau_dict) > 0:
                profile_dict.update({run:tau_dict})
                self.run_list.append(run)
        self.results['profiles'] = profile_dict


    def get_profile_files(self, run):
        fname_dict = {}

        # Treat initial and final profiles separately
        initial_profile = self.dir + f"profile_initial_N{self.n}_r{self.r0}_m{self.m0}_fback" + np.format_float_positional(self.b, trim='-') + f"_relabel{self.delta_t}_run{run}.txt"
        if path.exists(initial_profile):
            fname_dict.update({'i':initial_profile})
        final_profile = self.dir + f"profile_final_N{self.n}_r{self.r0}_m{self.m0}_fback" + np.format_float_positional(self.b, trim='-') + f"_relabel{self.delta_t}_run{run}.txt"
        if path.exists(final_profile):
            fname_dict.update({'f':final_profile})

        # Add rest of files
        fname_list = sorted(glob.glob(self.dir + f"profile_tau*_N{self.n}_r{self.r0}_m{self.m0}_fback" + np.format_float_positional(self.b, trim='-') + f"_relabel{self.delta_t}_run{run}.txt"))
        for fname in fname_list:
            var_dict = tools.get_variables(fname)
            if 'tau' in var_dict.keys():
                fname_dict.update({str(int(var_dict['tau'])):fname})
        return fname_dict

    def load_front_positions(self):
        front_dict = {}
        run_list = self.read_run_list()
        for run in run_list:
            position_file = self.dir + f"position_N{self.n}_r{self.r0}_m{self.m0}_fback" + np.format_float_positional(self.b, trim='-') + f"_relabel{self.delta_t}_run{run}.txt"
            if os.stat(position_file).st_size != 0:
                position_data = np.loadtxt(position_file, delimiter=',')
                front_dict[run] = position_data
        self.results['front_positions'] = front_dict

    def save(self, output_file):
        with open(output_file, 'wb') as f_out:
            pickle.dump(self, f_out)

    @staticmethod
    def get_tau_list(data_dir):
        file_list = sorted(glob.glob(data_dir + 'profile_tau*.txt'))
        tau_list = []
        for fpath in file_list:
            fname = fpath.split('/')[-1]
            variables_dict = tools.get_variables(fname)
            tau_list.append(int(variables_dict['tau']))
        return list(set(tau_list))

    @staticmethod
    def heterozygosity(f):
        h = 2 * f * (1 - f)
        return np.mean(h)


def test_class(kwargs):
    results = SimulationResults(**kwargs)
    results.compute_clone_size_counts()
    results.load_trees()
    print(results)
    print(results.results)
    columns = ['n', 'r0', 'm0', 'b', 'delta_t', 'run', 'clone_size', 'tree']
    print(results.table_format(columns))

    # Output results
    output_fpath = '../results/simulation_results_test.dat'
    results.save(output_fpath)


if __name__ == "__main__":
    test_dir = '../data/trees/n350/b10.0_relabel50/'
    data_kwargs = {'dir':test_dir, 'n':350, 'b':10.0, 'delta_t':50, 'r0':0.01, 'm0':0.4, 'a':0}
    test_class(data_kwargs)
