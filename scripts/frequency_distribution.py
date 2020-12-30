import numpy as np
import scipy.stats as stats
import os
from analyze_distributions import logit


class frequency_distribution:
    def __init__(self, frequencies_fpath=''):
        self.fpath = frequencies_fpath
        self.read_and_format_data()
        self._extract_simulation_parameters()


    def read_and_format_data(self, delimiter=','):
        if os.path.isfile(self.fpath):
            self.frequencies = np.loadtxt(self.fpath, delimiter=delimiter)
            self.num_timepoints = self.frequencies.shape[0]
            self.num_replicas = self.frequencies.shape[1]
            self.sampling_times_fpath = self._get_sampling_times_fpath()
            self.sampling_times = np.loadtxt(self.sampling_times_fpath, delimiter=delimiter)

        else:
            user_input = input(self.fpath + ' file not found. Input new file to read or c to cancel...\n')
            if user_input == 'c':
                return -1
            else:
                self.fpath = user_input
                read_and_format_data()


    def update_data(self, n=None, a=None, b=None):
        if n is not None:
            self.n = n
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

        self._update_fpath()
        self.read_and_format_data()


    def get_nonfixed_f(self, i, logit=False):
        nonfixed_index = (self.frequencies[i] != 0) * (self.frequencies[i] != 1)
        frequencies = self.frequencies[i][nonfixed_index]
        if logit == True:
            frequencies = logit(frequencies)
        return frequencies


    def get_h(self):
        h = 2 * (self.frequencies * (1 - self.frequencies)).mean(axis=1)
        return h


    def get_t(self, i, units='generations'):
        if units == 'generations':
            return self.sampling_times[i]
        elif units == 'ne':
            ne = self.get_ne()
            return self.sampling_times[i] / ne


    def get_ne(self, max_het = 0.1, min_surviving_replicas = 50):
        h = self.get_h()
        t = self.sampling_times
        p_nonfixed = np.array([len(self.get_nonfixed_f(i)) / self.num_replicas for i in range(self.num_timepoints)])
        fitting_index = (h < max_het) * (p_nonfixed > (min_surviving_replicas / self.num_replicas))

        y_fit = np.log(h[fitting_index])
        x_fit = t[fitting_index]
        slope, intercept, r_value, p_value, std_err = stats.linregress( x_fit, y_fit )
        return -1.0 / slope


    def _get_sampling_times_fpath(self):
        data_dir_path = '/'.join(self.fpath.split('/')[:-1])
        fname = self.fpath.split('/')[-1]
        new_fname_components = fname.split('_')
        new_fname_components[0] = 'sampling'
        new_fname_components[1] = 'times'
        return data_dir_path + '/' + '_'.join(new_fname_components)


    def _extract_simulation_parameters(self):
        name_root = self.fpath.split('/')[-1].split('.')#Get file name
        if 'csv' in name_root:
            name_root.remove('csv') #remove ending
        name_root = '.'.join(name_root) #reform name
        aux = [s for s in name_root.split('_')]

        for s in aux:
            if s[0] == 'A':
                self.a = float(s[1:])
            elif s[0] == 'B':
                self.b = float(s[1:])
            elif s[0] == 'N':
                self.n = int(s[1:])


    def _update_fpath(self):
        fpath_components = [component for component in self.fpath.split('_')]
        for i, s in enumerate(fpath_components):
            if s[0] == 'A':
                fpath_components[i] = 'A' + str(self.a)
            elif s[0] == 'B':
                fpath_components[i] = 'B' + str(self.b)
            elif s[0] == 'N':
                fpath_components[i] = 'N' + str(self.n)
        self.fpath = '_'.join(fpath_components)
        print(self.fpath)


def test_class():
    '''
    fake_fname = 'nofilehere.txt'
    fake_distribution = frequency_distribution(fake_fname)
    '''

    frequencies_fname = 'data/stochastic_fronts/fraction_samples_N1000000_A0.0_B0.0_points=auto_stochastic.csv'
    distribution = frequency_distribution(frequencies_fname)
    distribution.read_and_format_data()
    print(distribution.get_ne())
    distribution.update_data(b=3.33)
    print(distribution.get_ne())
    print(distribution.get_nonfixed_f(10, logit=True))


if __name__ == '__main__':
    test_class()
