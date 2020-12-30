import numpy as np
import pandas as pd
import pickle
import gzip
from simulation_results import SimulationResults

DataTable_columns = ['n', 'r0', 'm0', 'a', 'b', 'delta_t', 'run', 'clone_size', 'profiles', 'front_positions', 'tree']
raw_data_columns = ['clone_size', 'profiles', 'front_positions', 'tree']

class DataTable:
    def __init__(self, table_file=None):
        if table_file is None:
            self.table = pd.DataFrame(columns=DataTable_columns)
        else:
            self.load(table_file)

    def __repr__(self):
        return repr(self.table)

    def import_simulations(self, simulation_results, weighted_offspring=False):
        for row in simulation_results.table_format(DataTable_columns, weighted_offspring=weighted_offspring):
            self.table = self.table.append(row, ignore_index=True)

    def filtered_table(self, columns):
        return self.table[self.table[columns].notnull()]

    def filter_parameters(self, params_dict):
        table_parameter_strs = np.array([self.parameter_str(i) for i, _ in self.table.iterrows()])
        test_str = self.convert_parameters_to_str(params_dict)
        filtered_table = self.table.loc[np.where(table_parameter_strs == test_str)[0], :].copy()
        return filtered_table.reset_index(drop=True)

    def convert_parameters_to_str(self, params_dict):
        return f"N{params_dict['n']}_r{params_dict['r0']}_m{params_dict['m0']}_" + \
                f"fback{params_dict['b']}_relabel{params_dict['delta_t']}"

    def get_params_dict(self, row_index=None):
        params_dict = {}
        if row_index is None:
            for i, row in self.table.iterrows():
                params_dict[i] = {'n':row['n'],
                        'r0':row['r0'],
                        'm0':row['m0'],
                        'b':row['b'],
                        'delta_t':row['delta_t']}
        else:
            row = self.table.loc[row_index, :]
            params_dict = {'n':row['n'],
                    'r0':row['r0'],
                    'm0':row['m0'],
                    'b':row['b'],
                    'delta_t':row['delta_t']}
        return params_dict

    def convert_to_results(self):
        data_columns = self.table.columns
        results_columns = [column for column in data_columns if column not in raw_data_columns ]
        self.table = self.table[results_columns]

    def add_column(self, column):
        if column not in self.table.columns:
            self.table[column] = None

    def run_str(self, row):
        return self.parameter_str(row) + "_run{self.table.at[row, 'run']}"

    def parameter_str(self, row):
        return f"N{self.table.at[row, 'n']}_r{self.table.at[row, 'r0']}_m" + \
                f"{self.table.at[row, 'm0']}_fback{self.table.at[row, 'b']}" + \
                f"_relabel{self.table.at[row, 'delta_t']}"

    def get_parameter_sets(self):
        parameter_columns = ['n', 'r0', 'm0', 'a', 'b', 'delta_t']
        parameter_strs = []
        parameter_sets = []
        for i, row in self.table.iterrows():
            if self.parameter_str(i) not in parameter_strs:
                parameter_strs.append(self.parameter_str(i))
                parameter_sets.append(row[parameter_columns].to_dict())
        return parameter_sets

    def load(self, input_file):
        with open(input_file, 'rb') as f_in:
            self = pickle.load(f_in)

    def save(self, output_file, compress=False):
        if compress == True:
            with gzip.GzipFile(output_file, 'wb') as f_out:
                pickle.dump(self, f_out)
        else:
            with open(output_file, 'wb') as f_out:
                pickle.dump(self, f_out)

def load_table(input_file, compressed=False):
    if compressed == False:
        with open(input_file, 'rb') as f_in:
            table = pickle.load(f_in)
    else:
        with gzip.GzipFile(input_file, 'rb') as f_in:
            table = pickle.load(f_in)
    return table


def test_class(data_dir, kwargs):
    results = SimulationResults(**kwargs)
    results.compute_clone_size_counts(data_dir)
    results.load_trees(data_dir)

    table = DataTable()
    table.import_simulations(results)

    output_file = '../results/data_table_test.dat'
    table.save(output_file)
    print(table)

if __name__ == "__main__":
    test_dir = '../data/trees/n350/b10.0_relabel50/'
    data_kwargs = {'dir':test_dir, 'n':350, 'b':10.0, 'delta_t':50, 'r0':0.01, 'm0':0.4, 'a':0}
    test_class(test_dir, data_kwargs)
