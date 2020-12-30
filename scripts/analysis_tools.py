import numpy as np
import re
import glob
import subprocess
import tree
from os import path
from ete3 import Tree
#from tree import Tree


def separate_parameter_values(s):
    '''
    Splits s at the occurence of the first digit

    Parameters
    __________
    s : string formated as *_variableName#_*
    '''

    # Split s
    parts = re.split('(\d.*)', s)

    # Filter empty strings and return parts
    return list(filter(None, parts))


def get_variables(fname):
    # Remove path before file name
    fname = fname.split('/')[-1]

    # Remove file extension
    parts = fname.split(".")
    fname = ".".join(parts[:-1])

    # Split into components
    name_comps = fname.split("_")
    var_dict = {}
    for string in name_comps:
        params = separate_parameter_values(string)
        if len(params) == 2:
            var_dict[params[0]] = float(params[1])
    return var_dict


def wave_type(fname, ps_threshold=2.0, sf_threshold=4.0):
    '''
    Returns whether wave is pulled, semi-pushed, or fully-pushed by comparing value of fback parameter to thresholds.

    Parameters
    __________
    fname : name of data file; string
    ps_threshold : threshold for pulled--semi-pushed transition; float
    sf_threshold : threshold for semi-pushed--fully-pushed transition; float
    '''

    var_dict = get_variables(fname)

    if 'fback' not in var_dict.keys():
        print("Value of fback not found in file name!")
        return 0

    if var_dict["fback"] < ps_threshold:
        wave_type = "pulled"
    elif var_dict["fback"] < sf_threshold:
        wave_type = "semi-pushed"
    else:
        wave_type = "fully-pushed"

    return wave_type


def read_run_list(data_dir, file_ext=None):
    if file_ext is None:
        file_list = sorted(glob.glob(data_dir + 'position_*.txt'))
    else:
        file_list = sorted(glob.glob(data_dir + '*.' + file_ext))
    run_list = []
    for fname in file_list:
        var_dict = get_variables(fname)
        if 'run' in var_dict.keys():
            run_list.append(int(var_dict['run']))
    return list(np.unique(run_list))


def get_tau(fpath):
    fname = fpath.split("/")[-1]
    var_dict = get_variables(fname)
    return int(var_dict['tau'])


def generate_fname(b, relabel, head="", tail=""):
    return head + "fback" + str(b) + "_relabel" + str(relabel) + tail


def generate_parameter_string(n, r, m, b=""):
    string = "N" + str(n)
    string += "_r" + str(r)
    string += "_m" + str(m)
    if b != "":
        string += "_fback" + str(b)
    return string


def extract_simulation_parameters(dir_path):
    kwargs_list = splice_unique_fnames(dir_path)
    formated_kwargs = []

    for kwargs in kwargs_list:
        kwargs['n'] = int(kwargs.pop('N'))
        kwargs['r0'] = kwargs.pop('r')
        kwargs['m0'] = kwargs.pop('m')
        kwargs['b'] = kwargs.pop('fback')
        if 'relabel' in kwargs.keys():
            kwargs['delta_t'] = int(kwargs.pop('relabel'))
        elif 'tmix' in kwargs.keys():
            # Old simulations with different delta_t label
            kwargs['delta_t'] = int(kwargs.pop('tmix'))
        else:
            print(f'No delta_t found in {dir_path}. Setting value to 0.')
            kwargs['delta_t'] = 0
        if dir_path[-1] != '/':
            kwargs['dir'] = dir_path + '/'
        else:
            kwargs['dir'] = dir_path
        #kwargs.pop('run')
        formated_kwargs.append(kwargs)
    return formated_kwargs


def splice_unique_fnames(dir_path):
    '''
    Takes in path to dir containing simulation results and returns list of dictionaries with unique parameter values
    '''
    # Get list of unique position_*.txt files
    file_names = glob.glob(dir_path + '/position_*.txt')
    truncated_fnames = truncate_fnames(file_names, '_run', '.tst')
    unique_fnames = np.unique(truncated_fnames)
    params = []
    for fname in unique_fnames:
        params.append(get_variables(fname))
    return params


def truncate_fnames(fnames, truncation_key, extension=None):
    if extension is None:
        ext = ''
    else:
        ext = extension

    truncated_fnames = []
    for fname in fnames:
        truncation_index = fname.find(truncation_key)
        if truncation_index != -1:
            truncated_fnames.append(fname[:truncation_index] + ext)

    return truncated_fnames


def read_tree(fpath):
    try:
        bash_command = "sed 's/)$/);/g' " + fpath + " > temp.nwk"
        subprocess.call([bash_command], shell=True)
        tree = Tree("temp.nwk", format=1)
        subprocess.call(["rm -f temp.nwk"], shell=True)
        return tree
    except:
        print("Error reading Newick tree:", fpath, ".")
        return None


def read_custom_tree(fpath, insert_outer_brackets=True, generative_model=False):
    try:
        if generative_model == False:
            bash_command = "sed 's/)$/);/g' " + fpath + " > temp.nwk"
        elif insert_outer_brackets == True:
            bash_command = "sed 's/;$/);/g' " + fpath + " | sed 's/^(/^((/g' > temp.nwk"
        else:
            bash_command = f"cat {fpath} > temp.nwk"

        subprocess.call([bash_command], shell=True)
        t = tree.Tree("temp.nwk")
        subprocess.call(["rm -f temp.nwk"], shell=True)
        return t
    except:
        print("Error reading Newick tree:", fpath, ".")
        return None

def calculate_overlap(n1_dict, n2_dict):
    common_keys = get_common_keys(n1_dict, n2_dict)
    overlap = 0
    for key in common_keys:
        overlap += n1_dict[key] * n2_dict[key]
    return overlap


def normed_inner_product(arr1, arr2):
    '''
    Calculates the inner product between arr1 and arr2, normalized by the L2-norm of arr2.
    '''
    return inner_product(arr1, arr2) / inner_product(arr2, arr2)


def inner_product(arr1, arr2):
    return np.sum(arr1 * arr2)


def get_common_keys(dict1, dict2):
    '''
    Returns list of keys shared between dict1 and dict2
    '''
    return list(set(dict1.keys()).intersection(set(dict2.keys())))

def normalize_distribution(density):
    '''
    Normalizes density by the sum of its entries.
    '''
    return density / np.sum(density)


def test_tools():
    test_dir = '../data/tests/tau_m/'
    unique_params = splice_unique_fnames(test_dir)
    print(unique_params)


def distribution_distance(p_object, q_object, method='L2'):
    '''
    Takes two distribution dicts and calculates distance between them.

    Parameters
    __________
    p_object, q_object : distributions in dict format
    method : distance method ['L2', 'KL']
    '''

    if (type(p_object) is dict) and (type(q_object) is dict):
        common_keys = get_common_keys(p_object, q_object)
        p = convert_to_array(p_object, common_keys)
        q = convert_to_array(q_object, common_keys)
    elif type(p_object) is dict:
        sorted_keys = sorted(list(p_object.keys()))
        p = np.array([p_object[key] for key in sorted_keys])
        q = np.array(q_object)
    elif type(q_object) is dict:
        sorted_keys = sorted(list(q_object.keys()))
        q = np.array([q_object[key] for key in sorted_keys])
        p = np.array(p_object)
    else:
        p = np.array(p_object)
        q = np.array(q_object)

    p = norm_vector(p)
    q = norm_vector(q)

    if method == 'L2':
        diff = inner_product(p - q, p - q)
    elif method == 'KL':
        nonzero_index = np.where(p != 0)[0]
        diff = np.sum(p[nonzero_index] * np.log(p[nonzero_index] / q[nonzero_index]))
    else:
        diff = None
    return diff


def convert_to_array(dictionary, keys):
    '''
    Returns array containing values of dictionary at keys.
    '''
    values = []
    for key in keys:
        values.append(dictionary[key])
    return np.array(values)


def norm_vector(v):
    if inner_product(v, v) != 1:
        v = v / np.sqrt(inner_product(v, v))
    return v


def linear_cooperativity(A, B):
    '''Takes A and B for linear dispersal model and returns:
        0 if wave is pulled
        1 if wave is semi-pushed
        2 if wave is fully-pushed'''
    if 2*A + B <= 2:
        return 0
    elif 2*A + B < 4:
        return 1
    else:
        return 2


def bs_propagator(psi, psi_0, t, Ne):
    numerator = np.sin(np.exp(-t / Ne) * np.pi)
    denominator = 2 * np.pi * (np.cos(np.exp(-t / Ne) * np.pi) + np.cosh(np.exp(-t / Ne) * psi - psi_0))
    return numerator / denominator


def convert_profile_to_density(profiles_dict):
    for time in profiles_dict.keys():
        profiles_dict[time] = np.sum(profiles_dict[time] > 0, axis=1)
    return profiles_dict


def calculate_node_x(nodes, N, L=300):
    '''
    Takes list of tree nodes and returns dictionary with nodes at each position along the front.
    '''
    sorted_nodes ={}
    for x in range(L):
        sorted_nodes[x] = []

    for node in nodes:
        node_id = int(node)
        gen_factor = N * L
        x = (node_id % gen_factor) // N
        sorted_nodes[x].append(node)

    return sorted_nodes


if __name__ == '__main__':
    test_tools()
