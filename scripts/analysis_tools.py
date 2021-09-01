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


def velocity(g, m, f):
    D = m/2.
    if f >= -0.5:
        v = np.sqrt(D*g/2.)*(1 - 2.*f)
    else:
        v = 2.*np.sqrt(D*g*abs(f))
    return v

def velocity_Fisher(g, m, f):
    D = m/2.
    if f <= 0.0:
        v_F = 2.*np.sqrt(D*g*abs(f))
    else:
        v_F = 0.
    return v_F

def v_coop(r, m, B):
    D = m/2.
    if B > 2.:
        v = np.sqrt(D*r*B/2.)*(1 + 2./B)
    else:
        v = 2.*np.sqrt(D*r)
    return v

def vF_coop(r, m, B):
    D = m/2.
    vF = 2.*np.sqrt(D*r)
    return vF

def alpha_coop(B, r, m):
    D = m/2.
    v = v_coop(r, m, B)
    vF = 2 * np.sqrt(D * r)
    beta = np.sqrt(1 - (vF/v)**2)
    return 2 * beta / (1 - beta)

def growth(g, f, x):
    return g*(1 - x)*(x - f)

def profile(gf, migr, fstr, x):
    D = migr/2.
    if fstr > -0.5:
        prof = 1./(1. + np.exp(np.sqrt(gf/(2.*D))*x))
    else:
        prof = 1./(1. + np.exp(np.sqrt(gf*abs(fstr)/D)*x))
    return prof

def fixation_const(gf, migr, fstr, x_min, x_max, dx):
    x_arr = np.arange(x_min, x_max, dx)
    c_arr = profile(gf, migr, fstr, x_arr)
    v = velocity(gf, migr, fstr)
    D = migr/2.

    prelim_prob = c_arr**2*np.exp(v*x_arr/D)
    const = integrate.simps(prelim_prob, x_arr)
    return const

def fixation_probability(gf, migr, fstr, x_min, x_max, dx, x):
    c = profile(gf, migr, fstr, x)
    v = velocity(gf, migr, fstr)
    D = migr/2.
    const = fixation_const(gf, migr, fstr, x_min, x_max, dx)
    prob = c**2*np.exp(v*x/D)/const
    return prob

def ancestral_probability(gf, migr, fstr, x_min, x_max, dx, x):
    c = profile(gf, migr, fstr, x)
    v = velocity(gf, migr, fstr)
    D = migr/2.
    const = fixation_const(gf, migr, fstr, x_min, x_max, dx)
    prob = c**3*np.exp(2*v*x/D)/(const**2)
    return prob


def Neff(gf, migr, fstr, N, x_min, x_max, dx):
    x_arr = np.arange(x_min, x_max, dx)
    c_arr = profile(gf, migr, fstr, x_arr)
    v = velocity(gf, migr, fstr)
    D = migr/2.

    const = fixation_const(gf, migr, fstr, x_min, x_max, dx)
    function = c_arr**3*np.exp(2*v*x_arr/D)/(const**2)

    Ne = N/integrate.simps(function, x_arr)
    return Ne

def test_dx():
    Ne_arr = []
    dx_arr = np.logspace(-4, -1, 10)
    for dx in dx_arr:
        Ne = Neff(0.01, 0.25, -0.2, 10000, x_min, x_max, dx)
        Ne_arr.append(Ne)

def theory_exponent(fstr):
    if fstr <-0.5:
        exp = 0.
    elif fstr < -.25:
        exp = 2.*(1. + 2.*fstr)
    elif fstr < 0.5:
        exp = 1.
    return exp

def semipushed_theory(B):
    exp = 2.*(1. - 2./fstr)
    return exp

def stoch_semi(a):
    return (a - 2.)/2.

def front_growth(labels_flag, save_flag, label_size):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 8}
    matplotlib.rc('font', **font)

    gf = 0.01
    gf_pushed = 4*gf
    m = 1.25
    dx = 0.01
    x_min = -40
    x_max = 50
    f_pulled = -1.0
    f_pushed = 0.0

    min_f = -0.8
    max_f = -0.2
    f_arr = np.arange(min_f, max_f, 0.01)
    f_pulled_arr = np.arange(min_f, -0.5, 0.001)
    f_pushed_arr = np.arange(-0.5, max_f, 0.001)
    v_arr = np.array([velocity(gf, m, f) for f in f_arr])
    vF_arr = np.array([velocity_Fisher(gf, m, f) for f in f_arr])
    v_ratio_arr = v_arr/vF_arr

    min_v = 0.95*min(v_ratio_arr)
    max_v = 1.05*max(v_ratio_arr)

    y_pp_transition = np.arange(min_v, max_v, 0.001)
    x_pp_transition = -0.5*np.ones(len(y_pp_transition))

    x_array = np.arange(x_min, x_max, 0.01)
    pulled_profile = np.array([profile(gf, m, f_pulled, x) for x in x_array])
    pulled_growth = np.array(growth(gf, f_pulled, pulled_profile))
    pushed_profile = np.array([profile(gf_pushed, m, f_pushed, x) for x in x_array])
    pushed_growth = np.array(growth(gf_pushed, f_pushed, pushed_profile))


    fig = plt.figure(figsize=(cm2inch(17.8),cm2inch(5.4)))

    gf = 0.01
    m = 0.25
    dx = 0.01
    x_min = -30
    x_max = 30
    f_pulled = -1.0
    f_pseudo = -0.4
    f_pushed = -0.0

    x_array = np.arange(x_min, x_max, 0.1)
    x_array = np.append(x_array, [x_max])
    pulled_profile = np.array([profile(gf, m, f_pulled, x) for x in x_array])
    pulled_growth = np.array(growth(gf, f_pulled, pulled_profile))
    pulled_fixation = np.array([fixation_probability(gf, m, f_pulled, x_min, x_max, dx, x) for x in x_array])
    pulled_ancestry = np.array([ancestral_probability(gf, m, f_pulled, x_min, x_max, dx, x) for x in x_array])
    pseudo_profile = np.array([profile(gf, m, f_pseudo, x) for x in x_array])
    pseudo_growth = np.array(growth(gf, f_pseudo, pulled_profile))
    pseudo_fixation = np.array([fixation_probability(gf, m, f_pseudo, x_min, x_max, dx, x) for x in x_array])
    pseudo_ancestry = np.array([ancestral_probability(gf, m, f_pseudo, x_min, x_max, dx, x) for x in x_array])
    pushed_profile = np.array([profile(gf, m, f_pushed, x) for x in x_array])
    pushed_growth = np.array(growth(gf, f_pushed, pushed_profile))
    pushed_fixation = np.array([fixation_probability(gf, m, f_pushed, x_min, x_max, dx, x) for x in x_array])
    pushed_ancestry = np.array([ancestral_probability(gf, m, f_pushed, x_min, x_max, dx, x) for x in x_array])

    pulled_growth_fill = np.array([[elem]*len(pulled_profile) for elem in pulled_growth])
    pseudo_growth_fill = np.array([[elem]*len(pseudo_profile) for elem in pseudo_growth])
    pushed_growth_fill = np.array([[elem]*len(pushed_profile) for elem in pushed_growth])
    max_growth = max(pushed_growth)
    min_growth = min(pushed_growth)


    ax1 = fig.add_subplot(131)
    #ax1.set_title('pulled', fontsize=12, fontweight='bold')
    ax1.set_xlabel('position, x', fontsize=label_size, fontweight='bold')
    ax1.set_ylabel('population density, n', fontsize=label_size, fontweight='bold')
    ax1.set_xticks([-20, 0, 20, 40])
    ax1.set_xticklabels([])
    #ax1.set_yticks([0., 0.25, 0.5, 0.75, 1.])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim([0.0, 1.1])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    ax1.contourf(x_array[:-1], pulled_profile[:-1], pulled_growth_fill[:-1].T[:-1], 200, cmap=plt.cm.winter)
    ax1.fill_between(x_array, pulled_profile, y2=1.01*max(pulled_profile), color='w')
    #ax1.text(1.10*x_min, 1.03*1.1, 'A', fontsize=12, fontweight='bold', color='k')
    #ax1.plot(x_array, (1./max(pulled_fixation))*pulled_fixation, lw=2, c='r')
    #ax1.plot(x_array, (1./max(pulled_ancestry))*pulled_ancestry, lw=2, c='purple')
    #ax1.text(10, 0.80, 'ancestry', fontweight='bold', fontsize=8, color='r')
    #ax1.text(6, 0.30, 'diversity', fontweight='bold', fontsize=8, color='purple')

    growth_focus = x_array[np.argmax(pulled_growth)]
    ancestry_focus = x_array[np.argmax(pulled_fixation)]
    diversity_focus = x_array[np.argmax(pulled_ancestry)]
    #ax1.scatter([growth_focus, ancestry_focus, diversity_focus], [0.1, 0.1, 0.1], s=80, edgecolor='none', color=['darkolivegreen', 'r', 'purple'])
    #ax1.text(12, 0.75, 'ancestry\n'+'$\mathbf{\propto n^2 e^{v \zeta/D}}$', fontweight='bold', fontsize=8, color='r')
    #ax1.text(6, 0.30, 'diversity\n'+'$\mathbf{\propto \gamma_f}$ $\mathbf{n^3 e^{2v \zeta/D}}$', fontweight='bold', fontsize=8, color='purple')


    ax1 = fig.add_subplot(132)
    #ax1.set_title('semi-pushed', fontsize=12, fontweight='bold')
    ax1.set_xlabel('position, x', fontsize=label_size, fontweight='bold')
    #ax1.set_ylabel('population density, n', fontsize=label_size, fontweight='bold')
    ax1.set_xticks([-20, 0, 20, 40])
    ax1.set_xticklabels([])
    #ax1.set_yticks([0., 0.25, 0.5, 0.75, 1.])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim([0.0, 1.1])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    ax1.contourf(x_array[:-1], pseudo_profile[:-1], pseudo_growth_fill[:-1].T[:-1], 200, cmap=plt.cm.winter)
    ax1.fill_between(x_array, pseudo_profile, y2=1.01*max(pseudo_profile), color='w')
    #ax1.text(1.10*x_min, 1.03*1.1, 'B', fontsize=12, fontweight='bold', color='k')
    #ax1.plot(x_array, (1./max(pseudo_fixation))*pseudo_fixation, lw=2, c='r')
    #ax1.plot(x_array, (1./max(pseudo_ancestry))*pseudo_ancestry, lw=2, c='purple')
    growth_focus = x_array[np.argmax(pseudo_growth)]
    ancestry_focus = x_array[np.argmax(pseudo_fixation)]
    diversity_focus = x_array[np.argmax(pseudo_ancestry)]
    #ax1.scatter([growth_focus, ancestry_focus, diversity_focus], [0.1, 0.1, 0.1], s=80, edgecolor='none', color=['darkolivegreen', 'r', 'purple'], zorder=3)



    ax1 = fig.add_subplot(133)
    #ax1.set_title('fully-pushed', fontsize=12, fontweight='bold')
    ax1.set_xlabel('position, x', fontsize=label_size, fontweight='bold')
    #ax1.set_ylabel('population density, n', fontsize=label_size, fontweight='bold')
    ax1.set_xticks([-20, 0, 20, 40])
    ax1.set_xticklabels([])
    #ax1.set_yticks([0., 0.25, 0.5, 0.75, 1.])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim([0.0, 1.1])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    cax = ax1.contourf(x_array[:-2], pushed_profile[:-2], pushed_growth_fill[:-2].T[:-2], 200, cmap=plt.cm.winter)
    ax1.fill_between(x_array, pushed_profile, y2=1.01*max(pushed_profile), color='w')
    #ax1.text(1.10*x_min, 1.03*1.1, 'C', fontsize=12, fontweight='bold', color='k')
    #ax1.plot(x_array, (1./max(pushed_fixation))*pushed_fixation, lw=2, c='r')
    #ax1.plot(x_array, (1./max(pushed_ancestry))*pushed_ancestry, lw=2, c='purple')

    growth_focus = x_array[np.argmax(pushed_growth)]
    ancestry_focus = x_array[np.argmax(pushed_fixation)]
    diversity_focus = x_array[np.argmax(pushed_ancestry)]
    #ax1.scatter([growth_focus, ancestry_focus, diversity_focus], [0.1, 0.1, 0.1], s=80, edgecolor='none', color=['darkolivegreen', 'r', 'purple'], zorder=2)


    cbar = fig.colorbar(cax, ticks=[min_growth, max_growth])
    cbar.ax.set_yticklabels(['low', 'high'])

    ax1.text(42, 0.82, 'growth rate', fontsize=10, rotation=90)
    #cbar.ax.set_ylabel('growth rate', rotation=90)

    plt.tight_layout(pad=1.5)
    #plt.savefig('./Fig1pushed_pulled_growth.tiff', dpi=500)

    if save_flag != 0:
        plt.savefig('../figures/draft/front_growth.pdf')

if __name__ == '__main__':
    test_tools()
