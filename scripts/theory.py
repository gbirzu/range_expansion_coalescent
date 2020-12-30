import numpy as np
import scipy.special as special


'''
Colescent statistics based on Caliebe et al. (2007)
'''

def external_branch_distribution(t, alpha):
    '''
    Calculate distribution function for length of external branches Zn of coalescent

    Parameters
    ----------
    t : double
        time
    alpha : double
        Beta coalescent parameter
        Note that throughout:
            alpha = 1 corresponds to Bolthausen-Sznitman coalescent
            alpha = 2 corresponds to Kingman coalescent

    '''

    if alpha == 2:
        return 8 / ((2 + t) ** 3)
    elif alpha == 1:
        return np.exp(-t)
    else:
        return (1 + t / (alpha * special.gamma(alpha))) ** (-1 - alpha / (alpha - 1)) / ((alpha - 1) * special.gamma(alpha))


def total_branch_length(n, alpha, normalized=False):
    '''
    Calculates total length of coalescent

    Parameters
    ----------
    n : int
        number of samples
    alpha : double
        Beta coalescent parameter
    '''
    if alpha == 1:
        # Use formula in Drmota et al. 2007
        if normalized == False:
            return mu_1_BS(n)
        else:
            return mu_1_BS(n) / mu_1_BS(2)
    elif alpha > 1 and alpha < 2:
        # Use formula in Kerstig 2012
        if normalized == False:
            return c1(alpha) * n ** (2 - alpha)
        else:
            return n**(2 - alpha) / 2**(2 - alpha)
    elif alpha == 2:
        # Is already normalized; no need to check
        return harmonic_sum(n)

def mu_1_BS(x, asympt_cutoff=20):
    n = int(x)
    if n == 1:
        return 0
    elif n == 2:
        return 2
    elif n <= asympt_cutoff:
        # Use exact recursion (Eq. 8, Drmota et al. 2007)
        mu_1 = 1 / alpha_BS(n)
        for k in range(2, n):
            mu_1 += p_nk_BS(n, k) * mu_1_BS(k)
        return mu_1
    else:
        # Use asymptotic scaling (Eq. 24, Drmota et al. 2007)
        return n / np.log(n) + (2 - np.euler_gamma) * n / (np.log(n)**2)

def p_nk_BS(n, k):
    # Eq. 9, Drmota et al. 2007
    return n / ((n - 1) * (n - k) * (n - k + 1))

def alpha_BS(n):
    # Inline equation after Eq. 3 in Drmota et al. 2007
    return (n - 1) / n

def c1(alpha):
    return special.gamma(alpha) * alpha * (alpha - 1) / (2 - alpha)

# Eq. 8
def g(N, Pij, alpha):
    G = np.zeros((N+1,N+1))
    for m in range(2, N+1):
        for n in range(m, N+1):
            if n == m:
                G[n,m] = -1.0 / q(n, m, alpha)
            else:
                G[n,m] = np.dot(Pij[n,:], G[:,m])
    return G


def harmonic_sum(n):
    j_list = np.arange(1, n)
    return np.sum([1. / j for j in j_list])

def var_ln_BS(n):
    return mu_2_BS(n) - mu_1_BS(n)**2

def mu_2_BS(x, asympt_cutoff=20):
    n = int(x)
    if n == 1:
        return 0
    elif n == 2:
        return 2 * mu_1_BS(2) / alpha_BS(n)
    elif n <= asympt_cutoff:
        # Use exact recursion (Eq. 8, Drmota et al. 2007)
        mu_2 = 2 * mu_1_BS(n)
        for k in range(2, n):
            mu_2 += p_nk_BS(n, k) * mu_2_BS(k)
        return mu_2
    else:
        # Use aymptotic scaling (Eq. 25, Drmota et al. 2007)
        return n**2 / (np.log(n)**2) + (9 / 2 - 2 * np.euler_gamma) * n**2 / (np.log(n)**3)

def var_ln_kingman(n):
    # Asymptotic value (Lemma 7.1, Drmota et al. 2007)
    return 2 * np.log(n)


def sfs(f, alpha, n=10000):
    '''
    Site frequency spectrum prediction for different coalescents

    Parameters
    ----------
    f : double
        derived allele frequency
    alpha : double
        Beta coalescent parameter
    '''

    if alpha == 1:
        # Use Eq. S9 from Neher & Hallatschek 2013
        return (2 * f - 1) / (f * (1 - f) * (np.log(f) - np.log(1 - f)))
    elif alpha == 2:
        return 1 / f
    else:
        # Use Theorem 1 in Birkner et al. 2013
        i = f * n
        return C(alpha) * (2 - alpha) * np.exp(np.log(special.poch(alpha - 1, i - 1)) - special.gammaln(i + 1))


def C(alpha, A=1):
    return alpha * (alpha - 1) / (A * special.gamma(2 - alpha) * (2 - alpha))


def tk_ratio_kingman(i, j=2):
    return (2 - 2 / i) / (2 - 2 / j)


def tk_ratio_bs(i, j=2):
    '''
    returns ratio of merger times for Bolthausen-Sznitman coalescent special cases of i, j.
    Based on Brunet & Derrida, J. Stat. Mech. 2013
    '''
    if j != 2:
        return bs_theory(i, 2) / bs_theory(j, 2)
    else:
        if i == 2:
            return 1
        elif i == 3:
            return 5 / 4
        elif i == 4:
            return 25 / 18
        elif i == 5:
            return 427 / 288
        else:
            return 0


def test_functions():
    n_list = list(range(1, 7)) + [10] + [20] + [100] + [1000]
    print("n \t E(L_n) \t E(L_n^2) \t Var(L_n)")
    for n in n_list:
        m1 = mu_1_BS(n)
        m2 = mu_2_BS(n)
        print(n, '\t', mu_1_BS(n), '\t', mu_2_BS(n), '\t', var_ln_BS(n))

if __name__ == "__main__":
    test_functions()
