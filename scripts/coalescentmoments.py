import sys
import numpy as np
from scipy.special import beta, binom

'''
Calculate the site frequncy spectrum first and second moments
according to Fu 1995 and Birkner et al 2013.
'''

def sfs_moments(N, alpha, m2=True):
    '''
    Calculate first and second moments of the site frequency spectrum.

    Parameters
    ----------
    N : int
        Sample size
    alpha : float
        Beta coalescent parameter alpha
    m2 : Bool
        If True, return first and second moments.
        If False, return first moments only

    '''
    # If Kingman coalescent, use explicit Fu formulas to save time
    if alpha == 2.0:
        moments = fu_moments(N)
        if m2:
            return moments
        else:
            return moments[0]
    else:
        Pij = pij(N, alpha)
        G = g(N, Pij, alpha)
        Pnkb = pnkb(N, Pij, G)
        if m2:
            Peq = peq(N, Pij, G)
            Pun = pun(N, Pij, G, Pnkb)
            Pne = pne(N, Pij, G, Pnkb)
            return sfs_m1(N, G, Pnkb), sfs_m2(N, G, Pnkb, Peq, Pun, Pne)
        else:
            return sfs_m1(N, G, Pnkb)

def fold_sfs_moments(N, M1, M2):
    '''
    Calculate first and second moments of the folded site frequency spectrum.

    Parameters
    ----------
    N : int
        Sample size
    M1 : np.array(dtype=float)
        First moments of the unfolded site frequency spectrum
    M2 : np.array(dtype=float)
        Second moments of the unfolded site frequency spectrum

    '''

    M1_folded = np.zeros(N//2)
    for i in range(N//2):
         M1_folded[i] = (M1[i] + M1[N-i-2]) / (1.0 + (i==N-i-2))

    M2_folded = np.zeros((N//2, N//2))
    for i in range(N//2):
        for j in range(i):
            M2_folded[i,j] = (M2[i,j] + M2[N-i-2,j] + M2[i,N-j-2] + M2[N-i-2,N-j-2]) / ((1.0+(i==N-i-2))*(1.0+(j==N-j-2)))
            M2_folded[j,i] = M2_folded[i,j]
    return (M1_folded, M2_folded)


# Schweinsberg 2003 Eq. 12
def lamb(n, k, alpha):
    if alpha == 2:
        # Kingman coalescent limit
        if k == 2:
            return 1.0
        else:
            return 0.0
    return beta(k-alpha, n-k+alpha) / beta(2-alpha, alpha)
    # return gamma(k - 1) * gamma(n - k + 1) / gamma(n)

# Unnumbered equation before eq. 6, combined with lambda_bsc = 1 (from eq 3 with alpha=1)
def q(i, j, alpha):
    if i == j:
        return -1.0 * np.sum(q(i, k, alpha) for k in range(1,i))
        # return 1.0 - i
    elif i > j and j >= 1:
        return binom(i, i-j+1) * lamb(i, i-j+1, alpha)
        # return float(i) / ((i-j+1)*(i-j))
    else:
        sys.stderr.write('Error in q: i={}, j={}\n'.format(i,j))
        return 0.0

# Eq. 6
def pij(N, alpha):
    Pij = np.zeros((N+1,N+1))
    for i in range(2, N+1):
        for j in range(2, i+1):
            Pij[i,j] = q(i, j, alpha) / -q(i, i, alpha)
    return Pij

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

# Eq. A2
def pnkb(N, Pij, G):
    nprime = np.arange(N,dtype=float)
    Pnkb = np.zeros((N+1, N+1, N))
    for k in range(2, N+1):
        for n in range(k, N+1):
            if n == k:
                Pnkb[n,k,1] = 1
            else:
                for b in range(1, n-k+2):
                    nstart1 = n-b+1
                    nstart2 = max(k,b+1)

                    p1 = Pij[n, nstart1:n]
                    g1 = G[nstart1:n, k]
                    np1 = nprime[nstart1:n]
                    a1 = Pnkb[:,k,:].diagonal(offset=b-n)[1:b]
                    s1 = np.sum(p1*g1*(b-(n-np1))*a1/np1)

                    p2 = Pij[n, nstart2:n]
                    g2 = G[nstart2:n, k]
                    np2 = nprime[nstart2:n]
                    a2 = Pnkb[nstart2:n, k , b]
                    s2 = np.sum(p2*g2*(np2-b)*a2/np2)

                    Pnkb[n,k,b] = (s1 + s2) / G[n, k]
    return Pnkb

# Eq. A4
def peq(N, Pij, G):
    nprime = np.arange(N,dtype=float)
    Peq = np.zeros((N+1, N+1, N, N))
    for k in range(2, N+1):
        for n in range(k, N+1):
            # Boundary condition
            if n == k:
                Peq[n,k,1,1] = 1
            else:
                for i in range(1, n-k+2):
                    for j in range(1, n-k+3-i):

                        nstart1 = n-i+1
                        p1 = Pij[n, nstart1:n]
                        g1 = G[nstart1:n, k]
                        m1 = nprime[nstart1:n]
                        a1 = Peq[:,k,:,j].diagonal(offset=i-n)[1:i]
                        s1 = np.sum(p1*g1*(i-(n-m1))*a1/m1)

                        nstart2 = n-j+1
                        p2 = Pij[n, nstart2:n]
                        g2 = G[nstart2:n, k]
                        m2 = nprime[nstart2:n]
                        a2 = Peq[:,k,i,:].diagonal(offset=j-n)[1:j]
                        s2 = np.sum(p2*g2*(j-(n-m2))*a2/m2)

                        nstart3 = max(k, i+j+1)
                        p3 = Pij[n, nstart3:n]
                        g3 = G[nstart3:n, k]
                        m3 = nprime[nstart3:n]
                        a3 = Peq[nstart3:n,k,i,j]
                        s3 = np.sum(p3*g3*(m3-i-j)*a3/m3)

                        Peq[n,k,i,j] = (s1 + s2 + s3) / G[n, k]
    return Peq

# Eq. A6
def pun(N, Pij, G, Pnkb):
    nprime = np.arange(N,dtype=float)
    Pun = np.zeros((N+1, N+1, N, N+1, N))
    for l in range(2, N+1):
        for k in range(2, l):
            for n in range(l, N+1):

                if n == l:
                    Pun[n,k,:,l,1] = Pnkb[n,k,:] * (n-np.arange(N))/n
                else:
                    for i in range(1, n-k+2):
                        for j in range(1, n-l+2):

                            nstart1 = n-i+1
                            p1 = Pij[n, nstart1:n]
                            g1 = G[nstart1:n, l]
                            m1 = nprime[nstart1:n]
                            a1 = Pun[:,k,:,l,j].diagonal(offset=i-n)[1:i]
                            s1 = np.sum(p1*g1*(i-(n-m1))*a1/m1)

                            nstart2 = n-j+1
                            p2 = Pij[n, nstart2:n]
                            g2 = G[nstart2:n, l]
                            m2 = nprime[nstart2:n]
                            a2 = Pun[:,k,i,l,:].diagonal(offset=j-n)[1:j]
                            s2 = np.sum(p2*g2*(j-(n-m2))*a2/m2)

                            nstart3 = max(l, i+j+1)
                            p3 = Pij[n, nstart3:n]
                            g3 = G[nstart3:n, l]
                            m3 = nprime[nstart3:n]
                            a3 = Pun[nstart3:n,k,i,l,j]
                            s3 = np.sum(p3*g3*(m3-i-j)*a3/m3)

                            Pun[n,k,i,l,j] = (s1 + s2 + s3) / G[n, l]
    return Pun

# Eq. A8
def pne(N, Pij, G, Pnkb):
    nprime = np.arange(N,dtype=float)
    Pne = np.zeros((N+1, N+1, N, N+1, N))
    for l in range(2, N+1):
        for k in range(2, l):
            for n in range(l, N+1):
                if n == l:
                    Pne[n,k,:,l,1] = Pnkb[n,k,:] * np.arange(N)/n
                else:
                    for i in range(1, n-k+2):
                        for j in range(1, n-l+2):

                            if i > j:
                                nstart1 = n-(i-j)+1
                                p1 = Pij[n, nstart1:n]
                                g1 = G[nstart1:n, l]
                                m1 = nprime[nstart1:n]
                                a1 = Pne[:,k,:,l,j].diagonal(offset=i-n)[1+j:i]
                                s1 = np.sum(p1*g1*(i-j-(n-m1))*a1/m1)
                            else:
                                s1 = 0.0

                            nstart2 = n-j+1
                            p2 = Pij[n, nstart2:n]
                            g2 = G[nstart2:n, l]
                            m2 = nprime[nstart2:n]
                            a2 = Pne[nstart2:n, k, i-n+nstart2:i, l, j-(n-nstart2):j]
                            a2 = np.array([Pne[int(m), k, int(i-(n-m)), l, int(j-(n-m))] for m in m2])
                            s2 = np.sum(p2*g2*(j-(n-m2))*a2/m2)

                            nstart3 = max(l, i+1)
                            p3 = Pij[n, nstart3:n]
                            g3 = G[nstart3:n, l]
                            m3 = nprime[nstart3:n]
                            a3 = Pne[nstart3:n,k,i,l,j]
                            s3 = np.sum(p3*g3*(m3-i)*a3/m3)

                            Pne[n,k,i,l,j] = (s1 + s2 + s3) / G[n, l]
    return Pne

# Proposition 1
def sfs_m1(N, G, Pnkb):
    # 0th entry will be the singletons
    M1 = np.dot(np.arange(N+1)*G[N,:], Pnkb[N,:,:])
    return M1[1:] / 2

# Theorem 2. Note: the function has been modified from Rice et al. 2018 to include the variance term.
def sfs_m2(N, G, Pnkb, Peq, Pun, Pne):
    M2 = np.zeros((N, N))
    K = np.arange(N+1)
    for i in range(1,N):
        for j in range(1,i):
            s1 = np.dot(2.0*K*(K-1)*G[N,:]*G.diagonal(), Peq[N,:,i,j])
            s2 = 0
            for k in range(2,N+1):
                for l in range(k+1,N+1):
                    s2 += k*l* (Pun[N,k,i,l,j] + Pne[N,k,i,l,j] + Pun[N,k,j,l,i] + Pne[N,k,j,l,i]) * G[N,l] * G[l,k]
            M2[i,j] = s1 + s2
            M2[j,i] = s1 + s2
        s3 = np.dot(2.0*K*G[N,:]*(1 + G.diagonal()), Pnkb[N,:,i]) # Variance, i.e. the diagnonal term
        M2[i,i] = s3
    # 0th entry will be the singletons
    # Normalize
    return M2[1:,1:] / 4


def a_fu(n):
    return np.sum(1.0/np.arange(1,n))

def beta_fu(n, i):
    return 2.0 * n * (a_fu(n+1) - a_fu(i)) / ((n-i+1)*(n-i)) - 2.0/(n-i)

def sigma_fu(n):
    '''
    Calculates sigma matrix from Fu 1995 Eqs. (2) and (3).
    Note: this has been modified from Rice et al. 2018 to include the variance term.
    '''
    Sigma = np.zeros((n,n))
    for i in range(1,n):
        # Don't calculate variance terms
        for j in range(1,i):
            if i + j < n:
                Sigma[i,j] = (beta_fu(n, i+1) - beta_fu(n, i))/2
            elif i + j == n:
                Sigma[i,j] = (a_fu(n)-a_fu(i))/(n-i) + (a_fu(n)-a_fu(j))/(n-j) - (beta_fu(n,i)+beta_fu(n,j+1))/2 - 1.0/(i*j)
            else:
                Sigma[i,j] = (beta_fu(n,j) - beta_fu(n,j+1))/2 - 1.0/(i*j)
        # Add variance term
        # Note: compared to Fu 1995, there is an extra factor of two to account for the symmetrization below
        if i < n/2:
            Sigma[i, i] = (beta_fu(n, i + 1))/2
        elif i == n/2:
            Sigma[i, i] = (2*(a_fu(n) - a_fu(i))/(n - i) - 1/i**2)/2
        else:
            Sigma[i, i] = (beta_fu(n, i) - 1/i**2)/2
    # This was a problem when n was large. Must be some optimization for large matrices
    # Sigma += Sigma.T
    Sigma = Sigma + Sigma.T
    return Sigma[1:,1:]

def fu_moments(n):
    '''
    Calculates first and second moments using Eq. (1) from Fu 1995.
    Note: this has been modified from Rice et al. 2018 to include the variance term.
    '''

    m1 = 1.0 / np.arange(1,n)
    m2 = sigma_fu(n) + m1[:,None]*m1[None,:]
    return m1, m2


if __name__ == "__main__":
    # Bug testing: compare calculation to fu formula for kingman coalescent
    N = 10
    alpha = 2
    M1, M2 = sfs_moments(N, alpha)
    M1_fu, M2_fu = fu_moments(N)
    print('For n = {}:'.format(N))
    if np.allclose(M1_fu, M1):
        print('\tfirst moments equal Fu expectation.')
    else:
        print('\tfirst moments DO NOT equal Fu expectation!')
    if np.allclose(M2_fu, M2):
        print('\tsecond moments equal Fu expectation.')
    else:
        print('\tsecond moments DO NOT equal Fu expectation!')
