# -*- coding: utf-8 -*-
"""
Utilities necessary to analyze and understand the distribution of individuals in a branching process with a power-law distributed number of offspring.
"""
import numpy as np
import scipy as sp
from scipy.special import erfc
from scipy.special import gamma
import powerlaw as pl
from scipy.integrate import quad
def exact_complementary_cumulative(y, alpha):
    """Complementary cumulative distribution at long times for the number of individual divided by the mean number of individuals at this time over the surviving realizations. y is an ndarray of points, alpha is the tail of the offspring distribution. alpha='short' i for short tails. alpha='half' is for alpha=0.5. No other solutions are knows so far."""
    if alpha == 'short':
        return np.exp(-y)
    if alpha == 'half':
        return (1+2*y)*np.exp(y)*erfc(np.sqrt(y)) - 2*np.sqrt(y/np.pi)
    
def exact_moment_generating_function_it(s, alpha):
    """Inifinite-time limit of the scaled variable y/<y> over the surviving lineages. Moment generating function is E{e^(-sX)}, i.e. the Laplace transform of the probability distribution."""
    return 1-s/(1+s**alpha)**(1/alpha)

def exact_cumulative_laplace_it(s, alpha):
    """Inifinite-time limit of the scaled variable y/<y> over the surviving lineages. Laplace transform of the complementary cumilative distribution."""
    return 1/(1+s**alpha)**(1/alpha)

def theory_complementary_cumulative_it(y, alpha):
    """Inifinite-time limit of the scaled variable y/<y> over the surviving lineages. Moment generating function is E{e^(-sX)}, i.e. the Laplace transform of the probability distribution."""
    def integrand(x, y, alpha):
        d = np.sqrt(y**(2*alpha)+x**(2*alpha)+2*np.cos(np.pi*alpha)*(x*y)**alpha)
        return (1/np.pi)*np.exp(-x)*np.sin(np.arccos((y**alpha+x**alpha*np.cos(np.pi*alpha))/d)/alpha)*(d)**(-1/alpha)        
    integral = np.zeros(y.shape)
    for i in range(y.size):
        temp = quad(integrand, 0, +np.inf, args=(y[i],alpha))
        integral[i] = temp[0]
    return integral

def theory_complementary_cumulative(y, t, alpha, g=1):
    """Finite-time limit of the scaled variable y/<y> over the surviving lineages. Moment generating function is E{e^(-sX)}, i.e. the Laplace transform of the probability distribution. y is an array, but t and alpha are constants."""
    def integrand(x, y, t, alpha, g):
        s = (1+alpha*g*t)**(-1/alpha) # survival probability
        sa = 1-s**alpha # occurs often
        xa = x**alpha # occurs often
        sx = x*s # occurs often
        esx = (np.exp(sx)-1)/sx # occurs often
        esxa = esx**alpha
        d = np.sqrt(1 + (xa*esx*sa)**2 + 2*np.cos(np.pi*alpha)*xa*esx*sa)
        return (1/np.pi)*np.exp(-x*y)*esx*np.sin(np.arccos((1+np.cos(np.pi*alpha)*xa*esxa*sa)/d)/alpha)*(d)**(-1/alpha)        
    integral = np.zeros(y.shape)
    for i in range(y.size):
        temp = quad(integrand, 0, +np.inf, args=(y[i], t, alpha, g))
        integral[i] = temp[0]
    return integral

def theory_Pn(t, alpha, N=10**3, g=1):
    """ Computes the probabilities for P(t,n) by expanding the generating function in Taylor series using integration in the complex plain. Returns an array for Pn starting with n=0. P0=1-S."""
    s = (1+alpha*g*t)**(-1/alpha) # survival probability
    Pn = np.zeros(N)
    Pn[0] = 1-s
    def integrand(p, n, alpha, t, g, r=1.0):
        z = r*(np.cos(p)+1j*np.sin(p))
        return np.real(-0.5/np.pi*r**(-n)*(np.cos(n*p)-1j*np.sin(n*p))*(1-z)*(1+alpha*g*t*(1-z)**alpha)**(-1/alpha))
    for i in range(N-1):
        temp = quad(integrand, 0, 2*np.pi, args=(i+1, alpha, t, g))
        Pn[i+1] = temp[0]
    return Pn

def theory_complementary_cumulative_it_small_y(y, alpha):
    """Infinite time limit, small population size asymptotics."""
    return 1-y**(alpha)/alpha/gamma(1+alpha)
        
def theory_complementary_cumulative_it_large_y(y, alpha):
    """Infinite time limit, large population size asymptotics."""
    return y**(-1-alpha)/gamma(1-alpha)    


class random:
    """ Provides random variables with power law distribution. Not perfect, but works. The mean number of offspring is set to ak, which is for now always 1. """
    def __init__(self, distribution='power law', alpha=0.5, xmin=1, burnin_average=10**8, ak = 1):
        assert distribution=='power law'
        if distribution=='power law':
            self.rnd = pl.Power_Law(xmin = xmin, parameters = [2+alpha], discrete=True)
        self.burnin_average = burnin_average
        average_k = np.mean(self.rnd.generate_random(burnin_average))
        self.pz = 1-1/average_k # probability of no offspring
    def __call__(self, N=1):
        k = self.rnd.generate_random(N)
        o = np.random.random(N)>self.pz
        return k*o
    
class bp:
    """ Simulates branching process starting from 1 individual. """
    def __init__(self, Nr=2*10**8, Times = [0, 10, 25], distribution='power law', alpha=0.5, burnin_average=10**8, ak=1):
        """ Times must include zero."""
        self.rand = random(distribution=distribution, alpha=alpha, burnin_average=burnin_average, ak=ak)
        self.n = np.zeros((len(Times), Nr))
        self.n[0,:] = 1 # initialization
        self.Nr = Nr
        self.Times = Times
    def simulate_one(self, Times):
        t = 0
        n = 1
        counter = 1
        nt = np.zeros(len(Times))
        nt[0] = n
        while n>0 and t<Times[-1]:
            t += np.random.exponential(1.0/n, size=1)
            k = self.rand()
            n += k-1
            if t>Times[counter]:
                nt[counter] = n
                counter += 1
        return nt
    def simulate(self):
        for i in range(self.Nr):
            self.n[:,i] = self.simulate_one(self.Times)
        return True
