import numpy as np
from scipy.stats import poisson

np.random.seed(19)

N = 100

# Actual parameters that we do not know
# The EM Algorithm is trying to estimate these values
Pi1 = 0.2
Pi2 = 0.8
Lambda1 = 2
Lambda2 = 8

Z1 = np.random.poisson(Lambda1, round(N * Pi1))
Z2 = np.random.poisson(Lambda2, round(N * Pi2))
X = np.concatenate((Z1, Z2))


#def Poisson(x, l):
#poisson.pmf(x,l)
def P(X,l):
    # returns n by k matrix of poisson values.
    # lets you put lambdas in as an array
    pmf = np.zeros((np.shape(X)[0],np.shape(l)[0]))
    for k in range(K):
        pmf[:,k] = poisson.pmf(X,l[k])
    return pmf
        
    

# number of cluster components
K = 2

# initialize lambda and pi
def initialize_params(K):
    lambdas = np.random.rand(K)
    pis = np.array([1/K]*K)
    return lambdas, pis

# E step
def Estep(X, lambdas, pis):
    # returns n by k matrix of posterior probabiltiies
    gammaz = np.zeros((X.shape[0],lambdas.shape[0]))
    pmf = P(X,lambdas)
    for k in range(K):
        gammaz[:,k] = np.multiply(pis[k],pmf[:,k])
    d = np.sum(gammaz,axis=1)
    return np.divide(gammaz.T,d).T # this is p(k|l)


# M Step
def MStep(gammaz,X, lambdas):
    N = np.shape(gammaz)[0]
    N_k = np.sum(gammaz,axis=0)
    newlambdas = np.divide(np.sum(np.multiply(gammaz.T,X).T,axis=0),N_k)#shape(k,)
    newpis = np.divide(N_k,N)
    return newlambdas, newpis

def ExpMax(X,K,iterations):
    lambdas, pis = initialize_params(K)
    for s in range(10000):
        gamma = Estep(X,lambdas,pis)
        lambdas, pis = MStep(gamma,X,lambdas)
    return gamma, lambdas, pis

# Evaluate Log likelihood
def loglik(X,gammaz,lambdas,pis):
    q=np.multiply(pis,P(X,lambdas)) # this is q(k,l)
    Q = np.multiply(gammaz,np.log(q))
    return np.sum(Q)

z, l, p = ExpMax(X,2,5)



ans = loglik(X,z,l,p)

