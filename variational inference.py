

import numpy as np
import multiprocessing
import tensorflow as tf





####################### CONSTANTS ##############################
#s - number of samples
#k - number of latent dimensions
#o - number of otus

####################### VARIABLES ###############################
#                                                    dimensions
#MU - mean of latent distribution, (z) z ~ Normal --- (s x k) 
#SIGMA - std dev of z                             --- (s x k)
#THETA - loading matrix                           --- (k x o)


##########################################################
def calcPI(THETA, Z):
#calculates gibbs boltzmann distribution for a given theta and z matrix
#inputs: MU(s x k) mean of Z distribution, SIGMA (s x k) std dev of Z distribution, THETA (k x o) loading matrix
#outputs: PI (s x o), representing predicted abundances

    pi = np.exp(-1 * tf.linalg.matmul(Z, THETA))
    pi = pi/np.linalg.norm(pi, ord=1, axis=2, keepdims=True)
    #axis = 2 for parallel processing. Want to average over otu axis
    return pi

def calcX(n, N):
#calculates relative abundance X for probabilities n and N total counts 
#inputs: n (s x o) probabilities for each otu in each sample, N (float) (total counts)
#outpus: x (s x o) relative abundances
    return np.asarray(calcN(n, N) / N)
def calcN(Q, N):
#calculate multinomial distribution for N counts and proab
#inputs: Q (s x o), N (float)
#outputs: (s x o) count matrix. Each row sums to N
    return np.asarray([np.random.multinomial(N, Q[i]) for i in range(np.shape(Q)[0])])
 

def dmuAvg(X, THETA, MU, SIGMA, EPS):
#calculates the expectation of dmu. The number of draws from the distribution is np.shape(EPS)[0]
#inputs: X (s x o) training data of relative abu, THETA (k x o) loading matrix, SIGMA (s x k) std dev of z, MU (s x k) mean of z
#EPS (s x k) random gaussian noise. This is used to ensure the same noise is used in all derivatives for a given step
#outputs: dL/dmu (s x k) (expectation over z distribution)


    #Parallel calculation of dmu np.shape(EPS)[0] number of times
    deltaMU = pool.starmap(dmu, [(X, THETA, MU, SIGMA, EPS)])
    
    deltaMU = np.average(deltaMU, axis = 1)[0]

    return deltaMU
def dmu(X, THETA, MU, SIGMA, EPS):
#calculates derivative of likelihood wrt mu, not including KL divergence term
#inputs: X (s x o) - real data, THETA (k x o), MU (s x k), SIGMA (s x k), EPS (s x k) 
#EPS represents normal distribution (s, k)
#outputs: dL/dmu (s x k) (for single z matrix)


    z = MU + SIGMA * EPS
    print(z)
    dm = -1 * tf.linalg.matmul(X, np.transpose(THETA)) + tf.linalg.matmul(calcPI(THETA, z), np.transpose(THETA))

    return dm

def dsigmaAvg(X, THETA, MU, SIGMA, EPS):
#calculates the expectation of dsigma. The number of draws from the distribution is np.shape(EPS)[0]
#inputs: X (s x o) training data of relative abu, THETA (k x o) loading matrix, SIGMA (s x k) std dev of z, MU (s x k) mean of z
#EPS (s x k) random gaussian noise. This is used to ensure the same noise is used in all derivatives for a given step
#outputs: dL/dsigma (s x k) (expectation over z distribution)

    deltaSIGMA = pool.starmap(dsigma, [(X, THETA, MU, SIGMA, EPS)])
    deltaSIGMA = np.average(deltaSIGMA, axis = 1)[0]
   
    return deltaSIGMA

def dsigma(X, THETA, MU, SIGMA, EPS):
#calculates derivative of likelihood wrt sigma, not including KL divergence term
#inputs: X (s x o) - real data, THETA (k x o), MU (s x k), SIGMA (s x k), EPS (s x k) 
#EPS represents normal distribution (s, k)
#outputs: dL/dsigma (s x k) (for single z matrix)
    
        
    z = MU + EPS * SIGMA
    ds = -1 * EPS * tf.linalg.matmul(X, np.transpose(THETA)) + EPS * tf.linalg.matmul(calcPI(THETA, z), np.transpose(THETA))#TERM1
    
    return ds

def dthetaAvg(X, THETA, MU, SIGMA, EPS):
#calculates the expectation of dtheta. The number of draws from the distribution is np.shape(EPS)[0]
#inputs: X (s x o) training data of relative abu, THETA (k x o) loading matrix, SIGMA (s x k) std dev of z, MU (s x k) mean of z
#EPS (s x k) random gaussian noise. This is used to ensure the same noise is used in all derivatives for a given step
#outputs: dL/dtheta (k x o) (expectation over z distribution)


    deltaTHETA = pool.starmap(dtheta, [(X, THETA, MU, SIGMA, EPS)])
    deltaTHETA = np.average(deltaTHETA, axis = 1)[0]

    return deltaTHETA
def dtheta(X, THETA, MU, SIGMA, EPS):
#calculates derivative of likelihood wrt theta, not including KL divergence term
#inputs: X (s x o) - real data, THETA (k x o), MU (s x k), SIGMA (s x k), EPS (s x k) 
#EPS represents normal distribution (s, k)
#outputs: dL/dtheta (k x o) (for single z matrix)
    
    z = MU + SIGMA * EPS
   
    #reshape Z so that parallel processing can work properly
    ztranspose = z.reshape(np.shape(z)[0], np.shape(z)[2], np.shape(z)[1])
    
    dt = -1 * tf.linalg.matmul(ztranspose, X) + tf.linalg.matmul(ztranspose, calcPI(THETA, z))
    
    return dt

def dmuKL(MU):
#calculate dL/dmu wrt KL divergence term
#inputs: MU (s x k)
#outputs: dKL/dmu (s x k)
    return -1*MU
def dsigmaKL(SIGMA):
#calculate dL/dsigma wrt KL divergence term
#inputs: SIGMA (s x k)
#outputs: dKL/dsigma (s x k)
    return -1*(SIGMA - (1 / SIGMA))

def initialize(DATA, k):
#inialize theta, mu, sigma matrices with proper shapes from data and
#given number of latent dimensions, k
#inputs: data (s x o) training dataset of relative abundances
#outputs: theta (k x o), mu (s x k), sigma (s x k) all initialized 
    S = np.shape(DATA)[0]
    
    o = np.shape(DATA)[1]

    theta = np.random.rand(k, o)
    mu = np.zeros((S, k))
    sigma = np.ones((S, k))
    return theta, mu, sigma

def train(ndat, k = 5, BETA = 1, N = 10000, eta = 10**-4, naverage = 100, maxiter = 50000):
#Perform training on ndat, the training data set
#Hyperparameters: 
#k - number of latent dimensions
#BETA - relative weight of KL divergence compared to likelihood in loss function
#N - total number of counts from which we do multinomial sampling of data
#eta - step size 
#naverage - number of draws from distribution to calculate expectation over z
#maxiter - max iterations before convergence

    #Convert training dataset to multnomial counts and calculate relative abu
    xdat = calcX(ndat, N)

    #initialize variables
    theta, mu, sigma = initialize(ndat, k)

    #declare step sizes for all variables to be the same
    eta = 10**-4
    etam = eta
    etas = eta
    etat = eta

    #Declare convergence status and iteration number
    converged = False
    iter = 0

    while(converged == False):
        
 
        #draw random gaussian noise
        eps = np.random.normal(size = (naverage, np.shape(mu)[0], np.shape(mu)[1]))

        #Calculate derivates wrt all terms in likelihood function and all variables
        delm = dmuAvg(xdat, theta, mu, sigma, eps)
        dels = dsigmaAvg(xdat, theta, mu, sigma, eps)
        delt = dthetaAvg(xdat, theta, mu, sigma, eps)
        delmKL = dmuKL(mu)
        delsKL = dsigmaKL(sigma)


        #update variables
        mu = mu + etam * (delm + BETA * delmKL)
        sigma = sigma + etas * (dels + BETA * delsKL)
        theta = theta + etat * delt
        
        iter += 1
        #Convergence condition
        if iter > maxiter:#Or add alternative convergence condition
            converged = True
        

    return theta, mu, sigma

if __name__ == '__main__':
    
    #Set up paralle processing
    pool = multiprocessing.Pool()
    #Generate training data
    data = np.genfromtxt('ndat.csv', delimiter = ',')
    theta, mu, sigma = train(data)
    
    #save learned matrices as csv files
    np.savetxt("mu.csv", mu, delimiter = ',')
    np.savetxt("sigma.csv", sigma, delimiter = ',')
    np.savetxt("theta.csv", theta, delimiter = ',')



