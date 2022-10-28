import numpy as np

import scipy.spatial.distance as dist
import matplotlib.pyplot as plt


########CONSTANTS ###########
# s - number of samples 
# o - number of otu
# k - number of latent dimensions
###############################


##############CALCULATE METRICS BASED OFF OF 1 DATA SET
def calcDistanceDist(dat):
    #Input: dat (s x o) relative abundance matrix
    #Output: brayCurtis ((s^2 - s)/2 x 1) vector with Bray Curtis distance between
    #any two samples
    s = np.shape(dat)[0]
    
    brayCurtis = []
    for sample1 in range(s):
        for sample2 in range(s):
            if sample2 > sample1:
                brayCurtis.append(dist.braycurtis(dat[sample1, :], dat[sample2, :]))
    
    return np.asarray(brayCurtis)
def calcClosestNeighbor(dat):
    #Input: dat (s x o) relative abundance matrix
    #Output: closestNeighbor (s x 1) vector containing the distance between a given sample s 
    #and its closest neighbor
    s = np.shape(dat)[0]
    closestNeighbor = []
    for sample1 in range(s):
        sampleDistances = np.ones(s)
        for sample2 in range(s):    
            if sample2 != sample1:
                #print(np.linalg.norm(dat[sample1, :] - dat[sample2, :], ord = 2))
                sampleDistances[sample1] = dist.braycurtis(dat[sample1, :], dat[sample2, :])
        closestNeighbor.append(np.min(sampleDistances))
    #plt.
    #print(np.shape(closestNeighbor))
    #print(closestNeighbor)
   # exit()

    return np.asarray(closestNeighbor)
def calcOTUAverage(dat):
    #Input: dat (s x o) relative abundance matrix
    #Output: average (o x 1) average OTU abundance across samples
    return np.mean(dat, axis = 0)
def calcOTUVariance(dat):
    #Input: dat (s x o) relative abundance matrix
    #Output: var (o x 1) OTU variance across samples
    return np.var(dat, axis = 0)
def calcCovariance(dat):
    #Input: dat (s x o) relative abundance matrix
    #Output: cov (o x o) covariance matrix 
    return np.cov(dat, rowvar=False)
def calcTaylorsLaw(dat):
    #Input: dat (s x o) relative abundance matrix
    #Output: mean, var (both o x 1) 
    mean = calcOTUAverage(dat)
    variance = calcOTUVariance(dat)

    return mean, variance
def calcAFD(dat):
    #abundance fluctuation distribution
    return 0
    return 0
def calcMAD(dat):
    #Input: dat (s x o) relative abundance matrix
    #Output: mean abundance distribution (o x 1) functionally same as mean

    return calcOTUAverage(dat)
def calcSAD(dat):
    #species abundance distribution

    return 0
def calcOccupancyDist(dat):
    return 0
def calcShannon(dat):
#input: dat (s x o) relative abu matrix
#output: shannon (s x 1) diversities for each sample
    shannon = np.zeros(np.shape(dat))
    for s in range(np.shape(dat)[0]):
        for o in range(np.shape(dat)[1]):
            if dat[s, o] != 0:
                shannon[s, o] = - dat[s, o] *np.log(dat[s, o])
    shannon = np.sum(shannon, axis = 1)
    print(np.shape(shannon))
    return shannon
def calcSpeciesDist(fdat, rdat):
    #Same as SAD
    #Accepts count data (s x o)
    #calculates a distribution across samples for each otu
    #otu number of distributions. 
    #returns: hist for each otu, return histogram of distribution (o x bin number)
    speciesDist = []
    bins = []       
    for o in range(np.shape(fdat)[1]):
        minbin = np.min([np.min(fdat[:, o]), np.min(rdat[:, o])])
        maxbin = np.max([np.max(fdat[:, o]), np.max(rdat[:, o])])
        bins = np.linspace(minbin, maxbin, 15)
        bins = np.logspace(-4, 0, 15)
        fhist, fbin = np.histogram(fdat[:, o], bins, density = True)
        rhist, rbin = np.histogram(rdat[:, o], bins, density = True)

        realCum = np.cumsum(rhist)/np.sum(rhist)
        fakeCum = np.cumsum(fhist)/np.sum(fhist)
        print(np.shape(realCum))
        #print(realHist)
        #plt.ylim([0, 1])
        plt.xscale('log')
        plt.plot(bins[1:], realCum, 'ro')
        plt.plot(bins[1:], fakeCum, 'bo')
        plt.legend(["real", "fake"])
        plt.show()
        # exit()
        # rhist = rhist/np.sum(rhist)
        # fhist = fhist/np.sum(fhist)
        # plt.plot(fbin[1:], fhist, 'ro')
        # plt.plot(rbin[1:], rhist, 'bo')
        # plt.legend('')
        # plt.xlabel('Relative Abundance')
        # plt.ylabel('Probability')
        # plt.title('otu: ' + str(o))
        # plt.show()

    return 0
def calcQ(mu, sigma, theta, naverage = 10000):
#Input: mu (s x k), sigma (s x k), theta (k x o), naverage 
#Output: q (s x o) reconstructed training data set. 
#Function not in use 
    q = np.zeros((np.shape(mu)[0], np.shape(theta)[1]))

    for i in range(naverage):
        z = np.random.normal(mu, sigma)
        Q = np.exp(-1 * np.matmul(z, theta))
        Q = Q/np.linalg.norm(Q, ord=1, axis=1, keepdims=True)
        q += Q
    q = q / naverage
    return q



###############COMPARES METRICS FOR FAKE (fdat) and REAL (rdat) DATA SETS############
def compareTaylorsLaw(rdat, fdat):
#Input: real and fake data (rdat and fdat) (s x o)
#output:plots taylors law for real and fake data on same graph

    #calc taylors law
    fmean, fvar = calcTaylorsLaw(fdat)
    rmean, rvar = calcTaylorsLaw(rdat)

    #plots
    plt.plot(fmean, fvar, 'ro')
    plt.plot(rmean, rvar, 'bo')
    plt.legend(['Fake', 'Real'])
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Taylors Law")
    plt.xlabel("Mean")
    plt.ylabel("Var")
    plt.show()
    return 0
def compareMAD(rdat, fdat):
#Input: real and fake data (rdat and fdat) (s x o)
#output: plots Mean abu distribution for fake and real data

    #calc mean abu distribution
    realDist = calcMAD(rdat)
    fakeDist = calcMAD(fdat)
    
    #Create histogram and cdf for MAD
    bin = np.logspace(-4, 0, 15)
    realHist = np.histogram(realDist, bins = bin, density = True)[0]
    fakeHist = np.histogram(fakeDist, bins = bin, density = True)[0]
    realCum = np.cumsum(realHist)/np.sum(realHist)
    fakeCum = np.cumsum(fakeHist)/np.sum(fakeHist)
   
    #plots
    plt.xscale('log')
    plt.plot(bin[1:], realCum, 'ro')
    plt.plot(bin[1:], fakeCum, 'bo')
    plt.legend(["real", "fake"])
    plt.title("Mean Abu Distribution")
    plt.show()

    return 0
def compareAFD(rdat, fdat):
    return 0
def compareClosestNeighbor(rdat, fdat):
#Input: real and fake data (rdat and fdat) (s x o)
#output: plots distribution of distance to closes neighbor for real and fake data
    realDist = calcClosestNeighbor(rdat)
    fakeDist = calcClosestNeighbor(fdat)
    
    #Create histogram
    bin = np.linspace(0, 1, 30)
    realHist = np.histogram(realDist, bins = bin, density = True)[0]
    fakeHist = np.histogram(fakeDist, bins = bin, density = True)[0]
    #normalize
    realHist = realHist * (bin[1] - bin[0])
    fakeHist = fakeHist * (bin[1] - bin[0])

    #Create CDF
    realCum = np.cumsum(realHist)/np.sum(realHist)
    fakeCum = np.cumsum(fakeHist)/np.sum(fakeHist)



    #plots
    plt.plot(bin[1:], realHist, 'ro')
    plt.plot(bin[1:], fakeHist, 'bo')

    #Uncomment for CDF
    # plt.plot(bin[1:], realCum, 'ro')
    # plt.plot(bin[1:], fakeCum, 'bo')
    plt.legend(["real", "fake"])
    plt.title("Distance to Closest Neighbor Probability Dist")
    plt.show()
    return 0
def compareDistanceDist(rdat, fdat):
    #Input: real and fake data (rdat and fdat) (s x o)
    #output: distribution of BC distance to from given sample to all other samples
    realDist = calcDistanceDist(rdat)
    fakeDist = calcDistanceDist(fdat)
    bin = np.linspace(0, 1, 30)
    
    #Create histogram
    realHist = np.histogram(realDist, bins = bin, density = True)[0]
    fakeHist = np.histogram(fakeDist, bins = bin, density = True)[0]

    realHist = realHist * (bin[1] - bin[0])
    fakeHist = fakeHist * (bin[1] - bin[0])

    #Create CDF
    realCum = np.cumsum(realHist)/np.sum(realHist)
    fakeCum = np.cumsum(fakeHist)/np.sum(fakeHist)
  
    #plots
    plt.plot(bin[1:], realHist, 'ro')
    plt.plot(bin[1:], fakeHist, 'bo')
    #Uncomment for CDF
    # plt.plot(bin[1:], realCum, 'ro')
    # plt.plot(bin[1:], fakeCum, 'bo')
    plt.legend(["real", "fake"])
    plt.title("BC Distance between Two Samples")
    plt.show()
    return 0
def compareShannon(rdat, fdat):
#Input: real and fake data (rdat and fdat) (s x o)
#output: plots shannon diversity distribution for fake and real data
    realDist = calcShannon(rdat)
    fakeDist = calcShannon(fdat)

    #Create histogram
    minbin = np.min([np.min(fakeDist), np.min(realDist)])
    maxbin = np.max([np.max(fakeDist), np.max(realDist)])
    bin = np.linspace(minbin, maxbin, 15)
    realHist = np.histogram(realDist, bins = bin, density = True)[0]
    fakeHist = np.histogram(fakeDist, bins = bin, density = True)[0]

    realHist = realHist * (bin[1] - bin[0])
    fakeHist = fakeHist * (bin[1] - bin[0])

    #plots
    plt.plot(bin[1:], realHist, 'ro')
    plt.plot(bin[1:], fakeHist, 'bo')
 
    plt.legend(["real", "fake"])
    plt.title("Shannon Diversity Distribution")
    plt.show()
    return 0


def generateData(mu, sigma, theta, nsample = 772):

    #Input: mu (s x k), sigma (s x k), theta (k x o), nsamples 
    #output: fdat (nsamples x o)
    #generates fake data based on learned latent space distribution (mu, sigma) and theta


    fdat = np.zeros((nsample, np.shape(theta)[1]))

    numClust = np.shape(mu)[0]
    for s in range(nsample):
        clust = np.random.randint(0, numClust)
        z = np.random.normal(mu[clust], sigma[clust])
        Q = np.exp(-1 * np.matmul(z, theta))
        Q = Q[None, :]
        
        Q = Q/np.linalg.norm(Q, ord=1, axis=1, keepdims=True)
        fdat[s] = Q

 
  
    return fdat
def getData():

    thetaFile = "theta.csv"
    muFile = "mu.csv"
    sigmaFile = "sigma.csv"
    rdat = np.genfromtxt("ndat.csv", delimiter = ',')
    #rdat = rdat[0:100]
    MU = np.genfromtxt(muFile, delimiter = ',')
    SIGMA = np.genfromtxt(sigmaFile, delimiter = ',')
    THETA = np.genfromtxt(thetaFile, delimiter = ',')
    fdat = generateData(MU, SIGMA, THETA)
    return fdat, rdat



def main():
    #Returns ndat.csv as matrix as well as fake data with 
    FDAT, RDAT = getData()
    
    compareShannon(RDAT, FDAT)
    
    calcSpeciesDist(FDAT, RDAT)
    
    compareMAD(RDAT, FDAT)

    compareClosestNeighbor(RDAT, FDAT)

    compareDistanceDist(RDAT, FDAT)
    return 0
if __name__ == '__main__':
    
   
    main()  
