import numpy as np
import random
import pylab as plt
from math import *

def standardize(X):
    v = np.var(X, axis = 0)
    a = np.mean(X, axis = 0 )
    st = (X-np.array([a for _ in range(len(X))])) / v
    return st

def gaussian(x, mu, sigma):
    d = x.shape[0]
    det_sig = np.linalg.det(sigma)
    A = 1.0 / (2*np.pi)**(d/2.0) * 1.0 / det_sig**(0.5)
    gs = A * exp(np.dot(np.dot(mu,sigma)),mu)
    return gs

class GMM:
    def __init__(self, clusters, data):
        self.K = clusters
        self.X = standardize(data)
        self.gamma = np.zeros([len(self.X), self.K],dtype = np.float)
        for gm in self.gamma:
            gm[random.randrange(len(gm))] = 1.0
        d = self.X.shape[1]            
        self.mean = np.empty([self.K,d])
        self.cov  = np.empty([self.K,d,d])
        self.pi   = np.empty(self.K)
        self.LL   = np.inf

    def step(self):
        # maximization step
        Nk       = np.sum(self.gamma, axis = 0)
        self.mean = np.dot(self.gamma.T, self.X)
        for (n,x) in enumerate(self.X):
            for (k,mu) in enumerate(self.mean):
                self.cov[k] = self.gamma[n,k]*np.outer((x-mu),(x-mu)) / Nk[k]
        self.pi = Nk / self.X.shape[0]

        # expectation step
        for (k,gm) in enumerate(self.gamma):
            for(n,x) in enumerate(self.X):
                gm[n,k] = pi[k] * gaussian(x,self.mean[k],self.cov[k])

    def solve(self):
        for x in range(1,100):
            self.step()
            LL = self.logL()
            if(LL > self.LL * 0.99):
                break
            else:
                self.LL = LL

    def logL(self):
        ll = 0
        for n in range(self.X.shape[0]):
            tmp = 0
            for k in range(self.pi.shape[0]):
                tmp += self.pi[k] * gaussian(self.X[n], self.mean[k],self.cov[k])
            ll += tmp
        return ll


    def plot(self):
        plt.plot(self.X[:,0][self.gamma[:,0] == 1],self.X[:,1][self.gamma[:,0] == 1],'bo')
        plt.plot(self.X[:,0][self.gamma[:,1] == 1],self.X[:,1][self.gamma[:,1] == 1],'ro')
        plt.show()



if __name__ == '__main__':
    data = np.loadtxt("../dataset/faithful.csv", delimiter=",",skiprows=1,usecols=0)
    gm = GMM(2,data)
    gm.solve()
    gm.plot()