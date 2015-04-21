import numpy as np
import random
import pylab as plt

def standardize(X):
    v = np.var(X, axis = 0)
    a = np.mean(X, axis = 0 )
    st = (X-np.array([a for _ in range(len(X))])) / v
    return st

class Kmeans:
    def __init__(self, clusters, dimension, data):
        self.K = clusters
        self.D = dimension
        self.X = standardize(data)
        self.gamma = np.zeros([len(self.X), self.K],dtype = np.int)
        self.means = np.array([self.X[random.randrange(len(self.X))] for r in range(self.K)])
        self.J = np.inf

    def step(self):
        # expectation step
        dists = np.array([[np.linalg.norm(a-b) for a in self.means] for b in self.X])
        self.gamma = np.zeros([len(self.X), self.K], dtype = np.int)
        for x,y in zip(range(len(self.X)),np.argmin(dists, axis = 1)):
            self.gamma[x,y] = 1

        #maximization step
        self.means = (np.dot(self.gamma.T, self.X).T / np.sum(self.gamma, axis = 0)).T

    def solve(self):
        for x in range(1,100):
            self.step()
            j = self.distortion()
            if(j > self.J * 0.99):
                break
            else:
                self.J = j

    def plot(self):
        plt.plot(self.X[:,0][self.gamma[:,0] == 1],self.X[:,1][self.gamma[:,0] == 1],'bo')
        plt.plot(self.X[:,0][self.gamma[:,1] == 1],self.X[:,1][self.gamma[:,1] == 1],'ro')
        plt.show()

    def distortion(self):
        dists = np.array([[np.linalg.norm(a-b) for a in self.means] for b in self.X])
        return np.sum(self.gamma * dists)



if __name__ == '__main__':
    data = np.loadtxt("../dataset/faithful.csv", delimiter=",",skiprows=1,usecols=(1,2))
    k = Kmeans(2,2,data)
    k.solve()
    k.plot()