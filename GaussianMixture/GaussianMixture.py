import numpy as np
import random
import pylab as plt
from sklearn.utils import extmath
from sklearn.cluster import KMeans
import sys


na = np.newaxis


class DataFormatter:
    def __init__(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def standarize(self, X):
        return (X - self.mean[na, :]) / self.std[na, :]


def log_gaussian(X, mu, cov):
    d = X.shape[1]
    det_sig = np.linalg.det(cov)
    A = 1.0 / (2*np.pi)**(d/2.0) * 1.0 / det_sig**(0.5)
    x_mu = X - mu[na, :]
    inv_cov = np.linalg.inv(cov)
    ex = - 0.5 * np.sum(x_mu[:, :, na] * inv_cov[na, :, :] *
                        x_mu[:, na, :], axis=(1, 2))
    return np.log(A) + ex


class GMM:
    def __init__(self,
                 K=2,
                 max_iter=300,
                 diag=False):
        self.K = K
        self.data_form = None
        self.pi = None
        self.mean = None
        self.cov = None
        self.max_iter = max_iter
        self.diag = diag

    def fit(self, _X):
        self.data_form = DataFormatter(_X)
        X = self.data_form.standarize(_X)
        N = X.shape[0]
        D = X.shape[1]
        K = self.K

        # init parameters using K-means
        kmeans = KMeans(n_clusters=self.K)

        kmeans.fit(X)

        self.mean = kmeans.cluster_centers_

        self.cov = np.array([[[1 if i == j else 0
                             for i in range(D)]
                             for j in range(D)]
                             for k in range(K)])

        self.pi = np.ones(K) / K

        # Optimization
        for _ in range(self.max_iter):
            # E-step

            gam_nk = self._gam(X)

            # M-step
            Nk = np.sum(gam_nk, axis=0)

            self.pi = Nk / N

            self.mean = np.sum(gam_nk[:, :, na] * X[:, na, :],
                               axis=0) / Nk[:, na]

            x_mu_nkd = X[:, na, :] - self.mean[na, :, :]

            self.cov = np.sum(gam_nk[:, :, na, na] *
                              x_mu_nkd[:, :, :, na] *
                              x_mu_nkd[:, :, na, :],
                              axis=0) / Nk[:, na, na]

            if(self.diag):
                for k in range(K):
                    var = np.diag(self.cov[k])
                    self.cov[k] = np.array([[var[i] if i == j else 0
                                             for i in range(D)]
                                            for j in range(D)])

    def _gam(self, X):
        log_gs_nk = np.array([log_gaussian(X, self.mean[i], self.cov[i])
                              for i in range(self.K)]).T

        log_pi_gs_nk = np.log(self.pi)[na, :] + log_gs_nk

        log_gam_nk = log_pi_gs_nk[:, :] - extmath.logsumexp(log_pi_gs_nk, axis=1)[:, na]

        return np.exp(log_gam_nk)

    def predict(self, _X):
        X = self.data_form.standarize(_X)

        gam_nk = self._gam(X)

        return np.argmax(gam_nk, axis=1)


def experiment(max_iter, diag):
    train_data = np.loadtxt("../dataset/vowel/train.txt", delimiter=" ")
    test_data = np.loadtxt("../dataset/vowel/test.txt", delimiter=" ")

    K = 5

    gmm = GMM(K=K, max_iter=max_iter, diag=diag)

    X = train_data[:, :-1]

    gmm.fit(X)

    X_t = test_data[:, :-1]
    T = test_data[:, -1]

    Y_t = gmm.predict(X_t)

    for i in range(K):
        idx = Y_t == i
        plt.plot(X_t[idx, 0], X_t[idx, 1], 'o')

    plt.xlim(100, 900)
    plt.ylim(700, 2500)
    plt.savefig('result.png')

    Nk = np.zeros(K)
    err_rate = np.zeros(K)

    idx = 0
    # assume errors are not more than the half
    for i in range(K):
        Nk[i] = T[T == i+1].shape[0]
        Y = Y_t[idx:idx+Nk[i]]
        counts = np.bincount(Y)
        most_freq = np.max(counts)
        errs = Nk[i] - most_freq
        err_rate[i] = errs / Nk[i]
        idx += Nk[i]

    return np.mean(err_rate)


def main():
    iters = np.arange(1, 50, 2)

    err_diag = np.zeros_like(iters, dtype='Float64')
    for i, it in enumerate(iters):
        err_diag[i] = experiment(max_iter=it, diag=True)

    err_full = np.zeros_like(iters, dtype='Float64')
    for i, it in enumerate(iters):
        err_full[i] = experiment(max_iter=it, diag=False)

    plt.figure()
    plt.ylim(0, 10)
    plt.title('Error rate')
    plt.plot(iters, err_diag * 100, '-', label='diagonal')
    plt.plot(iters, err_full * 100, '-', label='full')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('error rate[%]')
    plt.savefig('err.png')


if __name__ == '__main__':
    main()
