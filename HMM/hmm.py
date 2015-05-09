import numpy as np
import pylab as plt

na = np.newaxis


def gaussian(x, mu, sigma):
    '''
    x: D
    mu: D
    sigma: D * D

    ret: scalar
    '''
    if(len(x.shape) == 1):
        d = x.shape[0]
        det_sig = np.linalg.det(sigma)
        if(det_sig == 0):
            print("det cov marix is 0")
            raise Exception('cov rank')
        A = 1.0 / (2*np.pi)**(d/2.0) * 1.0 / det_sig**(0.5)
        return A * np.exp(np.dot(np.dot(mu, sigma), mu))
    elif(len(x.shape() == 2)):
        d = x.shape[1]
        det_sig = np.linalg.det(sigma)
        A = 1.0 / (2*np.pi)**(d/2.0) * 1.0 / det_sig**(0.5)
        ex = np.exp(np.sum(mu[:, :, na] * sigma * mu[:, na, :], axis=(1, 2)))
        return A * ex
    else:
        raise("X dimension error")


class HMM:
    '''
    HMM implementation in PRML chapter 13
    '''
    def __init__(self, X, means, covs):
        self.X = X
        self.means = means
        self.covs = covs
        self.K = means.shape[0]

        # uniform distribution
        K = self.K
        self.A = np.array([[1 / K**2 for j in range(K)] for i in range(K)])
        self.pi = np.array([1 / K for i in range(K)])

        self.alpha = None
        self.beta = None
        self.likelihood = np.inf

    def Estep(self):
        '''
        execute alpha-beta recursion
        '''
        pi = self.pi
        X = self.X
        means = self.means
        covs = self.covs
        A = self.A
        N = X.shape[0]
        K = self.K

        # alpha recursion
        c = []
        alpha = []
        px_z = np.array([gaussian(X[0, :], mean, cov)
                         for mean, cov in zip(means, covs)])
        _alpha = pi * px_z
        c.append(np.sum(_alpha))
        alpha.append(_alpha / c[-1])

        for n in range(1, N):
            px_z = [gaussian(X[n, :], mean, cov)
                    for mean, cov in zip(means, covs)]
            _alpha = px_z * np.dot(alpha[n-1], A)
            c.append(np.sum(_alpha))
            alpha.append(_alpha / c[-1])

        c = np.array(c)
        self.c = c
        self.alpha = np.array(alpha)

        # beta recursion
        beta = []
        beta_N = np.ones(K)
        beta.append(beta_N)

        for n in reversed(range(0, N-1)):
            px_z = [gaussian(X[n, :], mean, cov)
                    for mean, cov in zip(means, covs)]
            _beta = np.dot(A, px_z * beta[0] / c[n+1])
            beta.insert(0, _beta)

        self.beta = np.array(beta)

        # likelihood function p(X)
        self.likelihood = np.sum(self.c)

    def viterbi(self):
        pi = self.pi
        X = self.X
        means = self.means
        covs = self.covs
        A = self.A
        N = X.shape[0]

        phi = []
        z = []
        px_z = np.array([gaussian(X[0, :], mean, cov)
                         for mean, cov in zip(means, covs)])
        phi.append(pi * px_z)

        for n in range(1, N):
            px_z = [gaussian(X[n, :], mean, cov)
                    for mean, cov in zip(means, covs)]

            _phi = px_z * np.max(phi[n-1][:, na] * A, axis=0)
            phi.append(_phi)
            z.append(np.argmax(_phi))

        self.phi = np.array(phi)
        self.z = np.array(z)

    def Mstep(self):
        '''
        update parameters {A, pi, (means, covs)}
        '''
        alpha = self.alpha
        beta = self.beta
        X = self.X
        N = X.shape[0]
        A = self.A
        means = self.means
        covs = self.covs

        # alpha, beta, gamma: N*K
        # xi: N*K*K
        # X: N*D
        # means: K*D
        # px_z: N*K
        # A: K*K
        gamma = alpha * beta  # ignoe likelihood func
        sum_gamma = np.sum(gamma, axis=0)
        px_z = np.array([[gaussian(X[n, :], mean, cov)
                         for mean, cov in zip(means, covs)]
                         for n in range(N)])
        xi = alpha[:-1, :, na] * px_z[1:, na, :] * \
            A[na, :, :] * beta[1:, na, :]

        means = np.sum(gamma[:, :, na] * X[:, na, :], axis=0) \
            / sum_gamma[:, na]

        # x_mean: N*K*D
        x_mean = X[:, na, :] - means[na, :, :]
        covs = np.sum(gamma[:, :, na, na] *
                      (x_mean[:, :, :, na] * x_mean[:, :, na, :]),
                      axis=0) / sum_gamma[:, na, na]

        pi = gamma[0, :] / np.sum(gamma[0, :])

        xi_sum = np.sum(xi[1:], axis=0)
        A = xi_sum / np.sum(xi_sum, axis=1)[:, na]

        self.means = means
        self.covs = covs
        self.pi = pi
        self.A = A

    def learn(self):
        for _ in range(10):
            self.Estep()
            self.Mstep()

        self.viterbi()

    def check_likelihood(self):
        alpha = self.alpha
        beta = self.beta
        N = self.X.shape[0]
        c = self.c

        L_alpha = np.sum(alpha[N-1, :])
        L_beta = np.sum(beta[0, :] * alpha[0, :])
        L_c = np.prod(c)

        print('Liklihood')
        print('alpha:', L_alpha)
        print('beta:', L_beta)
        print('c:', L_c)

def main():
    K = 20
    N = 200
    D = 3

    X = np.random.rand(N, D)

    # parameters should be calcurated with K-means or somethig
    # but use it this time
    means = np.mean(X.reshape(K, N/K, D), axis=1)
    cov = np.cov(X.T)
    covs = np.array([cov for _ in range(K)])

    hmm = HMM(X, means, covs)

    hmm.learn()

    hmm.check_likelihood()

if(__name__ == '__main__'):
    main()
