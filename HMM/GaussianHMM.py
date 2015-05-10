import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.cluster import KMeans
import sys
import warnings

warnings.simplefilter("error", RuntimeWarning)
na = np.newaxis
sr = sys.stdin.read

def gaussian(x, mu, sigma, fmt='normal'):
    '''
    x: D
    mu: D
    sigma: D * D

    ret: scalar
    '''
    d = x.shape[0]
    det_sig = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    if(det_sig == 0):
        print("det cov marix is 0")
        raise Exception('cov rank')
    if(fmt == 'normal'):
        A = 1.0 / (2*np.pi)**(d/2.0) * 1.0 / det_sig**(0.5)
        return A * np.exp(-1 / 2 * np.dot(np.dot(x - mu, inv_sigma), x - mu))
    elif(fmt == 'log'):
        ret = 0
        ret += np.log(1.0 / (2*np.pi)**(d/2.0))
        ret += np.log(1.0 / det_sig**(0.5))
        ret += -1 / 2 * np.dot(np.dot(x - mu, inv_sigma), x - mu)
        return ret
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
        self.A = np.array([[1 / K for j in range(K)] for i in range(K)])
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

            try:
                alpha.append(_alpha / c[-1])
            except RuntimeWarning:
                print('')
                print('c:', c[-1])
                print('n:', n)
                print('alpha:', _alpha)
                print('alpha[n-1]', alpha[n-1])
                print('px_z:', px_z)
                print('X[n]', X[n])
                print('means', means)
                print('covs', covs)
                raise

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
            if(px_z == float("nan")):
                print(c[n+1])
                raise
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
        px_z = np.array([gaussian(X[0, :], mean, cov, fmt='log')
                         for mean, cov in zip(means, covs)])

        phi.append(np.log(pi) + px_z)

        for n in range(1, N):
            px_z = [gaussian(X[n, :], mean, cov, fmt='log')
                    for mean, cov in zip(means, covs)]
            _phi = px_z + np.max(phi[n-1][:, na] + np.log(A), axis=0)
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
        c = self.c

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
        xi = 1 / c[:-1, na, na] * alpha[:-1, :, na] * px_z[1:, na, :] * \
            A[na, :, :] * beta[1:, na, :]

        K = means.shape[0]
        # _xi = []
        # for n in range(1, N):
        #     _xi.append([])
        #     for i in range(K):
        #         _xi[n-1].append([])
        #         for j in range(K):
        #             _xi[n-1][i].append(1 / c[n] * alpha[n-1][i] *  A[i, j] * px_z[n][j] * beta[n][j])

        # _xi = np.array(_xi)
        # if(not (_xi == xi).all()):
        #     print(_xi[10])
        #     print(xi[10])
        #     print((_xi == xi)[10])

        means = np.sum(gamma[:, :, na] * X[:, na, :], axis=0) \
            / sum_gamma[:, na]

        # x_mean: N*K*D
        x_mean = X[:, na, :] - means[na, :, :]
        covs = np.sum(gamma[:, :, na, na] *
                      (x_mean[:, :, :, na] * x_mean[:, :, na, :]),
                      axis=0) / sum_gamma[:, na, na]

        pi = gamma[0, :] / np.sum(gamma[0, :])

        xi_sum = np.sum(xi, axis=0)
        A = xi_sum / np.sum(xi_sum, axis=1)[:, na]

        self.means = means
        self.covs = covs
        self.pi = pi
        self.A = A

    def learn(self):
        for _ in range(1):
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


def synth():
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


def stock():
    # Downloading the data
    date1 = datetime.date(1995, 1, 1)  # start date
    date2 = datetime.date(2012, 1, 6)  # end date
    # get quotes from yahoo finance
    quotes = quotes_historical_yahoo("INTC", date1, date2)
    if len(quotes) == 0:
        raise SystemExit

    # unpack quotes
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])
    volume = np.array([q[2] for q in quotes])[1:]

    # take diff of close value
    # this makes len(diff) = len(close_t) - 1
    # therefore, others quantity also need to be shifted
    diff = close_v[1:] - close_v[:-1]
    dates = dates[1:]
    close_v = close_v[1:]

    # pack diff and volume for training
    X = np.column_stack([diff, volume])
    N = X.shape[0]
    D = X.shape[1]
    K = 5

    km = KMeans(n_clusters=K)
    cls = km.fit_predict(X)

    means = []
    covs = []
    for i in range(K):
        idx = (i == cls)
        means.append(np.mean(X[idx], axis=0))
        covs.append(np.cov(X[idx].T))
    means = np.array(means)
    covs = np.array(covs)
    print(means)
    print(covs)

    model = HMM(X, means, covs)
    model.learn()
    model.check_likelihood()

    print("means and vars of each hidden state")
    for i in range(K):
        print("%dth hidden state" % i)
        print("mean = ", model.means[i])
        print("var = ", np.diag(model.covs[i]))
        print("")


    hidden_states = model.z
    print("hidden states")
    print(hidden_states)
    trans = model.A
    print("transition matrix")
    print(trans)

    years = YearLocator()   # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')
    fig = pl.figure()
    ax = fig.add_subplot(111)

    for i in range(K):
        # use fancy indexing to plot data in each state
        idx = (hidden_states == i)
        ax.plot_date(dates[idx], close_v[idx], 'o', label="%dth hidden state" % i)
    ax.legend()

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    # format the coords message box
    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax.fmt_ydata = lambda x: '$%1.2f' % x
    ax.grid(True)

    fig.autofmt_xdate()
    pl.savefig("fig.png")


if(__name__ == '__main__'):
    stock()
