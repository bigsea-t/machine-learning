import numpy as np
import matplotlib.pyplot as plt

na = np.newaxis


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Perceptron:
    def __init__(self,
                 max_itr=100000,
                 alpha_0=0.5,
                 beta = 0.98,
                 threshold=1e-10):
        self.w = None
        self.max_itr = max_itr
        self.alpha = alpha_0
        self.beta = beta
        self.std = None
        self.mean = None
        self.old_err = float('Inf')
        self.threshold = threshold
        self.activate = sigmoid

    def _forward(self, x):
        y = self.activate(np.sum(x * self.w))
        return y

    def predict(self, _X):
        X = (_X[:, :] - self.mean[na, :]) / self.std[na, :]
        Y = np.array([self._forward(x) for x in X])
        return Y

    def backward(self, x, t, y):
        delta = (y - t) * y * (1 - y)  # 1
        Ew = x * delta # H
        return Ew

    def fit(self, X, T):
        N = X.shape[0]
        D = X.shape[1]

        self.w = np.random.rand(D) * 2 - 1

        for i in range(self.max_itr):
            err = 0
            Ew = 0
            for x, t in zip(X, T):
                y = self._forward(x)
                _Ew = self.backward(x, t, y)
                Ew += _Ew
                err += 1 / 2 * (y - t) * (y - t)

            err = err / N
            delta_err = np.absolute(self.old_err - err)
            self.old_err = err
            self.w -= self.alpha * Ew
            self.alpha *= self.beta
            if(delta_err < self.threshold):
                break

    def boundary(self, x1):
        return - (self.w[1] * x1 + self.w[0]) / self.w[2] 



def main():
    pcp = Perceptron()

    X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 2, 3, 3, 4, 0, 1, 2, 3, 4],
                  [3, 2, 2, 3, 1, 2, 1, 1, 0, 0]]).T

    T = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    pcp.fit(X, T)

    x1_0 = np.linspace(-1,5,600)
    x2_0 = pcp.boundary(x1_0)

    plt.plot(x1_0, x2_0, '-')

    X_1 = X[T==0, :]
    X_2 = X[T==1, :]
    plt.plot(X_1[:, 1], X_1[:, 2], 'ro')
    plt.plot(X_2[:, 1], X_2[:, 2], 'go')

    plt.xlim([-1,5])
    plt.ylim([-1,4])

    plt.savefig('fig.png')


if __name__ == '__main__':
    main()
