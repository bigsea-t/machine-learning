import numpy as np
import pylab as plt
import math

Threshold = 0.0000001

na = np.newaxis

def sigmoid(z):
    return 1/(1+np.exp(-z))

def err_func(y, y_):
    return 1 / 2 * (np.sum((y_- y) * (y_ - y)))

def reg():
    filename = './dataset.csv'
    raw_x = np.genfromtxt(filename, delimiter = ",", skiprows = 1, dtype = float,usecols=0)
    raw_y = np.genfromtxt(filename, delimiter = ",", skiprows = 1, dtype = None,usecols=1)
    expand_x = list(map(lambda x: [x,1],raw_x))
    bin_y = list(map(lambda y: 1 if y.strip() == b'JH' else 0, raw_y))
    X = np.array(expand_x)
    y = np.array(bin_y, dtype=float)
    w_0 = np.ones(X.shape[1])
    w = w_0
    err = []
    alpha = 10


    for count in range(10000):
        y_ = sigmoid(np.dot(X,w))


        B  = (y_ - y) * y_ * (1 - y_)
        dE = np.dot(B,X) + alpha * w

        #
        # test code for dE
        #
        # dE_ = []
        # for i in range(w.shape[0]):
        #     dE_.append(0)
        #     for n in range(X.shape[0]):
        #         dE_[i] += (y_[n] - y[n]) * y_[n] * (1 - y_[n]) * X[n][i] 
        #     dE_[i] += alpha * w[i]
        # 
        # print(dE == dE_)

        A = ( (-3 * y_* y_) + 2 * (1 + y) * y_ - y) * y_ * (1 - y_) 
        H = np.sum(A[:,na,na] * X[:,:,na] * X[:,na,:], axis = 0) + alpha * np.identity(X.shape[1])

        # 
        # test code for H
        # 
        # H_ = []
        # for i in range(w.shape[0]):
        #     H_[i:] = [[]]
        #     for j in range(w.shape[0]):
        #         H_[i][j:] = [0]
        #         for n in range(X.shape[0]):
        #             H_[i][j] += ((-3 * y_[n] * y_[n]) + 2 * (1 + y[n]) * y_[n] - y[n]) * y_[n] * (1 - y_[n]) * X[n][i] * X[n][j]
        #         if(i == j):
        #             H_[i][j]+=alpha
        #
        # H_ = np.array(H_)
        #
        # print(H == H_) 

        invH = np.linalg.inv(H)
        w = w - np.dot(invH, dE)

        alpha = alpha * 0.98

        err.append(err_func(y, y_))
        d_err = math.fabs(err[count] - err[count - 1]) if len(err) > 1 else Threshold 
        if(d_err < Threshold):
            print("w\n",w)
            print("loop count:",count)
            break

    space = np.array([[x/1000,1] for x in range(1000,2000)])

    plt.figure()
    plt.plot(space[:,0],sigmoid(np.dot(space,w)),
            'b-',
            label = 'sigmoid func')
    plt.plot(X[:,0], sigmoid(np.dot(X,w)),
            'ro',
            label = 'data point')
    plt.title('Logistic Regressin to classify school stage by height')
    plt.xlabel('height[m]')
    plt.ylabel('probability to be Junior hight school student')

    plt.savefig('regresson.png')

    plt.figure()
    plt.plot(range(count),err[1:],
            'b-')

    plt.title('error function')
    plt.xlabel('count')
    plt.ylabel('error func')

    plt.savefig('error.png')

    plt.figure()



def main():
    reg()    


if __name__ == '__main__':
    main()