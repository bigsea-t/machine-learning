import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from numpy import newaxis as na
import itertools

def one_others(data, test, i):
    X1 = data[data[:, -1] == i+1][:, :-1]
    X2 = data[data[:, -1] != i+1][:, :-1]
    Y1 = np.full(X1.shape[0], 0, dtype='int')
    Y2 = np.full(X2.shape[0], 1, dtype='int')

    X1t = test[test[:, -1] == i+1][:, :-1]
    X2t = test[test[:, -1] != i+1][:, :-1]
    Y1t = np.full(X1t.shape[0], 0, dtype='int')
    Y2t = np.full(X2t.shape[0], 1, dtype='int')

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)
    Xt = np.concatenate((X1t, X2t), axis=0)
    Yt = np.concatenate((Y1t, Y2t), axis=0)

    return X, Y, Xt, Yt

def one_one(data, test, i, j):
    X1 = data[data[:, -1] == i+1][:, :-1]
    X2 = data[data[:, -1] == j+1][:, :-1]
    Y1 = np.full(X1.shape[0], i, dtype='int')
    Y2 = np.full(X2.shape[0], j, dtype='int')

    X1t = test[test[:, -1] == i+1][:, :-1]
    X2t = test[test[:, -1] == j+1][:, :-1]
    Y1t = np.full(X1t.shape[0], i, dtype='int')
    Y2t = np.full(X2t.shape[0], j, dtype='int')

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)
    Xt = np.concatenate((X1t, X2t), axis=0)
    Yt = np.concatenate((Y1t, Y2t), axis=0)

    return X, Y, Xt, Yt

def data_all(data):
    X = data[:, :-1]
    Y = data[:, -1] - 1
    return X, Y

def for_svms(X, Y, Xt, Yt, figname='fig'):
    h = 0.01
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    lin_svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - h, X[:, 0].max() + h
    y_min, y_max = X[:, 1].min() - h, X[:, 1].max() + h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']


    for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Zt = clf.predict(Xt)
        errors = np.count_nonzero(Zt != Yt)
        e_rate = errors / Yt.shape[0]
        print(e_rate)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the test points
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.savefig(figname + '.png') 

def rbf_err(X, Y, Xt, Yt, gamma):
    C = 1.0  # SVM regularization parameter
    classes = 5
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X, Y)
    Zt = rbf_svc.predict(Xt)
    errors = np.count_nonzero(Zt != Yt)
    e_rate = errors / Yt.shape[0]
    return e_rate

def decide_by_majority(train, test, kernel='rbf', gamma=1):
    classes = 5
    C = 1
    svcs = []
    for i, j in itertools.combinations(range(classes),2):
        X, Y, Xt, Yt = one_one(train, test, i, j)
        svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X, Y)
        svcs.append(svc)
    
    X, Y = data_all(test)
    Zs = [svc.predict(X) for svc in svcs]

    cls = []
    for i in range(Zs[0].shape[0]):
        count = np.zeros(classes)
        for Z in Zs:
            count[Z[i]] += 1
        _cls = np.argmax(count)
        cls.append(_cls)
    cls = np.array(cls)

    errors = []
    for i in range(classes):
        idx = Y == i
        error = np.count_nonzero(Y[idx] != cls[idx])
        num = np.count_nonzero(idx)
        errors.append(error / num)
        print(errors[-1])
    return np.array(errors)





def main():
    train_data = np.loadtxt("../dataset/vowel/train.txt", delimiter=" ")
    test_data = np.loadtxt("../dataset/vowel/test.txt", delimiter=" ")

    classes = 5

    # scaling for features vector
    std = np.std(train_data[:, :-1], axis=0)
    mean = np.mean(train_data[:, :-1], axis=0)
 
    train_data[:, :-1] = (train_data[:, :-1] - mean[na, :]) / std[na, :]
    test_data[:, :-1] = (test_data[:, :-1] - mean[na, :]) / std[na, :]

    for i in range(classes):
        X, Y, Xt, Yt = one_others(train_data, test_data, i)
        for_svms(X, Y, Xt, Yt, figname=str(i)+'th_cls')

    gamma = np.linspace(0, 10, 201)
    errors = []
    for gm in gamma:
        e_rate = 0
        for i in range(classes):
            X, Y, Xt, Yt = one_others(train_data, test_data, i)
            e_rate += rbf_err(X, Y, Xt, Yt, gm)
        e_rate /= classes
        errors.append(e_rate)
    errors = np.array(errors)

    plt.plot(gamma, errors, '-')
    plt.xlabel('Gamma')
    plt.ylabel('Error rate')
    plt.title('RBF Kernel\'s error rate in various gamma')
    plt.savefig('rbf_err.png')

    decide_by_majority(train_data, test_data)

if __name__ == '__main__':
    main()