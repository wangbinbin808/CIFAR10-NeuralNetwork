import pickle, os
import numpy as np

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        return X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float"), np.array(Y)

def load_CIFAR10(root):
    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(root, f'data_batch_{b}')
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_CIFAR_batch(os.path.join(root, 'test_batch'))
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    # 将像素值归一化到 [0, 1]
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    return X_train, X_test