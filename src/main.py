import sys
sys.path.append("./tests")
from test_simple_ml import *
from simple_ml import train_softmax, parse_mnist, train_nn
import faulthandler; faulthandler.enable()

def run_mnist_with_softmax_regression():
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz", 
                            "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                            "data/t10k-labels-idx1-ubyte.gz")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.2, batch=100)

def run_mnist_with_nn():
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz", 
                         "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                            "data/t10k-labels-idx1-ubyte.gz")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=400, epochs=20, lr=0.2)

def main():
    # run_mnist_with_softmax_regression()
    # run_mnist_with_nn()
    test_softmax_regression_epoch_cpp()

if __name__ == '__main__':
    main()