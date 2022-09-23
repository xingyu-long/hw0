import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """

    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    ### BEGIN YOUR CODE
    from array import array

    # load image to X      
    with gzip.open(image_filename, 'rb') as f:
        # Users of Intel processors and other low-endian machines must flip the bytes of the header
        _, count, row, col = struct.unpack('>iiii', f.read(16))
        image_data = array("B", f.read())
        X = np.asarray(image_data, dtype=np.float32).reshape(count, row * col)
    # normalization
    X = normalize_data(X)
    # load image lable to Y
    with gzip.open(label_filename, 'rb') as f:
        _, count = struct.unpack('>ii', f.read(8))
        label_data = array("B", f.read())
        Y = np.asarray(label_data, dtype=np.uint8)

    return (X, Y)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # https://www.youtube.com/watch?v=ILmANxT-12I
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:,None]

    return np.average(-np.log(softmax(Z)[np.indices(y.shape)[0], y]))
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    """
    shape: 
        input: m * n
        theta: n * k
        output: m * k
        x: n * 1
        h(x): (theta)^T .* x -> (k * n) .* (n * 1) -> k * 1

    derivative (gradient) of the loss:
        result = x(z - e_{y})^T
        z = normalize(exp(h(x)))

        (n * 1) .* (k * 1 - k * 1)^T -> n * k
    
    mini-Batch:
        X = m' * n
        result = 1/m * X^T(Z - I_{y}) -> (n * m') .* (m' * k) -> n * k
        Z = normalize(exp(X .* theta)) -> (m' * n).* (n * k) -> m' * k
    
    update the theta by lr:
        theta = theta  - lr * (gradient of theta)
    """
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:,None]
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    for i in range(X.shape[0] // batch):
        batch_start, batch_end = i * batch, (i + 1) * batch
        batch_X = X[batch_start: batch_end]
        # print('batch_x,', batch_X.shape)
        # print(repr(batch_X))
        # print('dot result', repr(np.dot(batch_X, theta)))
        batch_Z = softmax(np.dot(batch_X, theta))
        # print('batch_z', batch_Z.shape)
        # print(repr(batch_Z))
        batch_y = y[batch_start: batch_end]
        # print('batch_y', batch_y.shape)
        # print(repr(batch_y))
        I_y = np.zeros(batch_Z.shape)
        I_y[np.indices((batch, ))[0], batch_y] = 1
        # print('I_y', I_y.shape)
        # print(repr(I_y))
        # print('subtract', repr(batch_Z - I_y))
        # print('transpose', repr(np.transpose(batch_X)))
        # print('dot, ', repr(np.dot(np.transpose(batch_X), batch_Z - I_y)))
        g = np.dot(np.transpose(batch_X), batch_Z - I_y) / batch
        # print('g', g.shape)
        # print(repr(g))
        # print('mut', repr(lr * g))
        theta -= lr * g
        # print('theta', repr(theta))
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    """
    Shape:
        d: d-dimensional hidden unit
        W_1: n * d
        W_2: d * k
        X: m * n

    minimize (RELU(X W_1)W_2, y)
    RELU(X W_1)W_2 -> (m * d) dot (d * k) -> m * k

    Z_1: m * d, ReLU(XW1)
    G_2: m * k, normalize(exp(Z_1 * W_2)) - I_y
    G_1: m * d, 1{Z_1 > 0} matmul (G_2(W_2)^T)


    Gradient for W_1 and W_2:
        1/m X^T * G1 -> (n * m) dot (m * d) -> n * d
        1/m (Z_1)^T * G2 -> (d * m) dot (m * k) -> d * k
    """
    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:,None]

    for i in range(X.shape[0] // batch):
        batch_start, batch_end = i * batch, (i + 1) * batch
        batch_X = X[batch_start: batch_end]
        batch_y = y[batch_start: batch_end]
        Z_1 = relu(np.dot(batch_X, W1))
        Z_2 = softmax(np.dot(Z_1, W2))
        
        I_y = np.zeros(Z_2.shape)
        I_y[np.indices((batch, ))[0], batch_y] = 1

        G_2 = Z_2 - I_y
        binary_matrix = (Z_1 > 0).astype(int)
        G_1 = np.multiply(binary_matrix, np.dot(G_2, np.transpose(W2)))
        # gradient for w1 and w2
        g_1 = np.dot(np.transpose(batch_X), G_1) / batch
        g_2 = np.dot(np.transpose(Z_1), G_2) / batch
        # apply gradient to w1 and w2
        W1 -= lr * g_1
        W2 -= lr * g_2

    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
