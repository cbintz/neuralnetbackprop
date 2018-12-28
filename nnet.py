# CS 451 HW 3
# Neural Net with backpropagation

# Corinne Bintz & Lillie Atkins

import random, math
import numpy as np

class Network(object):

    def __init__(self, sizes, debug=False):
        """
        Construct a new neural net with layer sizes given.  For
        example, if sizes = [2, 3, 1] then it would be a three-layer 
        network, with the first layer containing 2 neurons, the
        second layer 3 neurons, and the third layer 1 neuron.
        The biases and weights for the network are initialized randomly.
        If debug=True then repeatable "random" values are used.
        biases and weights are lists of length sizes-1.
        biases[i] is a column vector for layer i+1.
        weights[i] is a matrix for layers [i] and [i+1].
        """
        self.sizes = sizes
        self.debug = debug
        
        self.biases = []
        self.weights = []
         
        print(sizes)
        print(len(sizes))
        for i in range(len(sizes)-1):
            self.biases.append(rand_mat(sizes[i+1], 1, debug)) # add column vector in biases for layer i+1
            self.weights.append(rand_mat(sizes[i+1], sizes[i], debug)) # add matrix for layers i and i + 1


    def feedforward(self, a):
        """Return the output of the network if a is input"""
         #from one layer to next, compute z = w*a+b, a = g(z) 
        for i in range(len(self.sizes)-1): # for each layer
            z = self.weights[i]@a + self.biases[i] # multiply weight times input plus biases
            a = sigmoid(z) # a = g(z)
       
        return a # output of network

    def train(self, train_data, valid_data, epochs, mini_batch_size, alpha):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``train_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``valid_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
       
        self.report_accuracy(0, train_data, valid_data)

        # vectorize outputs y in training data ("one hot" encoding)
        ny = self.sizes[-1]
        train_data_vec = [(x, unit(y, ny)) for x, y in train_data]
        
        m = len(train_data)
        for j in range(epochs):
            if not self.debug:
                alpha = 0.9*alpha # decrease alpha each epoch
                random.shuffle(train_data_vec)
            numBatches = m//mini_batch_size # number of mini batches according to given mini batch size
            start = 0 # start index for indexing train_data_vec
            stop = mini_batch_size # stop index for indexing train_data_vec
            for i in range(numBatches):
                newBatch = train_data_vec[start: stop]  # divide train_data_vec into batches (lists of size mini_batch_size)
                self.update_mini_batch(newBatch, alpha) # call self.update_mini_batch on each mini-batch
                start += mini_batch_size # increase start index by size of mini_batch_size
                stop += mini_batch_size  # increase stop index by size of mini_batch_size
            self.report_accuracy(j+1, train_data, valid_data)

        
    def update_mini_batch(self, mini_batch, alpha):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        mini_batch is a list of tuples (x, y), and alpha
        is the learning rate.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        nL = len(self.sizes)
        m = len(mini_batch)
        for x, y in mini_batch: # for each mini_batch
           
            # then add each of their elements to grad_b, grad_w
            delta_b, delta_w = self.backprop(x,y) # results of self.backrop for current mini_batch
            for i in range(nL-1): # for each layer
               for eltIndex in range(len(grad_b[i])): # for each element in that layer
                    grad_b[i][eltIndex] += delta_b[i][eltIndex] # add each element of batch's biases delta to gradient sum
            for i in range(nL-1): # for each layer
               for eltIndex in range(len(grad_w[i])): # for each element in that layer
                    grad_w[i][eltIndex] += delta_w[i][eltIndex] # add each element of batch's weight delta to gradient sum
           
        # now, adjust the biases and weights by 
        # subtracting (alpha/m) times the gradients
        for i in range(nL-1): # for each layer
            for eltIndex in range(len(grad_b[i])): # for each element in that layer's bias vector
                    self.biases[i][eltIndex] -= ((alpha/m) *grad_b[i][eltIndex]) # adjust biases 
            for eltIndex in range(len(grad_w[i])): #for each element in that layer's weight vector
                    self.weights[i][eltIndex] -= ((alpha/m) *grad_w[i][eltIndex]) # adjust weights
       

    def backprop(self, x, y):
        """
        Return (grad_b, grad_w) representing the gradient of the cost 
        function for a single training example (x, y).  grad_b and
        grad_w are layer-by-layer lists of numpy arrays, similar
        to self.biases and self.weights.
        """
        n = len(self.sizes) # number of layers
        
        # forward pass through network
        a = [0] * n
        z = [0] * n
        a[0] = x # initial activation (z[0] is not used)
        for i in range(1, n): # 1 .. n-1
            b = self.biases[i-1]
            w = self.weights[i-1]
            z[i] = w@a[i-1] + b # compute z = w*a+b and store in z array
            a[i] = sigmoid(z[i]) # compute a = signmoid(z) and store in a array
           
        # backward pass
            
        delta = [0] * n
        i = n-1 # index of last layer
        delta[i] = (a[i] - y) * (sigmoid_grad(z[i])) # find delta for very last layer
        for i in range(n-2, 0, -1): # n-2 .. 1
            delta[i] = (self.weights[i].T@delta[i+1]) * (sigmoid_grad(z[i])) #find delta for all other layers

        # compute gradients
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for i in range(0, n-1):
            grad_b[i] = delta[i+1]
            grad_w[i] = delta[i+1]*a[i].T
        return (grad_b, grad_w)
        


    def evaluate(self, data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
    
        correct = 0 # sum variable
        for x, y in data:  # loop over (x, y) in data
            a = self.feedforward(x)  # run feedforward(x) 
            maxElt = np.argmax(a)  # find the index of the maximum element in the output vector
            correct += (maxElt==y) # if it is equal to y, it is correct
        
        return correct # return number of correct results
     
    def report_accuracy(self, epoch, train_data, valid_data):
        """report current accuracy on training and validation data"""
        tr, ntr = self.evaluate(train_data), len(train_data)
        te, nte = self.evaluate(valid_data), len(valid_data)
        print("Epoch %d: " % epoch, end='')
        print("train %d/%d (%.2f%%) " % (tr, ntr, 100*tr/ntr), end='')
        print("valid %d/%d (%.2f%%) " % (te, nte, 100*te/nte))

#### Helper functions

def sigmoid(z):
    """vectorized sigmoid function"""
    return 1/(1+np.exp(-z))

def sigmoid_grad(z):
    """vectorized gradient of the sigmoid function"""
    s = sigmoid(z)
    return s*(1-s)

def unit(j, n):
    """return n x 1 unit vector with oat index j and zeros elsewhere"""
    vector = np.zeros((n,1))
    for i in range(n):
        if (i==j):
            vector[i,0] = 1
    return vector
    

def rand_mat(rows, cols, debug):
    """
    return random matrix of size rows x cols; if debug make repeatable
    """
    eps = 0.12 # random values are in -eps...eps
    if debug:
        # use math.sin instead of random numbers
        vals = np.array([eps * math.sin(x+1) for x in range(rows * cols)])
        return np.reshape(vals, (rows, cols))
    else:
        return 2 * eps * np.random.rand(rows, cols) - eps
