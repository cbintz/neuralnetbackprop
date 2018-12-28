# run NN code

from data import load_mnist_data
import nnet
import numpy as np

from matplotlib import pyplot as plt 

def show(x):
    """ visualize a single training example """
    im = plt.imshow(np.reshape(1 - x, (28, 28)))
    im.set_cmap('gray')

print("loading MNIST dataset")
(train_data, valid_data) = load_mnist_data()

# reduce data sets for faster speed:
train_data = train_data[:50000]
valid_data = valid_data[:1000]


x, y = train_data[123]


net = nnet.Network([784, 100, 10]) # 1 hidden layer of size 100

print("training")
net.train(train_data, valid_data, epochs=7, mini_batch_size=5, alpha=2)

ncorrect = net.evaluate(valid_data)
print("Validation accuracy: %.3f%%" % (100 * ncorrect / len(valid_data)))
