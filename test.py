# testing code for CS 451 HW 3

#from unittest import TestCase
import unittest
from data import load_mnist_data
import nnet
import numpy as np

class HW3Test(unittest.TestCase):

    def test1_sigmoid(self):
        s = nnet.sigmoid(np.array([[0, 1, 2], [2, 4, -1]]))
        self.assertEqual(s.shape[0], 2)
        self.assertEqual(s.shape[1], 3)
        self.assertAlmostEqual(s[0, 0], 0.5)
        self.assertAlmostEqual(s[0, 1], 0.7310585786300049)
        self.assertAlmostEqual(s[0, 2], 0.8807970779778823)
        self.assertAlmostEqual(s[1, 2], 0.2689414213699951)
        
    def test2_sigmoid_grad(self):
        s = nnet.sigmoid_grad(np.array([[0, 1, 2], [2, 4, -1]]))
        self.assertEqual(s.shape[0], 2)
        self.assertEqual(s.shape[1], 3)
        self.assertAlmostEqual(s[0, 0], 0.25)
        self.assertAlmostEqual(s[0, 1], 0.1966119332414818)
        self.assertAlmostEqual(s[0, 2], 0.1049935854035066)
        self.assertAlmostEqual(s[1, 2], 0.1966119332414818)

    def test3_unit(self):
        e = nnet.unit(1, 4)
        self.assertEqual(e.shape[0], 4)
        self.assertEqual(e.shape[1], 1)
        self.assertEqual(e[0, 0], 0.0)
        self.assertEqual(e[1, 0], 1.0)
        self.assertEqual(e[2, 0], 0.0)
        self.assertEqual(e[3, 0], 0.0)

    def test4a_biases_sizes(self):
        net = nnet.Network([30, 20, 10], debug=True)
        self.assertEqual(len(net.biases), 2)
        self.assertEqual(net.biases[0].shape[0], 20)
        self.assertEqual(net.biases[0].shape[1], 1)
        self.assertEqual(net.biases[1].shape[0], 10)
        self.assertEqual(net.biases[1].shape[1], 1)
        self.assertAlmostEqual(net.biases[1][5, 0], -0.0335298597838711)
        
    def test4b_weights_sizes(self):
        net = nnet.Network([30, 20, 10], debug=True)
        self.assertEqual(len(net.weights), 2)
        self.assertEqual(net.weights[0].shape[0], 20)
        self.assertEqual(net.weights[0].shape[1], 30)
        self.assertEqual(net.weights[1].shape[0], 10)
        self.assertEqual(net.weights[1].shape[1], 20)
        self.assertAlmostEqual(net.weights[1][5, 15], 0.0283993672037143)

    def test5_feedforward(self):
        net = nnet.Network([3, 2], debug=True)
        a = np.array([[1], [2], [3]])
        f = net.feedforward(a)
        self.assertEqual(f.shape[0], 2)
        self.assertEqual(f.shape[1], 1)
        self.assertAlmostEqual(f[0, 0], 0.615617486546994)
        self.assertAlmostEqual(f[1, 0], 0.422521223234278)
        net2 = nnet.Network([3, 5, 3], debug=True)
        g = net2.feedforward(a)
        self.assertEqual(g.shape[0], 3)
        self.assertEqual(g.shape[1], 1)
        self.assertAlmostEqual(g[0, 0], 0.529519780213278)
        self.assertAlmostEqual(g[1, 0], 0.543296048678935)
        self.assertAlmostEqual(g[2, 0], 0.509071159489918)

    def test6a_backprop(self):
        net = nnet.Network([3, 2], debug=True)
        x = np.array([[2], [1], [4]])
        y = np.array([[0.3], [0.6]])
        gb, gw = net.backprop(x, y)
        self.assertEqual(len(gb), 1)
        self.assertEqual(len(gw), 1)
        self.assertEqual(gb[0].shape[0], 2)
        self.assertEqual(gb[0].shape[1], 1)
        self.assertEqual(gw[0].shape[0], 2)
        self.assertEqual(gw[0].shape[1], 3)
        self.assertAlmostEqual(gb[0][0, 0],  0.0750232605807354)
        self.assertAlmostEqual(gb[0][1, 0], -0.0437921866454942)
        self.assertAlmostEqual(gw[0][0, 2],  0.3000930423229416)
        self.assertAlmostEqual(gw[0][1, 1], -0.0437921866454942)
        self.assertAlmostEqual(gw[0][1, 2], -0.1751687465819769)

    def test6b_backprop(self):
        net = nnet.Network([3, 6, 2], debug=True)
        x = np.array([[2], [1], [4]])
        y = np.array([[0.3], [0.6]])
        gb, gw = net.backprop(x, y)
        self.assertEqual(len(gb), 2)
        self.assertEqual(len(gw), 2)
        self.assertEqual(gb[0].shape[0], 6)
        self.assertEqual(gb[0].shape[1], 1)
        self.assertEqual(gb[1].shape[0], 2)
        self.assertEqual(gb[1].shape[1], 1)
        self.assertEqual(gw[0].shape[0], 6)
        self.assertEqual(gw[0].shape[1], 3)
        self.assertEqual(gw[1].shape[0], 2)
        self.assertEqual(gw[1].shape[1], 6)
        self.assertAlmostEqual(gb[0][5, 0], -0.0001770445364953)
        self.assertAlmostEqual(gb[1][0, 0],  0.0564664628991693)
        self.assertAlmostEqual(gw[0][4, 2], -0.0042166992931269)
        self.assertAlmostEqual(gw[0][1, 1],  0.0009961251229913)
        self.assertAlmostEqual(gw[1][0, 0],  0.0348791419524805)
        print("remove triple quotes to run remaining 3 tests")

# comment out the last 3 tests for now so it runs faster

    def test7_update_minibatch(self):
        (train_data, _) = load_mnist_data()
        train_data_vec = [(x, nnet.unit(y, 10)) for x, y in train_data]
        mini_batch = train_data_vec[10:20]
        net = nnet.Network([784, 15, 10], debug=True)
        net.update_mini_batch(mini_batch, 5.5)
        self.assertEqual(net.biases[1].shape[0], 10)
        self.assertEqual(net.biases[1].shape[1], 1)
        self.assertAlmostEqual(net.biases[1][9, 0], -0.537912934538857)
        self.assertEqual(net.weights[0].shape[0], 15)
        self.assertEqual(net.weights[0].shape[1], 784)
        self.assertAlmostEqual(net.weights[0][9, 333], 0.099262147771176)

    def test8_evaluate(self):
        (train_data, valid_data) = load_mnist_data()
        net = nnet.Network([784, 5, 10], debug=True)
        nt = net.evaluate(train_data[:100])
        self.assertEqual(nt, 14)
        nv = net.evaluate(valid_data[9000:])
        self.assertEqual(nv, 102)

    def test9_train(self):
        (train_data, valid_data) = load_mnist_data()
        # reduce data sets for faster speed:
        train_data = train_data[:1000]
        valid_data = valid_data[:1000]
        net = nnet.Network([784, 12, 10], debug=True)
        net.train(train_data, valid_data, epochs=1, mini_batch_size=8, alpha=5)
        nv = net.evaluate(valid_data)
        self.assertEqual(nv, 503)


if __name__ == '__main__':
    unittest.main()
