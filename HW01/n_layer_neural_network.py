import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from HW01 import three_layer_neural_network

class Layer(object):
    """
    Single layer of the n-layer network
    """
    
    def __init__(self, input_dim, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        """
        :param input_dim: number neurons in the previous layer
        :param output_dim: number of hidden neurons in this layer
        :param actFun_type: type of activation function
        """

        # set input params
        np.random.seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize intermediate factors
        self.x = None
        self.z = None
        self.a = None
        
        # deltas
        self.delta = None
        self.db = None
        self.dW = None
        
        # set weights/bias
        self.W = np.random.randn(self.input_dim, self. output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self. output_dim))
        
        
    
    def feedforward(self, x, actFun):
        """
        :param x: input
        :param actFun: the activation function passed as an anonymous function
        :return: None
        """
        
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.a = actFun(self.z)
        return None
    
    
    def backprop(self, prev_delta, diff_actFun):
        """
        :param prev_delta: delta from previous layer
        :param diff_actFun: the differentiated activation function as an anonymous function
        :return: the resulting gradient
        """

        # print("PREV_DELT = ", prev_delta)
        delta = prev_delta * (diff_actFun(self.z))
        self.dW = np.dot(self.x.T, delta) + self.reg_lambda * self.W
        self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)

    

class DeepNeuralNetwork(three_layer_neural_network.NeuralNetwork):
    def __init__(self, layer_num, nn_layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=0):
        """
        Constructs a deep neural network
        
        :param layer_num: number of layers
        :param nn_layer_sizes: array of layer sizes
        :param actFun_type: activation function type
        :param reg_lambda: defaults to 0.01
        :param seed: randomization seed
        """

        self.layer_num = layer_num
        self.nn_layer_sizes = nn_layer_sizes
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.layers = []
        self.ACTIVATION_FUNC = lambda f: self.actFun(f, actFun_type)
        self.DIFF_ACTIVATION_FUNC = lambda g: self.diff_actFun(g, actFun_type)
        self.probs = None
        
        # start the HIDDEN layers
        for i in range(self.layer_num - 1):
            layer = Layer(nn_layer_sizes[i], nn_layer_sizes[i + 1], self.actFun_type)
            self.layers.append(layer)
            
        

    def feedforward(self, X, actFun):
        """
        :param x: input
        :param actFun: the activation function passed as an anonymous function
        :return: probabilities
        """
        carry_forward = X
        for layer in self.layers:
            layer.feedforward(carry_forward, self.ACTIVATION_FUNC)
            carry_forward = layer.a
        self.probs = carry_forward
        
        #  normalizing by e to avoid divide by 0 errors
        
        if self.actFun_type == "relu":
            exp_scores = np.exp(carry_forward)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return self.probs

    def backprop(self, X, y):
        """
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: nothing
        """
        # #void X

        prev_delta = self.probs
        prev_delta[range(len(X)), y] -= 1.0

        for layer in reversed(self.layers):
            prev_delta = layer.backprop(prev_delta, self.DIFF_ACTIVATION_FUNC)


    def calculate_loss(self, X, y):
        """
        Computes loss for prediction
        :param X: input data
        :param y: given labels
        :return: prediction loss
        """
        num_examples = len(X)
        self.feedforward(X, self.ACTIVATION_FUNC)
        data_loss = -1 * np.sum(np.log(self.probs[range(num_examples), y]))

        # Add regulationzation term to loss (optional)
        reg_sum = 0
        for layer in self.layers:
            reg_sum = reg_sum + np.sum(np.square(layer.W))
        data_loss += (self.reg_lambda / 2) * reg_sum
        
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, self.ACTIVATION_FUNC)
            # Backpropagation
            self.backprop(X,y)

            #regularization
            for layer in self.layers:
                layer.dW += self.reg_lambda * layer.W

            # Gradient descent parameter update
            for layer in self.layers:
                layer.W += -1 * epsilon * layer.dW
                layer.b += -1 * epsilon * layer.db
        
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" %
                      (i, self.calculate_loss(X, y)))




def main():
    # # generate and visualize Make-Moons dataset
    X, y = three_layer_neural_network.generate_data()
    num_samples, data_dimension = X.shape
    
    layer_sizes = [data_dimension, 25, 2]
    layer_num = len(layer_sizes)
    
    #  start the deep neural network
    model = DeepNeuralNetwork(layer_num, layer_sizes, actFun_type='relu')

    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)
    
if __name__ == "__main__":
    main()