# net.py
# A dead-simple neural network

import numpy as np

class Net(object):
    """
    Neural network class
    
    :Parameters:
        inp_range: list (num_inputs x 2)
            List of input ranges
        layers: list (1 x num_layers)
            Number of neurons in each layer
            
    :Internals:
        connect: a list of matrices
            ex: with 2 inputs, 2 hidden, 1 output
                connect[0] = 2x3
                connect[1] = 1x3
            so that
                hidden = activate( connect[0] * [input; 1])
                output = connect[1] * [hidden; 1]
    
    :Example:
        # Two inputs (ranges [1,2] and [3,4])
        # One hidden layer (2 neurons)
        # One output layer (1 neuron)
        net = Net([[1, 2], [3, 4]], [2, 1])
        
    :NOTE:
        num_layers MUST BE 2
    """
    
    def __init__(self, inp_range, layers):
        try:
            assert np.shape(inp_range)[1] == 2
        except:
            raise InputException("Invalid shape for inp_range")
        self.inp_range  = inp_range
        self.layer_size = [np.shape(inp_range)[0]] + layers
        try:
            assert len(layers) == 2
        except:
            raise InputException("Only 1 hidden layer supported")
        
        self.w_in  = np.random.random([self.layer_size[1], self.layer_size[0]+1])-0.5
        self.w_out = np.random.random([self.layer_size[2], self.layer_size[1]+1])-0.5
            
    def activate(self, layer):
        return np.array([1 / (1 + np.exp(-x)) for x in layer])
        
    def addBias(self, layer):
        return np.vstack((layer, [1]))
        
    def scaleInput(self, input):
        for i in range(len(input)):
            low = self.inp_range[i][0]
            hi  = self.inp_range[i][1]
            input[i] = -1.0 + 2.0 * (input[i] - low) / (hi - low)
        return input
        
    def sim(self, input):
        try:
            assert np.shape(input)[0] == self.layer_size[0]
        except:
            raise InputException("Invalid shape of input")
        
        # Get input
        layer = np.reshape(np.array(input), [-1, 1])
        layer = self.scaleInput(layer)
        layer = self.addBias(layer)
        
        # Get hidden layer
        layer = self.activate(np.dot(self.w_in, layer))
        layer = self.addBias(layer)
        
        # Get output
        layer = np.dot(self.w_out, layer)
        return layer
        
    def train_1(self, input, target, lr):
        error = 0
        order = range(len(input))
        np.random.shuffle(order)
        for i in order:
            # Forward propagation
            # Save intermediate results
            o = []
            
            # Input
            layer = np.array(input[i], copy=True)
            layer = np.reshape(layer, [-1, 1])
            layer = self.scaleInput(layer)
            layer = self.addBias(layer)
            o.append(layer)

            
            # Hidden
            layer = self.activate(np.dot(self.w_in, layer))
            layer = self.addBias(layer)
            o.append(layer)
            
            # Output
            layer = np.dot(self.w_out, layer)
            o.append(layer)
            output = layer
            
            # Backward propagation
            delta = [[] for _ in range(2)]
            delta[0] = np.zeros([self.layer_size[1], 1])
            
            # Output delta
            delta[1] = output - np.reshape(target[i], [-1, 1])
            
            # Hidden delta
            for j in range(self.layer_size[1]):
                for k in range(self.layer_size[2]):
                    delta[0][j] += delta[1][k] * self.w_out[k][j]
                delta[0][j] *= o[1][j] * (1 - o[1][j])
            
            # Changes in weights
            delta_w_out = -lr * np.dot(delta[1], np.reshape(o[1], [1, -1]))
            delta_w_in  = -lr * np.dot(delta[0], np.reshape(o[0], [1, -1]))
            self.w_out += delta_w_out
            self.w_in += delta_w_in           
            error += np.sum((output - np.reshape(target[i], [-1, 1]))**2)            

        return error
            
def main():
    net = Net([[1, 2], [3, 4]], [4, 2])
    
    input  = np.array([[1, 3], [2, 3], [1, 4], [2, 4]] * 1)
    target = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 1)
    
    print input
    print target
    
    for i in range(100):
        print net.train_1(input, target, 0.1)

    print net.sim([1, 3])
    print net.sim([2, 3])
    print net.sim([1, 4])
    print net.sim([2, 4])

if __name__ == "__main__":
    main()