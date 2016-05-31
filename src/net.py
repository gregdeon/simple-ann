# net.py
# A dead-simple neural network
#
# Potential improvements:
# - Refactor to avoid duplicate code in sim() and train_1()
# - Add train_many()
# - Support topologies other than 1 hidden layer
# - Add other training strategies (adaptive learning rate, momentum)
# - Add more error checking

import numpy as np

class Net(object):
    """
    Neural network class with:
    - 1 hidden layer
    - Arbitrary number of neurons in each layer
    - Sigmoid activation function f(x) = 1 / (1 + exp(-x)) 
    - Bias nodes in input/hidden layers
    - Gradient descent training
    
    :Parameters:
        inp_range: list (num_inputs x 2)
            List of input ranges
            Input values are scaled so that (min -> -1) and (max -> +1)
        hidden_count: int
            Number of neurons in the hidden layer
        output_count: int
            Number of neurons in the output layer (= number of output signals)
            
    :Internals:
        size_inp, size_hid, size_out: ints
            Number of neurons in input, hidden, and output layers
        w_in: 2D array
            Connections between input and hidden, so that
            hidden = activate(w_in * input)
        w_out: 2D array
            Connections between input and output, so that
            output = w_out * input
        
    
    :Example:
        # Two inputs (ranges [1,2] and [3,4])
        # One hidden layer (2 neurons)
        # One output layer (1 neuron)
        net = Net([[1, 2], [3, 4]], 2, 1)
    """
    
    def __init__(self, inp_range, hidden_count, output_count):
        # Make sure input matrix is Nx2 (ie: min + max for each)
        try:
            assert np.shape(inp_range)[1] == 2
        except:
            raise ValueError("Invalid shape for inp_range - need [min, max] for each node")
        self.inp_range  = inp_range
        
        # Size of each layer
        self.size_inp = np.shape(inp_range)[0]
        self.size_hid = hidden_count
        self.size_out = output_count
        
        # Random connections
        self.w_in  = np.random.random([self.size_hid, self.size_inp+1])-0.5
        self.w_out = np.random.random([self.size_out, self.size_hid+1])-0.5
            
            
    def _activate(self, layer):
        """
        Perform the activation function on each neuron in the layer
        Used on hidden layers
        """
        return np.array([1 / (1 + np.exp(-x)) for x in layer])
        
    def _addBias(self, layer):
        """
        Add a bias node to the current layer
        (ie: a neuron that always outputs 1)
        """
        return np.vstack((layer, [1]))
        
    def _scaleInput(self, input):
        """
        Scale the input values such that:
          Input       | Output
        --------------+--------
          min         | -1
          (min+max)/2 |  0
          max         | +1
          
        :Inputs:
            input: 1D list of length size_inp
        
        :Outputs:
            1D list of same length, with all values scaled (as described)
        """
        for i in range(len(input)):
            low = self.inp_range[i][0]
            hi  = self.inp_range[i][1]
            input[i] = -1.0 + 2.0 * (input[i] - low) / (hi - low)
        return input
        
    def sim(self, input):
        """
        Find the output values, given a set of input values
        
        :Inputs:
            input: a 1D array of length size_inp
            
        :Returns:
            A 1D array of length size_out
        """
        # Make sure the input is the right size
        try:
            size = np.shape(input)[0]
            assert size == self.size_inp
        except:
            raise ValueError("Expected input of size {}; got {}".format(size, self.size_inp))
        
        # Get input
        layer = np.reshape(np.array(input), [-1, 1])
        layer = self._scaleInput(layer)
        layer = self._addBias(layer)
        
        # Get hidden layer
        layer = self._activate(np.dot(self.w_in, layer))
        layer = self._addBias(layer)
        
        # Get output
        layer = np.dot(self.w_out, layer)
        return layer
        
    def train_1(self, input, target, lr):
        """
        Perform forward propagation to get the output values,
        then backward propagation to update the weights
        Repeat for a number of tests
        
        :Inputs:
            input: 2D array
                A <tests>x<size_inp> array. The input for test i is input[i, :]
            target: 2D array
                A <tests>x<size_out> array. The target for test i is target[i, :]
            lr: float
                Learning rate - preferably in the range (0, 1)
        """
        # Keep track of total error in tests
        error = 0
        
        # Optional: reorder test cases randomly (does this help?)
        order = range(len(input))
        #np.random.shuffle(order)
        
        for i in order:
            # Forward propagation
            # Save intermediate results
            o = []
            
            # Input
            layer = np.array(input[i], copy=True)
            layer = np.reshape(layer, [-1, 1])
            layer = self._scaleInput(layer)
            layer = self._addBias(layer)
            o.append(layer)

            
            # Hidden
            layer = self._activate(np.dot(self.w_in, layer))
            layer = self._addBias(layer)
            o.append(layer)
            
            # Output
            layer = np.dot(self.w_out, layer)
            o.append(layer)
            output = layer
            
            # Backward propagation
            delta = [[] for _ in range(2)]
            delta[0] = np.zeros([self.size_hid, 1])
            
            # Output delta
            delta[1] = output - np.reshape(target[i], [-1, 1])
            
            # Hidden delta
            for j in range(self.size_hid):
                for k in range(self.size_out):
                    delta[0][j] += delta[1][k] * self.w_out[k][j]
                delta[0][j] *= o[1][j] * (1 - o[1][j])
            
            # Changes in weights
            delta_w_out = -lr * np.dot(delta[1], np.reshape(o[1], [1, -1]))
            delta_w_in  = -lr * np.dot(delta[0], np.reshape(o[0], [1, -1]))
            self.w_out += delta_w_out
            self.w_in += delta_w_in           
            # SSE calculation
            error += np.sum((output - np.reshape(target[i], [-1, 1]))**2)            

        return error
            

def main():
    """
    Test code: learn XOR with 3 hidden nodes
    """
    
    # Set up neural net
    net = Net([[0, 1], [0, 1]], 3, 1)
    
    # Set up dataset
    input  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = np.array([[0],    [1],    [1],    [0]   ])
    
    # Print it to check
    print input
    print target
    
    # Train for 100 epochs
    for i in range(100):
        print net.train_1(input, target, 0.5)

    # Check that we've learned everything
    print net.sim([0, 0])       # 0
    print net.sim([0, 1])       # 1
    print net.sim([1, 0])       # 1
    print net.sim([1, 1])       # 0
    #print net.sim([1, 1, 1])    # ValueError()

if __name__ == "__main__":
    main()