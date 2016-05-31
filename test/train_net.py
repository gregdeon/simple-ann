# train-net.py
# Use the neural network module to detect simple signals

import numpy as np
import matplotlib.pyplot as plt
import random

from src.net import Net

def main():
    """ Step 1: make dataset """
    random.seed()
    # Make 3 inputs - 1 base and 2 added inputs
    sig_len = 10
    y_base = np.array([1, 2, 3, 2,  6,  5, 0, -1, 2, 4])
    y_add1 = np.array([0, 0, 1, 0, -2,  0, 0,  1, 1, 0])
    y_add2 = np.array([1, 0, 0, 1,  2, -1, 0,  0, 0, 0])
    
    # Set up a bunch of random signals to detect
    y_num = 100
    signal1 = np.array([random.randint(0,1) for i in range(y_num)])
    signal2 = np.array([random.randint(0,1) for i in range(y_num)])
    signal = np.array([signal1, signal2])
    
    # Add up the inputs accordingly
    y_list = np.zeros([y_num, len(y_base)])
    for i in range(y_num):
        y_sum = np.array([y_base[j] + signal1[i]*y_add1[j] + signal2[i]*y_add2[j] 
                for j in range(sig_len)])
        y_list[i] = y_sum
    
    # Add noise
    noise = np.random.random([y_num, len(y_base)]) / 10
    y_list += noise
    
    """ Step 2: train neural network """
    # Set up input and signals
    input = np.array(y_list)
    signal = signal.transpose()
    
    # Set up min and max for each input 
    # Can give the network a good idea of input ranges or just a rough range
    limits = [[y_base[i]-2, y_base[i]+2] for i in range(10)]
    #limits = [[-20, 20]]*10
    
    # Make network
    net = Net(limits, 2, 2)
    errorList = net.train_many(input, signal, 0.1, 100, 0.001)
    print "\n".join(map(str, errorList))
        
    """ Step 3: check results """
    # Print results by hand
    #for i in range(y_num):
    #    print y_list[i]
    #    print signal1[i]
    #    print signal2[i]
    #    print net.sim(y_list[i, :])
        
    # Plot error vs. training epochs
    plt.semilogy(errorList)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('SSE')
    plt.show()