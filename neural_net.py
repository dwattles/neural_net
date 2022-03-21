# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:42:49 2022

@author: dwatt
"""
import struct
import numpy as np
import matplotlib.pyplot as plt
import os

def load_dat():
    with open('train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
        #print(magic, n)
    
    with open ('train-images.idx3-ubyte', 'rb') as images:
        magic, num, nrows, ncols = struct.unpack('>IIII', images.read(16))
        #print(magic, num, nrows, ncols)
        train_images = np.fromfile(images, dtype=np.uint8).reshape(num,784)
        
    with open('t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    
    with open ('t10k-images.idx3-ubyte', 'rb') as images:
        magic, num, nrows, ncols = struct.unpack('>IIII', images.read(16))
        test_images = np.fromfile(images, dtype=np.uint8).reshape(num,784)
    
    return train_images, train_labels, test_images, test_labels    
        
#display examples for 2
def display_dat(image_array,label_array):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(64):
        image = image_array[label_array==2][i].reshape(28,28)
        ax[i].imshow(image, cmap='Greys', interpolation='nearest')
    plt.show()
        
train_x, train_y, test_x, test_y = load_dat()

#test
display_dat(train_x,train_y)

def encode(y):
    one_hot = np.zeros((10, y.shape[0]))
    for i, val in enumerate(y):
        one_hot[val,i]=1.0
        
    return one_hot

#test
#y=np.array([2,1,3,4])
#print(encode(y))

#activation
def sigmoid(x):
    
    return (1/(1+np.exp(-x)))

def sigmoid_gradient(x):
    
    return  (sigmoid(x)*(1-sigmoid(x)))

def cost_function(y_enc, output):
    term1 = -y_enc*np.log(output)
    term2 = (1-y_enc)*np.log(1-output)
    cost = np.sum(term1-term2)
    
    return cost

def add_bias(X, where):
    if where == 'column':
        X_bias = np.ones((X.shape[0], X.shape[1]+1))
        X_bias[:, 1:] = X
    elif where == 'row':
        X_bias = np.ones((X.shape[0]+1,X.shape[1]))
        X_bias[1:, :] = X 
        
    return X_bias


def init_weights(n_features, n_hidden, n_output=10):
    w1 = np.random.uniform(-1.0,1.0,size=n_hidden*(n_features+1))
    w1 = w1.reshape(n_hidden,n_features+1)
    w2 = np.random.uniform(-1.0,1.0,size=n_hidden*(n_hidden+1))
    w2 = w2.reshape(n_hidden,n_hidden+1)
    w3 = np.random.uniform(-1.0,1.0,size=n_output*(n_hidden+1))
    w3 = w3.reshape(n_output,n_hidden+1)
    
    return w1,w2,w3

def forward_pass(x, w1, w2, w3):
    a1 = add_bias(x, where='column')
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)
    a2 = add_bias(a2, where='row')
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)
    a3 = add_bias(a3, where='row')
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)
    
    return a1, z2, a2, z3, a3, z4, a4

def predict(x, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4 = forward_pass(x, w1, w2, w3)
    ypred = np.argmax(a4, axis=0)
    
    return ypred

#backprop
def gradient(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    delta4 = a4-y_enc
    z3 = add_bias(z3, where='row')
    delta3 = w3.T.dot(delta4)*sigmoid_gradient(z3)
    delta3 = delta3[1:, :]
    z2 = add_bias(z2, where='row')
    delta2 = w2.T.dot(delta3)*sigmoid_gradient(z2)
    delta2 = delta2[1:,:]
    
    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)
    
    return grad1, grad2, grad3

def run(X,y,X_t,y_t):
    X_copy, y_copy = X.copy(), y.copy()
    y_enc = encode(y)
    epochs = 50
    batch = 50
    
    w1,w2,w3 = init_weights(784, 75, 10)
    
    alpha = 0.001
    eta = 0.001
    dec = 0.00001
    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)
    
    for t in range(epochs):
        total_cost = []
        shuffle = np.random.permutation(y_copy.shape[0])
        X_copy, y_enc = X_copy[shuffle], y_enc[:, shuffle]
        eta /= (1+dec*t)
        
        mini = np.array_split(range(y_copy.shape[0]), batch)
        
        for step in mini:
            a1, z2, a2, z3, a3, z4, a4 = forward_pass(X_copy[step], w1, w2, w3)
            cost = cost_function(y_enc[:,step], a4)
            
            total_cost.append(cost)
            
            grad1, grad2, grad3 = gradient(a1, a2, a3, a4, z2, z3, z4, y_enc[:,step], w1, w2, w3)
            delta_w1, delta_w2, delta_w3 = eta*grad1, eta*grad2, eta*grad3
            
            w1 -= delta_w1 + delta_w1_prev
            w2 -= delta_w2 + delta_w2_prev
            w3 -= delta_w3 + delta_w3_prev
            
            delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3
            
        print('epoch #',t)
        
    y_pred = predict(X_t, w1, w2, w3)
    acc = np.sum(y_t == y_pred, axis=0)/X_t.shape[0]
    print('accuracy', acc*100)
    
    return 1

train_x, train_y, test_x, test_y = load_dat()

t = run(train_x, train_y, test_x, test_y)

