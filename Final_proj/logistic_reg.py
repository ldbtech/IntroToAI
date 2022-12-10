from audioop import bias
from matplotlib.font_manager import _Weight
import numpy as np
import random
"""
    Logistic Regression Explaination:
        . Class will take two tuples of train and test sets in the form of => (X, Y)
        . activation_function => Uses softmax formula.
        . init_parameters => Initialize parameters using Adam Initialization or Random.
        . cost =>
        . forward_propagation =>
        . backward_propagation =>
"""
class LogisticRegression:

    def __init__(self, train, test):
        self.X_train = train[0]
        self.y_train = train[1]
        self.X_test = test[0]
        self.y_test = test[1]
    
    def activation_function(self, Z_func):
        e_x = np.exp(Z_func)
        return e_x / np.sum(e_x)

    def init_parameters(self, dim):
        _Weight1 = np.dot(np.random.randn(dim[0], dim[1]), 0.001)
        bias1 = np.zeros((dim[0], 1))

        return {"w1" : _Weight1, "b1" : bias1}
    
    def forward_propagation(self, parameters,  X, Y):
        m_training = X.shape[1]
        # Forward Propagation
        Activation = self.activation_function(np.dot(parameters['w1'].T, X) + parameters['b1'])
        cost = self.cost(m_training, Activation, Y)

        # Backward Propagation using Gradient Descent:
        dw = 1/m_training*np.dot(X, (Activation - Y).T)
        db = 1/m_training * np.sum(Activation - Y)

        cost = np.squeeze(np.array(cost))

        cache = {'dw':dw, 'db':db, 'cost':cost}

        return cache

    
    def cost(self, m_training, Activation, Y):
        return -1/m_training * (np.dot(Y, np.log(Activation).T)+np.dot((1-Y), np.log(1-Activation).T))
        
    def backward_propagation(self):
        pass

