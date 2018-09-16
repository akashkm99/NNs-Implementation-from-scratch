import numpy as np


class SGD():

    def __init__(self,param,lr=1e-4,momentum=0):
        
        self.len = len(param)
        self.parameters = param
        self.v = [0]*self.len
        self.lr = lr
        self.momentum = momentum

    def step(self):
        
        for idx,param in enumerate(self.parameters):

            self.v[idx] = self.momentum*self.v[idx] + param.grad
            param.data = param.data - self.lr*self.v[idx]

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0     
