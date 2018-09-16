import numpy as np


class Sigmoid():

    def forward(self,input):
        
        exp = np.exp(-1*input)
        self.output = 1/(1 + exp)
        
        return self.output

    def backward(self,grad_in):
        
        self.grad_out = grad_in * (self.output) * (1 - self.output)
        return self.grad_out

    def __call__ (self,input):
        return self.forward(input)

class Relu():

    def forward(self,input):
        
        self.input = input
        self.output = np.maximum(0,self.input)
        return self.output

    def backward(self,grad_in):
        
        self.positive = (self.input > 0).astype(np.float32)
        self.grad_out = grad_in*self.positive
        return self.grad_out

    def __call__ (self,input):
        return self.forward(input)


        