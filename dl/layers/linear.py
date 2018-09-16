import numpy as np 
from ..parameters import Parameter

class Linear():
    
    def __init__(self,input_size,output_size,l2=0,bias=True):
        
        self.input_size = input_size
        self.output_size = output_size
        self.bias_true = bias
        self.l2 = l2
        
        initial_value = self.normal_initialization()

        self.weight = Parameter(initial_value)
        
        if self.bias_true:
            self.bias = Parameter(np.zeros([1,self.output_size]))

    def xavier_initialization(self):

         return np.random.randn(self.input_size,self.output_size) / np.sqrt(self.input_size)
    
    def normal_initialization(self):

         return np.random.randn(self.input_size,self.output_size)*0.08

    def relu_initialization(self):

        return np.random.randn(self.input_size,self.output_size) * np.sqrt(2.0/self.input_size)
    
    def glorot_initialization(self):

        return np.random.randn(self.input_size,self.output_size) * np.sqrt(2.0/(self.input_size+self.output_size))

    def forward(self,input):
        
        self.input = input
        self.output = np.matmul(self.input,self.weight.data)
        if self.bias_true:
            self.output = self.output + self.bias.data
        return self.output 

    def backward(self,grad_in):
        
        self.grad_out = np.matmul(grad_in,self.weight.data.T)
        self.weight.grad = np.matmul(self.input.T,grad_in) + self.l2*self.weight.data
        self.bias.grad = np.sum(grad_in,0,keepdims=True)
        
        return self.grad_out

 
    def __call__(self, input):
        return self.forward(input)


if __name__ == "__main__" and __package__ is None:
    # print 'hi'
    import dl
    __package__ = "dl.layers"
    from ..parameters import Parameter 
    # from parameters import Parameter


        
