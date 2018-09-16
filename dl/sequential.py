from collections import OrderedDict
from parameters import Parameter


class Sequential():

    def __init__(self,*args):
        self.module_list = args
        self.parameters = []

        for module in self.module_list:
            for name,param in module.__dict__.items():
                if isinstance(param,Parameter):
                    self.parameters.append(param)

        # print self.parameters

    def forward(self,input):
        for module in self.module_list:
            input = module(input)
        return input

    def backward(self,grad_in):
        
        for module in reversed(self.module_list):
            grad_in = module.backward(grad_in)

    def __call__(self,input):
        return self.forward(input)