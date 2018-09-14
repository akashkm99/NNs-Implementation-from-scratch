import numpy as np


class Parameter():

    def __init__(self,data,requires_grad=True):
        
        self.data = data
        if requires_grad:
            self.grad = np.zeros_like(self.data)
