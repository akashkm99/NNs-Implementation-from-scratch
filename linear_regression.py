#from dl.layers.linear import Linear
from dl.layers import Linear

import numpy as np


x =  np.random.randn(1024,5)

y = 2.0*x[:,0] + 4.0*x[:,1] + 5.0*x[:,2] + 6.0*x[:,3] + 3.0*x[:,4] + 1.0

y = np.expand_dims(y,1)


fc1 = Linear(5,1)
lr = 1e-3



for epoch in range(10):
    for iteration in range(32):

        x_batch = x[32*iteration:32*(iteration+1)]
        y_batch = y[32*iteration:32*(iteration+1)]

        y_hat = fc1(x_batch)

        # print x_batch.shape,y_batch.shape,y_hat.shape

        rmse = np.sqrt(np.mean((y_hat - y_batch)**2))
        grad_in = (y_hat - y_batch)#,keepdims=True)
        # print grad_in.shape
        _ = fc1.backward(grad_in)
        
        fc1.bias.data -=  lr*fc1.bias.grad
        fc1.weight.data -=  lr*fc1.weight.grad
        
        
    print 'Epoch: %d RMSE Error: %f' %(epoch,rmse) 


print fc1.weight.data, fc1.bias.data
