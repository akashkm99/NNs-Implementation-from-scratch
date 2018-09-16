from dl import Parameter,Sequential
from dl.layers import Linear,Sigmoid,Relu,Softmax_crossentropy



fc1 = Linear(784,1000)
a1 = Sigmoid()
fc2 = Linear(1000,500)
a2 = Sigmoid()
fc3 = Linear(250,10)

# model = Sequential(fc1,a1,fc2,a2,fc3)
model = Sequential(Linear(784,1000),Sigmoid(),Linear(1000,500),Sigmoid(),Linear(250,10))