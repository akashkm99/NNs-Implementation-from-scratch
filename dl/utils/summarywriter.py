import numpy as np
import matplotlib.pyplot as plt



class SummaryWriter():

    def __init__(self,path,title=None,xlab=None,ylab=None,legend=None):
        self.path = path
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self.legend = legend
        self.data = []

    def add_scalar(self,list):
        
        self.data.append(list)

    def close(self):
        
        self.data = np.array(self.data)
        plt.plot(self.data[:,0],self.data[:,1],label=self.legend[0])
        plt.plot(self.data[:,0],self.data[:,2],label=self.legend[1])
        plt.title(self.title)
        plt.xlabel(self.xlab)
        plt.ylabel(self.ylab)
        plt.legend()
        plt.savefig(self.path)

        # plt.show()
    