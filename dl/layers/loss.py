import numpy as np

class Softmax_crossentropy():

    def forward(self,target,logit,return_softmax=False):

        
        len_ = target.shape[0]
        y = np.zeros([len_,10])
        y[np.arange(len_),target] = 1

        self.target = target
        self.y = y

        normalized_logit = logit - np.max(logit,1,keepdims=True)
        self.normalized_logit_exp = np.exp(normalized_logit)
        self.normalized_logit_sum = np.sum(self.normalized_logit_exp,1)
        normalized_log_sum =  np.log(self.normalized_logit_sum)
        
        self.softmax = self.normalized_logit_exp/np.expand_dims(self.normalized_logit_sum,1)
        self.output = -1*(normalized_logit[np.arange(len_),target]  - normalized_log_sum)
        
        if return_softmax:
            return self.softmax,self.output
        
        y_pred = np.argmax(self.softmax,1)
        return y_pred,self.output 

    def backwad(self):

        self.grad = (self.softmax - self.y)
        return self.grad 

    def __call__ (self,target,logit):
        return self.forward(target,logit)