from dl import Parameter,Sequential
from dl.layers import Linear,Sigmoid,Relu,Softmax_crossentropy
from dl.optimizer import SGD
from dl.utils import SummaryWriter


import numpy as np
from load_mnist import load_mnist

import pandas as pd


X_train,Y_train,x_test,y_test = load_mnist()

b_size = 64
lr = 1e-3
momentum = 0.9 

writer = SummaryWriter("./plots/loss_sigmoid_1e-3.png",title="Plot of Cross Entopy Loss vs No of Iters",xlab="No. of Iters",ylab="Cross Entopy Loss",legend=["Train","Test"])
writer1 = SummaryWriter("./plots/acc_sigmoid_1e-3.png",title="Plot of Accuracy vs No of Iters",xlab="No. of Iters",ylab="Accuracy",legend=["Train","Test"])


loss_fn = Softmax_crossentropy()


def evaluate(y_actual,y_pred):
    
    confusion_matrix = np.zeros([10,10],dtype=np.int)
    dict_ = {}

    for i in zip(y_actual.astype(np.int),y_pred.astype(np.int)):
        confusion_matrix[i] += 1
    
    true_label = np.sum(confusion_matrix,1).astype(np.float)
    pred_label = np.sum(confusion_matrix,0).astype(np.float)
    tp = confusion_matrix[np.arange(10),np.arange(10)].astype(np.float)

    precision = tp/pred_label
    recall = tp/true_label
    f_score = 2*precision*recall/(precision+recall)

    dict_ = {'Precision':precision,'Recall':recall,'F-Score':f_score}
    stats = pd.DataFrame(dict_)

    return confusion_matrix,stats


def train(x_train,y_train,x_val,y_val,model,opt):
    
    len_train = x_train.shape[0]

    def train_data_gen():

        while True:
            for idx in range(len_train//b_size):
                yield x_train[b_size*idx:b_size*(idx+1)],y_train[b_size*idx:b_size*(idx+1)]

            yield x_train[b_size*(idx+1):],y_train[b_size*(idx+1):]

    y_pred = np.array([]);y_actual=np.array([]);loss_v=np.array([])
    gen = train_data_gen()
    flag = False
    for idx in range(8000):
        
        x_batch,y_batch = next(gen)

        output = model(x_batch)
        result = loss_fn(y_batch,output)

        y_pred = np.concatenate([y_pred,result[0]])
        loss_v = np.concatenate([loss_v,result[1]])
        y_actual = np.concatenate([y_actual,y_batch])

        model.backward(loss_fn.backwad())
        opt.step()

        if (idx+1)%200 == 0:

            if (idx+1) == 8000:
                flag = True

            loss = np.mean(loss_v)
            accuracy = np.sum((y_actual == y_pred).astype(np.float32))/y_actual.shape[0]

            confusion_matrix,val_stats,val_loss_std,val_loss,val_acc = validate(x_val,y_val,model,flag)

            print 'Iteration: %d Train Loss: %f Train Accuracy: %f Val Loss: %f Val Accuracy: %f' %(idx+1,loss,accuracy,val_loss,val_acc) 
            writer.add_scalar([idx,loss,val_loss])
            writer1.add_scalar([idx,accuracy,val_acc])
            if flag:
                print "Confusion Matrix"
                print confusion_matrix
                print "Statistics"
                print val_stats
                print "Overall Accuracy: %f" %(val_acc)
                print "Average Error: %f Standard Deviation on Error: %f" %(val_loss,val_loss_std)

            y_pred = np.array([]);y_actual=np.array([]);loss_v=np.array([]);flag=False

    writer.close()
    writer1.close()

def validate(x_val,y_val,model,eval=False):

    len_val = x_val.shape[0]

    def val_data_gen():

        for idx in range(len_val//b_size):
            yield x_val[b_size*idx:b_size*(idx+1)], y_val[b_size*idx:b_size*(idx+1)]
        
        yield x_val[b_size*(idx+1):],y_val[b_size*(idx+1):]
        
    y_pred = np.array([]);y_actual=np.array([]);loss_v=np.array([])

    for x_batch,y_batch in val_data_gen():
        
        output = model(x_batch)
        result = loss_fn(y_batch,output)

        y_pred = np.concatenate([y_pred,result[0]])
        loss_v = np.concatenate([loss_v,result[1]])
        y_actual = np.concatenate([y_actual,y_batch])

    loss = np.mean(loss_v)
    accuracy = np.sum(y_actual == y_pred).astype(np.float32)/(y_actual.shape[0])

    if eval:
        confusion_matrix,stats = evaluate(y_actual,y_pred)
        loss_std = np.std(loss_v)
        return confusion_matrix,stats,loss_std,loss,accuracy

    return None,None,None,loss,accuracy


def k_fold_validation(X_train,Y_train,k=5):
    
    length = X_train.shape[0]
    fold_len = length/k

    for i in range(k):
        # ith fold is validation
        
        # model = Sequential(Linear(784,1000),Relu(),Linear(1000,500),Relu(),Linear(500,250),Relu(),Linear(250,10))

        # model = Sequential(Linear(784,1000),Sigmoid(),Linear(1000,500),Sigmoid(),Linear(500,250),Sigmoid(),Linear(250,10))

        model = Sequential(Linear(784,1000,l2=0.1),Sigmoid(),Linear(1000,500,l2=0.9),Sigmoid(),Linear(500,250,l2=0.9),Sigmoid(),Linear(250,10,l2=0.9))

        opt = SGD(model.parameters,lr=lr,momentum=momentum)

        opt.zero_grad()

        val_idx  = np.arange(i*fold_len,(i+1)*fold_len)
        train_idx = np.arange((i+1)*fold_len,(i+k-1)*fold_len)%length

        x_val = X_train[val_idx];y_val = Y_train[val_idx]
        x_train = X_train[train_idx];y_train = Y_train[train_idx]

        print 'K-Fold Validation {}/{}'.format(i+1,k)
        print '-'*30
        train(x_train,y_train,x_val,y_val,model,opt)
        break


if __name__ == '__main__':
    k_fold_validation(X_train,Y_train,5)