# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:00:36 2018

@author: Michael
"""
import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_rcv1
import random
import math
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
def problem1a(rcv1):
    global article_number
    global feature_number
    target = rcv1.target
    data = rcv1.data
    article_number = target.shape[0]
    feature_number = data.shape[1] 
    label = np.zeros((article_number,1),dtype=np.int32)
    CCAT = target.toarray()[:,33]
    for i in range(0,article_number):
        if CCAT[i] == 1:
            label[i][0] = 1
        else:
            label[i][0] = -1
    return CCAT,label,data
def problem1b(rcv1,train_number):
    train_data = rcv1.data[0:train_number,:]
    train_label = label[0:train_number][:]
    test_data = rcv1.data[train_number:article_number,:]
    test_label = label[train_number:article_number,:]
    return train_data,train_label,test_data,test_label
def problem2(train_data,train_label,test_data,test_label,lambda1,T,A_t_number):
    error_rate,error_rate_test = pegasos(train_data,train_label,test_data,test_label,lambda1,T,A_t_number)
    min_error = [min(error_rate),error_rate.index(min(error_rate))]
    min_error_test = [min(error_rate_test),error_rate_test.index(min(error_rate_test))]
    x = range(1,T+1)
    plt.figure()
    plt.plot(x,error_rate)
    plt.xlabel('Iterations')
    plt.ylabel('Train error')
    plt.title('PEGASOS')
    plt.show()
    plt.figure()
    plt.plot(x,error_rate_test)
    plt.xlabel('Iterations')
    plt.ylabel('Test error')
    plt.title('PEGASOS')
    plt.show()
    return error_rate,error_rate_test,min_error,min_error_test
def problem3(train_data,train_label,test_data,test_label,const,lambda2,T,A_t_number):
    error_rate,error_rate_test = adagrad(train_data,train_label,test_data,test_label,const,lambda2,T,A_t_number)
    min_error = [min(error_rate),error_rate.index(min(error_rate))]
    min_error_test = [min(error_rate_test),error_rate_test.index(min(error_rate_test))]
    x = range(1,T+1)
    plt.figure()
    plt.plot(x,error_rate)
    plt.xlabel('Iterations')
    plt.ylabel('Train error')
    plt.title('ADAGRAD')
    plt.show()
    plt.figure()
    plt.plot(x,error_rate_test)
    plt.xlabel('Iterations')
    plt.ylabel('Test error')
    plt.title('ADAGRAD')
    plt.show()
    return error_rate,error_rate_test,min_error,min_error_test
def problem4(train_data,train_label,test_data,test_label,layers,nodes):
    K.tensorflow_backend._get_available_gpus()
    train_label[train_label==-1] = 0
    test_label[test_label==-1] = 0
    sgd = SGD(lr=0.1,decay = 0.0005)
    model = Sequential()
    model.add(Dense(nodes, input_dim=47236,activation='relu'))
    for layer in range(layers-1):
        model.add(Dense(nodes, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    model.fit(train_data, train_label, epochs=5, batch_size=10)
    loss, accuracy = model.evaluate(test_data, test_label, batch_size = 10)
    error_DNN = 1 - accuracy
    return loss, accuracy, error_DNN
def adagrad(train_data,train_label,test_data,test_label,const,lambda2,T,A_t_number):
#    lr = 1/math.sqrt(T)
    lr =0.1
    w = sparse.csc_matrix((1,train_data.shape[1]))
    accumulate_gradient = 0
    error_rate = []
    error_rate_test = []
    for t in range(1,T+1):
        wx_ = 0
        A_t = random.sample(range(train_data.shape[0]),A_t_number)
        for a_t in A_t:
            x = train_data[a_t,:]
            y = sparse.csc_matrix(train_label[a_t,:])
            wx = w.dot(x.transpose())
            ywx = wx[0,0]*train_label[a_t]
            if ywx < 1:
                wx_ += y*x          
        gradient = lambda2*w - wx_/A_t_number
        accumulate_gradient += gradient.power(2)
        G = np.sqrt(accumulate_gradient)
        w_prime = w - lr*sparse.csr_matrix(gradient.toarray()/(const + np.sqrt(accumulate_gradient.toarray()))) 
        w = min([1,1/math.sqrt(lambda2)/sparse.linalg.norm(G.multiply(w_prime))])*w_prime
        #Train error
        error_rate.append(error(train_data,train_label,w))
        #Test error for problem 5
        error_rate_test.append(error(test_data,test_label,w))
    return error_rate,error_rate_test
def pegasos(train_data,train_label,test_data,test_label,lambda1,T,A_t_number):
    w = sparse.csc_matrix((1,train_data.shape[1]))
    error_rate = []
    error_rate_test = []
    for t in range(1,T+1):
        wx_ = 0
        A_t = random.sample(range(train_data.shape[0]),A_t_number)
        for a_t in A_t:
            x = train_data[a_t,:]
            y = sparse.csc_matrix(train_label[a_t,:])
            wx = w.dot(x.transpose())
            ywx = wx[0,0]*train_label[a_t]
            if ywx < 1:
                wx_ += y*x         
        gradient = lambda1*w - wx_/A_t_number
        w_prime = w - (1/(t*lambda1))*gradient
        w = min([1,1/math.sqrt(lambda1)/sparse.linalg.norm(w_prime)])*w_prime
        #Train error
        error_rate.append(error(train_data,train_label,w))
        #Test error for problem 5
        error_rate_test.append(error(test_data,test_label,w))
    return error_rate,error_rate_test
def error(data,label,w):
    predict = w.dot(data.transpose()).toarray()
    wx = predict.transpose()*label
    res = np.sum(wx<0)
    error_rate = res/label.shape[0] 
    return error_rate
if __name__ == '__main__':
    rcv1 = fetch_rcv1()
    CCAT,label,data = problem1a(rcv1)
    train_data,train_label,test_data,test_label = problem1b(rcv1,100000)
    Train_error_P, Test_error_P, min_train_P, min_test_P = problem2(train_data,train_label,test_data,test_label,0.0001,2000,50)
    Train_error_A, Test_error_A,min_train_A,min_test_A= problem3(train_data,train_label,test_data,test_label,1e-7,0.0001,2000,50)
    loss1, accuracy1, error_DNN1 = problem4(train_data,train_label,test_data,test_label,5,375)
#    loss2, accuracy2, error_DNN2 = problem4(train_data,train_label,test_data,test_label,2,100)
#    loss3, accuracy3, error_DNN3 = problem4(train_data,train_label,test_data,test_label,3,100)
#    y = [error_DNN1,error_DNN2,error_DNN3]
#    x = [1,2,3]
#    plt.figure()    
#    plt.plot(x,y)
#    plt.xlabel('Hidden layers')
#    plt.ylabel('Train error')
#    plt.title('DNN')
#    plt.show()