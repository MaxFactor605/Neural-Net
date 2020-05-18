#!/usr/bin/env python3
import numpy as np
import os
import shelve
from PIL import Image as Img
import matplotlib.pyplot as plt
from tkinter import *
TRAIN = False
HEIGHT = 100
WIDHT = 100
NUM_LABELS = 10
INPUT_LAYER_SIZE = 401
HIDDEN_LAYER_SIZE = 26
LAMBDA = 15
ALPHA = 0.01
GRAD_CHECK = False
NUM_ITERS = 10
TEST = False

class MLDraw(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.pack()
        self.canvas = Canvas(self, bg='white', height=HEIGHT+15, width=WIDHT+15)
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', lambda event: self.freedraw(event))
        self.canvas.bind('<Button-1>', self.freedraw)
        self.thick = 2
        self.makebuttons()
        self.prediction = Label(self)
        self.prediction.pack(side=BOTTOM)

    def freedraw(self, event):
        self.canvas.create_oval(event.x - self.thick, event.y - self.thick, event.x + self.thick,
                                event.y + self.thick, fill='black', width=0)

    def makebuttons(self):
        butframe = Frame(self)
        butframe.pack(side=TOP, expand=YES, fill='x')
        Button(butframe, text='Read', command=self.predict).pack(side=LEFT)
        Button(butframe, text='Clear', command=self.clear).pack(side=RIGHT)

    def get_data(self):
        self.canvas.postscript(file='temp.jpg')
        im = Img.open('temp.jpg')
        rez_im = im.resize((20, 20))
        rez_im.save('temp1.jpg')
        data = np.asarray(rez_im)
        X = np.zeros((20, 20))
        for y in range(20):
            for x in range(20):
                X[x][y] = 1 if max(data[x][y]) == 255 else 0
        X = X.reshape((INPUT_LAYER_SIZE - 1, 1))
        return X
    def predict(self):
        X = self.get_data()
        a1 = np.vstack((np.array([1]), X.reshape((INPUT_LAYER_SIZE - 1, 1))))
        z2 = Theta1 @ a1
        a2 = np.vstack((np.array([1]), sigmoid(z2)))
        z3 = Theta2 @ a2
        a3 = sigmoid(z3)
        self.prediction.config(text='Prediction => {0}'.format(list(a3).index(max(a3))))


    def clear(self):
        self.prediction.config(text='')
        self.canvas.delete(ALL)


def get_data(filename):
    dbase = shelve.open(filename)
    X = dbase['1'][0].T
    Y = dbase['1'][1].T
    for t in range(2, len(dbase.keys())):
        X = np.vstack((X, dbase[str(t)][0].T))
        Y = np.vstack((Y, dbase[str(t)][1].T))
    dbase.close()
    print('y => ',Y.shape, 'x => ', X.shape)
    return X, Y

def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))


def save_weights(T1, T2):
    dbase = shelve.open('weights.db')
    dbase['Theta1'] = T1
    dbase['Theta2'] = T2
    dbase.close()


def sigmoid_gradient(Z):
    return sigmoid(Z)*(1-sigmoid(Z))


def theta_initilization(filename):
    if os.path.exists(filename):
        dbase = shelve.open(filename)
        Theta1, Theta2 = dbase['Theta1'], dbase['Theta2']
    else:
        Theta1 = np.random.uniform(-0.24, 0.24,[HIDDEN_LAYER_SIZE-1,INPUT_LAYER_SIZE])
        Theta2 = np.random.uniform(-0.24, 0.24,[NUM_LABELS, HIDDEN_LAYER_SIZE])
        dbase = shelve.open(filename)
        dbase['Theta1'] = Theta1
        dbase['Theta2'] = Theta2
    dbase.close()
    return Theta1, Theta2


def cost_function(X, Y, Theta1, Theta2):
    m = X.shape[0]
    h = forward_propagation(X, Theta1, Theta2)
    return (1/m) * sum(sum(-Y.T*np.log(h.T) - (1-Y).T*np.log(1-h.T))) + (LAMBDA/(2*m))*(sum(sum(Theta1[:,1:] ** 2)) + sum(sum(Theta2[:, 1:]**2)))


def test_cost_function(X, Y, Theta1, Theta2):
    m = X.shape[0]
    h = forward_propagation(X, Theta1, Theta2)
    return (1/m) * sum(sum(-Y.T*np.log(h.T) - (1-Y).T*np.log(1-h.T)))

def cost_function_for_check(X, Y, theta):

    Theta1 = theta[:(HIDDEN_LAYER_SIZE-1)*INPUT_LAYER_SIZE].reshape((HIDDEN_LAYER_SIZE-1, INPUT_LAYER_SIZE))
    Theta2 = theta[Theta1.size:].reshape((NUM_LABELS, HIDDEN_LAYER_SIZE))
    m = X.shape[0]
    h = forward_propagation(X, Theta1, Theta2)
    return (1/m) * sum(sum(-Y.T*np.log(h.T) - (1-Y).T*np.log(1-h.T))) + (LAMBDA/(2*m))*(sum(sum(Theta1[:,1:] ** 2)) + sum(sum(Theta2[:, 1:]**2)))

def forward_propagation(X, Theta1, Theta2):
    m = X.shape[0]
    h = np.zeros((m, NUM_LABELS))
    for i in range(0, m):
        a1 = np.vstack((np.array([1]), X[i].reshape((INPUT_LAYER_SIZE-1,1))))
        z2 = Theta1@a1
        a2 = np.vstack((np.array([1]), sigmoid(z2)))
        z3 = Theta2@a2
        a3 = sigmoid(z3)

        h[i] = a3.reshape((NUM_LABELS))

    return h



def back_propagation(X, Y, Theta1, Theta2):
    delta1 = np.zeros(np.shape(Theta1))
    delta2 = np.zeros(np.shape(Theta2))
    for i in range(0, m):
        a1 = np.vstack((np.array([1]), X[i].reshape((INPUT_LAYER_SIZE-1,1))))
        z2 = Theta1@a1
        a2 = np.vstack((np.array([1]), sigmoid(z2)))
        z3 = Theta2@a2
        a3 = sigmoid(z3)
        error3 = a3 - Y[i].reshape((NUM_LABELS,1))
        error2 = ((Theta2).T@error3)[1:] * sigmoid_gradient(z2)

        delta1 = delta1 + error2@a1.T
        delta2 = delta2 + error3@a2.T

    Theta1_grad = (1/m) * delta1[:, 0].reshape((HIDDEN_LAYER_SIZE - 1, 1))
    Theta1_grad = np.hstack((Theta1_grad, (1/m) * delta1[:, 1:] + (LAMBDA/m) * Theta1[:, 1:]))
    Theta2_grad = (1/m) * delta2[:, 0].reshape((NUM_LABELS, 1))
    Theta2_grad = np.hstack((Theta2_grad, (1/m) * delta2[:, 1:] + (LAMBDA/m) * Theta2[:, 1:]))
    return Theta1_grad, Theta2_grad


def gradient_check(X, Y, Theta1, Theta2):
    theta = Theta1.reshape((Theta1.size,1))
    theta = np.vstack((theta, Theta2.reshape((Theta2.size, 1))))
    e = 0.0001
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    for p in range(0, theta.size):
        perturb[p] = e
        loss1 = cost_function_for_check(X, Y, theta - perturb)
        loss2 = cost_function_for_check(X, Y, theta + perturb)
        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return numgrad


def gradient_descent(X, Y, Theta1, Theta2):
    J_history = []
    J_test_history = []
    for i in range(1, NUM_ITERS):
        print('GRADIENT DESCENT ITERATION => ', i)
        Theta1_grad, Theta2_grad = back_propagation(X, Y, Theta1, Theta2)
        Theta1 = Theta1 - ALPHA*Theta1_grad
        Theta2 = Theta2 - ALPHA*Theta2_grad
        J_history.append(test_cost_function(X, Y, Theta1, Theta2))
        J_test_history.append(test_cost_function(Xtest, Ytest, Theta1, Theta2))
    return Theta1, Theta2,  J_history, J_test_history




if __name__ == '__main__':
    print('Initialize Theta...')
    Theta1, Theta2 = theta_initilization('weights.db')
    if TRAIN:
        X, Y = get_data('Train_set.db')
        Xtest, Ytest = get_data('Test_set.db')
        m = X.shape[0]
        Theta1, Theta2, J_history, J_test_history = gradient_descent(X, Y, Theta1, Theta2)
        print(J_history)
        print(J_test_history)
        plt.plot(J_test_history, label='Test')
        plt.plot(J_history, label='Train')
        plt.legend()
        plt.show()
        save_weights(Theta1, Theta2)
    elif GRAD_CHECK:
        X, Y = get_data('Train_set.db')
        m = X.shape[0]
        print('Start gradient_check')
        numgrad = gradient_check(X,Y, Theta1, Theta2)
        print('Start back_propagation')
        grad1, grad2= back_propagation(X,Y, Theta1, Theta2)
        grad = grad1.reshape((grad1.size, 1))
        grad = np.vstack((grad, grad2.reshape((grad2.size, 1))))
        print('Numgrad\t\t\tGrad')
        print('{0}      {1}'.format(numgrad[:10], grad[:10]))
        print('Diffrence => ', ((sum(abs(numgrad - grad) ** 2)) ** (1/2))/(sum(abs(numgrad + grad) ** 2)) ** (1/2))
    elif TEST:
        print('get train data')
        Xtrain, Ytrain = get_data('Train_set.db')
        print('get test data')
        Xtest, Ytest = get_data('Test_set.db')
        print('Train Cost => {0} \nTest Cost => {1}\nDifference => {2}'.format(test_cost_function(Xtrain, Ytrain, Theta1, Theta2), test_cost_function(Xtest, Ytest, Theta1, Theta2),
                                                                               test_cost_function(Xtest, Ytest, Theta1, Theta2)-test_cost_function(Xtrain, Ytrain, Theta1, Theta2)))
    else:
        root = Tk()
        MLDraw(root).pack()
        root.mainloop()

