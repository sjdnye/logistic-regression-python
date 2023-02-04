import numpy as np
import matplotlib.pyplot as plt
import random
import math
import numdifftools as nd
import tensorflow as tf

raw_data = np.load('data2d.npz')
X = raw_data['X']
y = raw_data['y']
data = (X,y)
w = tf.Variable(tf.random.uniform(shape = [np.shape(X)[1]], minval = 0, maxval = 1, dtype = tf.float32), name = 'w')
b = tf.Variable(tf.random.uniform(shape = [], minval = 0, maxval = 1, dtype = tf.float32), name = 'b')
show_function = 2 if np.shape(X)[1] == 2 else 5


learning_rate = 0.005
step = 0
stop_flag = 0


def Phi(x):
    return 1/(1 + tf.math.exp(-x))

def C(w,b,data):
    result = 0
    X = data[0]
    y = data[1]
    for i in range(len(X)):
        result += tf.math.pow(Phi( tf.reduce_sum(tf.multiply(w, X[i])) + b) - y[i] , 2)

    return result  



def classify(X, y, w, b):
    global stop_flag
    phi_result = []
    y_result = []
    misclassified_num = 0

    for i in range(len(X)):
        phi_result.append(1 / (1 + tf.math.exp(-tf.reduce_sum(tf.multiply(w, X[i])) - b)))

    y_result = np.array([ 0 if phi_result[i] < 0.5 else 1 for i in range(len(phi_result)) ])

    for i in range(len(y)):
        if y_result[i] != y[i]:
            misclassified_num = misclassified_num + 1
    
    print(f"Training error ratio in this step is --> {misclassified_num / len(y)}\n ---------------------------------------------------") 
    if(misclassified_num / len(y) < 0.2):
        stop_flag = 1

    final_result = {
        'X' : X,
        'y' : y_result
    }

    return final_result

def scatter_2d_data(w, b, data):
    X = data['X']
    y = data['y']

    plt.cla()


    x_values = X[:, 0]
    y_values = (w[0] * X[:, 0] + b) / (-w[1])
    plt.plot(x_values, y_values, '-', color='g')

    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='r')
        elif y[i] == 0:
            plt.scatter(X[i, 0], X[i, 1], color='b') 
    plt.draw()
    plt.pause(.1)         

def scatter_5d_data(w,b, data):
    X = data['X']
    y = data['y']

    plt.cla()
    
    x_values = X[:, 1]
    y_values = (w[1] * X[:, 1] + b) / (-w[2])
    plt.plot(x_values, y_values, '-', color='g')

    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], color='r')
        elif y[i] == 0:
            plt.scatter(X[i, 1], X[i, 2], color='b')
    plt.draw()
    plt.pause(.1) 

 

# print(dc_dw)
# print(dc_db) 

# with tf.GradientTape() as gfg:
#     gfg.watch(w)
#     cost_function = C(w, b, data)
# res = gfg.gradient(cost_function,w)   
# print(f"result: {res}") 
    

 

while True:

    step = step + 1

    with tf.GradientTape() as tape:
        cost_function = C(w, b, data)  

    [dc_dw, dc_db] = tape.gradient(cost_function, [w, b]) 
    classified_data = classify(X, y, w, b)
    print(f"Cost function in step ${step} is --> { C(w, b, data)}")
    
    if show_function == 2:
        scatter_2d_data(w, b, classified_data)
    else:
        scatter_5d_data(w, b, classified_data)  

    w = tf.Variable(initial_value=w - learning_rate * dc_dw)
    b = tf.Variable(initial_value=b - learning_rate * dc_db)

    if  np.linalg.norm(np.array(dc_dw)) < 10 and np.linalg.norm(np.array(dc_db)) < 0.5:
        print(f"Best solution in step ${step}")
        break

    if stop_flag == 1:
        print(f"Best solution in last step (${step})")
        break

plt.show()
