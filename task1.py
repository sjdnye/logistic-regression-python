import numpy as np
import matplotlib.pyplot as plt
import random
import math
import numdifftools as nd


raw_data = np.load('data2d.npz')
X_global = raw_data['X']
y_global = raw_data['y']
X = X_global
y = y_global
data = (X_global,y_global)

w = np.random.rand(np.shape(X_global)[1])
b = random.uniform(0, 1)

# plt.scatter(X[:,0],X[:,1])
# plt.show()

show_function = 2 if np.shape(X_global)[1] == 2 else 5

stop_flag = 0
learning_rate = 0.0005
step = 0


def Phi(x):
    return 1/(1 + np.exp(-x))


def C(w,b,data):
    result = 0
    X = data[0]
    y = data[1]
    for i in range(len(X)):
        result += pow((Phi(np.dot(X[i], w) + b) - y[i]), 2)

    return result   


def compute_dC_dw(w,b,data):
    result = 0
    X = data[0]
    y = data[1]
    for i in range(len(X)):
        result += ( (X[i] * np.exp(-w.T @ X[i] - b)) / (pow(1 + np.exp(-w.T @ X[i] - b),2))) * ( 1 / (1 + np.exp(-w.T @ X[i] - b)) - y[i])

    return 2 * result    

def compute_dC_db(w,b,data):
    result = 0
    X = data[0]
    y = data[1]
    for i in range(len(X)):
        result += ( ( np.exp(-w.T @ X[i] - b)) / (pow(1 + np.exp(-w.T @ X[i] - b),2))) * ( 1 / (1 + np.exp(-w.T @ X[i] - b)) - y[i])

    return 2 * result    


def compute_dC_dw_numeric(w,b, data):
    eps = 1e-6
    
    compute_1 = C(w, b, data)
    compute_2 = C(w + eps, b, data)

    result = (compute_1 - compute_2) / eps 
    return result

def compute_dC_db_numeric(w,b, data):
    eps = 1e-6
    
    compute_1 = C(w, b, data)
    compute_2 = C(w, b + eps, data)

    result = (compute_1 - compute_2) / eps 
    return result

def classify(X, y, w, b):
    global stop_flag
    phi_result = []
    y_result = []
    misclassified_num = 0

    for i in range(len(X)):
        phi_result.append((1 / (1 + np.exp(-w.T @ X[i] - b))))

    y_result = np.array([ 0 if phi_result[i] < 0.5 else 1 for i in range(len(phi_result)) ])

    for i in range(len(y)):
        if y_result[i] != y[i]:
            misclassified_num = misclassified_num + 1
    
    print(f"Training error ratio in this step is --> {misclassified_num / len(y)}\n ---------------------------------------------------") 
    if(misclassified_num / len(y) < 0.02):
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
    y_values = (w[1] * X[:, 2] + b) / (-w[2])
    plt.plot(x_values, y_values, '-', color='g')

    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], color='r')
        elif y[i] == 0:
            plt.scatter(X[i, 1], X[i, 2], color='b')
    plt.draw()
    plt.pause(.1) 

            
    



dC_dw = compute_dC_dw(w,b, data)
dC_db = compute_dC_db(w,b, data)
# print(dC_dw)
# print(dC_db)    
dC_dw_n = compute_dC_dw_numeric(w,b, data)
dC_db_n = compute_dC_db_numeric(w,b, data)
# print(dC_dw_n)
# print(dC_db_n) 
#------------------------------------------------------ length of difference between actual and numeric gradient ----------------------------------------------------------
print(f"Absolute error of actual and numeric gradients with respect to w is --> ${np.linalg.norm(dC_dw - dC_dw_n)}")
print(f"Relative error of actual and numeric gradients with respect to w is --> ${np.linalg.norm(dC_dw - dC_dw_n) / np.linalg.norm(dC_dw) }")
print(f"Absolute error of actual and numeric gradients with respect to b is --> ${np.linalg.norm(dC_db - dC_db_n)}")
print(f"Relative error of actual and numeric gradients with respect to b is --> ${np.linalg.norm(dC_db - dC_db_n) / np.linalg.norm(dC_db) }")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


while True:

    step = step + 1

    dC_dw = compute_dC_dw(w,b, data)
    dC_db = compute_dC_db(w,b, data)

    classified_data = classify(X, y, w, b)
    print(f"Cost function in step ${step} is --> { C(w, b, data)}")
    
    if show_function == 2:
        scatter_2d_data(w, b, classified_data)
    else:
        scatter_5d_data(w, b, classified_data)    

    w = w - learning_rate * dC_dw
    b = b - learning_rate * dC_db

    if  np.linalg.norm(np.array(dC_dw)) < 20 and np.linalg.norm(np.array(dC_db)) < 1:
        print(f"Best solution in step ${step}")
        break

    if stop_flag == 1:
        print(f"Best solution in last step ${step}")
        break

plt.show()
