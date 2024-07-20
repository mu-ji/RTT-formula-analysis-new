'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this file include functions related to using machine learning solution to predict distance based on RTT
'''

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def ridge_training(X,Y):
    ridge_model = Ridge()
    ridge_model.fit(X,Y)
    return ridge_model

def lasso_training(X,Y):
    lasso_model = Lasso()
    lasso_model.fit(X,Y)
    return lasso_model

def ridge_predicting(ridge_model,test_x):
    ridge_predictions = ridge_model.predict(test_x)
    return ridge_predictions

def lasso_predicting(lasso_model,test_x):
    lasso_predictions = lasso_model.predict(test_x)
    return lasso_predictions

def data_process_ML(train_set_file):
    with open(train_set_file, 'r') as train_file:
        lines = train_file.readlines()

    X = np.array([0,0,0,0])
    for i in range(len(lines)):
        line = lines[i].strip().split(' ')
        line = [float(num) for num in line]
        
        rtt_mean = np.mean(line[:200])
        rtt_var = np.var(line[:200])
        rssi_mean = np.mean(line[200:])
        rssi_var = np.var(line[200:])
        X = np.vstack((X, np.array([rtt_mean,rtt_var,rssi_mean,rssi_var])))

    X = X[1:,:]  #去掉第一行的0

    degree = 2
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    return X_poly

def generate_train_test_y():
    distance_list = [i for i in range(1,12)]
    train_y = np.array([0])
    for distance in distance_list:
        for j in range(int(len(train_x)/len(distance_list))):
            train_y = np.vstack((train_y,distance))
    train_y = train_y[1:,:]

    test_y = np.array([0])
    for distance in distance_list:
        for j in range(int(len(test_x)/len(distance_list))):
            test_y = np.vstack((test_y,distance))
    test_y = test_y[1:,:]
    return train_y, test_y

def transform_error(error_array):
    error_list = []
    for i in range(11):
        error_list.append(error_array.reshape(220,)[i*20:(i+1)*20])
    error_array = np.array(error_list).T
    return error_array

def traditional_prediction(data):
    prediction_list = []
    for i in range(len(data)):
        prediction = (data[i][1] - 20074.659)/2*300000000/16000000*0.4
        prediction_list.append(np.array(prediction))
    predictions = np.array(prediction_list)
    return predictions

train_x = data_process_ML('train_set/outdoor_train_set.txt')
test_x = data_process_ML('test_set/outdoor_test_set.txt')

train_x = data_process_ML('train_set/indoor_with_people_walking_train_set.txt')
test_x = data_process_ML('test_set/indoor_with_people_walking_test_set.txt')

train_x = data_process_ML('train_set/indoor_without_people_walking_train_set.txt')
test_x = data_process_ML('test_set/indoor_without_people_walking_test_set.txt')
'''
train_y,test_y = generate_train_test_y()

ridge_model = ridge_training(train_x,train_y)
ridge_predictions = ridge_predicting(ridge_model,test_x)

lasso_model = lasso_training(train_x,train_y)
lasso_predictions = lasso_predicting(lasso_model,test_x)

ridge_error = ridge_predictions - test_y
ridge_error = transform_error(ridge_error)
lasso_error = lasso_predictions.reshape((220,1)) - test_y
lasso_error = transform_error(lasso_error)

traditional_predictions = traditional_prediction(test_x)
traditional_error = traditional_predictions.reshape((220,1)) - test_y
traditional_error = transform_error(traditional_error)


boxprops = dict(facecolor='lightblue', color='blue')
plt.violinplot(ridge_error,positions=[i-0.2 for i in range(1,23,2)],showmeans=True,widths=0.3)
boxprops = dict(facecolor='red', color='maroon')
plt.violinplot(lasso_error,positions=[i+0.2 for i in range(1,23,2)],showmeans=True,widths=0.3)
#boxprops = dict(facecolor='green', color='green')
#plt.violinplot(traditional_error,positions=[i+0.5 for i in range(2,34,3)],showmeans=True)

rect_ridge = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue')
rect_lasso = plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='orange')
#rect_traditional = plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='green')
plt.legend([rect_ridge, rect_lasso], ['ridge model error', 'lasso model error'])

labels = (['{} meters'.format(i) for i in range(1,12)])
plt.xticks([i+1 for i in range(1,23,2)], labels)
plt.title('Ridge and Lasso model prediction error in different distance(oindoor environment without people walking)')
plt.ylabel('error(meters)')
plt.grid()
plt.show()
'''
