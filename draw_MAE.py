import numpy as np
import matplotlib.pyplot as plt
import machine_learning as ML
import neural_network_with_GMM as NN
import torch
import torch.nn as nn

ML_train_x = ML.data_process_ML('train_set/outdoor_train_set.txt')
ML_test_x = ML.data_process_ML('test_set/outdoor_test_set.txt')
NN_train_x = NN.data_process_NN('train_set/outdoor_train_set.txt')
NN_test_x = NN.data_process_NN('test_set/outdoor_test_set.txt')

ML_train_x = ML.data_process_ML('train_set/indoor_with_people_walking_train_set.txt')
ML_test_x = ML.data_process_ML('test_set/indoor_with_people_walking_test_set.txt')
NN_train_x = NN.data_process_NN('train_set/indoor_with_people_walking_train_set.txt')
NN_test_x = NN.data_process_NN('test_set/indoor_with_people_walking_test_set.txt')

ML_train_x = ML.data_process_ML('train_set/indoor_without_people_walking_train_set.txt')
ML_test_x = ML.data_process_ML('test_set/indoor_without_people_walking_test_set.txt')
NN_train_x = NN.data_process_NN('train_set/indoor_without_people_walking_train_set.txt')
NN_test_x = NN.data_process_NN('test_set/indoor_without_people_walking_test_set.txt')

train_y,test_y = ML.generate_train_test_y()

def compute_error(ML_train_x,ML_test_x,NN_train_x,NN_test_x,train_y,test_y):
    ridge_model = ML.ridge_training(ML_train_x,train_y)
    ridge_predictions = ML.ridge_predicting(ridge_model,ML_test_x)

    lasso_model = ML.lasso_training(ML_train_x,train_y)
    lasso_predictions = ML.lasso_predicting(lasso_model,ML_test_x)

    ridge_error = ridge_predictions - test_y
    ridge_error = ML.transform_error(ridge_error)
    lasso_error = lasso_predictions.reshape((220,1)) - test_y
    lasso_error = ML.transform_error(lasso_error)

    X = torch.from_numpy(NN_train_x[:,:]).float()
    Y = torch.from_numpy(train_y).float()
    NN_test_X = torch.from_numpy(NN_test_x[:,:]).float()
    NN_test_Y = torch.from_numpy(test_y).float()

    X_hat = torch.from_numpy(NN_train_x[:,6:]).float()
    test_X_het = torch.from_numpy(NN_test_x[:,6:]).float()

    NN_with_GMM_model = NN.neural_network_with_GMM_train(X,Y)
    NN_without_GMM_model = NN.neural_network_without_GMM_train(X_hat,Y)

    NN_with_GMM_predictions = NN.neural_network_predicting(NN_with_GMM_model,NN_test_X)
    NN_without_GMM_predictions = NN.neural_network_predicting(NN_without_GMM_model,test_X_het)

    NN_with_GMM_error = NN_with_GMM_predictions - test_y
    NN_with_GMM_error = NN.transform_error(NN_with_GMM_error)
    NN_without_GMM_error = NN_without_GMM_predictions - test_y
    NN_without_GMM_error = NN.transform_error(NN_without_GMM_error)

    traditional_predictions = ML.traditional_prediction(ML_test_x)
    traditional_error = traditional_predictions.reshape((220,1)) - test_y
    traditional_error = ML.transform_error(traditional_error)

    ridge_error = np.mean(np.abs(ridge_error))
    lasso_error = np.mean(np.abs(lasso_error))
    NN_with_GMM_error = np.mean(np.abs(NN_with_GMM_error))
    NN_without_GMM_error = np.mean(np.abs(NN_without_GMM_error))
    traditional_error = np.mean(np.abs(traditional_error))
    return [ridge_error,lasso_error,NN_with_GMM_error,NN_without_GMM_error,traditional_error]

ML_train_x = ML.data_process_ML('train_set/outdoor_train_set.txt')
ML_test_x = ML.data_process_ML('test_set/outdoor_test_set.txt')
NN_train_x = NN.data_process_NN('train_set/outdoor_train_set.txt')
NN_test_x = NN.data_process_NN('test_set/outdoor_test_set.txt')
train_y,test_y = ML.generate_train_test_y()
error_list_outdoor = compute_error(ML_train_x,ML_test_x,NN_train_x,NN_test_x,train_y,test_y)

ML_train_x = ML.data_process_ML('train_set/indoor_without_people_walking_train_set.txt')
ML_test_x = ML.data_process_ML('test_set/indoor_without_people_walking_test_set.txt')
NN_train_x = NN.data_process_NN('train_set/indoor_without_people_walking_train_set.txt')
NN_test_x = NN.data_process_NN('test_set/indoor_without_people_walking_test_set.txt')
train_y,test_y = ML.generate_train_test_y()
error_list_indoor_without = compute_error(ML_train_x,ML_test_x,NN_train_x,NN_test_x,train_y,test_y)

ML_train_x = ML.data_process_ML('train_set/indoor_with_people_walking_train_set.txt')
ML_test_x = ML.data_process_ML('test_set/indoor_with_people_walking_test_set.txt')
NN_train_x = NN.data_process_NN('train_set/indoor_with_people_walking_train_set.txt')
NN_test_x = NN.data_process_NN('test_set/indoor_with_people_walking_test_set.txt')
train_y,test_y = ML.generate_train_test_y()
error_list_indoor_with = compute_error(ML_train_x,ML_test_x,NN_train_x,NN_test_x,train_y,test_y)

plt.figure()
plt.bar([0.6,0.8,1,1.2,1.4],error_list_outdoor,color=['red','blue','purple','green','orange'],width=0.2)
plt.bar([2.6,2.8,3,3.2,3.4],error_list_indoor_without,color=['red','blue','purple','green','orange'],width=0.2)
plt.bar([4.6,4.8,5,5.2,5.4],error_list_indoor_with,color=['red','blue','purple','green','orange'],width=0.2)

rect_ridge = plt.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='red')
rect_lasso = plt.Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='blue')
rect_with = plt.Rectangle((0, 0), 1, 1, facecolor='purple', edgecolor='purple')
rect_wthout = plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='green')
rect_traditional = plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='orange')
plt.legend([rect_ridge, rect_lasso, rect_with, rect_wthout, rect_traditional], ['Ridge model error','lasso model error','NN with GMM error', 'NN without GMM error', 'traditional error'])

labels = (['outdoor','indoor without people walking','indoor with prople walking'])
plt.xticks([i for i in range(1,6,2)], labels)
plt.grid()
plt.ylabel('MAE (meters)')
plt.show()