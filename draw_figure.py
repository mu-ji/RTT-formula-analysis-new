import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import torch

import machine_learning as ml
import neural_network_with_GMM as nn_GMM
import curve_fit as cf

def compute_err(distance_list, pre_list):
    error_list = []
    for i in range(len(distance_list)):
        error = np.abs(distance_list[i] - pre_list[i])
        error_list.append(error)
    
    return np.mean(error_list)

def read_data(distance):
    f = open ('indoor_with_people_walking/distance{}.txt'.format(distance), 'r')
    time_list = []
    rssi_list = []
    data = f.readlines()
    #print(data)
    for i in range(len(data)):
        time_list.append(float(data[i].split(' ')[0]))
        rssi_list.append(float(data[i].split(' ')[1]))

    time = np.array(time_list)
    rssi = np.array(rssi_list)

    data = np.vstack((time,rssi))
    return data

def GMM_filter(data):
    #best_aic,best_bic = compute_number_of_components(data,1,5)
    #n_components = best_aic  # 设置成分数量
    n_components = 2
    gmm = GaussianMixture(n_components=n_components)

    try:
        gmm.fit(data)
    except:
        lenghts = len(data)
        gmm.fit(data.reshape((lenghts,1)))
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    return means,covariances,weights

def construct_train_and_test_data(data_list):
    train_data = []
    test_data = []
    for i in data_list:
        train_data.append(i[:,:60000])
        test_data.append(i[:,60000:])
    return train_data,test_data

def build_trainset_and_testset(train_data,test_data,n,p):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    train_x_time_component1_mean = []
    train_x_time_component1_var = []
    train_x_time_component2_mean = []
    train_x_time_component2_var = []
    train_x_time_component1_weight = []
    train_x_time_component2_weight = []
    train_x_rssi_mean = []
    train_x_rssi_var = []
    train_x_time_mean = []
    train_x_time_var = []

    test_x_time_component1_mean = []
    test_x_time_component1_var = []
    test_x_time_component2_mean = []
    test_x_time_component2_var = []
    test_x_time_component1_weight = []
    test_x_time_component2_weight = []
    test_x_rssi_mean = []
    test_x_rssi_var = []
    test_x_time_mean = []
    test_x_time_var = []

    for i in range(len(train_data)):
        for j in range(n):
            k = np.random.randint(60000 - p)
            #print('1:',train_data[i,:,k:k+p])
            train_x.append(train_data[i,:,k:k+p])
            train_y.append(i)

            means,covariances,weights = GMM_filter(train_data[i,0,k:k+p])
            train_x_time_component1_mean.append(means[0][0])
            train_x_time_component1_var.append(covariances[0][0][0])
            train_x_time_component2_mean.append(means[1][0])
            train_x_time_component2_var.append(covariances[1][0][0])
            train_x_time_component1_weight.append(weights[0])
            train_x_time_component2_weight.append(weights[1])

            train_x_rssi_mean.append(np.mean(train_data[i,1,k:k+p]))
            train_x_rssi_var.append(np.var(train_data[i,1,k:k+p]))
            train_x_time_mean.append(np.mean(train_data[i,0,k:k+p]))
            train_x_time_var.append(np.var(train_data[i,0,k:k+p]))

        k = np.random.randint(60000 - p)
        test_x.append(test_data[i,:,k:k+p])
        test_y.append(i)

        means,covariances,weights = GMM_filter(test_data[i,0,k:k+p])
        test_x_time_component1_mean.append(means[0][0])
        test_x_time_component1_var.append(covariances[0][0][0])
        test_x_time_component2_mean.append(means[1][0])
        test_x_time_component2_var.append(covariances[1][0][0])
        test_x_time_component1_weight.append(weights[0])
        test_x_time_component2_weight.append(weights[1])

        test_x_rssi_mean.append(np.mean(test_data[i,1,k:k+p]))
        test_x_rssi_var.append(np.var(test_data[i,1,k:k+p]))
        test_x_time_mean.append(np.mean(test_data[i,0,k:k+p]))
        test_x_time_var.append(np.var(test_data[i,0,k:k+p]))

        train_packet = [train_x,train_y,
                        train_x_time_component1_mean,train_x_time_component1_var,
                        train_x_time_component2_mean,train_x_time_component2_var,
                        train_x_time_component1_weight,train_x_time_component2_weight,
                        train_x_rssi_mean,train_x_rssi_var,
                        train_x_time_mean,train_x_time_var]
        
        test_packet = [test_x,test_y,
                        test_x_time_component1_mean,test_x_time_component1_var,
                        test_x_time_component2_mean,test_x_time_component2_var,
                        test_x_time_component1_weight,test_x_time_component2_weight,
                        test_x_rssi_mean,test_x_rssi_var,
                        test_x_time_mean,test_x_time_var]
    
    return train_packet,test_packet


def multi_prediction():
    sample_times = 100
    distance_list = [i for i in range(1,12)]
    data_list = []
    train_data = []
    test_data = []
    for i in distance_list:
        data = read_data(i)
        data_list.append(data)

    train_data,test_data = construct_train_and_test_data(data_list)
    train_dataset = np.array(train_data)       #(16,2,1000)
    train_dataset[:,0:1,:] -= 20074.659
    test_dataset = np.array(test_data)         #(16,2,1000)
    test_dataset[:,0:1,:] -= 20074.659

    train_packet,test_packet = build_trainset_and_testset(train_dataset,test_dataset,sample_times,200)

    train_x = np.array(train_packet[0])
    train_y = np.array(train_packet[1])

    train_x_time_component1_mean = np.array(train_packet[2])
    train_x_time_component1_var = np.array(train_packet[3])
    train_x_time_component2_mean = np.array(train_packet[4])
    train_x_time_component2_var = np.array(train_packet[5])
    train_x_time_component1_weight = np.array(train_packet[6])
    train_x_time_component2_weight = np.array(train_packet[7])

    train_x_rssi_mean = np.array(train_packet[8])
    train_x_rssi_var = np.array(train_packet[9])
    train_x_time_mean = np.array(train_packet[10])
    train_x_time_var = np.array(train_packet[11])

    test_x = np.array(test_packet[0])
    test_y = np.array(test_packet[1])

    test_x_time_component1_mean = np.array(test_packet[2])
    test_x_time_component1_var = np.array(test_packet[3])
    test_x_time_component2_mean = np.array(test_packet[4])
    test_x_time_component2_var = np.array(test_packet[5])
    test_x_time_component1_weight = np.array(test_packet[6])
    test_x_time_component2_weight = np.array(test_packet[7])

    test_x_rssi_mean = np.array(test_packet[8])
    test_x_rssi_var = np.array(test_packet[9])
    test_x_time_mean = np.array(test_packet[10])
    test_x_time_var = np.array(test_packet[11])

    train_merged_array = np.concatenate((train_x_time_component1_mean[:, np.newaxis],
                                            train_x_time_component1_var[:, np.newaxis],
                                            train_x_time_component2_mean[:, np.newaxis],
                                            train_x_time_component2_var[:, np.newaxis],
                                            train_x_time_component1_weight[:, np.newaxis],
                                            train_x_time_component2_weight[:, np.newaxis],
                                            train_x_rssi_mean[:, np.newaxis],
                                            train_x_rssi_var[:, np.newaxis],
                                            train_x_time_mean[:, np.newaxis],
                                            train_x_time_var[:, np.newaxis]), axis=1)

    X = train_merged_array[:,:]
    Y = train_y

    ridge_model = ml.ridge_training(X,Y)
    lasso_model = ml.lasso_training(X,Y)

    test_merged_array = np.concatenate((test_x_time_component1_mean[:, np.newaxis],
                                            test_x_time_component1_var[:, np.newaxis],
                                            test_x_time_component2_mean[:, np.newaxis],
                                            test_x_time_component2_var[:, np.newaxis],
                                            test_x_time_component1_weight[:, np.newaxis],
                                            test_x_time_component2_weight[:, np.newaxis],
                                            test_x_rssi_mean[:, np.newaxis],
                                            test_x_rssi_var[:, np.newaxis],
                                            test_x_time_mean[:, np.newaxis],
                                            test_x_time_var[:, np.newaxis]), axis=1)
    test_x = test_merged_array[:,:]
    test_x = torch.from_numpy(test_merged_array[:,:]).float()

    nn_with_GMM_model = nn_GMM.neural_network_with_GMM_train(X,Y,sample_times)
    nn_without_GMM_model = nn_GMM.neural_network_without_GMM_train(X,Y,sample_times)

    curve_model = cf.fit_curve(distance_list)

    ridge_predictions = ml.ridge_predicting(ridge_model,test_x)
    lasso_predictions = ml.lasso_predicting(lasso_model,test_x)
    curve_predictions = cf.curve_predicting(curve_model,test_x[:,6:])
    nn_with_GMM_predictions = nn_GMM.neural_network_predicting(nn_with_GMM_model,test_x)
    nn_without_GMM_predictions = nn_GMM.neural_network_predicting(nn_without_GMM_model,test_x[:,6:])

    print(ridge_predictions)
    print(lasso_predictions)
    print(nn_with_GMM_predictions)
    print(nn_without_GMM_predictions)

    ridge_error = []
    lasso_error = []
    with_GMM_error = []
    without_GMM_error = []

    for i in range(len(distance_list)):
        ridge_error.append(np.abs(ridge_predictions[i] - distance_list[i]))
        lasso_error.append(np.abs(lasso_predictions[i] - distance_list[i]))
        with_GMM_error.append(np.abs(nn_with_GMM_predictions[i] - distance_list[i]))
        without_GMM_error.append(np.abs(nn_without_GMM_predictions[i] - distance_list[i]))

    return ridge_error,lasso_error,with_GMM_error,without_GMM_error
'''
plt.figure()
ax = plt.subplot(221)
ax.plot(distance_list,distance_list,c='r',label='true value')
ax.plot(distance_list,[i + 1 for i in distance_list],c='r',label='+1 error boundray',linestyle='--')
ax.plot(distance_list,[i - 1 for i in distance_list],c='r',label='-1 error boundray',linestyle='--')
ax.scatter(distance_list,ridge_predictions,c='b',label='ridge predictions')
ax.scatter(distance_list,lasso_predictions,c='g',label='lasso predictions')
ax.scatter(distance_list,curve_predictions,c='y',label='curve predictions')
ax.set_title('Machine learning results')
ax.set_xlabel('True distance (m)')
ax.set_ylabel('Model predicted distance (m)')
plt.legend()

ax = plt.subplot(222)
ax.plot(distance_list,distance_list,c='r',label='true value')
ax.plot(distance_list,[i + 1 for i in distance_list],c='r',label='+1 error boundray',linestyle='--')
ax.plot(distance_list,[i - 1 for i in distance_list],c='r',label='-1 error boundray',linestyle='--')
ax.scatter(distance_list,nn_with_GMM_predictions,c='b',label='NN with GMM predictions')
ax.scatter(distance_list,nn_without_GMM_predictions,c='g',label='NN without GMM predictions')
ax.set_title('Neural network results')
ax.set_xlabel('True distance (m)')
ax.set_ylabel('Model predicted distance (m)')
plt.legend()

plt.show()
'''

ridge_error_list = []
lasso_error_list = []
with_GMM_error_list = []
without_GMM_error_list = []

for times in range(10):
    ridge_error,lasso_error,with_GMM_error,without_GMM_error = multi_prediction()
    ridge_error_list.append(ridge_error)
    lasso_error_list.append(lasso_error)
    with_GMM_error_list.append(with_GMM_error)
    without_GMM_error_list.append(without_GMM_error)

print(ridge_error_list)

