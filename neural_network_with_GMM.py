'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this file include functions related to using neural network and GMM to predict distance based on RTT
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import multi_GMM_with_NN as MN

class RegressionNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegressionNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.elu = nn.ELU()
            self.fc3 = nn.Linear(hidden_size, output_size)



        def forward(self, x): 
            out = self.fc1(x)
            out = self.elu(out)
            out = self.fc2(out)
            out = self.elu(out)
            out = self.fc3(out)
            return out
        
def neural_network_with_GMM_train(X,Y):
    input_size = 10
    hidden_size = 256
    output_size = 1

    GMM_model = RegressionNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(GMM_model.parameters(), lr=0.001)

    num_epochs = 1000
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        # 前向传播
        outputs = GMM_model(X)
        loss = criterion(outputs, Y) #+ 0.0001*data_loss(X,Y,outputs)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练过程中的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss.append(loss.item())

        #predictions = model(test_x)
        #tt_loss = criterion(predictions,test_y)
        #test_loss.append(tt_loss.item())

    return GMM_model

def neural_network_predicting(model,test_x):
    # 在测试集上进行预测
    with torch.no_grad():
        predictions = model(test_x)

    predictions = predictions.numpy()
    return predictions

def neural_network_without_GMM_train(X,Y):
    input_size = 4
    hidden_size = 256
    output_size = 1

    model = RegressionNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, Y) #+ 0.0001*data_loss(X,Y,outputs)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练过程中的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss.append(loss.item())

        #predictions = model(test_x)
        #tt_loss = criterion(predictions,test_y)
        #test_loss.append(tt_loss.item())

    return model
    
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

def data_process_NN(train_set_file):
    with open(train_set_file, 'r') as train_file:
        lines = train_file.readlines()

    X = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(len(lines)):
        line = lines[i].strip().split(' ')
        line = [float(num) for num in line]     #200rtt + 200rssi

        means,covariances,weights = GMM_filter(np.array(line)[:200] - 20074.659)
        
        rtt_mean = np.mean(line[:200]) - 20074.659
        rtt_var = np.var(line[:200])
        rssi_mean = np.mean(line[200:])
        rssi_var = np.var(line[200:])
        X = np.vstack((X, np.array([float(means[0][0]),float(covariances[0][0]),float(means[1][0]),float(covariances[1][0]),
                                    float(weights[0]),float(weights[1]),
                                    rssi_mean,rssi_var,rtt_mean,rtt_var])))

    X = X[1:,:]  #去掉第一行的0
    return X

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
        prediction = (data[i][8])/2*300000000/16000000*0.4
        prediction_list.append(np.array(prediction))
    predictions = np.array(prediction_list)
    return predictions

train_x = data_process_NN('train_set/outdoor_train_set.txt')
test_x = data_process_NN('test_set/outdoor_test_set.txt')

train_x = data_process_NN('train_set/indoor_with_people_walking_train_set.txt')
test_x = data_process_NN('test_set/indoor_with_people_walking_test_set.txt')

train_x = data_process_NN('train_set/indoor_without_people_walking_train_set.txt')
test_x = data_process_NN('test_set/indoor_without_people_walking_test_set.txt')

train_y,test_y = generate_train_test_y()

X = torch.from_numpy(train_x[:,:]).float()
Y = torch.from_numpy(train_y).float()
test_X = torch.from_numpy(test_x[:,:]).float()
test_Y = torch.from_numpy(test_y).float()

X_hat = torch.from_numpy(train_x[:,6:]).float()
test_X_het = torch.from_numpy(test_x[:,6:]).float()

NN_with_GMM_model = neural_network_with_GMM_train(X,Y)
NN_without_GMM_model = neural_network_without_GMM_train(X_hat,Y)

NN_with_GMM_predictions = neural_network_predicting(NN_with_GMM_model,test_X)
NN_without_GMM_predictions = neural_network_predicting(NN_without_GMM_model,test_X_het)

with_GMM_error = NN_with_GMM_predictions - test_y
with_GMM_error = transform_error(with_GMM_error)
without_GMM_error = NN_without_GMM_predictions - test_y
without_GMM_error = transform_error(without_GMM_error)

traditional_predictions = traditional_prediction(test_x)
traditional_error = traditional_predictions.reshape((220,1)) - test_y
traditional_error = transform_error(traditional_error)


def trimmed_data(error):
    trimmed_data = []
    for i in range(error.shape[1]):
        column_data = error[:, i]  # 取出每一组数据
        trimmed_column = np.clip(column_data, np.percentile(column_data, 10), np.percentile(column_data, 90))
        trimmed_data.append(trimmed_column)

    # 转换为numpy数组
    trimmed_data = np.column_stack(trimmed_data)
    return trimmed_data

with_GMM_error = trimmed_data(with_GMM_error)
without_GMM_error = trimmed_data(without_GMM_error)
traditional_error = trimmed_data(traditional_error)



train_x = MN.data_process_NN('train_set/indoor_with_people_walking_train_set.txt')
test_x = MN.data_process_NN('test_set/indoor_with_people_walking_test_set.txt')

train_x = MN.data_process_NN('train_set/indoor_without_people_walking_train_set.txt')
test_x = MN.data_process_NN('test_set/indoor_without_people_walking_test_set.txt')

X = torch.from_numpy(train_x[:,:]).float()
Y = torch.from_numpy(train_y).float()
test_X = torch.from_numpy(test_x[:,:]).float()
test_Y = torch.from_numpy(test_y).float()

NN_with_multi_GMM_model = MN.neural_network_with_GMM_train(X,Y)
NN_with_multi_GMM_predictions = MN.neural_network_predicting(NN_with_multi_GMM_model,test_X)

with_multi_GMM_error = NN_with_multi_GMM_predictions - test_y
with_multi_GMM_error = MN.transform_error(with_multi_GMM_error)

def trimmed_data(error):
    trimmed_data = []
    for i in range(error.shape[1]):
        column_data = error[:, i]  # 取出每一组数据
        trimmed_column = np.clip(column_data, np.percentile(column_data, 10), np.percentile(column_data, 90))
        trimmed_data.append(trimmed_column)

    # 转换为numpy数组
    trimmed_data = np.column_stack(trimmed_data)
    return trimmed_data

with_multi_GMM_error = trimmed_data(with_multi_GMM_error)

boxprops = dict(facecolor='lightblue', color='blue')
plt.violinplot(without_GMM_error,positions=[i-0.4 for i in range(1,23,2)],showmeans=True,widths=0.3)
boxprops = dict(facecolor='red', color='maroon')
plt.violinplot(with_GMM_error,positions=[i+0.4 for i in range(1,23,2)],showmeans=True,widths=0.3)
boxprops = dict(facecolor='green', color='green')
plt.violinplot(with_multi_GMM_error,positions=[i for i in range(1,23,2)],showmeans=True,widths=0.3)

rect_1 = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue')
rect_2 = plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='orange')
rect_3 = plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='green')
plt.legend([rect_1, rect_2, rect_3], ['NN without GMM error','NN with multi GMM error', 'NN with GMM error'])

labels = (['{} meters'.format(i) for i in range(1,12)])
plt.xticks([i for i in range(1,23,2)], labels)
plt.title('Three kinds of NNs prediction error in different distance(indoor environment without people walking)')
plt.ylabel('error(meters)')
plt.grid()
plt.show()
