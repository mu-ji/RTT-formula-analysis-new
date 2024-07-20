'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this file include some function related to GMM
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

def read_data(distance):
    f = open ('wiredexperiment\distance{}.txt'.format(distance), 'r')
    time_list = []
    rssi_list = []
    data = f.readlines()
    for i in range(len(data)):
        time_list.append(int(data[i].split(' ')[0]))
        rssi_list.append(int(data[i].split(' ')[1]))
    time = np.array(time_list)
    rssi = np.array(rssi_list)
    return time,rssi

#rtt, rssi = read_data(15)
#X = np.column_stack((rtt, rssi))

def select_components_number(X):
    min_aic = np.inf  # 初始化AIC的最小值为正无穷
    best_n_components = None  # 初始化最佳成分数量为None

    for n_components in range(1, 5):  # 尝试1到10个成分数量
        gmm = GaussianMixture(n_components=n_components)  # 创建GMM模型
        gmm.fit(X)  # 拟合模型
        aic = gmm.aic(X)

        if aic < min_aic:  # 更新最小AIC值和最佳成分数量
            min_aic = aic
            best_n_components = n_components

    return best_n_components
'''
gmm = GaussianMixture(n_components=best_n_components)
gmm.fit(X)

means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

print('means:',means)
print('covariances:',covariances)
print('weights:',weights)
'''
def filter_components(gmm):
    weights = gmm.weights_
    means = gmm.means_
    '''
    # 筛选权重大于0.1的分量
    selected_indices = np.where(weights > 0.2)[0]
    selected_weights = weights[selected_indices]
    selected_means = means[selected_indices]

    # 找到具有最小均值的分量
    min_mean_index = np.argmin(selected_means[:, 0])  # 假设你想按照第一个特征的均值进行比较
    selected_weights = selected_weights[min_mean_index]
    selected_means = selected_means[min_mean_index]
    '''
    selected_indices = np.where(weights > 0.2)[0]
    selected_weights = weights[selected_indices]
    selected_means = means[selected_indices]

    # 对权重进行归一化
    normalized_weights = normalize(selected_weights.reshape(1, -1), norm='l1')[0]

    # 加权求和得到新的分布的均值
    selected_means = np.average(selected_means, axis=0, weights=normalized_weights)
    
    return selected_means

def filter_components_1(gmm):
    weights = gmm.weights_
    means = gmm.means_

    # 找到均值中的最大值和最小值的索引
    min_mean_index = np.argmin(means[:, 0])  # 假设你想按照第一个特征的均值进行比较
    max_mean_index = np.argmax(means[:, 0])  # 假设你想按照第一个特征的均值进行比较

    # 剔除均值中的最大值和最小值
    filtered_indices = np.arange(len(means))
    filtered_indices = np.delete(filtered_indices, [min_mean_index, max_mean_index])

    # 筛选权重大于0.1的分量
    selected_indices = filtered_indices[weights[filtered_indices] > 0.1]
    selected_weights = weights[selected_indices]
    selected_means = means[selected_indices]

    # 对权重进行归一化
    normalized_weights = normalize(selected_weights.reshape(1, -1), norm='l1')[0]

    # 加权求和得到新的分布的均值
    new_mean = np.sum(normalized_weights[:, np.newaxis] * selected_means, axis=0)

    return new_mean
