'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this file include functions related to using curve to fix to predict distance based on RTT
'''

from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def read_data(distance):
    f = open ('indoor_with_people_walking\distance{}.txt'.format(distance), 'r')
    time_list = []
    rssi_list = []
    data = f.readlines()
    for i in range(len(data)):
        time_list.append(float(data[i].split(' ')[0]))
        rssi_list.append(float(data[i].split(' ')[1]))
    time = np.array(time_list)
    rssi = np.array(rssi_list)
    return time,rssi

def sample_from_data(x,n):
    sample_size = n
    sample_indices = np.random.choice(x.shape[0], size=sample_size, replace=False)
    sample = x[sample_indices, :]
    return sample 

def fit_curve(distance_list):
    point_list = []
    for distance in distance_list:
        rtt, rssi = read_data(distance)
        X = np.column_stack((rtt, rssi))            #(1000,2)
        X[:,0] -= 20074.659
        sample_x_list = []
        sample_y_list = []
        sample_distance_list = []
        for i in range(1000):
            sample = sample_from_data(X,200)
            sample_x = np.mean(sample[:,0])
            sample_y = np.mean(sample[:,1])
            point_list.append([sample_x,sample_y,distance])
            sample_x_list.append(sample_x)
            sample_y_list.append(sample_y)
            sample_distance_list.append(distance)

    kmeans = KMeans(n_clusters=len(distance_list))
    kmeans.fit(point_list)
    centroids = kmeans.cluster_centers_

    data = centroids
    x = data[:, :2]  # 前两列作为特征
    y = data[:, 2]  # 第三列作为目标变量

    # 多项式特征转换
    poly = PolynomialFeatures(degree=2)  # 选择多项式的阶数
    X_poly = poly.fit_transform(x)

    # 多项式回归模型
    model = LinearRegression()
    model.fit(X_poly, y)

    return model

def curve_predicting(model,test_x):
    c1 = test_x[:,0]
    c2 = test_x[:,2]
    test_x = np.column_stack((c2,c1))
    poly = PolynomialFeatures(degree=2)
    test_x = poly.fit_transform(test_x)
    y_pred = model.predict(test_x)
    return y_pred