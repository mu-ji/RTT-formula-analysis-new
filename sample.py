import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

import my_GMM

adapter_length = 0.1
light_speed = 299792458
coefficient = 0.4

def read_data(distance):
    f = open ('experiment3_outdoor\distance{}.txt'.format(distance), 'r')
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
    sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
    sample = X[sample_indices, :]
    return sample 

distance_list = [i for i in range(1,16)]

plt.figure()
ax = plt.subplot(111,projection='3d')

point_list = []
for distance in distance_list:
    rtt, rssi = read_data(distance)
    X = np.column_stack((rtt, rssi))            #(1000,2)
    X[:,0] -= 20074.659
    sample_x_list = []
    sample_y_list = []
    sample_distance_list = []
    for i in range(100):
        sample = sample_from_data(X,200)
        sample_x = np.mean(sample[:,0])
        sample_y = np.mean(sample[:,1])
        point_list.append([sample_x,sample_y,distance])
        sample_x_list.append(sample_x)
        sample_y_list.append(sample_y)
        sample_distance_list.append(distance)
    
    color = cm.viridis(distance/len(distance_list))
    ax.scatter3D(sample_x_list,sample_y_list,sample_distance_list,s = 0.5,c = color)


kmeans = KMeans(n_clusters=15)
kmeans.fit(point_list)
centroids = kmeans.cluster_centers_

x = centroids[:, 0]
y = centroids[:, 1]

print(centroids)
ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c = 'r')
ax.set_xlabel("RTT")
ax.set_ylabel('RSSI')
ax.set_zlabel('distance')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

data = centroids
X = data[:, :2]  # 前两列作为特征
print(X)
y = data[:, 2]  # 第三列作为目标变量

# 多项式特征转换
poly = PolynomialFeatures(degree=2)  # 选择多项式的阶数
X_poly = poly.fit_transform(X)

# 多项式回归模型
model = LinearRegression()
model.fit(X_poly, y)

# 预测
y_pred = model.predict(X_poly)

# 可视化结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='b', label='原始数据')
ax.scatter(X[:, 0], X[:, 1], y_pred, c='r', label='拟合结果')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()


def read_data(distance):
    f = open ('experiment3_outdoor\distance{}.txt'.format(distance), 'r')
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
    sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
    sample = X[sample_indices, :]
    return sample

distance_list = [i for i in range(1,16)]
predict_list = []
for distance in distance_list:
    rtt, rssi = read_data(distance)
    X = np.column_stack((rtt, rssi))            #(1000,2)
    X[:,0] -= 20074.659

    sample_predict = []
    for i in range(10):
        sample = sample_from_data(X,200)
        sample_x = np.mean(sample[:,0])
        sample_y = np.mean(sample[:,1])

        distance_pre = model.predict(poly.fit_transform([[sample_x,sample_y]]))
        sample_predict.append(distance_pre)
    
    predict_list.append(np.mean(sample_predict))

def compute_err(distance_list, pre_list):
    error_list = [] 
    for i in range(len(distance_list)):
        error = np.abs(distance_list[i] - pre_list[i])
        error_list.append(error)
    
    return np.mean(error_list)

joblib.dump(model, 'new_3_model.pkl')

plt.figure()
plt.scatter(distance_list,predict_list,c='b',label='model predict error = {}'.format(compute_err(distance_list,predict_list)))
plt.plot(distance_list, distance_list, c = 'r', label = 'true distance')
plt.plot(distance_list, [i+1 for i in distance_list], c = 'r', linestyle = '--', label = '+1 error boundray')
plt.plot(distance_list, [i-1 for i in distance_list], c = 'r', linestyle = '--', label = '-1 error boundray')

plt.legend()
plt.show()