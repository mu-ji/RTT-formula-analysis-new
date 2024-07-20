import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# 输入数据
data = np.array([[1.5857, -57.09465, 7.],
                 [0.9514, -49.12205, 2.],
                 [4.9146, -63.85465, 14.],
                 [3.1389, -58.49075, 12.],
                 [0.7197, -43.3471, 1.],
                 [2.8672, -59.77885, 9.],
                 [1.6451, -52.49905, 4.],
                 [1.9227, -54.6239, 5.],
                 [3.1409, -62.61865, 11.],
                 [2.5658, -64.9417, 13.],
                 [4.0269, -59.1634, 15.],
                 [1.4381, -51.0658, 3.],
                 [2.5605, -59.5398, 10.],
                 [1.7471, -54.87165, 6.],
                 [1.6909, -56.9237, 8.]])

# 提取特征和目标变量
X = data[:, :2]  # 前两列作为特征
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
ax.scatter(X[:, 0], X[:, 1], y, c='b', label='true data')
ax.scatter(X[:, 0], X[:, 1], y_pred, c='r', label='predict data')
ax.set_xlabel('RTT')
ax.set_ylabel('RSSI')
ax.set_zlabel('distance')
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

joblib.dump(model, 'new_model.pkl')

plt.figure()
plt.scatter(distance_list,predict_list,c='b',label='model predict error = {}'.format(compute_err(distance_list,predict_list)))
plt.plot(distance_list, distance_list, c = 'r', label = 'true distance')
plt.plot(distance_list, [i+1 for i in distance_list], c = 'r', linestyle = '--', label = '+1 error boundray')
plt.plot(distance_list, [i-1 for i in distance_list], c = 'r', linestyle = '--', label = '-1 error boundray')

plt.legend()
plt.show()