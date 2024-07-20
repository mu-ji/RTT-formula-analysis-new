import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义第一个全连接网络提取RTT特征
rtt_net = nn.Sequential(
    nn.Linear(200, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 5),
    nn.ReLU()
)

# 定义第二个全连接网络提取RSSI特征
rssi_net = nn.Sequential(
    nn.Linear(200, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 5),
    nn.ReLU()
)

# 定义第三个全连接网络接收两个特征拼接后的输入，输出距离预测
distance_net = nn.Sequential(
    nn.Linear(5 + 5, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 1)
)

# 定义优化器
rtt_optimizer = torch.optim.SGD(rtt_net.parameters(), lr=0.001)
rssi_optimizer = torch.optim.SGD(rssi_net.parameters(), lr=0.001)
distance_optimizer = torch.optim.SGD(distance_net.parameters(), lr=0.001)

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

def compute_err(distance_list, pre_list):
    error_list = []
    for i in range(len(distance_list)):
        error = np.abs(distance_list[i] - pre_list[i])
        error_list.append(error)
    
    return np.mean(error_list)

def sample_from_data(X,n):
    sample_size = n
    sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
    sample = X[sample_indices, :]
    return sample

distance_list = [i for i in range(1,16)]

RTT_train_list = []
RSSI_train_list = []
distance_train_list = []

RTT_test_list = []
RSSI_test_list = []
distance_test_list = []

for distance in distance_list:
    rtt, rssi = read_data(distance)
    X = np.column_stack((rtt, rssi))            #(1000,2)
    X[:,0] -= 20074.659
    sample_x_list = []
    sample_y_list = []
    for i in range(50):
        sample = sample_from_data(X[:500,:],200)
        sample_x = sample[:,0]
        sample_y = sample[:,1]
        RTT_train_list.append(sample_x)
        RSSI_train_list.append(sample_y)
        distance_train_list.append(distance)
    
    for i in range(10):
        sample = sample_from_data(X[501:,:],200)
        sample_x = sample[:,0]
        sample_y = sample[:,1]
        RTT_test_list.append(sample_x)
        RSSI_test_list.append(sample_y)
        distance_test_list.append(distance)

num_epochs = 2000

criterion = nn.MSELoss()


train_losses = []
test_losses = []

def list_to_tensor(list_to_change):
    # 将列表中的数组转换为张量
    tensor_data = [torch.tensor(arr) for arr in list_to_change]

    # 将张量数据堆叠成一个张量
    tensor_data = torch.stack(tensor_data)
    return tensor_data.float()

RTT_train_list = list_to_tensor(RTT_train_list)
RSSI_train_list = list_to_tensor(RSSI_train_list)
distance_train_list = list_to_tensor(distance_train_list)

RTT_test_list = list_to_tensor(RTT_test_list)
RSSI_test_list = list_to_tensor(RSSI_test_list)
distance_test_list = list_to_tensor(distance_test_list)


for epoch in range(num_epochs):
    # 前向传播
    rtt_features = rtt_net(RTT_train_list)
    rssi_features = rssi_net(RSSI_train_list)
    combined_features = torch.cat((rtt_features, rssi_features), dim=1)
    predictions = distance_net(combined_features)

    # 计算损失
    train_loss = criterion(predictions, distance_train_list)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.item():.4f}')

    train_losses.append(train_loss.item())

    # 反向传播和优化
    rtt_optimizer.zero_grad()
    rssi_optimizer.zero_grad()
    distance_optimizer.zero_grad()
    train_loss.backward()
    rtt_optimizer.step()
    rssi_optimizer.step()
    distance_optimizer.step()

    with torch.no_grad():
        test_rtt_features = rtt_net(RTT_test_list)
        test_rssi_features = rssi_net(RSSI_test_list)
        test_combined_features = torch.cat((test_rtt_features, test_rssi_features), dim=1)
        test_predictions = distance_net(test_combined_features)
        test_loss = criterion(test_predictions, distance_test_list)
        test_losses.append(test_loss.item())

with torch.no_grad():
    test_rtt_features = rtt_net(RTT_test_list)
    test_rssi_features = rssi_net(RSSI_test_list)
    test_combined_features = torch.cat((test_rtt_features, test_rssi_features), dim=1)
    test_predictions = distance_net(test_combined_features).numpy()

ax = plt.subplot(111)
ax.scatter([i for i in range(len(distance_test_list))],test_predictions, c = 'b',label = 'ML_predictions')
ax.plot([i for i in range(len(distance_test_list))],distance_test_list,c = 'r',label = 'true value')

plt.legend()
plt.show()