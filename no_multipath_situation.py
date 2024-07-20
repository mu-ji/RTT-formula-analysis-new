'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script using to estimate parameters in no multi path situation formula
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats

def read_data(distance):
    f = open ('wiredexperiment\distance{}.txt'.format(distance), 'r')
    time_list = []
    data = f.readlines()
    for i in range(len(data)):
        time_list.append(int(data[i].split(' ')[0]))
    time = np.array(time_list)
    return time



def compute_err(distance_list, pre_list):
    error_list = []
    for i in range(len(distance_list)):
        error = np.abs(distance_list[i] - pre_list[i])
        error_list.append(error)
    
    return np.mean(error_list)


wired_signal_speed = 222222222.222       #4.5ns/meter
adapter_length = 0.1

distance_list = [i for i in range(1,12)]

def main():
    
    wired_signal_speed = 222222222.222       #4.5ns/meter
    adapter_length = 0.1

    distance_list = [i for i in range(1,12)]

    m_list = []
    Q_list = []
    means = []
    varance = []

    def pdf_distance(params):
        """计算新正态分布与观测数据的 KL 散度之和"""
        mu, sigma = params
        kl_divergences = []

        for i in range(len(m_list)):
            mean_i = m_list[i]
            variance_i = Q_list[i]
            
            distance_between_pdf = (mu-mean_i)**2 + (sigma - np.sqrt(variance_i))**2

            print(distance_between_pdf)
            kl_divergences.append(distance_between_pdf)

        return np.sum(kl_divergences)

    for i in distance_list:
        data = read_data(i)
        data_mean = np.mean(data)
        data_var = np.var(data)
        T_los = (i + adapter_length)*2/wired_signal_speed*16000000
        T_waiting = 16000
        means.append(data_mean)
        varance.append(data_var)
        m_list.append(data_mean - T_los - T_waiting)
        Q_list.append(data_var)
    
    # 初始化新正态分布的参数 
    mu_init, sigma_init = 4000, 1

    # 最小化 KL 散度之和
    result = minimize(pdf_distance, [mu_init, sigma_init])

    # 获取优化后的参数
    mu_opt, sigma_opt = result.x
    print(mu_opt, sigma_opt)

    traditional_pre = []
    new_model_pre = []
    for i in means:
        traditional_pre.append(((i-20074.659)/2)/16000000*222222222.222)
        new_model_pre.append(((i - T_waiting - mu_opt)/2)/16000000*wired_signal_speed)
    
    return new_model_pre, mu_opt


new_model_pre,mu_opt = main()



fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111)


for i in distance_list:
    data = read_data(i)
    sample_list = []
    sample_distance = [i]*10
    for j in range(10):
        n = np.random.randint(200)
        sample_list.append(np.mean(data[n:n+200]))
    T_waiting = 16000
    if i == 1:
        ax.scatter([i+0.2 for i in sample_distance], [((i - T_waiting - mu_opt)/2)/16000000*wired_signal_speed for i in sample_list], c = 'none', marker = 'o', s=50, label = 'prediction',  edgecolors='r')
        ax.scatter([i+0.2 for i in sample_distance], [((i - 20074.659)/2)/16000000*wired_signal_speed for i in sample_list], c = 'blue', marker = '+', s = 50,label = 'tradition')
    else:
        ax.scatter([i+0.2 for i in sample_distance], [((i - T_waiting - mu_opt)/2)/16000000*wired_signal_speed for i in sample_list], c = 'none', s = 50, marker = 'o',  edgecolors='r')
        ax.scatter([i+0.2 for i in sample_distance], [((i - 20074.659)/2)/16000000*wired_signal_speed for i in sample_list], c = 'blue', marker = '+', s = 50)
ax.plot([i+0.2 for i in distance_list], [i+0.2 for i in distance_list], c = 'black', label = 'true distance')
ax.plot([i+0.2 for i in distance_list], [i+0.2+1 for i in distance_list], c = 'y', linestyle = '--', label = '+1 error boundray')
ax.plot([i+0.2 for i in distance_list], [i+0.2-1 for i in distance_list], c = 'y', linestyle = '--', label = '-1 error boundray')
ax.set_aspect(1)
ax.grid(True)
ax.set_xlim((1,13))
my_x_ticks = np.arange(0, 13, 1)
ax.set_xticks(my_x_ticks)
ax.set_ylim((1,13))
my_y_ticks = np.arange(0, 13, 1)
ax.set_yticks(my_y_ticks)
ax.set_xlabel('True Distance(m)', fontdict={'weight': 'normal', 'size': 15})

ax.set_ylabel('Measured Distance(m)', fontdict={'weight': 'normal', 'size': 15})
#ax.set_title('new model in wired experiment with error based time')

plt.rcParams.update({'font.size': 13})
plt.legend()
plt.show()