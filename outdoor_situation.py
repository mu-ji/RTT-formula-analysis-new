'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script using to estimate parameters in multi path situation formula in outdoor environment
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats

adapter_length = 0.1
light_speed = 299792458
coefficient = 0.4

distance_list = [i for i in range(1,16)]

def read_data(distance):
    f = open ('5msexperiment\distance{}.txt'.format(distance), 'r')
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

#store mean and varcance of different distance data
means = []
varances = []
#store mu part and var part which need to be optimize
mu_list = []
var_list = []

for i in distance_list:
    data = read_data(i)
    data_mean = np.mean(data)
    data_var = np.var(data)
    means.append(data_mean)
    varances.append(data_var)

    T_los = (i + adapter_length)*2/light_speed*16000000
    T_waiting = 16000
    m = 4074.6992363565078
    q = 0.9710031598953147

    mu_list.append(data_mean - T_los - T_waiting - m)
    var_list.append(data_var - q**2)

def pdf_distance(params):
    """计算新正态分布与观测数据的 KL 散度之和"""
    p, u, R = params
    kl_divergences = []

    for i in range(len(mu_list)):
        mean_i = mu_list[i]
        variance_i = var_list[i]
        
        distance_between_pdf = ((1-p)*u-mean_i)**2 + (((1-p)**2)*R**2 - variance_i)**2

        print(distance_between_pdf)
        kl_divergences.append(distance_between_pdf)

    return np.sum(kl_divergences)

# 初始化新正态分布的参数 
p_init, u_init, R_init = 0.5, 1, 1

# 最小化 KL 散度之和
result = minimize(pdf_distance, [p_init, u_init, R_init], bounds=[(0,1),(0,100),(0,100)])


# 获取优化后的参数
p_opt, u_opt, R_opt = result.x
print(p_opt, u_opt, R_opt)

traditional_pre = []
new_model_pre = []
for i in means:
    traditional_pre.append(((i-20074.659)/2)/16000000*light_speed*coefficient)
    new_model_pre.append(((i - T_waiting - m - (1-p_opt)*u_opt)/2)/16000000*light_speed)

    

plt.figure()
ax = plt.subplot(111)
ax.plot(distance_list, traditional_pre, c = 'b', label = 'traditional error={}'.format(compute_err(distance_list,traditional_pre)))
ax.plot(distance_list, new_model_pre, c = 'y', label = 'new_model error={}'.format(compute_err(distance_list,new_model_pre)))
ax.plot(distance_list, distance_list, c = 'r', label = 'true distance')
ax.plot(distance_list, [i+1 for i in distance_list], c = 'r', linestyle = '--', label = '+1 error boundray')
ax.plot(distance_list, [i-1 for i in distance_list], c = 'r', linestyle = '--', label = '-1 error boundray')
ax.set_xlabel('true distance')
ax.set_ylabel('predict distance')
ax.set_title('new model in wired experiment with error based time')


plt.legend()
plt.show()