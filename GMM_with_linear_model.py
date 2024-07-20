import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import my_GMM
'''
_1meters = [20075.82145926,-42.29040458]
_2meters = [20076.0990676,-40.86421911]
_3meters = [20075.51841965,-41]
_4meters = [20076,-46.34755332]
_5meters = [20075.71800434,-41]
_6meters = [20076.58923553,-40.00053277]
_7meters = [20075.7248954,-49]
_8meters = [20076,-43.78178368]
_9meters = [20080,-57.23442518]
_10meters = [20076.23611112,-45]
_11meters = [20077.75339198,-48.64819349]
_12meters = [20077.64651158,-49.0005168 ]
_13meters = [20079.83693458,-58.00130976]
_14meters = [20080.657173,-54.33227828]
_15meters = [20078.75658129,-59.20187227]
'''
adapter_length = 0.1
light_speed = 299792458
coefficient = 0.4

def read_data(distance):
    f = open ('experiment3_outdoor\distance{}.txt'.format(distance), 'r')
    time_list = []
    rssi_list = []
    data = f.readlines()
    for i in range(len(data)):
        time_list.append(int(data[i].split(' ')[0]))
        rssi_list.append(int(data[i].split(' ')[1]))
    time = np.array(time_list)
    rssi = np.array(rssi_list)
    return time,rssi

def compute_err(distance_list, pre_list):
    error_list = []
    for i in range(len(distance_list)):
        error = np.abs(distance_list[i] - pre_list[i])
        error_list.append(error)
    
    return np.mean(error_list)

distance_list = [i for i in range(1,16)]

rtt_means = []
rtt_varances = []
for i in distance_list:
    rtt, rssi = read_data(i)
    rtt_mean = np.mean(rtt)
    rtt_var = np.var(rtt)
    rtt_means.append(rtt_mean)
    rtt_varances.append(rtt_var)
'''
new_model_list = [_1meters[0],_2meters[0],_3meters[0],_4meters[0],_5meters[0],_6meters[0],
                  _7meters[0],_8meters[0],_9meters[0],_10meters[0],_11meters[0],_12meters[0],
                  _13meters[0],_14meters[0],_15meters[0]]
'''
new_model_list = []
for i in distance_list:
    rtt, rssi = read_data(i)
    X = np.column_stack((rtt, rssi))

    best_components = my_GMM.select_components_number(X)

    gmm = GaussianMixture(n_components=best_components)
    gmm.fit(X)

    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    print('means:',means)
    print('weight:',weights)
    selected_means = my_GMM.filter_components_1(gmm)
    print(selected_means)
    new_model_list.append(selected_means[0])


traditional_pre = []
new_model_pre = []


for i in range(len(rtt_means)):
    traditional_pre.append(((rtt_means[i]-20074.659)/2)/16000000*light_speed*coefficient)
    new_model_pre.append(((new_model_list[i]-20074.659)/2)/16000000*light_speed*coefficient)


plt.figure()
ax = plt.subplot(111)
ax.scatter(distance_list, traditional_pre, c = 'b', label = 'traditional error={}'.format(compute_err(distance_list,traditional_pre)))
ax.scatter(distance_list, new_model_pre, c = 'y', label = 'new_model error={}'.format(compute_err(distance_list,new_model_pre)))
ax.plot(distance_list, distance_list, c = 'r', label = 'true distance')
ax.plot(distance_list, [i+1 for i in distance_list], c = 'r', linestyle = '--', label = '+1 error boundray')
ax.plot(distance_list, [i-1 for i in distance_list], c = 'r', linestyle = '--', label = '-1 error boundray')
ax.set_xlabel('true distance')
ax.set_ylabel('predict distance')
ax.set_title('new model in wired experiment with error based time')
plt.legend()
plt.show()