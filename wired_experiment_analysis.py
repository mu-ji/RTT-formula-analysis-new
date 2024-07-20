import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from collections import Counter

distance_list = [i for i in range(1,11)]

input_file = 'wiredexperiment/distance{}.txt'.format(1)
with open(input_file, 'r') as f_in:
    lines = f_in.readlines()
print(len(lines))
print(lines[1][:5])

def extract_RTT(input_file):
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
    RTT_array = np.zeros((len(lines),))

    for i in range(len(lines)):
        RTT_array[i] = int(lines[i][:5])

    return RTT_array

def generate_sample(RTT_array):
    sample_number = int(len(RTT_array)/200)
    sample_array = np.zeros((sample_number,))
    for i in range(sample_number):
        sample_array[i] = np.mean(RTT_array[i*200+i:(i+1)*200+i])
    
    return sample_array

RTT_array_1 = extract_RTT(input_file)
'''
print(RTT_array_1[200:400])
counts = Counter(RTT_array_1[400:600])
print(counts)
'''
sample_array_1 = generate_sample(RTT_array_1)
distance1_RTT_mean = np.mean(sample_array_1)
print(distance1_RTT_mean)

estimate_error_list = []
for i in distance_list:
    input_file = 'wiredexperiment/distance{}.txt'.format(i)
    RTT_array = extract_RTT(input_file)
    sample_array = generate_sample(RTT_array)
    sample_array -= distance1_RTT_mean
    distance_estimate = sample_array*222222222.222/(2*16000000)
    distance_estimate_error = distance_estimate - (i-1)
    #print(distance_estimate_error)
    for i in distance_estimate_error:
        estimate_error_list.append(i)

mu = np.mean(estimate_error_list)
sigma = np.std(estimate_error_list)

counts, bins = np.histogram(estimate_error_list, bins=20,  density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2
bins += (bins[1]-bins[0])/2
plt.figure()   
plt.plot(bins[:-1], counts, linestyle='-', color='b', label = 'RTT Error', marker='o')
#plt.bar(bin_centers, counts, width=bins[1]-bins[0], edgecolor='b', facecolor='none')

x = np.linspace(np.array(estimate_error_list).min(), np.array(estimate_error_list).max(), 100)
pdf = norm.pdf(x, loc=mu, scale=sigma)
plt.plot(x, pdf, marker='', linestyle='-', color='r', label = 'RTT Error - Gaussian model')

plt.legend()
plt.grid()
plt.xlabel('RTT Error (m)', fontsize=10)
plt.ylabel('Probability', fontsize=10)
plt.savefig('wired_rtt_error_distribution.svg',dpi=1000,format='svg')
plt.show()

