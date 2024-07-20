import numpy as np
import csv

every_distance_sample_number = 1000
distance_list = [i for i in range(1,12)]

one_sample_length = 1000


def sample_data(input_file, output_file, one_sample_length):
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    n = np.random.randint(0,len(lines)/2-one_sample_length)
    rtt_list = []
    rssi_list = []
    for i in range(n,n + one_sample_length):
        line = lines[i].strip()
        values = line.split(' ')
        rtt_list.append(values[0])
        rssi_list.append(values[1])
    
    with open(output_file, 'a') as f_out:
        for i in range(one_sample_length):
            f_out.write(rtt_list[i] + ' ')
        for i in range(one_sample_length):
            f_out.write(rssi_list[i] + ' ')
        f_out.write('\n')

#input_file is the raw data
#output_file is sample data
input_file = 'indoor_without_people/distance{}.txt'
output_file = 'train_set/indoor_without_people_walking_train_set1000.txt'

for distance in distance_list:
    for times in range(every_distance_sample_number):
        sample_data(input_file.format(distance),output_file,one_sample_length)

