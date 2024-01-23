import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from sklearn import svm
from sklearn.model_selection import train_test_split
from tsfresh.feature_extraction.feature_calculators import fft_coefficient, fft_aggregated, spkt_welch_density, \
    fourier_entropy, welch
from math import floor

from On_Periodicity_Detection_and_Structural_Periodic_Similarity.periodicity import signalProcess
from milestone1 import read_PAMAP2_dataset
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, append, linspace, array, arange, sin, cos
import cmath
from Fourier_Analysis import FA

# '''
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# PAMAP2_dataset
activity_list = ['lying', 'sitting', 'standing', 'walking', 'running',
                 'cycling', 'Nordic_walking', 'ascending_stairs', 'descending_stairs',
                 'vacuum_cleaning', 'ironing', 'rope_jumping']

new_source_data_X = []
new_source_data_Label = []
train_target_data_X = []
train_target_data_Label = []
test_target_data_X = []
test_target_data_Label = []


activities_required = ['lying']
sensor_channels_required = ['IMU_Chest', 'IMU_Hand']  # ['IMU_Hand', 'IMU_Chest', 'IMU_Ankle']
# activities_required = activity_list  # activity_list  # ['lying', 'sitting', 'standing', 'walking', 'running'] # activity_list  # activity_list  # 12 common activities ['rope_jumping']
source_user = '1'  # 1 # 5 # 6
target_user = '6'
Sampling_frequency = 100  # HZ
Num_windows = 5
DATASET_NAME = 'PAMAP2'
pamap2_ds = read_PAMAP2_dataset.READ_PAMAP2_DATASET(source_user, target_user, bag_window_second=Num_windows,
                                                    bag_overlap_rate=0.5,
                                                    instances_window_second=0.1, instances_overlap_rate=0.5,
                                                    sampling_frequency=Sampling_frequency)

source_required_X_bags, source_required_Y_bags, source_required_X_bags_with_instances, source_required_Y_bags_with_instances, \
target_required_X_bags, target_required_Y_bags, target_required_X_bags_with_instances, target_required_Y_bags_with_instances \
    = pamap2_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required,
                                                                           activities_required)

source_samples = []
for a_window in source_required_X_bags:
    a_sample = []
    for a_channel in range(18):
        a_signal = a_window[:, a_channel]
        t = linspace(0, Num_windows, len(a_signal))
        fa = FA(t, a_signal, Sampling_frequency)
        Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = fa.Fourier_Transform()
        a_sample.extend(Ampts)
    source_samples.append(a_sample)
source_mean = np.mean(np.array(source_samples), axis=0)

X_train_input, X_test_input, Y_train_input, Y_test_input = train_test_split(target_required_X_bags,
                                                                            target_required_Y_bags,
                                                                            test_size=0.1,
                                                                            stratify=target_required_Y_bags,
                                                                            random_state=1)
target_train_samples = []
for a_window in X_test_input:
    a_sample = []
    for a_channel in range(18):
        a_signal = a_window[:, a_channel]
        t = linspace(0, Num_windows, len(a_signal))
        fa = FA(t, a_signal, Sampling_frequency)
        Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = fa.Fourier_Transform()
        a_sample.extend(Ampts)
    target_train_samples.append(a_sample)
target_train_mean = np.mean(np.array(target_train_samples), axis=0)

# range_list = [x * 0.4 for x in range(-5, 5)] # (-3, 3, 0.2)
# for alpha in range_list:
#    for beta in range_list:
new_source = np.array(source_samples) - source_mean + target_train_mean
new_source_data_X = new_source
new_source_data_Label = source_required_Y_bags
train_target_data_X = target_train_samples
train_target_data_Label = Y_test_input

target_test_samples = []
for a_window in X_train_input:
    a_sample = []
    for a_channel in range(18):
        a_signal = a_window[:, a_channel]
        t = linspace(0, Num_windows, len(a_signal))
        fa = FA(t, a_signal, Sampling_frequency)
        Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = fa.Fourier_Transform()
        a_sample.extend(Ampts)
    target_test_samples.append(a_sample)

test_target_data_X = target_test_samples
test_target_data_Label = Y_train_input

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
for a_activity in ['sitting', 'standing', 'walking', 'running',
                   'cycling', 'Nordic_walking', 'ascending_stairs', 'descending_stairs',
                   'vacuum_cleaning', 'ironing', 'rope_jumping']:
    activities_required = [a_activity]
    sensor_channels_required = ['IMU_Chest', 'IMU_Hand']  # ['IMU_Hand', 'IMU_Chest', 'IMU_Ankle']
    # activities_required = activity_list  # activity_list  # ['lying', 'sitting', 'standing', 'walking', 'running'] # activity_list  # activity_list  # 12 common activities ['rope_jumping']
    source_user = '1'  # 1 # 5 # 6
    target_user = '6'
    Sampling_frequency = 100  # HZ
    Num_windows = 5
    DATASET_NAME = 'PAMAP2'
    pamap2_ds = read_PAMAP2_dataset.READ_PAMAP2_DATASET(source_user, target_user, bag_window_second=Num_windows,
                                                        bag_overlap_rate=0.5,
                                                        instances_window_second=0.1, instances_overlap_rate=0.5,
                                                        sampling_frequency=Sampling_frequency)

    source_required_X_bags, source_required_Y_bags, source_required_X_bags_with_instances, source_required_Y_bags_with_instances, \
    target_required_X_bags, target_required_Y_bags, target_required_X_bags_with_instances, target_required_Y_bags_with_instances \
        = pamap2_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required,
                                                                               activities_required)

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    source_samples = []
    for a_window in source_required_X_bags:
        a_sample = []
        for a_channel in range(18):
            a_signal = a_window[:, a_channel]
            t = linspace(0, Num_windows, len(a_signal))
            fa = FA(t, a_signal, Sampling_frequency)
            Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = fa.Fourier_Transform()
            a_sample.extend(Ampts)
        source_samples.append(a_sample)
    source_mean = np.mean(np.array(source_samples), axis=0)

    X_train_input, X_test_input, Y_train_input, Y_test_input = train_test_split(target_required_X_bags,
                                                                                target_required_Y_bags,
                                                                                test_size=0.1,
                                                                                stratify=target_required_Y_bags,
                                                                                random_state=1)
    target_train_samples = []
    for a_window in X_test_input:
        a_sample = []
        for a_channel in range(18):
            a_signal = a_window[:, a_channel]
            t = linspace(0, Num_windows, len(a_signal))
            fa = FA(t, a_signal, Sampling_frequency)
            Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = fa.Fourier_Transform()
            a_sample.extend(Ampts)
        target_train_samples.append(a_sample)
    target_train_mean = np.mean(np.array(target_train_samples), axis=0)

    # range_list = [x * 0.4 for x in range(-5, 5)] # (-3, 3, 0.2)
    # for alpha in range_list:
    #    for beta in range_list:
    new_source = np.array(source_samples) - source_mean + target_train_mean
    new_source_data_X = np.vstack((new_source_data_X, new_source))
    new_source_data_Label = new_source_data_Label + source_required_Y_bags
    train_target_data_X = np.vstack((train_target_data_X, target_train_samples))
    train_target_data_Label = train_target_data_Label + Y_test_input

    target_test_samples = []
    for a_window in X_train_input:
        a_sample = []
        for a_channel in range(18):
            a_signal = a_window[:, a_channel]
            t = linspace(0, Num_windows, len(a_signal))
            fa = FA(t, a_signal, Sampling_frequency)
            Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = fa.Fourier_Transform()
            a_sample.extend(Ampts)
        target_test_samples.append(a_sample)

    test_target_data_X = np.vstack((test_target_data_X, target_test_samples))
    test_target_data_Label = test_target_data_Label + Y_train_input

print()
clf = svm.LinearSVC(dual=False, multi_class='ovr')
clf.fit(train_target_data_X, train_target_data_Label)
score = clf.score(test_target_data_X, test_target_data_Label)
print(str(score))

a = np.vstack((train_target_data_X, new_source_data_X))
b = train_target_data_Label + new_source_data_Label
clf2 = svm.LinearSVC(dual=False, multi_class='ovr')
clf2.fit(a, b)
score2 = clf2.score(test_target_data_X, test_target_data_Label)
print(str(score2))
print('before_TL_' + str(score) + '_after_TL_' + str(score2))

'''
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# get main period
# 2. check periodicity
def check_periodicity(a_signal):
time_series_list = {}
time_series_list['time_series'] = a_signal
data_input = {"data": time_series_list}
sp = signalProcess(data_input)
periodicity = sp.getPrimaryPeriods()

period_flag = True
if not periodicity:
    # No periodicity
    print('_No periodicity')
    period_flag = False
else:

    shortPeriodList = []
    p = periodicity['periods']

    for key in p:
        print(periodicity['periods'][key])
        shortPeriodList.append(periodicity['periods'][key][0])
        if len(periodicity['periods'][key]) > 3:
            period_flag = False

    if shortPeriodList[0] is None and len(shortPeriodList) > 1:
        preValue = shortPeriodList[1]
    else:
        preValue = shortPeriodList[0]
        for period_value in shortPeriodList:
            if abs(period_value - preValue) > 6:
                period_flag = False
        if not period_flag:
            print('_No periodicity')
        else:
            print('_Has periodicity')

a_source_signal = source_required_X_bags[15][:, 0]
check_periodicity(a_source_signal)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for i in range(len(source_required_X_bags)):

for j in range(9):
a_source_signal = source_required_X_bags[i][:, j]
source_t = linspace(0, Num_windows, len(a_source_signal))
source_FA = FA(source_t, a_source_signal, Sampling_frequency)
Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = source_FA.Fourier_Transform()
a_sample.extend(Pxx_den.tolist())


    new_f =source_FA.Reverse_Fourier_Transform(fourier_coef)

    #dominant_detailed_frequency, all_mask_list, specified_frequecy_list, freqs, Pxx_den = source_FA.TL_get_dominant_detailed_frequency_and_mask()
    #source_coefficient_list = source_FA.TL_Fourier_Transform_Rebuild(specified_frequecy_list)
    #new_f = source_FA.TL_Inverse_Fourier_Transform_Rebuild(source_coefficient_list, source_t)
    plt.plot(source_t, a_source_signal, 'b-')
    plt.plot(source_t, new_f, 'r-')
    plt.show()
    print()


    for a_freq in specified_frequecy_list:
        if a_freq in specified_frequecy_list_dict_key:
            specified_frequecy_list_dict_key[a_freq] += 1
        else:
            specified_frequecy_list_dict_key[a_freq] = 1

    # source_coefficient_list = source_FA.TL_Fourier_Transform_Rebuild(specified_frequecy_list)
sorted_value_specified_frequecy_list_dict_key = {k: v for k, v in sorted(specified_frequecy_list_dict_key.items(), key=lambda item: item[1])}
sorted_value_specified_frequecy_list_dict_key = {k: v for k, v in sorted_value_specified_frequecy_list_dict_key.items() if v >= 0.1*len(source_required_X_bags)}
sorted_key_specified_frequecy_list_dict_key = dict(sorted(specified_frequecy_list_dict_key.items()))
print()
import json
sorted_value_specified_frequecy_list_dict_key = {'sorted_value_specified_frequecy_list_dict_key': sorted_value_specified_frequecy_list_dict_key}
#sorted_key_specified_frequecy_list_dict_key = {'sorted_key_specified_frequecy_list_dict_key': sorted_key_specified_frequecy_list_dict_key}
with open(a_activity + '_' + str(j) + '_' + a_position + '.txt', 'w') as file:
    file.write(json.dumps(sorted_value_specified_frequecy_list_dict_key))
    #file.write(json.dumps(sorted_key_specified_frequecy_list_dict_key))
    file.close()




a_source_signal = source_required_X_bags[10][:, 0]
source_t = linspace(0, Num_windows, len(a_source_signal))
#a_source_signal = 3 * sin(2 * pi * 1.2 * source_t) + 0.5 * sin(2 * pi * 10.2 * source_t)
source_FA = FA(source_t, a_source_signal, Sampling_frequency)
source_dominant_detailed_frequency, _, source_specified_frequecy_list = source_FA.TL_get_dominant_detailed_frequency_and_mask()
source_dominant_detailed_amp = source_FA.TL_get_dominant_detailed_frequency_s_amp(source_dominant_detailed_frequency)
source_coefficient_list = source_FA.TL_Fourier_Transform_Rebuild(source_specified_frequecy_list)
plt.plot(source_t, a_source_signal, 'b-')
plt.show()
plt.close()


a_target_signal = target_required_X_bags[10][:, 0]
target_t = linspace(0, Num_windows, len(a_target_signal))
#a_target_signal = 1 * sin(2 * pi * 2.4 * source_t) + 0.5 * sin(2 * pi * 7.4 * source_t)
target_FA = FA(target_t, a_target_signal, Sampling_frequency)
target_dominant_detailed_frequency, _, _ = target_FA.TL_get_dominant_detailed_frequency_and_mask()
target_dominant_detailed_amp = target_FA.TL_get_dominant_detailed_frequency_s_amp(target_dominant_detailed_frequency)
plt.plot(target_t, a_target_signal, 'b-')
plt.show()
plt.close()


new_s_coefficient_list, scale_value_freq = target_FA.TL_freq_phase_amp(source_dominant_detailed_frequency, target_dominant_detailed_frequency, source_dominant_detailed_amp, target_dominant_detailed_amp, source_coefficient_list)
new_source_t = linspace(0, 15, 1500)
a_new_source_signal = source_FA.TL_Inverse_Fourier_Transform_Rebuild(new_s_coefficient_list, new_source_t)
plt.plot(new_source_t, a_new_source_signal, 'b-')
plt.show()
plt.close()
'''
