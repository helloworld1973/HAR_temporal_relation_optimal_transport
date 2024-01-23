import numpy as np
from read_dataset import read_DSADS_dataset
from feature_extraction.feature_core import get_feature
from SOT.SOT import norm_max

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# DSADS_dataset

activity_list = ['sitting', 'standing', 'lying_on_back', 'lying_on_right', 'ascending_stairs', 'descending_stairs',
                 'standing_in_an_elevator_still', 'moving_around_in_an_elevator',
                 'walking_in_a_parking_lot', 'walking_on_a_treadmill_in_flat',
                 'walking_on_a_treadmill_inclined_positions',
                 'running_on_a_treadmill_in_flat', 'exercising on a stepper', 'exercising on a cross trainer',
                 'cycling_on_an_exercise_bike_in_horizontal_positions',
                 'cycling_on_an_exercise_bike_in_vertical_positions',
                 'rowing', 'jumping', 'playing_basketball']

activities_required = activity_list #['lying_on_back', 'walking_in_a_parking_lot', 'ascending_stairs', 'descending_stairs']

sensor_channels_required = ['T_x_acc', 'T_y_acc', 'T_z_acc',
                            'T_x_gyro', 'T_y_gyro', 'T_z_gyro',  # Torso
                            'RA_x_acc', 'RA_y_acc', 'RA_z_acc',
                            'RA_x_gyro', 'RA_y_gyro', 'RA_z_gyro',  # Right Arm
                            'LL_x_acc', 'LL_y_acc', 'LL_z_acc',
                            'LL_x_gyro', 'LL_y_gyro', 'LL_z_gyro']  # Left Leg

sensor_channels_required = ['RA_x_acc', 'RA_y_acc', 'RA_z_acc',
                            'RA_x_gyro', 'RA_y_gyro', 'RA_z_gyro'] # Right Arm

# ['rowing'] # activity_list #['standing', 'lying_on_back', 'lying_on_right', 'descending_stairs', 'exercising on a stepper']  # activity_list  # 12 common activities ['rope_jumping']
source_user = '2'  # 2, 3, 4, 5, 7, 8
target_user = '3'
Sampling_frequency = 25  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
DATASET_NAME = 'DSADS'
dsads_ds = read_DSADS_dataset.READ_DSADS_DATASET(source_user, target_user, bag_window_second=Num_Seconds,
                                                 bag_overlap_rate=Window_Overlap_Rate,
                                                 instances_window_second=0.1, instances_overlap_rate=0.5,
                                                 sampling_frequency=Sampling_frequency)
source_required_X_bags, source_required_Y_bags, source_required_amount, _, _, \
target_required_X_bags, target_required_Y_bags, target_required_amount, _, _ = \
    dsads_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required, activities_required)


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# feature extraction because each window only 7 data points, 4 frequency features can not calculated:
# #feature_all.append(self.fft_shape_std() ** 2)
# #feature_all.append(self.fft_shape_std())
# #feature_all.append(self.fft_shape_skew())
# #feature_all.append(self.fft_shape_kurt())
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def get_features(required_X_bags):
    samples = []
    for a_window in required_X_bags:
        a_sample = []
        Chest_a_x = a_window[:, 0]
        Chest_a_y = a_window[:, 1]
        Chest_a_z = a_window[:, 2]
        Chest_g_x = a_window[:, 3]
        Chest_g_y = a_window[:, 4]
        Chest_g_z = a_window[:, 5]

        '''
        Hand_a_x = a_window[:, 6]
        Hand_a_y = a_window[:, 7]
        Hand_a_z = a_window[:, 8]
        Hand_g_x = a_window[:, 9]
        Hand_g_y = a_window[:, 10]
        Hand_g_z = a_window[:, 11]

        Ankle_a_x = a_window[:, 12]
        Ankle_a_y = a_window[:, 13]
        Ankle_a_z = a_window[:, 14]
        Ankle_g_x = a_window[:, 15]
        Ankle_g_y = a_window[:, 16]
        Ankle_g_z = a_window[:, 17]
        '''
        Chest_a_xyz = np.sqrt(np.square(Chest_a_x) + np.square(Chest_a_y) + np.square(Chest_a_z))
        Chest_g_xyz = np.sqrt(np.square(Chest_g_x) + np.square(Chest_g_y) + np.square(Chest_g_z))
        #Hand_a_xyz = np.sqrt(np.square(Hand_a_x) + np.square(Hand_a_y) + np.square(Hand_a_z))
        #Hand_g_xyz = np.sqrt(np.square(Hand_g_x) + np.square(Hand_g_y) + np.square(Hand_g_z))
        #Ankle_a_xyz = np.sqrt(np.square(Ankle_a_x) + np.square(Ankle_a_y) + np.square(Ankle_a_z))
        #Ankle_g_xyz = np.sqrt(np.square(Ankle_g_x) + np.square(Ankle_g_y) + np.square(Ankle_g_z))

        a_sample.extend(get_feature(Chest_a_xyz))
        a_sample.extend(get_feature(Chest_g_xyz))
        #a_sample.extend(get_feature(Hand_a_xyz))
        #a_sample.extend(get_feature(Hand_g_xyz))
        #a_sample.extend(get_feature(Ankle_a_xyz))
        #a_sample.extend(get_feature(Ankle_g_xyz))

        samples.append(a_sample)
    return samples


source_samples = get_features(source_required_X_bags)
target_samples = get_features(target_required_X_bags)

source_samples = norm_max(np.array(source_samples))
target_samples = norm_max(np.array(target_samples))
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# data persistent
with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
    np.save(f, np.array(source_samples))  # work for DSADS
with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(source_required_Y_bags))
with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
    np.save(f, np.array(target_samples))  # work for DSADS
with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(target_required_Y_bags))
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_raw.npy',
          'wb') as f:
    np.save(f, np.array(source_required_X_bags))  # work for DSADS

with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_raw.npy',
          'wb') as f:
    np.save(f, np.array(target_required_X_bags))  # work for DSADS


start_source = 0
start_target = 0
for index, a_act in enumerate(activities_required):
    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
            Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
        np.save(f, np.array(source_samples[start_source: start_source + source_required_amount[index]]))  # work for DSADS

    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(source_required_Y_bags[start_source: start_source + source_required_amount[index]]))

    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
            Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
        np.save(f, np.array(target_samples[start_target: start_target + target_required_amount[index]]))  # work for DSADS

    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(target_required_Y_bags[start_target: start_target + target_required_amount[index]]))
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    with open(
            DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_raw.npy',
            'wb') as f:
        np.save(f, np.array(source_required_X_bags[start_source: start_source + source_required_amount[index]]))  # work for DSADS
    with open(
            DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_raw.npy',
            'wb') as f:
        np.save(f, np.array(target_required_X_bags[start_target: start_target + target_required_amount[index]])) # work for DSADS

    start_source += source_required_amount[index]
    start_target += target_required_amount[index]
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
print()