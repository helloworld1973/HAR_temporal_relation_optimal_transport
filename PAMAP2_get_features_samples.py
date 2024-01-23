import numpy as np
from SOT.SOT import norm_max
from read_dataset import read_PAMAP2_dataset
from feature_extraction.feature_core import get_feature

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# PAMAP2_dataset
activity_list = ['lying', 'sitting', 'standing', 'walking', 'running',
                 'cycling', 'Nordic_walking', 'ascending_stairs', 'descending_stairs',
                 'vacuum_cleaning', 'ironing']
activities_required = activity_list  # ['lying', 'walking', 'ascending_stairs', 'descending_stairs']
sensor_channels_required = ['IMU_Hand']  # ['IMU_Hand', 'IMU_Chest', 'IMU_Ankle']
# activities_required = ['lying']  # ['lying', 'sitting', 'standing', 'walking', 'running'] # activity_list  # activity_list  # 12 common activities ['rope_jumping']
source_user = '1'  # 1 # 5 # 6
target_user = '5'
Sampling_frequency = 100  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
DATASET_NAME = 'PAMAP2'
pamap2_ds = read_PAMAP2_dataset.READ_PAMAP2_DATASET(source_user, target_user, bag_window_second=Num_Seconds,
                                                    bag_overlap_rate=Window_Overlap_Rate,
                                                    instances_window_second=0.1, instances_overlap_rate=0.5,
                                                    sampling_frequency=Sampling_frequency)

source_required_X_bags, source_required_Y_bags, source_required_amount, _, _, \
target_required_X_bags, target_required_Y_bags, target_required_amount, _, _ = \
    pamap2_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required, activities_required)


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# feature extraction
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def get_features(required_X_bags):
    samples = []
    for a_window in required_X_bags:
        a_sample = []

        Hand_a_x = a_window[:, 0]
        Hand_a_y = a_window[:, 1]
        Hand_a_z = a_window[:, 2]
        Hand_g_x = a_window[:, 3]
        Hand_g_y = a_window[:, 4]
        Hand_g_z = a_window[:, 5]

        #Chest_a_x = a_window[:, 0]
        #Chest_a_y = a_window[:, 1]
        #Chest_a_z = a_window[:, 2]
        #Chest_g_x = a_window[:, 3]
        #Chest_g_y = a_window[:, 4]
        #Chest_g_z = a_window[:, 5]

        #Hand_a_x = a_window[:, 9]
        #Hand_a_y = a_window[:, 10]
        #Hand_a_z = a_window[:, 11]
        #Hand_g_x = a_window[:, 12]
        #Hand_g_y = a_window[:, 13]
        #Hand_g_z = a_window[:, 14]

        #Ankle_a_x = a_window[:, 18]
        #Ankle_a_y = a_window[:, 19]
        #Ankle_a_z = a_window[:, 20]
        #Ankle_g_x = a_window[:, 21]
        #Ankle_g_y = a_window[:, 22]
        #Ankle_g_z = a_window[:, 23]

        #Chest_a_xyz = np.sqrt(np.square(Chest_a_x) + np.square(Chest_a_y) + np.square(Chest_a_z))
        #Chest_g_xyz = np.sqrt(np.square(Chest_g_x) + np.square(Chest_g_y) + np.square(Chest_g_z))
        Hand_a_xyz = np.sqrt(np.square(Hand_a_x) + np.square(Hand_a_y) + np.square(Hand_a_z))
        Hand_g_xyz = np.sqrt(np.square(Hand_g_x) + np.square(Hand_g_y) + np.square(Hand_g_z))
        #Ankle_a_xyz = np.sqrt(np.square(Ankle_a_x) + np.square(Ankle_a_y) + np.square(Ankle_a_z))
        #Ankle_g_xyz = np.sqrt(np.square(Ankle_g_x) + np.square(Ankle_g_y) + np.square(Ankle_g_z))

        #a_sample.extend(get_feature(Chest_a_xyz))
        #a_sample.extend(get_feature(Chest_g_xyz))
        a_sample.extend(get_feature(Hand_a_xyz))
        a_sample.extend(get_feature(Hand_g_xyz))
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
    # np.save(f, np.array(source_samples))  # work for DSADS
    np.save(f, np.array(source_samples, dtype=object))  # work for OPPT PAMAP2
with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(source_required_Y_bags))
with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
    # np.save(f, np.array(target_samples))  # work for DSADS
    np.save(f, np.array(target_samples, dtype=object))  # work for OPPT PAMAP2
with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(target_required_Y_bags))
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_raw.npy',
          'wb') as f:
    # np.save(f, np.array(source_samples))  # work for DSADS
    np.save(f, np.array(source_required_X_bags, dtype=object))  # work for OPPT PAMAP2
with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_raw.npy',
          'wb') as f:
    # np.save(f, np.array(target_samples))  # work for DSADS
    np.save(f, np.array(target_required_X_bags, dtype=object))  # work for OPPT PAMAP2

start_source = 0
start_target = 0
for index, a_act in enumerate(activities_required):
    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
            Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
        # np.save(f, np.array(source_samples))  # work for DSADS
        np.save(f, np.array(source_samples[start_source: start_source + source_required_amount[index]],
                            dtype=object))  # work for OPPT PAMAP2
    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(source_required_Y_bags[start_source: start_source + source_required_amount[index]]))
    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
            Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
        # np.save(f, np.array(target_samples))  # work for DSADS
        np.save(f, np.array(target_samples[start_target: start_target + target_required_amount[index]],
                            dtype=object))  # work for OPPT PAMAP2
    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(target_required_Y_bags[start_target: start_target + target_required_amount[index]]))
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    with open(
            DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_raw.npy',
            'wb') as f:
        # np.save(f, np.array(source_samples))  # work for DSADS
        np.save(f, np.array(source_required_X_bags[start_source: start_source + source_required_amount[index]],
                            dtype=object))  # work for OPPT PAMAP2
    with open(
            DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_raw.npy',
            'wb') as f:
        # np.save(f, np.array(target_samples))  # work for DSADS
        np.save(f, np.array(target_required_X_bags[start_target: start_target + target_required_amount[index]],
                            dtype=object))  # work for OPPT PAMAP2

    start_source += source_required_amount[index]
    start_target += target_required_amount[index]
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
