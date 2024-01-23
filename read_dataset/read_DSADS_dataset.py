import numpy as np
from gtda.time_series import SlidingWindow


# sampling frequency: 25Hz
class READ_DSADS_DATASET:
    def __init__(self, source_user, target_user, bag_window_second, bag_overlap_rate, instances_window_second,
                 instances_overlap_rate, sampling_frequency):
        self.File_Path = 'E:\\Python Projects\\TL_DATASETS_REPO\\Daily and Sports Activities Data Set\\data\\'
        self.source_user = source_user
        self.target_user = target_user
        self.bag_window_second = bag_window_second
        self.bag_overlap_rate = bag_overlap_rate
        self.instances_window_second = instances_window_second
        self.instances_overlap_rate = instances_overlap_rate
        self.sampling_frequency = sampling_frequency

        self.Activity_Mapping = {'sitting': 'a01',
                                 'standing': 'a02',
                                 'lying_on_back': 'a03',
                                 'lying_on_right': 'a04',
                                 'ascending_stairs': 'a05',
                                 'descending_stairs': 'a06',
                                 'standing_in_an_elevator_still': 'a07',
                                 'moving_around_in_an_elevator': 'a08',
                                 'walking_in_a_parking_lot': 'a09',
                                 'walking_on_a_treadmill_in_flat': 'a10',
                                 'walking_on_a_treadmill_inclined_positions': 'a11',
                                 'running_on_a_treadmill_in_flat': 'a12',
                                 'exercising on a stepper': 'a13',
                                 'exercising on a cross trainer': 'a14',
                                 'cycling_on_an_exercise_bike_in_horizontal_positions': 'a15',
                                 'cycling_on_an_exercise_bike_in_vertical_positions': 'a16',
                                 'rowing': 'a17',
                                 'jumping': 'a18',
                                 'playing_basketball': 'a19'}

        self.Activity_Mapping_ID = {'a01': 0,
                                    'a02': 1,
                                    'a03': 2,
                                    'a04': 3,
                                    'a05': 4,
                                    'a06': 5,
                                    'a07': 6,
                                    'a08': 7,
                                    'a09': 8,
                                    'a10': 9,
                                    'a11': 10,
                                    'a12': 11,
                                    'a13': 12,
                                    'a14': 13,
                                    'a15': 14,
                                    'a16': 15,
                                    'a17': 16,
                                    'a18': 17,
                                    'a19': 18}

        self.User_Mapping = {'1': 'p1',
                             '2': 'p2',
                             '3': 'p3',
                             '4': 'p4',
                             '5': 'p5',
                             '6': 'p6',
                             '7': 'p7',
                             '8': 'p8'}

        self.all_60_files = ['s01.txt', 's02.txt', 's03.txt', 's04.txt', 's05.txt', 's06.txt', 's07.txt', 's08.txt',
                             's09.txt', 's10.txt',
                             's11.txt', 's12.txt', 's13.txt', 's14.txt', 's15.txt', 's16.txt', 's17.txt', 's18.txt',
                             's19.txt', 's20.txt',
                             's21.txt', 's22.txt', 's23.txt', 's24.txt', 's25.txt', 's26.txt', 's27.txt', 's28.txt',
                             's29.txt', 's30.txt',
                             's31.txt', 's32.txt', 's33.txt', 's34.txt', 's35.txt', 's36.txt', 's37.txt', 's38.txt',
                             's39.txt', 's40.txt',
                             's41.txt', 's42.txt', 's43.txt', 's44.txt', 's45.txt', 's46.txt', 's47.txt', 's48.txt',
                             's49.txt', 's50.txt',
                             's51.txt', 's52.txt', 's53.txt', 's54.txt', 's55.txt', 's56.txt', 's57.txt', 's58.txt',
                             's59.txt', 's60.txt']

        self.Sensor_Dict = {'T_x_acc': 0, 'T_y_acc': 1, 'T_z_acc': 2,
                            'T_x_gyro': 3, 'T_y_gyro': 4, 'T_z_gyro': 5,
                            'T_x_mag': 6, 'T_y_mag': 7, 'T_z_mag': 8,

                            'RA_x_acc': 9, 'RA_y_acc': 10, 'RA_z_acc': 11,
                            'RA_x_gyro': 12, 'RA_y_gyro': 13, 'RA_z_gyro': 14,
                            'RA_x_mag': 15, 'RA_y_mag': 16, 'RA_z_mag': 17,

                            'LA_x_acc': 18, 'LA_y_acc': 19, 'LA_z_acc': 20,
                            'LA_x_gyro': 21, 'LA_y_gyro': 22, 'LA_z_gyro': 23,
                            'LA_x_mag': 24, 'LA_y_mag': 25, 'LA_z_mag': 26,

                            'RL_x_acc': 27, 'RL_y_acc': 28, 'RL_z_acc': 29,
                            'RL_x_gyro': 30, 'RL_y_gyro': 31, 'RL_z_gyro': 32,
                            'RL_x_mag': 33, 'RL_y_mag': 34, 'RL_z_mag': 35,

                            'LL_x_acc': 36, 'LL_y_acc': 37, 'LL_z_acc': 38,
                            'LL_x_gyro': 39, 'LL_y_gyro': 40, 'LL_z_gyro': 41,
                            'LL_x_mag': 42, 'LL_y_mag': 43, 'LL_z_mag': 44}

    def find_sensor_channel_ID_by_sensor_channel_name(self, sensor_channel_name):
        sensor_channel_ID = self.Sensor_Dict[sensor_channel_name]
        return sensor_channel_ID

    def find_activity_ID_by_activity_name(self, activity_name):
        activity_ID = self.Activity_Mapping_ID[activity_name]
        return activity_ID

    def find_activity_folder_by_activity_name(self, activity_name):
        activity_folder = self.Activity_Mapping[activity_name]
        return activity_folder

    def find_user_name_by_user_ID(self, user_ID):
        user_name = self.User_Mapping[user_ID]
        return user_name

    def generate_data_with_required_sensor_channels_and_activities_and_a_user(self, which_user, activities_folder_list,
                                                                              sensor_channels_ID_list_required):
        user_name = self.find_user_name_by_user_ID(which_user)
        required_X_bags = []
        required_Y_bags = []
        required_amount = []
        required_X_bags_with_instances = []
        required_Y_bags_with_instances = []

        for a_activity_folder in activities_folder_list:
            x_a_activity_a_user_complete_data = []
            y_a_activity_a_user_complete_data = []
            for a_sub_txt_file_name in self.all_60_files:
                file_path = self.File_Path + a_activity_folder + '\\' + user_name + '\\' + a_sub_txt_file_name
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        a_row_column = line.replace('\n', '').split(',')
                        new_a_row_column = [float(i) for i in a_row_column]
                        x_a_activity_a_user_complete_data.append(new_a_row_column)
                        y_a_activity_a_user_complete_data.append(
                            self.find_activity_ID_by_activity_name(a_activity_folder))
            x_a_activity_a_user_complete_data = np.array(x_a_activity_a_user_complete_data)
            x_a_activity_a_user_complete_data = x_a_activity_a_user_complete_data[:, sensor_channels_ID_list_required]

            '''
            unify unit as
            gyr: rad / s
            acc: m / s^2
            mag: mT
            
            # magnetic data may not very useful, x^2 + y^2 + z^2 not same, other electrical device effect
            x_a_activity_a_user_complete_data[:, 6] = x_a_activity_a_user_complete_data[:, 6] / 10
            x_a_activity_a_user_complete_data[:, 7] = x_a_activity_a_user_complete_data[:, 7] / 10
            x_a_activity_a_user_complete_data[:, 8] = x_a_activity_a_user_complete_data[:, 8] / 10
            '''
            required_X_bags_a_activity, required_Y_bags_a_activity, required_X_bags_with_instances_a_activity, required_Y_bags_with_instances_a_activity = self.data_segment(
                x_a_activity_a_user_complete_data, y_a_activity_a_user_complete_data)

            required_X_bags.extend(required_X_bags_a_activity)
            required_Y_bags.extend(required_Y_bags_a_activity)
            required_amount.append(len(required_Y_bags_a_activity))
            required_X_bags_with_instances.extend(required_X_bags_with_instances_a_activity)
            required_Y_bags_with_instances.extend(required_Y_bags_with_instances_a_activity)

        return required_X_bags, required_Y_bags, required_amount, required_X_bags_with_instances, required_Y_bags_with_instances

    def data_segment(self, data_x, data_y):
        sliding_bag = SlidingWindow(size=int(self.sampling_frequency * self.bag_window_second), stride=int(
            self.sampling_frequency * self.bag_window_second * (1 - self.bag_overlap_rate)))
        X_bags = sliding_bag.fit_transform(data_x)
        Y_bags = sliding_bag.resample(data_y)  # last occur label
        Y_bags = Y_bags.tolist()

        required_X_bags = []
        required_Y_bags = Y_bags
        for a_X_bag in X_bags:
            required_X_bags.append(a_X_bag)

        required_X_bags_with_instances = []
        required_Y_bags_with_instances = []
        for i in range(len(required_X_bags)):
            sliding_instance = SlidingWindow(size=int(self.sampling_frequency * self.instances_window_second),
                                             stride=int(
                                                 self.sampling_frequency * self.instances_window_second * (
                                                         1 - self.instances_overlap_rate)))
            a_X_bag_instances = sliding_instance.fit_transform(required_X_bags[i])
            num_instances = a_X_bag_instances.shape[0]
            a_Y_bag_instances = [required_Y_bags[i]] * num_instances

            required_X_bags_with_instances.append(a_X_bag_instances)
            required_Y_bags_with_instances.append(a_Y_bag_instances)

        return required_X_bags, required_Y_bags, required_X_bags_with_instances, required_Y_bags_with_instances

    def generate_data_with_required_sensor_channels_and_activities(self, sensor_channels_required, activities_required):
        sensor_channels_ID_list_required = [self.find_sensor_channel_ID_by_sensor_channel_name(a_channel_name) for
                                            a_channel_name in sensor_channels_required]
        activities_folder_list = [self.find_activity_folder_by_activity_name(a_activity_name) for a_activity_name in
                                  activities_required]

        source_required_X_bags, source_required_Y_bags, source_required_amount, source_required_X_bags_with_instances, source_required_Y_bags_with_instances = self.generate_data_with_required_sensor_channels_and_activities_and_a_user(
            self.source_user, activities_folder_list, sensor_channels_ID_list_required)
        target_required_X_bags, target_required_Y_bags, target_required_amount, target_required_X_bags_with_instances, target_required_Y_bags_with_instances = self.generate_data_with_required_sensor_channels_and_activities_and_a_user(
            self.target_user, activities_folder_list, sensor_channels_ID_list_required)

        return source_required_X_bags, source_required_Y_bags, source_required_amount, source_required_X_bags_with_instances, source_required_Y_bags_with_instances, target_required_X_bags, target_required_Y_bags, target_required_amount, target_required_X_bags_with_instances, target_required_Y_bags_with_instances
