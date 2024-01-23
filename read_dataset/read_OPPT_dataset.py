import zipfile
from io import BytesIO
import numpy as np
from gtda.time_series import SlidingWindow
from pandas import Series


# Sampling_frequency = 30 HZ
class READ_OPPT_DATASET:
    def __init__(self, source_user, target_user, bag_window_second, bag_overlap_rate, instances_window_second,
                 instances_overlap_rate, sampling_frequency):
        self.bag_window_second = bag_window_second
        self.bag_overlap_rate = bag_overlap_rate
        self.instances_window_second = instances_window_second
        self.instances_overlap_rate = instances_overlap_rate
        self.sampling_frequency = sampling_frequency

        self.File_Path = 'E:\\Python Projects\\TL_DATASETS_REPO\\OpportunityUCIDataset.zip'
        self.sub_folders = ['-ADL2.dat', '-ADL3.dat']
        # ['-ADL1.dat', '-ADL2.dat', '-ADL3.dat', '-ADL4.dat', '-ADL5.dat']

        self.Source_Data_Files = ['OpportunityUCIDataset/dataset/' + source_user + i for i in self.sub_folders]
        self.Target_Data_Files = ['OpportunityUCIDataset/dataset/' + target_user + i for i in self.sub_folders]
        self.l = "locomotion"
        self.NB_SENSOR_CHANNELS = 114
        # self.OVERLAP = 0.5
        self.Activity_Mapping = {'Stand': 1, 'Walk': 2, 'Sit': 3, 'Lie': 4}
        '''
        self.Activity_Mapping = {'Open Door 1': 1, 'Open Door 2': 2, 'Close Door 1': 3, 'Close Door 2': 4,
                                 'Open Fridge': 5, 'Close Fridge': 6, 'Open Dishwasher': 7, 'Close Dishwasher': 8,
                                 'Open Drawer 1': 9, 'Close Drawer 1': 10, 'Open Drawer 2': 11, 'Close Drawer 2': 12,
                                 'Open Drawer 3': 13, 'Close Drawer 3': 14, 'Clean Table': 15, 'Drink from Cup': 16,
                                 'Toggle Switch': 17, 'Others': 0}
        '''
        self.Sensor_Dict = {'timestamp': 0,
                            'ACC_RKN^_X': 1, 'ACC_RKN^_Y': 2, 'ACC_RKN^_Z': 3,  # right knee high
                            'ACC_HIP_X': 4, 'ACC_HIP_Y': 5, 'ACC_HIP_Z': 6,  # hip
                            'ACC_LUA^_X': 7, 'ACC_LUA^_Y': 8, 'ACC_LUA^_Z': 9,  # left upper arm high
                            'ACC_RUA__X': 10, 'ACC_RUA__Y': 11, 'ACC_RUA__Z': 12,  # right upper arm low
                            'ACC_LH_X': 13, 'ACC_LH_Y': 14, 'ACC_LH_Z': 15,  # left hand
                            'ACC_BACK_X': 16, 'ACC_BACK_Y': 17, 'ACC_BACK_Z': 18,  # back
                            'ACC_RKN__X': 19, 'ACC_RKN__Y': 20, 'ACC_RKN__Z': 21,  # right knee low
                            'ACC_RWR_X': 22, 'ACC_RWR_Y': 23, 'ACC_RWR_Z': 24,  # right wrist
                            'ACC_RUA^_X': 25, 'ACC_RUA^_Y': 26, 'ACC_RUA^_Z': 27,  # right upper arm high
                            'ACC_LUA__X': 28, 'ACC_LUA__Y': 29, 'ACC_LUA__Z': 30,  # left upper arm low
                            'ACC_LWR_X': 31, 'ACC_LWR_Y': 32, 'ACC_LWR_Z': 33,  # left wrist
                            'ACC_RH_X': 34, 'ACC_RH_Y': 35, 'ACC_RH_Z': 36,  # right hand

                            # back
                            'IMU_BACK_ACC_X': 37, 'IMU_BACK_ACC_Y': 38, 'IMU_BACK_ACC_Z': 39, 'IMU_BACK_GYRO_X': 40,
                            'IMU_BACK_GYRO_Y': 41, 'IMU_BACK_GYRO_Z': 42, 'IMU_BACK_MAG_X': 43, 'IMU_BACK_MAG_Y': 44,
                            'IMU_BACK_MAG_Z': 45,
                            # right upper arm
                            'IMU_RUA_ACC_X': 46, 'IMU_RUA_ACC_Y': 47, 'IMU_RUA_ACC_Z': 48, 'IMU_RUA_GYRO_X': 49,
                            'IMU_RUA_GYRO_Y': 50, 'IMU_RUA_GYRO_Z': 51, 'IMU_RUA_MAG_X': 52, 'IMU_RUA_MAG_Y': 53,
                            'IMU_RUA_MAG_Z': 54,
                            # right lower arm
                            'IMU_RLA_ACC_X': 55, 'IMU_RLA_ACC_Y': 56, 'IMU_RLA_ACC_Z': 57, 'IMU_RLA_GYRO_X': 58,
                            'IMU_RLA_GYRO_Y': 59, 'IMU_RLA_GYRO_Z': 60, 'IMU_RLA_MAG_X': 61, 'IMU_RLA_MAG_Y': 62,
                            'IMU_RLA_MAG_Z': 63,
                            # left upper arm
                            'IMU_LUA_ACC_X': 64, 'IMU_LUA_ACC_Y': 65, 'IMU_LUA_ACC_Z': 66, 'IMU_LUA_GYRO_X': 67,
                            'IMU_LUA_GYRO_Y': 68, 'IMU_LUA_GYRO_Z': 69, 'IMU_LUA_MAG_X': 70, 'IMU_LUA_MAG_Y': 71,
                            'IMU_LUA_MAG_Z': 72,
                            # left lower arm
                            'IMU_LLA_ACC_X': 73, 'IMU_LLA_ACC_Y': 74, 'IMU_LLA_ACC_Z': 75, 'IMU_LLA_GYRO_X': 76,
                            'IMU_LLA_GYRO_Y': 77, 'IMU_LLA_GYRO_Z': 78, 'IMU_LLA_MAG_X': 79, 'IMU_LLA_MAG_Y': 80,
                            'IMU_LLA_MAG_Z': 81}

    def find_sensor_channel_ID_by_sensor_channel_name(self, sensor_channel_name):
        sensor_channel_ID = self.Sensor_Dict[sensor_channel_name]
        return sensor_channel_ID

    def find_activity_ID_by_activity_name(self, activity_name):
        activity_ID = self.Activity_Mapping[activity_name]
        return activity_ID

    def seg_samples_in_same_activity(self, source_index_list):
        samples_list = []
        a_sample = []
        pre_value = source_index_list[0]
        for a_index in source_index_list:
            if a_index - pre_value > 30:
                samples_list.append(a_sample)
                a_sample = []
                a_sample.append(a_index)
            else:
                a_sample.append(a_index)
            pre_value = a_index
        samples_list.append(a_sample)
        return samples_list

    def data_segment(self, data_x, data_y, activities_ID_list):
        sliding_bag = SlidingWindow(size=int(self.sampling_frequency * self.bag_window_second), stride=int(
            self.sampling_frequency * self.bag_window_second * (1 - self.bag_overlap_rate)))
        X_bags = sliding_bag.fit_transform(data_x)
        Y_bags = sliding_bag.resample(data_y)  # last occur label
        Y_bags = Y_bags.tolist()

        required_X_bags = []
        required_Y_bags = []
        required_amount = []
        for a_activity_ID in activities_ID_list:
            a_activity_index_list = [i for i, j in enumerate(Y_bags) if j == a_activity_ID]
            X_bags_a_activity = X_bags[a_activity_index_list]
            Y_bags_a_activity = np.array(Y_bags)[a_activity_index_list]
            required_X_bags.extend(X_bags_a_activity)
            required_Y_bags.extend(Y_bags_a_activity)
            required_amount.append(len(Y_bags_a_activity))

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

        return required_X_bags, required_Y_bags, required_amount, required_X_bags_with_instances, required_Y_bags_with_instances

    def generate_data_with_required_sensor_channels_and_activities(self, sensor_channels_required, activities_required):
        source_data_x, source_data_y, target_data_x, target_data_y = self.generate_data_with_required_sensor_channels(
            sensor_channels_required)
        activities_ID_list = [self.find_activity_ID_by_activity_name(a_activity_name) for a_activity_name in
                              activities_required]

        # drop row if it contains null value
        mask = np.any(np.isnan(source_data_x), axis=1)
        source_data_x = source_data_x[~mask]
        source_data_y = source_data_y[~mask]
        mask = np.any(np.isnan(target_data_x), axis=1)
        target_data_x = target_data_x[~mask]
        target_data_y = target_data_y[~mask]

        source_required_X_bags, source_required_Y_bags, source_required_amount, source_required_X_bags_with_instances, source_required_Y_bags_with_instances = self.data_segment(
            source_data_x, source_data_y, activities_ID_list)
        target_required_X_bags, target_required_Y_bags, target_required_amount, target_required_X_bags_with_instances, target_required_Y_bags_with_instances = self.data_segment(
            target_data_x, target_data_y, activities_ID_list)

        return source_required_X_bags, source_required_Y_bags, source_required_amount, source_required_X_bags_with_instances, source_required_Y_bags_with_instances, \
               target_required_X_bags, target_required_Y_bags, target_required_amount, target_required_X_bags_with_instances, target_required_Y_bags_with_instances, \
               source_data_x, target_data_x

    def generate_data_with_required_sensor_channels(self, sensor_channels_required):
        source_data_x, source_data_y, target_data_x, target_data_y = self.generate_data()
        sensor_channels_ID_list = [self.find_sensor_channel_ID_by_sensor_channel_name(a_channel_name) for a_channel_name
                                   in sensor_channels_required]
        source_data_x = source_data_x[:, sensor_channels_ID_list]
        target_data_x = target_data_x[:, sensor_channels_ID_list]
        '''
        unify to unit
        gyr: rad / s
        acc: m / s^2
        mag: mT
        '''
        source_data_x[:, 0] = source_data_x[:, 0] / 1000 * 9.8
        source_data_x[:, 1] = source_data_x[:, 1] / 1000 * 9.8
        source_data_x[:, 2] = source_data_x[:, 2] / 1000 * 9.8
        target_data_x[:, 0] = target_data_x[:, 0] / 1000 * 9.8
        target_data_x[:, 1] = target_data_x[:, 1] / 1000 * 9.8
        target_data_x[:, 2] = target_data_x[:, 2] / 1000 * 9.8
        # gyro
        source_data_x[:, 3] = source_data_x[:, 3] / 1000
        source_data_x[:, 4] = source_data_x[:, 4] / 1000
        source_data_x[:, 5] = source_data_x[:, 5] / 1000
        target_data_x[:, 3] = target_data_x[:, 3] / 1000
        target_data_x[:, 4] = target_data_x[:, 4] / 1000
        target_data_x[:, 5] = target_data_x[:, 5] / 1000
        '''
        # mag
        source_data_x[:, 6] = source_data_x[:, 6] / 1000 / 10
        source_data_x[:, 7] = source_data_x[:, 7] / 1000 / 10
        source_data_x[:, 8] = source_data_x[:, 8] / 1000 / 10
        target_data_x[:, 6] = target_data_x[:, 6] / 1000 / 10
        target_data_x[:, 7] = target_data_x[:, 7] / 1000 / 10
        target_data_x[:, 8] = target_data_x[:, 8] / 1000 / 10
        '''
        return source_data_x, source_data_y, target_data_x, target_data_y

    def generate_data(self):
        source_data_x = np.empty((0, self.NB_SENSOR_CHANNELS))
        source_data_y = np.empty((0))
        target_data_x = np.empty((0, self.NB_SENSOR_CHANNELS))
        target_data_y = np.empty((0))

        zf = zipfile.ZipFile(self.File_Path)
        print('Processing dataset files ...')

        source_data_x_all = []
        source_data_y_all = []
        for i, filename in enumerate(self.Source_Data_Files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {0}'.format(filename))
                # np.savetxt(target_filename+fnrr+".csv", data, delimiter=",")
                x, y = self.process_dataset_file(data, self.l)
                source_data_x = np.vstack((source_data_x, x))
                source_data_y = np.concatenate([source_data_y, y])
                if i == 0:
                    source_data_x_all = source_data_x
                    source_data_y_all = source_data_y
                else:
                    source_data_x_all = np.concatenate((source_data_x_all, source_data_x), axis=0)
                    source_data_y_all = np.concatenate((source_data_y_all, source_data_y), axis=0)

            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))

        target_data_x_all = []
        target_data_y_all = []
        for i, filename in enumerate(self.Target_Data_Files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {0}'.format(filename))
                # np.savetxt(target_filename+fnrr+".csv", data, delimiter=",")
                x, y = self.process_dataset_file(data, self.l)
                target_data_x = np.vstack((target_data_x, x))
                target_data_y = np.concatenate([target_data_y, y])
                if i == 0:
                    target_data_x_all = target_data_x
                    target_data_y_all = target_data_y
                else:
                    target_data_x_all = np.concatenate((target_data_x_all, target_data_x), axis=0)
                    target_data_y_all = np.concatenate((target_data_y_all, target_data_y), axis=0)

            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))

        return source_data_x_all, source_data_y_all, target_data_x_all, target_data_y_all

    '''
    def sliding_windows_construction(self, source_data_x, WINDOW_SIZE, OVERLAP):
        height = len(source_data_x)
        i = 0
        list = []
        while i + WINDOW_SIZE < height:
            period_list = []
            start = i
            end = i + WINDOW_SIZE
            period_list.append(start)
            period_list.append(end)
            list.append(period_list)
            i = end + 1 - int(WINDOW_SIZE * OVERLAP)
        return list
    '''
    '''
    def load_data(self, len_seq):
        source_data_x, source_data_y, target_data_x, target_data_y = self.generate_data(self.File_Path, self.l)

        # sliding windows construction
        source_data_x_windows = self.sliding_windows_construction(source_data_x, len_seq, self.OVERLAP)
        target_data_x_windows = self.sliding_windows_construction(target_data_x, len_seq, self.OVERLAP)

        source_data_x_list = []
        source_data_y_list = []
        for window in source_data_x_windows:
            startIndex = window[0]
            endIndex = window[1]
            a_x = source_data_x[startIndex:endIndex]
            a_y = source_data_y[endIndex]
            source_data_x_list.append(a_x)
            source_data_y_list.append(a_y)

        target_data_x_list = []
        target_data_y_list = []
        for window in target_data_x_windows:
            startIndex = window[0]
            endIndex = window[1]
            a_x = target_data_x[startIndex:endIndex]
            a_y = target_data_y[endIndex]
            target_data_x_list.append(a_x)
            target_data_y_list.append(a_y)

        source_data_x_list = np.array(source_data_x_list)
        source_data_y_list = np.array(source_data_y_list)
        target_data_x_list = np.array(target_data_x_list)
        target_data_y_list = np.array(target_data_y_list)

        return source_data_x_list, source_data_y_list, target_data_x_list, target_data_y_list
    '''

    def process_dataset_file(self, data, label):
        """Function defined as a pipeline to process individual OPPORTUNITY files

        :param data: numpy integer matrix
            Matrix containing data samples (rows) for every sensor channel (column)
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numy integer array
            Processed sensor data, segmented into features (x) and labels (y)
        """

        # Select correct columns
        data = self.select_columns_opp(data)

        # Colums are segmentd into features and labels
        data_x, data_y = self.divide_x_y(data, label)
        data_y = self.adjust_idx_labels(data_y, label)
        data_y = data_y.astype(int)

        # Perform linear interpolation
        # data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

        # Remaining missing data are converted to zero
        # data_x[np.isnan(data_x)] = 0

        # All sensor channels are normalized
        # scaler = MinMaxScaler()
        # data_x = scaler.fit_transform(data_x)

        return data_x, data_y

    def select_columns_opp(self, data):
        """Selection of the 113 columns employed in the OPPORTUNITY challenge

        :param data: numpy integer matrix
            Sensor data (all features)
        :return: numpy integer matrix
            Selection of features
        """

        # included-excluded
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        return np.delete(data, features_delete, 1)

    def divide_x_y(self, data, label):
        """Segments each sample into features and label

        :param data: numpy integer matrix
            Sensor data
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numpy integer array
            Features encapsulated into a matrix and labels as an array
        """

        data_x = data[:, 0:114]
        if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
        if label == 'locomotion':
            data_y = data[:, 114]  # Locomotion label
        elif label == 'gestures':
            data_y = data[:, 115]  # Gestures label

        return data_x, data_y

    def adjust_idx_labels(self, data_y, label):
        """Transforms original labels into the range [0, nb_labels-1]

        :param data_y: numpy integer array
            Sensor labels
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer array
            Modified sensor labels
        """

        if label == 'locomotion':  # Labels for locomotion are adjusted
            data_y[data_y == 4] = 3
            data_y[data_y == 5] = 4
        elif label == 'gestures':  # Labels for gestures are adjusted
            data_y[data_y == 406516] = 1
            data_y[data_y == 406517] = 2
            data_y[data_y == 404516] = 3
            data_y[data_y == 404517] = 4
            data_y[data_y == 406520] = 5
            data_y[data_y == 404520] = 6
            data_y[data_y == 406505] = 7
            data_y[data_y == 404505] = 8
            data_y[data_y == 406519] = 9
            data_y[data_y == 404519] = 10
            data_y[data_y == 406511] = 11
            data_y[data_y == 404511] = 12
            data_y[data_y == 406508] = 13
            data_y[data_y == 404508] = 14
            data_y[data_y == 408512] = 15
            data_y[data_y == 407521] = 16
            data_y[data_y == 405506] = 17
        return data_y
