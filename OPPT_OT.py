import math
import pickle
import numpy as np
import ot
import SOT.SOT as sot
import HMM.HMM as HMM
import Hidden_Markov_Model.hmm as myhmm

for source_user in ['S1', 'S2', 'S3']:
    for target_user in ['S1', 'S2', 'S3']:
        source_user = str(source_user)
        target_user = str(target_user)
        # problem mappings: 8_3  4_7   4_2  3_8

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # get features read files
        #source_user = '3'  # 2, 3, 4, 5, 7, 8
        #target_user = '8'
        Sampling_frequency = 30  # HZ
        Num_Seconds = 0.3
        Window_Overlap_Rate = 0.5
        DATASET_NAME = 'OPPT'
        activities_required = ['Stand', 'Walk', 'Sit', 'Lie']
        n_state = 4
        n_activities = 4
        n_feature = 76
        Cov_Type = 'diag'
        # ['lying', 'walking', 'ascending_stairs', 'descending_stairs']
        # /////////////////
        with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            all_source_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            all_target_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy', 'rb') as f:
            all_source_labels = np.load(f)
        with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy', 'rb') as f:
            all_target_labels = np.load(f)
        # /////////////////
        with open(DATASET_NAME + '_Stand_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            stand_source_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_Stand_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            stand_target_bags = np.load(f, allow_pickle=True)
        # /////////////////
        with open(DATASET_NAME + '_Walk_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            walking_source_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_Walk_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            walking_target_bags = np.load(f, allow_pickle=True)
        # /////////////////
        with open(DATASET_NAME + '_Sit_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            sit_source_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_Sit_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            sit_target_bags = np.load(f, allow_pickle=True)
        # /////////////////
        with open(DATASET_NAME + '_Lie_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            lie_source_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_Lie_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            lie_target_bags = np.load(f, allow_pickle=True)
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        # '''
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # HMM model generation for each activity for each user
        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=stand_source_bags, user='source', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Stand')
        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=stand_target_bags, user='target', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Stand')

        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=walking_source_bags, user='source', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Walk')
        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=walking_target_bags, user='target', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Walk')

        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=sit_source_bags, user='source', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Sit')
        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=sit_target_bags, user='target', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Sit')

        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=lie_source_bags, user='source', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Lie')
        HMM.HMM_with_specified_I_T_Matrix(dataset_name=DATASET_NAME, bags=lie_target_bags, user='target', n_state=n_state, Cov_Type=Cov_Type,
                                          activities_required='Lie')
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # '''

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # read model
        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Stand' + '_' + 'source' + '_' + Cov_Type + ".pkl", "rb") as file:
            Model_l_s = pickle.load(file)
        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Stand' + '_' + 'target' + '_' + Cov_Type + ".pkl", "rb") as file:
            Model_l_t = pickle.load(file)

        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Walk' + '_' + 'source' + '_' + Cov_Type + ".pkl", "rb") as file:
            Model_w_s = pickle.load(file)
        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Walk' + '_' + 'target' + '_' + Cov_Type + ".pkl", "rb") as file:
            Model_w_t = pickle.load(file)

        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Sit' + '_' + 'source' + '_' + Cov_Type + ".pkl",
                  "rb") as file:
            Model_d_s = pickle.load(file)
        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Sit' + '_' + 'target' + '_' + Cov_Type + ".pkl",
                  "rb") as file:
            Model_d_t = pickle.load(file)

        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Lie' + '_' + 'source' + '_' + Cov_Type + ".pkl",
                  "rb") as file:
            Model_a_s = pickle.load(file)
        with open(DATASET_NAME + '_GaussianHMM_' + str(n_state) + '_' + 'Lie' + '_' + 'target' + '_' + Cov_Type + ".pkl",
                  "rb") as file:
            Model_a_t = pickle.load(file)
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # get mean and variance
        Model_l_s_mean = Model_l_s.means_
        Model_l_s_variance = Model_l_s.covars_
        Model_l_t_mean = Model_l_t.means_
        Model_l_t_variance = Model_l_t.covars_

        Model_w_s_mean = Model_w_s.means_
        Model_w_s_variance = Model_w_s.covars_
        Model_w_t_mean = Model_w_t.means_
        Model_w_t_variance = Model_w_t.covars_

        Model_d_s_mean = Model_d_s.means_
        Model_d_s_variance = Model_d_s.covars_
        Model_d_t_mean = Model_d_t.means_
        Model_d_t_variance = Model_d_t.covars_

        Model_a_s_mean = Model_a_s.means_
        Model_a_s_variance = Model_a_s.covars_
        Model_a_t_mean = Model_a_t.means_
        Model_a_t_variance = Model_a_t.covars_
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # get_features_in_each_state
        def get_features_in_each_state(bags, model):
            features_list = []
            likely_state_sequence = model.predict(bags)
            for i in range(0, n_state):
                index = np.where(likely_state_sequence == i)
                features_list.append(bags[index])
            return features_list


        stand_s_features_list = get_features_in_each_state(stand_source_bags, Model_l_s)
        stand_t_features_list = get_features_in_each_state(stand_target_bags, Model_l_t)
        walking_s_features_list = get_features_in_each_state(walking_source_bags, Model_l_s)
        walking_t_features_list = get_features_in_each_state(walking_target_bags, Model_l_t)
        sit_s_features_list = get_features_in_each_state(sit_source_bags, Model_l_s)
        sit_t_features_list = get_features_in_each_state(sit_target_bags, Model_l_t)
        lie_s_features_list = get_features_in_each_state(lie_source_bags, Model_l_s)
        lie_t_features_list = get_features_in_each_state(lie_target_bags, Model_l_t)

        s_list_mean = [Model_l_s_mean, Model_w_s_mean, Model_a_s_mean, Model_d_s_mean]
        s_list_variance = [Model_l_s_variance, Model_w_s_variance, Model_a_s_variance, Model_d_s_variance]
        t_list_mean = [Model_l_t_mean, Model_w_t_mean, Model_a_t_mean, Model_d_t_mean]
        t_list_variance = [Model_l_t_variance, Model_w_t_variance, Model_a_t_variance, Model_d_t_variance]
        s_list_features_list = [stand_s_features_list, walking_s_features_list, sit_s_features_list,
                                lie_s_features_list]
        t_list_features_list = [stand_t_features_list, walking_t_features_list, sit_t_features_list,
                                lie_t_features_list]

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # SOT version

        l_t_index = Model_l_t.predict(stand_target_bags)
        w_t_index = Model_w_t.predict(walking_target_bags) + 4
        a_t_index = Model_a_t.predict(sit_target_bags) + 8
        d_t_index = Model_d_t.predict(lie_target_bags) + 12
        index = l_t_index.tolist() + w_t_index.tolist() + a_t_index.tolist() + d_t_index.tolist()

        tsot = sot.SOT('', '', '', 0.1, 1, 0.1)
        pred, acc = tsot.fit_predict_for_HMM(xns=np.array(s_list_mean).reshape(16, n_feature),
                                             yns=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                                             xnt=np.array(t_list_mean).reshape(16, n_feature), Ty=all_target_labels-1,
                                             Tx=all_target_bags, index=index)
        print(str(source_user) + '_to_' + str(target_user) + ':' + str(acc))




