import math
import pickle
import numpy as np
import ot
import SOT.SOT as sot
import HMM.HMM as HMM
import Hidden_Markov_Model.hmm as myhmm


def TROT(n_state, reg_e, reg_cl, reg_ta):
    for source_user in [2, 3, 4, 5, 7, 8]:
        for target_user in [2, 3, 4, 5, 7, 8]:
            if source_user == target_user:
                continue
            else:
                source_user = str(source_user)
                target_user = str(target_user)
                Sampling_frequency = 25  # HZ
                Num_Seconds = 0.3
                Window_Overlap_Rate = 0.5
                DATASET_NAME = 'DSADS'
                activities_required = ['sitting', 'standing', 'lying_on_back', 'lying_on_right', 'ascending_stairs',
                                       'descending_stairs',
                                       'standing_in_an_elevator_still', 'moving_around_in_an_elevator',
                                       'walking_in_a_parking_lot', 'walking_on_a_treadmill_in_flat',
                                       'walking_on_a_treadmill_inclined_positions',
                                       'running_on_a_treadmill_in_flat', 'exercising on a stepper',
                                       'exercising on a cross trainer',
                                       'cycling_on_an_exercise_bike_in_horizontal_positions',
                                       'cycling_on_an_exercise_bike_in_vertical_positions',
                                       'rowing', 'jumping', 'playing_basketball']
                n_feature = 38
                Cov_Type = 'diag'
                # ['lying', 'walking', 'ascending_stairs', 'descending_stairs']
                # /////////////////
                with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                        Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
                    all_source_bags = np.load(f)
                with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                        Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
                    all_target_bags = np.load(f)
                with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy', 'rb') as f:
                    all_source_labels = np.load(f)
                with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy', 'rb') as f:
                    all_target_labels = np.load(f)
                # /////////////////

                s_list_mean = []
                t_list_mean = []
                s_clusters_index_list = []
                index_list = []
                for index, a_act in enumerate(activities_required):
                    # source
                    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                            Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
                        a_act_source_bags = np.load(f, allow_pickle=True)
                        Model_a_act_s = HMM.HMM_with_specified_I_T_Matrix(DATASET_NAME, bags=a_act_source_bags,
                                                                          user='source',
                                                                          n_state=n_state,
                                                                          Cov_Type=Cov_Type, activities_required=a_act)
                        Model_a_act_s_mean = Model_a_act_s.means_
                        s_list_mean.append(Model_a_act_s_mean)

                        s_clusters_index_list.extend([index] * n_state)

                    # target
                    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                            Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
                        a_act_target_bags = np.load(f, allow_pickle=True)
                        Model_a_act_t = HMM.HMM_with_specified_I_T_Matrix(DATASET_NAME, bags=a_act_target_bags,
                                                                          user='target',
                                                                          n_state=n_state,
                                                                          Cov_Type=Cov_Type, activities_required=a_act)
                        Model_a_act_t_mean = Model_a_act_t.means_
                        t_list_mean.append(Model_a_act_t_mean)
                        a_act_index = Model_a_act_t.predict(a_act_target_bags) + n_state * index
                        index_list.extend(a_act_index.tolist())
                # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                tsot = sot.SOT('', '', '', reg_e=reg_e, reg_cl=reg_cl, reg_ta=reg_ta)
                pred, acc = tsot.fit_predict_for_HMM(
                    xns=np.array(s_list_mean).reshape(n_state * len(activities_required), n_feature),
                    yns=s_clusters_index_list,
                    xnt=np.array(t_list_mean).reshape(n_state * len(activities_required), n_feature),
                    Ty=all_target_labels,
                    Tx=all_target_bags, index=index_list)
                print(source_user + '_to_' + target_user + ':' + str(acc))

                # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
