import numpy as np
import sklearn
import time
import random
import pickle
import HMM.HMM as HMM
import SOT.SOT as sot
from sklearn.metrics import classification_report, confusion_matrix


n_feature = 38
Cov_Type = 'diag'
Num_Seconds = 0.3
Window_Overlap_Rate = 0.5
S_T_file_path_pairs = []

# -------------------------------------------------------------------------------------------------------------------
# read data PAMAP2
DATASET_NAME = 'PAMAP2'
activities_required = ['lying', 'sitting', 'standing', 'walking', 'running',
                       'cycling', 'Nordic_walking', 'ascending_stairs', 'descending_stairs',
                       'vacuum_cleaning', 'ironing']
user_list = ['1', '5', '6']

for source_user in user_list:
    for target_user in user_list:
        if source_user == target_user:
            continue
        else:
            S_feats = DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy'
            S_labels = DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy'
            T_feats = DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy'
            T_labels = DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy'
            S_T_file_path_pairs.append([S_feats, S_labels, T_feats, T_labels, DATASET_NAME])
# --------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# read data DSADS
DATASET_NAME = 'DSADS'
activities_required = ['sitting', 'standing', 'lying_on_back', 'lying_on_right', 'ascending_stairs',
                       'descending_stairs', 'standing_in_an_elevator_still', 'moving_around_in_an_elevator',
                       'walking_in_a_parking_lot', 'walking_on_a_treadmill_in_flat',
                       'walking_on_a_treadmill_inclined_positions',
                       'running_on_a_treadmill_in_flat', 'exercising on a stepper',
                       'exercising on a cross trainer',
                       'cycling_on_an_exercise_bike_in_horizontal_positions',
                       'cycling_on_an_exercise_bike_in_vertical_positions',
                       'rowing', 'jumping', 'playing_basketball']
user_list = ['2', '3', '4', '5', '7', '8']

for source_user in user_list:
    for target_user in user_list:
        if source_user == target_user:
            continue
        else:
            S_feats = DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy'
            S_labels = DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy'
            T_feats = DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy'
            T_labels = DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy'
            S_T_file_path_pairs.append([S_feats, S_labels, T_feats, T_labels, DATASET_NAME])
# -------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------
# read data OPPT
DATASET_NAME = 'OPPT'
activities_required = ['Stand', 'Walk', 'Sit', 'Lie']
user_list = ['S1', 'S2', 'S3']

for source_user in user_list:
    for target_user in user_list:
        if source_user == target_user:
            continue
        else:
            S_feats = DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy'
            S_labels = DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy'
            T_feats = DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy'
            T_labels = DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy'
            S_T_file_path_pairs.append([S_feats, S_labels, T_feats, T_labels, DATASET_NAME])


# -------------------------------------------------------------------------------------------------------------------


def split_vali_test_with_temporal_order(T_feats, T_labels):
    unique_labels = list(set(T_labels))

    T_vali_feats = np.array([[]])
    T_test_feats = np.array([[]])
    T_vali_labels = np.array([[]])
    T_test_labels = np.array([[]])
    for index, i in enumerate(unique_labels):
        indices = [j for j, x in enumerate(T_labels) if x == i]
        split_index = int(len(indices) / 2)
        vali_indices = indices[0: split_index]
        test_indices = indices[split_index: len(indices)]
        if index == 0:
            T_vali_feats = T_feats[vali_indices]
            T_test_feats = T_feats[test_indices]
            T_vali_labels = T_labels[vali_indices]
            T_test_labels = T_labels[test_indices]
        else:
            T_vali_feats = np.concatenate((T_vali_feats, T_feats[vali_indices]), axis=0)
            T_test_feats = np.concatenate((T_test_feats, T_feats[test_indices]), axis=0)
            T_vali_labels = np.concatenate((T_vali_labels, T_labels[vali_indices]), axis=0)
            T_test_labels = np.concatenate((T_test_labels, T_labels[test_indices]), axis=0)

    return T_vali_feats, T_vali_labels, T_test_feats, T_test_labels


def TROT(DATASET_NAME, source_user, target_user, vali_or_test, n_state, reg_e, reg_cl, reg_ta, target_bags,
         target_labels):
    activities_required = []

    if DATASET_NAME == 'OPPT':
        activities_required = ['Stand', 'Walk', 'Sit', 'Lie']

    elif DATASET_NAME == 'DSADS':
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

    elif DATASET_NAME == 'PAMAP2':
        activities_required = ['lying', 'sitting', 'standing', 'walking', 'running',
                               'cycling', 'Nordic_walking', 'ascending_stairs', 'descending_stairs',
                               'vacuum_cleaning', 'ironing']

    def set_data_read_range_with_vali_or_test(vali_or_test, a_act_source_bags):
        start_index = -1
        end_index = -1

        if vali_or_test == "validation":
            start_index = 0
            end_index = int(len(a_act_source_bags) / 2)

        elif vali_or_test == "test":
            start_index = int(len(a_act_source_bags) / 2)
            end_index = int(len(a_act_source_bags))

        return start_index, end_index

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
            start_index, end_index = set_data_read_range_with_vali_or_test(vali_or_test, a_act_target_bags)
            Model_a_act_t = HMM.HMM_with_specified_I_T_Matrix(DATASET_NAME,
                                                              bags=a_act_target_bags[start_index:end_index, :],
                                                              user='target',
                                                              n_state=n_state,
                                                              Cov_Type=Cov_Type, activities_required=a_act)
            Model_a_act_t_mean = Model_a_act_t.means_
            t_list_mean.append(Model_a_act_t_mean)
            a_act_index = Model_a_act_t.predict(a_act_target_bags[start_index:end_index, :]) + n_state * index
            index_list.extend(a_act_index.tolist())

    # -----------------------------------------------------------------------------------------------------------------
    tsot = sot.SOT(reg_e=reg_e, reg_cl=reg_cl, reg_ta=reg_ta)
    pred, acc = tsot.fit_predict_for_HMM(
        xns=np.array(s_list_mean).reshape(n_state * len(activities_required), n_feature),
        yns=s_clusters_index_list,
        xnt=np.array(t_list_mean).reshape(n_state * len(activities_required), n_feature),
        Ty=target_labels,
        Tx=target_bags, index=index_list)
    cm = confusion_matrix(target_labels, pred)
    classify_results = classification_report(target_labels, pred)

    return acc, cm, classify_results


def get_param_optimal():
    n_state = [2, 3, 4, 5, 6, 7, 8]
    reg_e_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2, 3]
    reg_cl_list = [0, 0.03, 0.07, 0.1, 0.2, 0.4, 0.8, 1.6, 2, 3]
    reg_ta_list = [0, 0.03, 0.07, 0.1, 0.2, 0.4, 0.8, 1.6, 2, 3]

    params_list = []
    for a_state in n_state:
        for a_reg_e in reg_e_list:
            for a_reg_cl in reg_cl_list:
                for a_reg_ta in reg_ta_list:
                    param = {"n_state": a_state, "reg_e": a_reg_e, "reg_cl": a_reg_cl, "reg_ta": a_reg_ta}
                    params_list.append(param)
    return params_list


# -------------------------------------------------------------------------------------------------------------------
# data process and split as training set, validation set, testing set


for S_feats, S_labels, T_feats, T_labels, DATASET_NAME in S_T_file_path_pairs:

    if DATASET_NAME == 'PAMAP2' or DATASET_NAME == 'OPPT':
        with open(S_feats, 'rb') as f:
            all_source_bags = np.load(f, allow_pickle=True).astype(float)
        with open(T_feats, 'rb') as f:
            all_target_bags = np.load(f, allow_pickle=True).astype(float)
    elif DATASET_NAME == 'DSADS':
        with open(S_feats, 'rb') as f:
            all_source_bags = np.load(f).astype(float)
        with open(T_feats, 'rb') as f:
            all_target_bags = np.load(f).astype(float)

    if DATASET_NAME == 'OPPT':
        with open(S_labels, 'rb') as f:
            all_source_labels = (np.load(f) - 1).astype(int)
        with open(T_labels, 'rb') as f:
            all_target_labels = (np.load(f) - 1).astype(int)
    elif DATASET_NAME == 'DSADS' or DATASET_NAME == 'PAMAP2':
        with open(S_labels, 'rb') as f:
            all_source_labels = np.load(f).astype(int)
        with open(T_labels, 'rb') as f:
            all_target_labels = np.load(f).astype(int)

    results = {}
    times = {}
    file_pre = 'TROT_Ideal_' + DATASET_NAME + '_' + S_feats.split('_')[2] + "_" + T_feats.split('_')[2]
    printFile = open(file_pre + '_results.txt', 'w')

    T_vali_feats, T_vali_labels, T_test_feats, T_test_labels = split_vali_test_with_temporal_order(all_target_bags,
                                                                                                   all_target_labels)
    with open(file_pre + '_T_vali_feats.pkl', 'wb') as f:
        pickle.dump(T_vali_feats, f)
    with open(file_pre + '_T_vali_labels.pkl', 'wb') as f:
        pickle.dump(T_vali_labels, f)
    with open(file_pre + '_T_test_feats.pkl', 'wb') as f:
        pickle.dump(T_test_feats, f)
    with open(file_pre + '_T_test_labels.pkl', 'wb') as f:
        pickle.dump(T_test_labels, f)

    # validation step
    best_accuracy = 0
    best_params = {}
    hyperparas_list = get_param_optimal()

    for param_index, a_combination_of_params in enumerate(hyperparas_list):
        startTime = time.time()
        print(str(startTime))
        acc, cm, classify_results = TROT(DATASET_NAME, S_feats.split("_")[2], T_feats.split("_")[2], "validation",
                   n_state=a_combination_of_params["n_state"], reg_e=a_combination_of_params["reg_e"],
                   reg_cl=a_combination_of_params["reg_cl"], reg_ta=a_combination_of_params["reg_ta"],
                   target_bags=T_vali_feats, target_labels=T_vali_labels)

        # print results
        print('validation set_' + DATASET_NAME + '_param_' + str(param_index) + "_" + S_feats.split('_')[2] + "_" +
              T_feats.split('_')[2] + ":" + str(acc), file=printFile)
        print('validation set_' + DATASET_NAME + '_param_' + str(param_index) + "_" + S_feats.split('_')[2] + "_" +
              T_feats.split('_')[2] + ":" + str(classify_results), file=printFile)
        print('validation set_' + DATASET_NAME + '_param_' + str(param_index) + "_" + S_feats.split('_')[2] + "_" +
              T_feats.split('_')[2] + ":" + str(cm), file=printFile)
        for i in a_combination_of_params:
            print(i, a_combination_of_params[i], file=printFile)
        print('#####################################', file=printFile)
        # ---------------------------------------------------------------------------------------------

        if acc > best_accuracy:
            best_accuracy = acc
            best_params = a_combination_of_params

        dict_index = DATASET_NAME + '_param_' + str(param_index) + "_" + S_feats.split('_')[2] + "_" + \
                     T_feats.split('_')[2]
        times[dict_index] = time.time() - startTime
        results[dict_index] = acc

    # print results
    print('Best params result validation set_' + DATASET_NAME + "_" + S_feats.split('_')[
        2] + "_" + T_feats.split('_')[2] + ":" + str(best_accuracy), file=printFile)
    for i in best_params:
        print(i, best_params[i], file=printFile)
    print('#####################################', file=printFile)
    # -------------------------------------------------------------------------------------------------

    # test step
    acc, cm, classify_results = TROT(DATASET_NAME, S_feats.split("_")[2], T_feats.split("_")[2], "test", n_state=best_params["n_state"],
               reg_e=best_params["reg_e"],
               reg_cl=best_params["reg_cl"], reg_ta=best_params["reg_ta"],
               target_bags=T_test_feats, target_labels=T_test_labels)

    # print results
    print('testing set_' + DATASET_NAME + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[2] + ":" + str(acc),
          file=printFile)
    print('testing set_' + DATASET_NAME + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[2] + ":" + str(classify_results),
          file=printFile)
    print('testing set_' + DATASET_NAME + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[2] + ":" + str(cm),
          file=printFile)
    print('---------------------------------------------------------------------------------------', file=printFile)
    # -------------------------------------------------------------------------------------------------

    printFile.close()

    with open(file_pre + '_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(file_pre + '_times.pkl', 'wb') as f:
        pickle.dump(times, f)
