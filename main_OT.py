import copy
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from experiments_comparation.pyotda import ot
import time
from sklearn.decomposition import PCA
import SOT.SOT as sot
import random
import pickle

algo = 'OT'

adaptationAlgoUsed = [algo]  # "JDOT", "TCA" error

n_feature = 38
Cov_Type = 'diag'
Num_Seconds = 0.3
Window_Overlap_Rate = 0.5
S_T_file_path_pairs = []

'''
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
'''

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

'''
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
'''

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


def getLabel(trainData, trainLabels, testData, type_classifier="1NN"):
    """
    :param trainData:
    :param trainLabels:
    :param testData:
    :param type_classifier: Only nNN and SVM_x implemented. With x a float and n an integer.
    :return: The prediction of the label of testData using the train data to learn a classifier
    """
    if "NN" in type_classifier:
        clf = sklearn.neighbors.KNeighborsClassifier(int(type_classifier[0:-2]))
        clf.fit(trainData, trainLabels)
        prediction = clf.predict(testData)
    elif "SVM" in type_classifier:
        C = float(type_classifier.split("_")[1])
        trainData, trainLabels = sklearn.utils.shuffle(trainData, trainLabels)
        clf = sklearn.linear_model.SGDClassifier(max_iter=2000, tol=10 ** (-4), alpha=C)
        clf.fit(trainData, trainLabels)
        prediction = clf.predict(testData)
    return prediction


def getAccuracy(trainData, trainLabels, testData, testLabels, type_classifier="1NN"):
    """
    :param trainData:
    :param trainLabels:
    :param testData:
    :param testLabels:
    :param type_classifier:
    :return: The accuracy of the test data train with the train data. Only NN and SVM are implemented
    """
    prediction = getLabel(trainData, trainLabels, testData, type_classifier)
    cm = confusion_matrix(testLabels, prediction)
    classify_results = classification_report(testLabels, prediction)
    return 100 * float(sum(prediction == testLabels)) / len(testData), cm, classify_results


def adaptData(algo, Sx, Sy, Tx, Ty, param=None):
    """
    Main function of the code that launch a method.
    :param algo: Name of the method to use.
    :param Sx: Source features.
    :param Sy: Source labels.
    :param Tx: Target features.
    :param Ty: Target labels.
    :param param: List of parameters needed for each method.
    :return: The adapted data source and target. It also return the labels unchanged.
    """
    if algo == "Tused":
        # Cheating method that use the target dataset to learn the classifier.
        # This can be usefull for a baseline that we probably can't beat in domain adaptation.
        Sy = Ty
        sourceAdapted = Tx
        targetAdapted = Tx

    if algo == "NA":
        # No Adaptation
        sourceAdapted = Sx
        targetAdapted = Tx

    elif algo == "SA":
        # Subspace Alignment, described in:
        # Unsupervised Visual Domain Adaptation Using Subspace Alignment, 2013,
        # Fernando et al.

        pcaS = sklearn.decomposition.PCA(n_components=param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(n_components=param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        Xa = XS.dot(np.transpose(XS)).dot(XT)

        sourceAdapted = Sx.dot(Xa)
        targetAdapted = Tx.dot(XT)

    elif algo == "TCA":
        # Domain adaptation via transfer component analysis. IEEE TNN 2011
        d = param["d"]  # subspace dimension
        Ns = Sx.shape[0]
        Nt = Tx.shape[0]
        L_ss = (1. / (Ns * Ns)) * np.full((Ns, Ns), 1)
        L_st = (-1. / (Ns * Nt)) * np.full((Ns, Nt), 1)
        L_ts = (-1. / (Nt * Ns)) * np.full((Nt, Ns), 1)
        L_tt = (1. / (Nt * Nt)) * np.full((Nt, Nt), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        X = np.vstack((Sx, Tx))
        K = np.dot(X, X.T)  # linear kernel
        H = (np.identity(Ns + Nt) - 1. / (Ns + Nt) * np.ones((Ns + Nt, 1)) *
             np.ones((Ns + Nt, 1)).T)
        inv = np.linalg.pinv(np.identity(Ns + Nt) + K.dot(L).dot(K))
        D, W = np.linalg.eigh(inv.dot(K).dot(H).dot(K))
        W = W[:, np.argsort(-D)[:d]]  # eigenvectors of d highest eigenvalues
        sourceAdapted = np.dot(K[:Ns, :], W)  # project source
        targetAdapted = np.dot(K[Ns:, :], W)  # project target

    elif algo == "CORAL":
        # Return of Frustratingly Easy Domain Adaptation. AAAI 2016
        from scipy.linalg import sqrtm
        Cs = np.cov(Sx, rowvar=False) + np.eye(Sx.shape[1])
        Ct = np.cov(Tx, rowvar=False) + np.eye(Tx.shape[1])
        Ds = Sx.dot(np.linalg.inv(np.real(sqrtm(Cs))))  # whitening source
        Ds = Ds.dot(np.real(sqrtm(Ct)))  # re-coloring with target covariance
        sourceAdapted = Ds
        targetAdapted = Tx

    elif algo == "OT":
        # Optimal Transport with class regularization described in:
        # Domain adaptation with regularized optimal transport, 2014.
        # Courty et al.
        transp3 = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=0, norm="median",
                                              max_iter=1, max_inner_iter=100, log=False,
                                              tol=10 ** -7)
        transp3.fit(Xs=Sx, ys=Sy, Xt=Tx)

        sourceAdapted = transp3.transform(Xs=Sx)
        targetAdapted = Tx

    elif algo == "OTDA":
        # Optimal Transport with class regularization described in:
        # Domain adaptation with regularized optimal transport, 2014.
        # Courty et al.
        transp3 = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp3.fit(Xs=Sx, ys=Sy, Xt=Tx)

        sourceAdapted = transp3.transform(Xs=Sx)
        targetAdapted = Tx

    elif algo == "MLOT":
        ML_init_temps = param["ML_init"]
        # pcaS = sklearn.decomposition.PCA(min(param["d"], Sx.shape[0], Sx.shape[1]),
        # svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(min(param["d"], Tx.shape[0], Tx.shape[1]), svd_solver=param["svd_solver"]).fit(
            Tx)
        # XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)

        if algo == "MLOT_id":
            # The pca is not applied
            source_in_target_subspace = Sx
            target_in_target_subspace = Tx
        elif algo == "MLOT":
            # The pca is apply only on the target dataset at this point, this can be seen as a preprocess.
            # The source PCA is apply during the SinkhornMLTranport fit.
            source_in_target_subspace = Sx
            target_in_target_subspace = Tx.dot(XT.dot(np.transpose(XT)))
            param["ML_init"] = "SS"
        transp3 = ot.da.SinkhornMLTransport(reg_e=param["reg_e"],
                                            reg_cl=param["reg_cl"],
                                            reg_l=param["reg_l"],
                                            norm="median",
                                            max_iter=param["max_iter"],
                                            max_inner_iter_grad=param["max_inner_iter_grad"],
                                            max_inner_iter_sink=param["max_inner_iter_sink"],
                                            svd_solver=param["svd_solver"],
                                            dimension=param["d"],
                                            ML_init=param["ML_init"],
                                            margin=param["margin"],
                                            mini_batch_size=5000)
        param["ML_init"] = ML_init_temps
        transp3.fit(Xs=source_in_target_subspace,
                    ys=Sy,
                    Xt=target_in_target_subspace,
                    yt=Ty)

        transp3.xt_ = Tx
        sourceAdapted = transp3.transform(Xs=source_in_target_subspace)
        targetAdapted = Tx

    elif algo == "LMNN":
        # Large Margin Nearest Neighbor
        from experiments_comparation.pyotda.ot import lmnn_original
        LMNN = lmnn_original.LargeMarginNearestNeighbor(k=3, mu=0.5,
                                                        margin=param["margin"],
                                                        nFtsOut=param["d"],
                                                        maxCst=int(1e7),
                                                        randomState=None,
                                                        maxiter=param["max_iter"])
        LMNN.fit(X=Sx, y=Sy)

        sourceAdapted = Sx @ (LMNN.L_).T @ (LMNN.L_)
        targetAdapted = Tx

    elif algo == "JDOT":
        # WARNING : here we use the transport plan learned to adapt the source
        # and we do not use the prediction.
        from JDOT import jdot
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        lb.fit(Sy)
        Sy_01 = lb.transform(Sy)
        # WARNING : we use SVM method as NN method is not immediately implemented
        clf_jdot, dic = jdot.jdot_svm(X=Sx, y=Sy_01, Xtest=Tx, ytest=[],
                                      gamma_g=1,
                                      numIterBCD=param["max_iter"],  # To stay fair, this will also be cross validate
                                      alpha=param["reg_l"],  # from 10-5 to 1.
                                      lambd=1e1,  # Used for the classifier
                                      method='emd',
                                      reg_sink=1,
                                      ktype='linear')
        transp = dic["G"] / np.sum(dic["G"], 1)[:, None]  # Barycentric mapping
        sourceAdapted = transp @ Tx  # WARNING : here we use the transport plan learned to adapt the source
        # and we do not use the prediction.
        targetAdapted = Tx

    elif algo == "SOT":
        tsot = sot.SOT('ACT', d=param["clusters"], reg_e=param["reg_e"], reg_cl=param["reg_cl"], reg_ce=param["reg_ce"])
        spath = './test_MDA_JCPOT_ACT_diag_SG.json'
        tpath = './test_MDA_JCPOT_ACT_19_diag_TG.json'
        tmodelpath = './test_MDA_JCPOT_ACT_19_diag_H'
        pred, acc = tsot.fit_predict(Sx=all_source_bags, Sy=all_source_labels, Tx=all_target_bags,
                                     Ty=all_target_labels,
                                     sfilepath=spath, sourcename='D', tfilepath=tpath, tmodelpath=tmodelpath,
                                     targetname='H')

        cm = confusion_matrix(all_target_labels, pred)
        classify_results = classification_report(all_target_labels, pred)

        sourceAdapted = acc, cm, classify_results
        targetAdapted = acc



    elif algo == "GFK":
        gfk = GFK(dim=param["d"])
        _, _, _, Xs_new, Sy_new, Xt_new = gfk.fit_predict(Xs=Sx, Ys=Sy, Xt=Tx, Yt=Ty)
        sourceAdapted = Xs_new
        targetAdapted = Xt_new
        Sy = Sy_new

    return sourceAdapted, targetAdapted, Sy, Ty


def get_param_optimal(algo):
    d_list = [int(n_feature / 10 * (i + 1)) for i in range(0, 9)]
    svd_solver_list = ["full", "randomized", "arpack"]
    reg_e_list = [0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 3]
    reg_cl_list = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 3]
    reg_l_list = [0.001, 0.01, 0.1, 1, 10, 100]
    reg_ce_list = [0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 3]
    max_iter_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50]
    margin_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
    cluster_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 100, 150, 200]

    if algo == "SA":
        number_iteration_list = [1, 10]
        params_list = []
        for a_d in d_list:
            for a_svd in svd_solver_list:
                if a_svd == "full":
                    param = {"d": a_d, "svd_solver": a_svd, "numberIteration": 1}
                    params_list.append(param)
                else:
                    for a_iter in number_iteration_list:
                        param = {"d": a_d, "svd_solver": a_svd, "numberIteration": a_iter}
                        params_list.append(param)
        return params_list

    elif algo == "TCA" or algo == "GFK":
        params_list = []
        for a_d in d_list:
            param = {"d": a_d}
            params_list.append(param)
        return params_list

    elif algo == "CORAL" or algo == "NA" or algo == "Tused":
        params_list = []
        param = {"d": 1}
        params_list.append(param)
        return params_list

    elif algo == "OT":
        params_list = []
        for a_reg_e in reg_e_list:
            param = {"reg_e": a_reg_e, "reg_cl": 0, "max_iter": 1, "max_inner_iter": 100, "tol": 10 ** -7,
                     "norm": "median"}
            params_list.append(param)
        return params_list

    elif algo == "OTDA":
        params_list = []
        for a_reg_e in reg_e_list:
            for a_reg_cl in reg_cl_list:
                param = {"reg_e": a_reg_e, "reg_cl": a_reg_cl, "norm": "median"}
                params_list.append(param)
        return params_list

    elif algo == "MLOT":
        params_list = []
        for a_reg_e in reg_e_list:
            for a_reg_cl in reg_cl_list:
                for a_reg_l in reg_l_list:
                    for a_max_iter in max_iter_list:
                        for a_margin in margin_list:
                            for a_d in d_list:
                                param = {"reg_e": a_reg_e, "reg_cl": a_reg_cl, "reg_l": a_reg_l, "norm": "median",
                                         "max_iter": a_max_iter, "max_inner_iter_grad": 1, "max_inner_iter_sink": 10,
                                         "margin": a_margin, "d": a_d, "numberIteration": 1, "ML_init": "SS",
                                         "svd_solver": "full"}
                                params_list.append(param)
        return params_list

    elif algo == "LMNN":
        params_list = []
        for a_margin in margin_list:
            for a_d in d_list:
                for a_max_iter in max_iter_list:
                    param = {"margin": a_margin, "d": a_d, "max_iter": a_max_iter, "k": 3, "mu": 0.5,
                             "maxCst": int(1e7),
                             "randomState": None}
                    params_list.append(param)
        return params_list

    elif algo == "JDOT":
        params_list = []
        reg_l_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        for a_reg_l in reg_l_list:
            for a_max_iter in max_iter_list:
                param = {"reg_l": a_reg_l, "max_iter": a_max_iter, "gamma_g": 1, "lambd": 1e1, "method": 'emd',
                         "reg_sink": 1, "ktype": 'linear'}
                params_list.append(param)
        return params_list

    elif algo == "SOT":
        params_list = []
        for a_cluster in cluster_list:
            for a_reg_e in reg_e_list:
                for a_reg_cl in reg_cl_list:
                    for a_reg_ce in reg_ce_list:
                        param = {"clusters": a_cluster, "reg_e": a_reg_e, "reg_cl": a_reg_cl, "reg_ce": a_reg_ce}
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

    # --------------------------------------------------------------------------------------------------------------------
    best_hyper_params = {}
    best_hyper_params_accuracy = {}
    test_set_final_accuracy = {}
    results = {}
    times = {}
    file_pre = algo + '_' + DATASET_NAME + '_' + S_feats.split('_')[2] + "_" + T_feats.split('_')[2]
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

    # hyper-parameters turning
    for a_algo in adaptationAlgoUsed:
        startTime = time.time()
        best_accuracy = 0
        best_params = {}

        # select a a_combination_of_params
        time_before_loop = time.time()
        hyperparas_list = get_param_optimal(a_algo)
        number_iteration_cross_val = 0
        run_hours = 6  # 42 cross user, 10.5 days
        while time.time() - time_before_loop < 3600 * run_hours and number_iteration_cross_val < 80 and len(
                hyperparas_list) > 0:
            a_combination_of_params = random.choice(hyperparas_list)
            hyperparas_list.remove(a_combination_of_params)

            try:
                # select a combination of hyper-paras
                if a_algo in ["OT", "OTDA", "NA", "Tused", "TCA", "CORAL", "LMNN", "JDOT", "GFK", "SOT", "MLOT"]:
                    a_combination_of_params["numberIteration"] = 1

                for iteration in range(a_combination_of_params["numberIteration"]):
                    from datetime import datetime

                    print("Start Time =", datetime.now().strftime("%H:%M:%S"))

                    np.random.seed(iteration * 45 + 4988612)
                    random.seed(iteration * 65 + 8965321)

                    accuracy_value = -1
                    cm = -1
                    classify_results = -1
                    dict_index = ''
                    if a_algo == "SOT":
                        acc_cm_classify_results, _, _, _ = adaptData(a_algo, all_source_bags, all_source_labels,
                                                                     T_vali_feats,
                                                                     T_vali_labels,
                                                                     a_combination_of_params)
                        accuracy_value, cm, classify_results = acc_cm_classify_results
                        dict_index = DATASET_NAME + "_" + a_algo + "_" + str(iteration) + "_" + S_feats.split('_')[
                            2] + "_" + T_feats.split('_')[2]
                    else:
                        subSa, Ta, subSay, Tay = adaptData(a_algo, all_source_bags, all_source_labels, T_vali_feats,
                                                           T_vali_labels,
                                                           a_combination_of_params)

                        dict_index = DATASET_NAME + "_" + a_algo + "_" + str(iteration) + "_" + S_feats.split('_')[
                            2] + "_" + \
                                     T_feats.split('_')[2]

                        my_dict = {}
                        my_dict[dict_index + " subSa"] = subSa
                        my_dict[dict_index + " subSay"] = subSay
                        my_dict[dict_index + " Ta"] = Ta
                        my_dict[dict_index + " Tay"] = Tay

                        pickle_out = open(dict_index + ".pickle", "wb")
                        pickle.dump(my_dict, pickle_out)
                        pickle_out.close()
                        accuracy_value, cm, classify_results = getAccuracy(subSa, subSay, Ta, Tay,
                                                                           type_classifier='1NN')

                    results[dict_index] = accuracy_value, cm, classify_results
                    times[dict_index] = time.time() - startTime

                    # print results
                    print('validation set_' + DATASET_NAME + "_" + a_algo + "_iter_" + str(
                        iteration) + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[
                              2] + ":" + str(accuracy_value), file=printFile)
                    print('validation set_' + DATASET_NAME + "_" + a_algo + "_iter_" + str(
                        iteration) + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[
                              2] + ":" + str(classify_results), file=printFile)
                    print('validation set_' + DATASET_NAME + "_" + a_algo + "_iter_" + str(
                        iteration) + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[
                              2] + ":" + str(cm), file=printFile)
                    for i in a_combination_of_params:
                        print(i, a_combination_of_params[i], file=printFile)
                    print('####################################################', file=printFile)
                    # -------------------------------------------------------------------------------------------------

                    if accuracy_value > best_accuracy:
                        best_accuracy = copy.deepcopy(accuracy_value)
                        best_params = copy.deepcopy(a_combination_of_params)

            except:
                print("Error with this setting :", file=printFile)
                for i in a_combination_of_params:
                    print(i, a_combination_of_params[i], file=printFile)

            print("End Time =", datetime.now().strftime("%H:%M:%S"))
            number_iteration_cross_val += 1

        try:
            # print results
            print('Best params result validation set_' + DATASET_NAME + "_" + a_algo + S_feats.split('_')[2] + "_" +
                  T_feats.split('_')[2] + ":" + str(best_accuracy), file=printFile)
            for i in best_params:
                print(i, best_params[i], file=printFile)
            # -------------------------------------------------------------------------------------------------

            best_hyper_params[
                DATASET_NAME + "_" + a_algo + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[2]] = best_params
            best_hyper_params_accuracy[
                DATASET_NAME + "_" + a_algo + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[2]] = best_accuracy

            # --------------------------------------------------------------------------------------------------------------------
            # test set accuracy
            accuracy_value = -1
            cm = -1
            classify_results = -1
            if a_algo == "SOT":
                acc_cm_classify_results, _, _, _ = adaptData(a_algo, all_source_bags, all_source_labels, T_test_feats,
                                                             T_test_labels,
                                                             best_params)
                accuracy_value, cm, classify_results = acc_cm_classify_results
            else:
                subSa, Ta, subSay, Tay = adaptData(a_algo, all_source_bags, all_source_labels, T_test_feats,
                                                   T_test_labels,
                                                   best_params)
                accuracy_value, cm, classify_results = getAccuracy(subSa, subSay, Ta, Tay, type_classifier='1NN')

            test_set_final_accuracy[
                DATASET_NAME + "_" + a_algo + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[
                    2]] = accuracy_value

            # print results
            print('testing set_' + DATASET_NAME + "_" + a_algo + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[
                2] + ":" + str(accuracy_value), file=printFile)
            print('testing set_' + DATASET_NAME + "_" + a_algo + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[
                2] + ":" + str(classify_results), file=printFile)
            print('testing set_' + DATASET_NAME + "_" + a_algo + "_" + S_feats.split('_')[2] + "_" + T_feats.split('_')[
                2] + ":" + str(cm), file=printFile)
            print('---------------------------------------------------------------------------------------',
                  file=printFile)
        except:
            print('best paras error!', file=printFile)

    printFile.close()
    with open(file_pre + '_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(file_pre + '_times.pkl', 'wb') as f:
        pickle.dump(times, f)

    with open(file_pre + '+best_hyper_params.pkl', 'wb') as f:
        pickle.dump(best_hyper_params, f)
    with open(file_pre + '_best_hyper_params_accuracy.pkl', 'wb') as f:
        pickle.dump(best_hyper_params_accuracy, f)
    with open(file_pre + '_test_set_final_accuracy.pkl', 'wb') as f:
        pickle.dump(test_set_final_accuracy, f)

# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
# -------------------------------------------------------------------------------------------------------------------