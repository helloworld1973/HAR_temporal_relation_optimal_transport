import numpy as np
import time
from gmr import GMM
from scipy import linalg
from sklearn import svm
from sklearn.mixture import GaussianMixture
import pickle
import SOT.SOT as sot
import ot
from hmmlearn.hmm import GMMHMM, GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.model_selection import train_test_split
from tools.feature_comparation import compare_two_users_features_a_class

# '''
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# features read files
DATASET_NAME = 'PAMAP2'
# DATASET_NAME = 'DSADS'
# DATASET_NAME = 'OPPT'

source_user = '1'  # 5
target_user = '6'
Num_Seconds = 0.3
Window_Overlap_Rate = 0.5
activities_required = 'all'  # ['lying', 'walking', 'ascending_stairs', 'descending_stairs']

with open(DATASET_NAME + '_' + activities_required + '_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
    source_bags = np.load(f, allow_pickle=True)
with open(DATASET_NAME + '_' + activities_required + '_' + str(source_user) + '_Y_labels.npy', 'rb') as f:
    source_labels = np.load(f)
with open(DATASET_NAME + '_' + activities_required + '_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
    target_bags = np.load(f, allow_pickle=True)
with open(DATASET_NAME + '_' + activities_required + '_' + str(target_user) + '_Y_labels.npy', 'rb') as f:
    target_labels = np.load(f)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#'''
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# SOT
Sx, Sy, Tx, Ty = sot.load_from_file(source_bags, source_labels, target_bags, target_labels)
tsot = sot.SOT('ACT', 'C:\\Users\\xye685.UOA\\PycharmProjects\\Milestone1\\SOT\\', 50, 3.5, 1, 3)
spath = 'C:\\Users\\xye685.UOA\\PycharmProjects\\Milestone1\\SOT\\data\\test_MDA_JCPOT_ACT_diag_SG.json'
tpath = 'C:\\Users\\xye685.UOA\\PycharmProjects\\Milestone1\\SOT\\data\\test_MDA_JCPOT_ACT_19_diag_TG.json'
tmodelpath = 'C:\\Users\\xye685.UOA\\PycharmProjects\\Milestone1\\SOT\\model\\test_MDA_JCPOT_ACT_19_diag_H'
pred, acc = tsot.fit_predict(Sx, Sy, Tx, Ty, spath, 'D', tpath, tmodelpath, 'H')
print(acc)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#'''

# '''
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# HMM
n_state = 4
Feat = source_bags.shape[1]
Cov_Type = 'diag'


def HMM(bags, user, Train_ratio, n_state, Feat, Cov_Type, activities_required):
    Start = time.time()
    Iter = 1000
    N = len(bags)
    Data = bags
    Train_Data = Data[0:int(N * Train_ratio), :]
    Test_Data = Data[int(N * Train_ratio):N, :]

    Model = GaussianHMM(n_components=n_state, params='mc', init_params='mc', tol=pow(10, -5),
                        n_iter=Iter, covariance_type=Cov_Type)

    transmat = np.zeros((n_state, n_state))
    for i in range(n_state - 1):
        transmat[i][i + 1] = 1.0
        # transmat[i][i] = 0.2
    transmat[n_state - 1][0] = 1.0
    # transmat[n_state - 1][n_state - 1] = 0.2
    Model.transmat_ = transmat

    startprob = np.zeros(n_state)
    startprob[0] = 1.0
    Model.startprob_ = startprob

    Model.fit(Train_Data)
    # BIC AIC not work, because of the big value of HMM log_likelihood, make the monotical increasing function
    # log_likelihood = Model.score(Train_Data)
    # num_params = 2 * (n_state * Feat) + n_state * (n_state - 1) + (n_state - 1)
    # AIC = -2 * Model.score(Train_Data) + 2 * num_params
    # BIC = -2 * Model.score(Train_Data) + num_params * np.log(Train_Data.shape[0])
    # print('AIC:' + str(AIC) + '__BIC:' + str(BIC))
    END = time.time()
    print('Total Time Takes in seconds', END - Start)
    # likely_state_sequence = Model.predict(Test_Data)
    # posterior_probability = Model.predict_proba(Test_Data)
    with open('GaussianHMM_' + str(n_state) + '_' + activities_required + '_' + user + '_' + Cov_Type + ".pkl",
              "wb") as file:
        pickle.dump(Model, file)
    # with open('GaussianHMM_'+ str(Max_state) + '_' + Cov_Type + ".pkl", "rb") as file: Model = pickle.load(file)
    print()
    return Model


source_moddel = HMM(bags=source_bags, user='source', Train_ratio=1.0, n_state=n_state, Feat=Feat, Cov_Type=Cov_Type, activities_required=activities_required)
target_moddel = HMM(bags=target_bags, user='target', Train_ratio=1.0, n_state=n_state, Feat=Feat, Cov_Type=Cov_Type, activities_required=activities_required)
print()
s_transmat = source_moddel.transmat_
t_transmat = target_moddel.transmat_

s_means = source_moddel.means_  # .reshape(Max_state, Feat)
t_means = target_moddel.means_  # .reshape(Max_state, Feat)

s_covars = source_moddel.covars_  # .reshape(Max_state, Feat)
t_covars = target_moddel.covars_  # .reshape(Max_state, Feat)

# C_means = ot.dist(s_means, t_means)
# C_covars = ot.dist(s_covars, t_covars)
print()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# GaussianMixture
gmm = GMM(n_components=n_state, priors=np.repeat(1 / n_state, n_state), means=s_means, covariances=s_covars)
a = gmm.to_responsibilities(source_bags)
b = gmm.to_probability_density(source_bags)
# mvns = gmm.extract_mvn(component_idx=1)
# Save object gmm to file 'file'
pickle.dump(gmm, open("gmm_" + n_state, "wb"))
# Load object from file 'file'
gmm = pickle.load(open("gmm_" + n_state, "rb"))
print()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# '''

'''

with open(DATASET_NAME + '_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(Window_Overlap_Rate) + '_X_raw.npy', 'rb') as f:
    source_raws = np.load(f, allow_pickle=True)
with open(DATASET_NAME + '_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(Window_Overlap_Rate) + '_X_raw.npy', 'rb') as f:
    target_raws = np.load(f, allow_pickle=True)
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

'''
#source_samples = []
#target_samples = []
for i in range(11):
    s_index_list = np.where(source_labels == i)
    s_samples_in_a_class = source_bags[s_index_list]

    t_index_list = np.where(target_labels == i)
    t_samples_in_a_class = target_bags[t_index_list]

    matrix = compare_two_users_features_a_class(s_data_bags=s_samples_in_a_class, t_data_bags=t_samples_in_a_class,data_name=str(i))

    #source_temporal_dimension = [[i / s_samples_in_a_class.shape[0]] for i in range(s_samples_in_a_class.shape[0])]
    #s_samples_in_a_class = np.append(s_samples_in_a_class, source_temporal_dimension, axis=1)

    #target_temporal_dimension = [[i / t_samples_in_a_class.shape[0]] for i in range(t_samples_in_a_class.shape[0])]
    #t_samples_in_a_class = np.append(t_samples_in_a_class, target_temporal_dimension, axis=1)

    # matrix = compare_two_users_features_a_class(s_data_bags=s_samples_in_a_class, t_data_bags=t_samples_in_a_class,data_name=str(i))

    #source_samples.extend(s_samples_in_a_class)
    #target_samples.extend(t_samples_in_a_class)
    #print()

    # Eu_distance = lambda x, y: np.sqrt(np.sum(np.square(x - y)))
    # d, cost_matrix, acc_cost_matrix, path = dtw(t_samples_in_a_class,s_samples_in_a_class, dist=Eu_distance)
    # import matplotlib.pyplot as plt
    # plt.imshow(cost_matrix.T, cmap='gray', interpolation='nearest')
    
    # plt.plot(path[0], path[1], 'w')
    
    # plt.show()
    # plt.close()
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

'''
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# regression feature selection
# load and summarize the dataset
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


matrix = compare_two_users_features_a_class(s_data_bags=source_bags, t_data_bags=target_bags, data_name='all')

# add temporal dimension
# source_temporal_dimension = [[i / source_bags.shape[0]] for i in range(source_bags.shape[0])]
# source_bags = np.append(source_bags, source_temporal_dimension, axis=1)

# target_temporal_dimension = [[i / target_bags.shape[0]] for i in range(target_bags.shape[0])]
# target_bags = np.append(target_bags, target_temporal_dimension, axis=1)

#matrix = compare_two_users_features_a_class(s_data_bags=np.array(source_samples), t_data_bags=np.array(target_samples), data_name='all')
print()
'''

'''
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# classification
target_X_train_input, target_X_test_input, target_Y_train_input, target_Y_test_input = train_test_split(target_bags,
                                                                                                        target_labels,
                                                                                                        test_size=0.3,
                                                                                                        stratify=target_labels)

clf = svm.LinearSVC(dual=False, multi_class='ovr')
# clf = RandomForestClassifier(n_estimators=100)
clf.fit(target_X_train_input, target_Y_train_input)
score = clf.score(target_X_test_input, target_Y_test_input)
print(str(score))

# use 0.7 userA train, test on 0.3 userA = 0.98
# use 0.7 userB train, test on 0.3 userB = 0.98
# use 0.7 userA train, test on 0.3 userB = 0.82
# use 0.1 userB train, test on 0.9 userB = 0.928
# use 0.05 userB train, test on 0.95 userB = 0.8916
# use 0.02 userB train, test on 0.98 userB = 0.818
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''
