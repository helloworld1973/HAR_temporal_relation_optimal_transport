from hmmlearn.hmm import GMMHMM, GaussianHMM
import time
import numpy as np
import pickle


# from pomegranate import *

def HMM_with_specified_I_T_Matrix(dataset_name, bags, user, n_state, Cov_Type, activities_required):
    Start = time.time()
    Iter = 1000

    # Train_Data = Data[0:Train_num, :]
    # Test_Data = Data[int(N * Train_ratio):N, :]

    Model = GaussianHMM(n_components=n_state, params='mc', init_params='mc', tol=pow(10, -5),
                        n_iter=Iter, covariance_type=Cov_Type)

    transmat = np.zeros((n_state, n_state))
    for i in range(n_state - 1):
        transmat[i][i + 1] = 1.0
        # transmat[i][i] = 0.7
    transmat[n_state - 1][0] = 1.0
    # transmat[n_state - 1][n_state - 1] = 0.7
    Model.transmat_ = transmat

    startprob = np.zeros(n_state)
    startprob[0] = 1.0
    Model.startprob_ = startprob

    Model.fit(bags)
    # BIC AIC not work, because of the big value of HMM log_likelihood, make the monotical increasing function
    # log_likelihood = Model.score(Train_Data)
    # num_params = 2 * (n_state * Feat) + n_state * (n_state - 1) + (n_state - 1)
    # AIC = -2 * Model.score(Train_Data) + 2 * num_params
    # BIC = -2 * Model.score(Train_Data) + num_params * np.log(Train_Data.shape[0])
    # print('AIC:' + str(AIC) + '__BIC:' + str(BIC))
    END = time.time()
    #print('Total Time Takes in seconds', END - Start)
    likely_state_sequence = Model.predict(bags)
    posterior_probability = Model.predict_proba(bags)
    with open(dataset_name + '_GaussianHMM_' + str(n_state) + '_' + activities_required + '_' + user + '_' + Cov_Type + ".pkl",
              "wb") as file:
        pickle.dump(Model, file)
    # with open('GaussianHMM_'+ str(Max_state) + '_' + Cov_Type + ".pkl", "rb") as file: Model = pickle.load(file)
    return Model


def HMM_with_no_restricts(bags, user, n_state, Cov_Type, activities_required):
    Start = time.time()
    Iter = 1000

    Model = GaussianHMM(n_components=n_state, params='mct', init_params='mct', tol=pow(10, -5),
                        n_iter=Iter, covariance_type=Cov_Type)
    startprob = np.zeros(n_state)
    startprob[0] = 1.0
    Model.startprob_ = startprob

    Model.fit(bags)
    END = time.time()
    print('Total Time Takes in seconds', END - Start)

    #with open('GaussianHMM_' + str(n_state) + '_' + activities_required + '_' + user + '_' + Cov_Type + ".pkl",
    #          "wb") as file:
    #    pickle.dump(Model, file)

    return Model
