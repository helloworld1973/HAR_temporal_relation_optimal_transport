from Hidden_Markov_Model.hmm import GaussianHMM
from Hidden_Markov_Model.GaussianHMM_test import ContrastHMM

n_state =4
n_feature = 2  # 表示观测值的维度
X_length = 1000

test_hmm = GaussianHMM(n_state, n_feature)
standard_hmm = ContrastHMM(n_state, n_feature)
X, Z = standard_hmm.module.sample(X_length)
test_hmm.train(X)
print()