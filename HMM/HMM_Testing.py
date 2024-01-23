# I will first implement this in a form of Jupyter Notebook.
from HMM.Hidden_Markov_Model import *
from Hidden_Markov_Model.auto_HMM import *

import time







Start = time.time()
Train_ratio = 0.1
Cov_Type = 'diag'
Max_state = 3
Max_mixture = 4
Iter = 1000
Feat = 2
N = 2000
T = 50
flag = 1
Path = 'C:\\Users\\xye685.UOA\\PycharmProjects\\Auto_HMM-main\\Exam_4_25_2020.csv'
Data = pd.read_csv(Path)


Exam_HMM = Supervised_HMM(Train_ratio, Cov_Type, Max_state, Max_mixture, Iter, Feat, N, T, Data, flag)
Exam_HMM.Best_States()
END = time.time()
print('Total Time Takes in seconds', END - Start)
