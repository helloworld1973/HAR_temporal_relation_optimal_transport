import numpy as np

from ARMA.ARIMA import get_ARIMA_coefficient
from read_dataset import read_PAMAP2_dataset


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# get features read files
DATASET_NAME = 'PAMAP2'
# DATASET_NAME = 'DSADS'
# DATASET_NAME = 'OPPT'
user = 'source'  # 'target'
n_state = 4
n_activities = 4
n_feature = 114
Cov_Type = 'diag'
# ['lying', 'walking', 'ascending_stairs', 'descending_stairs']
Num_Seconds = 5
Window_Overlap_Rate = 0.5
source_user = '1'  # 5
target_user = '6'

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

# /////////////////
with open('E:\\Python Projects\\Milestone1_new\\' + DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_raw.npy', 'rb') as f:
    all_source_bags = np.load(f, allow_pickle=True)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

p_q_lag = 5

big_list = []
for channel_index in range(0, all_source_bags.shape[2]):
    source_ARIMA_coefficient_list = get_ARIMA_coefficient(all_source_bags[:, :, channel_index], p_q_lag)
    big_list.append(source_ARIMA_coefficient_list)
    print()

print()