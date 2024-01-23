import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)


def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    return model_fit


def get_ARIMA_coefficient(samples_list, p_q_lag):
    coefficient_list = []
    for aSample in samples_list:
        model_fit = ARIMA_Model(aSample, (p_q_lag, 2, p_q_lag))
        a_model_coefficient = model_fit.params.tolist()
        coefficient_list.append(a_model_coefficient)
    return coefficient_list


if __name__ == '__main__':
    source_time_series_data = get_MMAct_Dataset(source_subject='subject1', activity_name='closing', which_axis=1)
    target_time_series_data = get_MMAct_Dataset(source_subject='subject3', activity_name='closing', which_axis=1)
    '''
    source_time_series_data, target_time_series_data = generate_raw_time_series_body_acc_x()
    source_list = source_time_series_data[500:800]
    target_list = target_time_series_data[500:800]
    '''
    p_q_lag = 5
    source_ARIMA_coefficient_list = get_ARIMA_coefficient(source_time_series_data, p_q_lag)
    target_ARIMA_coefficient_list = get_ARIMA_coefficient(target_time_series_data, p_q_lag)

    # vertical mean
    source_mean_ARIMA_coefficient_list = np.mean(source_ARIMA_coefficient_list, axis=0)
    target_mean_ARIMA_coefficient_list = np.mean(target_ARIMA_coefficient_list, axis=0)

    tsne = Show_tSNE(source_ARIMA_coefficient_list, target_ARIMA_coefficient_list)
    tsne.display_tSNE()

    # target_model_fit = ARIMA_Model(target_list, (p_q_lag, 0, p_q_lag))  # p d q
    # target_fit_seq = target_model_fit.fittedvalues

    # predict_seq = model_fit.plot_predict(dynamic=False)
    # plt.show()
    print()
