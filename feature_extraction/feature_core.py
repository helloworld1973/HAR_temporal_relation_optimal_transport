import numpy as np
from feature_extraction.feature_time import Feature_time
from feature_extraction.feature_fft import Feature_fft

# from feature_time import Feature_time
# from feature_fft import Feature_fft

def get_feature(arr):
    '''
    Get features of an array
    :param arr: input 1D array
    :return: feature list
    '''
    feature_list = list()
    # get time domain features
    feature_time = Feature_time(arr).time_all()
    feature_list.extend(feature_time)
    # get frequency domain features
    feature_fft = Feature_fft(arr).fft_all()
    feature_list.extend(feature_fft)
    return feature_list
