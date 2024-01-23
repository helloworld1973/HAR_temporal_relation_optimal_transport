import math
from math import floor, ceil, sqrt

import numpy as np
from numpy import pi, append, linspace, array, arange, sin, cos
from numpy.random import rand, seed
from scipy import fftpack, signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import argrelextrema
# from mpldatacursor import datacursor
from statsmodels.tsa.stattools import acf


class FA():

    def __init__(self, t, Signal, f_s):
        self.t = t
        self.signal = Signal
        self.N = len(t)
        self.f_s = f_s

    def Fourier_Transform(self):
        Freqs = np.fft.rfftfreq(self.N, 1.0 / self.f_s)
        fourier_coef = np.fft.rfft(self.signal)
        Ampts = (2 / self.N) * abs(fourier_coef)
        Phases = np.angle(fourier_coef)
        f, Pxx_den = signal.periodogram(self.signal, self.f_s)
        return Freqs, Ampts, Phases, f, Pxx_den, fourier_coef

    def Reverse_Fourier_Transform(self, fourier_coef):
        reconstruct_signal = np.fft.irfft(fourier_coef)
        return reconstruct_signal

    def PlotFFT(self):
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(80, 60))

        # signal plot
        axs[0].plot(self.t, self.signal)
        axs[0].grid()
        axs[0].set_xlabel('time (s)', fontsize=16)
        axs[0].set_ylabel('Voltage', fontsize=16)
        axs[0].set_title('Signal', fontweight='bold', fontsize=20)

        Freqs, Ampts, Phases, f, Pxx_den, fourier_coef = self.Fourier_Transform()

        axs[1].plot(Freqs, Ampts, 'r')
        # datacursor(h)
        axs[1].grid()
        axs[1].set_xlabel('frequency (Hz)', fontsize=16)
        axs[1].set_ylabel('Voltage', fontsize=16)
        axs[1].set_title('amplitude', fontweight='bold', fontsize=20)

        axs[2].plot(Freqs, Phases, 'r')
        # datacursor(h)
        axs[2].grid()
        axs[2].set_xlabel('frequency (Hz)', fontsize=16)
        axs[2].set_ylabel('degree', fontsize=16)
        axs[2].set_title('phase', fontweight='bold', fontsize=20)

        axs[3].plot(f, Pxx_den, 'r')
        # datacursor(h)
        axs[3].grid()
        axs[3].set_xlabel('frequency (Hz)', fontsize=16)
        axs[3].set_ylabel('PSD (V**2/Hz)', fontsize=16)
        axs[3].set_title('periodgram', fontweight='bold', fontsize=20)
        plt.show()

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        sm.graphics.tsa.plot_acf(x=self.signal, lags=int(len(self.signal) / 2) - 1, ax=ax1, color='blue')
        acf_list = acf(x=self.signal, nlags=int(len(self.signal) / 2) - 1)
        # for local maxima
        local_max_index = argrelextrema(acf_list, np.greater)
        period_values = [acf_list[i] for i in local_max_index]
        plt.show()

    def get_frequency_scaling_S2T(self, source_Pxx_den, source_freq, target_Pxx_den, target_freq):
        source_Pxx_den = source_Pxx_den.tolist()
        target_Pxx_den = target_Pxx_den.tolist()

        s_Pxx_den_index = sorted(range(len(source_Pxx_den)), key=lambda k: source_Pxx_den[k])
        s_dominant_freq = source_freq.tolist()[s_Pxx_den_index[-1]]
        s_second_dominant_freq = source_freq.tolist()[s_Pxx_den_index[-2]]

        t_Pxx_den_index = sorted(range(len(target_Pxx_den)), key=lambda k: target_Pxx_den[k])
        t_dominant_freq = target_freq.tolist()[t_Pxx_den_index[-1]]
        t_second_dominant_freq = target_freq.tolist()[t_Pxx_den_index[-2]]

        return s_Pxx_den_index[-1], s_dominant_freq, t_Pxx_den_index[
            -1], t_dominant_freq, s_dominant_freq / t_dominant_freq, \
               s_second_dominant_freq, t_second_dominant_freq, \
               s_Pxx_den_index[-2], t_Pxx_den_index[-2]

    def TL_get_dominant_detailed_frequency_and_mask_ACF(self, f_index, Pxx_den, freqs):
        dominant_detailed_frequency = -1
        if 1 < f_index < 22:  # corresponding frequency 4.4    f<4.373, then need autocorrelation, improve time domain resolution

            high_den = Pxx_den[f_index - 1]
            low_den = Pxx_den[f_index + 1]
            high_boundry_period_point = ceil(1. / freqs[f_index - 1] * self.f_s)
            low_boundry_period_point = floor(1. / freqs[f_index + 1] * self.f_s)
            range_weight = 1  # 可以拿公式推一下 看看能不能从 spectral leakage 推回raw original
            if low_den > high_den:
                high_boundry_period_point = ceil(1. / freqs[f_index] * self.f_s)
                range_weight = (1 - low_den / Pxx_den[f_index])
                low_boundry_period_point = floor(
                    1. / (freqs[f_index] + (freqs[f_index + 1] - freqs[f_index]) * 1.2 / 2) * self.f_s)

            else:
                low_boundry_period_point = floor(1. / freqs[f_index] * self.f_s)
                range_weight = high_den / Pxx_den[f_index]
                high_boundry_period_point = ceil(
                    1. / (freqs[f_index] - (freqs[f_index] - freqs[f_index - 1]) * 1.2 / 2) * self.f_s)

            if high_boundry_period_point >= 250:
                high_boundry_period_point = 249
            if low_boundry_period_point >= 250:
                low_boundry_period_point = 249

            search_range = [i for i in range(low_boundry_period_point, high_boundry_period_point + 1)]

            # autocorrelation
            acf_list = acf(x=self.signal, nlags=floor(len(self.signal) / 2) - 1)
            acf_search_list = acf_list[search_range]
            # for local maxima
            # local_max = argrelextrema(acf_search_list, np.greater)[0].tolist()
            acf_search_list_index = np.argmax(acf_search_list.tolist())

            local_max_index = search_range[acf_search_list_index]
            dominant_detailed_frequency = self.f_s / local_max_index
        else:
            dominant_detailed_frequency = freqs[f_index]

        return dominant_detailed_frequency

    def TL_get_dominant_detailed_frequency_and_mask(self):
        # periodgram method
        '''
        # Run permuted (randomized sampling) periodogram for noise level estimation of time-series and set 95%
        max_power = []
        for i in range(0, 1000):
            Qp = np.random.permutation(self.signal)
            ftmp, Ptmp = signal.periodogram(Qp, fs=self.f_s)
            max_power.append(np.percentile(Ptmp, 95))
        thresh = np.percentile(max_power, 99)
        '''
        # Compute actual periodogram from well-ordered time-series
        freqs, Pxx_den = signal.periodogram(self.signal, fs=self.f_s)

        # Mask powers above noise theshold
        Pxx_den[0] = 0.0001
        # Pmask = Pxx_den > thresh

        all_mask_list = [i for i in range(len(freqs))]

        specified_frequecy_list = np.asarray(freqs).tolist()
        '''
        mask_index_list = []
        pre_mask_flag = False
        for i, a_mask in enumerate(Pmask):
            if a_mask == True and pre_mask_flag == True:
                mask_index_list.append(i)
            elif a_mask == True and pre_mask_flag == False:
                mask_index_list.append(i)
                pre_mask_flag = True
            elif a_mask == False and pre_mask_flag == True:
                all_mask_list.append(mask_index_list)
                mask_index_list = []
                pre_mask_flag = False
        '''
        '''
        specified_frequecy_list = []
        for a_mask_list in all_mask_list:
            # local_max_index = a_mask_list[np.argmax(Pxx_den[a_mask_list])]
            local_max_dominant_detailed_frequency = self.TL_get_dominant_detailed_frequency_and_mask_ACF(
                a_mask_list, Pxx_den, freqs)
            specified_frequecy_list.append(local_max_dominant_detailed_frequency)
        '''
        max_index = np.argmax(Pxx_den)
        dominant_detailed_frequency = self.TL_get_dominant_detailed_frequency_and_mask_ACF(max_index, Pxx_den, freqs)

        return dominant_detailed_frequency, all_mask_list, specified_frequecy_list, freqs, Pxx_den

    def TL_get_dominant_detailed_frequency_s_amp(self, dominant_detailed_frequency):
        complex_num = np.sum(np.array(self.signal) * np.exp(
            -2j * np.pi * dominant_detailed_frequency * np.arange(0, self.N / self.f_s, 1 / self.f_s)))
        amp = (2 / 500) * abs(complex_num)
        return amp

    def TL_Fourier_Transform_Rebuild(self, specified_frequecy_list):
        coefficient_list = []
        for a_frequency in specified_frequecy_list:
            complex_num = np.sum(np.array(self.signal) * np.exp(
                -2j * np.pi * a_frequency * np.arange(0, self.N / self.f_s, 1 / self.f_s)))
            amp = (2 / 500) * abs(complex_num)
            phase = np.angle(complex_num)
            coefficient_list.append((a_frequency, amp, phase, complex_num))
        return coefficient_list

    def TL_Inverse_Fourier_Transform_Rebuild(self, coefficient_list, t):
        new_f = 0
        for i, coefficient in enumerate(coefficient_list):
            frequency = coefficient[0]
            amp = coefficient[1]
            phase = coefficient[2]
            new_f += amp * cos((2 * pi * frequency * t + phase))
        return new_f

    def TL_freq_phase_amp(self, s_dominant_detailed_frequency, t_dominant_detailed_frequency, s_dominant_amp,
                          t_dominant_amp, s_coefficient_list):
        scale_value_freq = s_dominant_detailed_frequency / t_dominant_detailed_frequency
        scale_value_amp = s_dominant_amp / t_dominant_amp
        new_s_coefficient_list = []
        for a_coefficient in s_coefficient_list:
            frequency = a_coefficient[0] / scale_value_freq
            amp = a_coefficient[1]  # / scale_value_amp
            phase = a_coefficient[2] / scale_value_freq
            new_s_coefficient_list.append((frequency, amp, phase))
        return new_s_coefficient_list, scale_value_freq


if __name__ == '__main__':
    N = 500
    fs = 100
    t = linspace(0, 5, N)
    f = 3 * sin(2 * pi * 1.26 * t) + 0.5 * sin(2 * pi * 10.2 * t)

    SignalFFT = FA(t, f, fs)
    SignalFFT.Fourier_Transform()
    dominant_detailed_frequency, Pmask, specified_frequecy_list = SignalFFT.TL_get_dominant_detailed_frequency_and_mask()
    coefficient_list = SignalFFT.TL_Fourier_Transform_Rebuild(specified_frequecy_list)
    recover_f = SignalFFT.TL_Inverse_Fourier_Transform_Rebuild(coefficient_list, t)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(80, 60))
    axs[0].plot(t, f)
    axs[0].grid()
    axs[0].set_xlabel('time (s)', fontsize=16)
    axs[0].set_ylabel('Voltage', fontsize=16)
    axs[0].set_title('Signal', fontweight='bold', fontsize=20)

    axs[1].plot(t, recover_f)
    axs[1].grid()
    axs[1].set_xlabel('time (s)', fontsize=16)
    axs[1].set_ylabel('Voltage', fontsize=16)
    axs[1].set_title('Signal', fontweight='bold', fontsize=20)
    plt.show()

    print()
'''
    # Example - 2: Square wave
    # f = array([])
    # for i in t:
    #     if i <= 0.5:
    #         f = append(f, 1)
    #     else:
    #         f = append(f, -1)

    # Example - 3: Signal with noise
    # seed(1)
    # f = sin( 2* pi* 5* t) + 0.5 * sin(2 * pi * 10 * t) + rand(len(t))

    SignalFFT = FFT(t, f, 100)
    Freqs1, Ampts1, Phases1, f1, Pxx_den1, fourier_coef1 = SignalFFT.Fourier_Transform()
    SignalFFT.PlotFFT()

    # t = linspace(0, 4, 1600)
    f2 = 3 * sin(2 * pi * 5 * t) + 0.7 * sin(2 * pi * 10 * t)
    SignalFFT2 = FFT(t, f2, 400)
    Freqs2, Ampts2, Phases2, f2, Pxx_den2, fourier_coef2 = SignalFFT2.Fourier_Transform()
    SignalFFT2.PlotFFT()

    s_dominant_freq_index, s_dominant_freq, t_dominant_freq_index, t_dominant_freq, scale_value_freq, \
    s_second_dominant_freq, t_second_dominant_freq, s_second_dominant_freq_index, t_second_dominant_freq_index = \
        SignalFFT2.get_frequency_scaling_S2T(Pxx_den1, Freqs1, Pxx_den2, Freqs2)

    t_dominant_freq = t_dominant_freq * scale_value_freq
    t_dominant_amp = Ampts2[t_dominant_freq_index]
    t_dominant_phase = Phases2[t_dominant_freq_index] * scale_value_freq

    t_second_dominant_freq = t_second_dominant_freq * scale_value_freq
    t_second_dominant_amp = Ampts2[t_second_dominant_freq_index]
    t_second_dominant_phase = Phases2[t_second_dominant_freq_index] * scale_value_freq

    t = linspace(0, 4, 1600)
    new_f2 = t_dominant_amp * sin(2 * pi * t_dominant_freq * t + t_dominant_phase) + t_second_dominant_amp * sin(
        2 * pi * t_second_dominant_freq * t + t_second_dominant_phase)
    SignalFFT3 = FFT(t, new_f2, 1600)
    Freqs3, Ampts3, Phases3, f3, Pxx_den3, fourier_coef3 = SignalFFT3.Fourier_Transform()
    SignalFFT3.PlotFFT()
    print()
'''
