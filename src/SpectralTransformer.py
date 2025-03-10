from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def circular(arr):
    new_shape = (arr.shape[0], arr.shape[1] // 2)
    fft_result_avg = np.zeros(new_shape, "complex128")
    power_spectrum_avg = np.zeros(new_shape)
    for i in range(1, arr.shape[1]):
        fft_result, power_spectrum = calculate_power_spectrum(np.roll(arr, i, axis=1))
        fft_result_avg += fft_result / arr.shape[1]
        power_spectrum_avg += power_spectrum / arr.shape[1]

    return fft_result_avg, power_spectrum_avg


def calculate_power_spectrum(arr, axis=1):
    # Perform Fourier Transform along the specified axis
    fft_result = np.fft.fft(arr, axis=axis)

    # FT of real valued signals are conjugate symmetric
    nyquist = fft_result.shape[1] // 2
    fft_result = fft_result[:, :nyquist]

    # Calculate the power spectrum (square of the magnitude of the FFT result)
    power_spectrum = np.abs(fft_result) ** 2

    return fft_result, power_spectrum


class SpectralTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, temp="fourier"):
        self.temp = temp

    def fit(self, X, y=None):
        """
        Learn something from the data if needed.

        X : array-like or dataframe of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or None
        """
        # This transformer doesn't learn anything from the data,
        # so we just return self.
        return self

    def transform(self, X):
        # max_length = 0
        # for x in X:
        #     num_langs, sample_len = x.shape
        #     max_length = max(max_length, sample_len)

        return [circular(x)[1] for x in X]
