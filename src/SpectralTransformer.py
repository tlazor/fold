from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def broadcast_vector_to_axis(vec, reference_array, axis):
    """
    vec: 1D array of length N.
    reference_array: the array we want to multiply with.
    axis: the axis along which vec should align.

    Returns a reshaped version of 'vec' that broadcasts
    over all other axes of 'reference_array'.
    """
    shape = [1] * reference_array.ndim  # e.g. [1,1,1,...]
    shape[axis] = vec.shape[0]  # put length=N at the desired axis
    return vec.reshape(shape)


def circular_optimized(arr, axis=1):
    """
    Computes the average of the shifted FFT results (in a closed-form way),
    and then the power spectrum of that average. Equivalent to:

       for i in range(1, N):
           fft_i, pow_i = calculate_power_spectrum(np.roll(arr, i, axis=axis))
           fft_result_avg += fft_i / N
           power_spectrum_avg += pow_i / N

    except done via a single FFT + complex weighting.
    """
    N = arr.shape[axis]

    # 1) Do one FFT on the original data
    fft_full = np.fft.fft(arr, axis=axis)

    # 2) Build shift_factors for frequencies:
    #    freq=0 -> (N-1)/N;  freq>0 -> -1/N
    shift_factors = np.full(N, -1 / N, dtype=fft_full.dtype)
    shift_factors[0] = (N - 1) / N

    # 3) Reshape so that axis=1 in the final multiplication
    shift_factors = broadcast_vector_to_axis(shift_factors, fft_full, axis)

    # 4) Multiply => the “averaged” FFT across all shifts
    fft_avg_full = fft_full * shift_factors

    # 5) Keep first half frequencies (to replicate your original shape/truncation)
    half = N // 2
    fft_result_avg = np.take(fft_avg_full, indices=range(0, half), axis=axis)

    # 6) Compute the power spectrum of that average
    power_spectrum_avg = np.abs(fft_result_avg) ** 2

    return fft_result_avg, power_spectrum_avg


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

        return [circular_optimized(x)[1] for x in X]
