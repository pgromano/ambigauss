from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema

def lorentzian(X, height, center, width):
    """A Lorentzian distribution for fitting peaks.

    Parameters
    ----------
    X : numpy.array
        X values for distribution
    center : float
        the center of the distribution
    width : float
        full width at half maximum.

    Returns
    -------
    distribution : numpy array
        normalized Lorentzian distribution
    """
    distribution = height * 0.5 * width / ((X - center)**2 + (0.5 * width)**2)
    return distribution


def multilorentzian(X, parameters):
    """Build array of Lorentzian from multiple gaussian parameters.
    """
    # Initialize y array.
    y = np.zeros(X.shape[0])

    # Count number of peaks
    n_peaks = parameters.shape[0] // 3

    # Reshape parameters
    parameters = np.reshape(parameters, (n_peaks, 3))

    for peak_i in range(n_peaks):

        # Get parameters for peak i
        center = parameters[peak_i, 0]
        height = parameters[peak_i, 1]
        width = parameters[peak_i, 2]

        # Add peak to y
        y += lorentzian(X, height, center, width)

    return y


def objective(parameters, X, y):
    return np.sum((y - multilorentzian(X, parameters))**2)


class LorentzianSpectrum(BaseEstimator, RegressorMixin):
    def __init__(self, n_peaks=1, heights=None, centers=None, widths=None,
                 bounds=None, tol=None, method=None):
        self.n_peaks = n_peaks
        self.heights = heights
        self.centers = centers
        self.widths = widths
        self.bounds = bounds
        self.tol = tol
        self.method = method

    def _init_params(self, X, y):
        # Initialize Lorentzian centers
        if self.centers is None:
            # Estimate peaks
            index = argrelextrema(np.diff(X), np.greater)[0]
            index = index[np.argsort(y[index])]
#            index = find_peaks_cwt(y, widths=np.repeat(np.diff(X).mean(), 100))
            if self.n_peaks == -1:
                # Define number of peaks
                self.n_peaks = len(index)
            else:
                # If n_peaks defined slice indices accordingly
                index = index[-self.n_peaks:]
            self.centers = X[index]

        # Initialize Lorentzian heights
        if self.heights is None:
            self.heights = np.ones(self.n_peaks)

        # Initialize Lorentzian width
        if self.widths is None:
            self.widths = np.ones(self.n_peaks)

        self._params = np.concatenate([self.heights,
                                       self.centers,
                                       self.widths])

    def fit(self, X, y):
        self._init_params(X, y)
        results = minimize(objective, self._params, args=(X, y), bounds=self.bounds
                           method=self.method, tol=self.tol)
        self._params = results.x
        self.heights, self.centers, self.widths = np.reshape(results.x, (self.n_peaks, 3)).T
        return self

    def predict(self, X):
        return multilorentzian(X, self._params)
