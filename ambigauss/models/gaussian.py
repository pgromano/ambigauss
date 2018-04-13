import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from scipy.signal import argrelextrema


def gaussian(X, height, center, width):
    """A Gaussian distribution for fitting peaks.

    Parameters
    ----------
    X : numpy.array
        X values for distribution
    height : float
        the amplitude of the distribution
    center : float
        the center of the distribution
    width : float
        full width at half maximum.

    Returns
    -------
    distribution : numpy array
        normalized Gaussian distribution
    """
    distribution = height * np.exp(-(X - center)**2 / (2 * width**2))
    return distribution


def multigaussian(X, parameters):
    """Build array of gaussians from multiple gaussian parameters.
    """
    # Initialize y array.
    y = np.zeros(X.shape[0])

    # Count number of peaks
    n_peaks = parameters.shape[0] // 3

    # Reshape parameters
    parameters = np.reshape(parameters, (n_peaks, 3))

    for peak_i in range(n_peaks):

        # Get parameters for peak i
        height = parameters[peak_i, 0]
        center = parameters[peak_i, 1]
        width = parameters[peak_i, 2]

        # Add peak to y
        y += gaussian(X, height, center, width)

    return y


def objective(parameters, X, y):
    return np.sum((y - multigaussian(X, parameters))**2)


class GaussianSpectrum(BaseEstimator, RegressorMixin):
    """Fit multiple Gaussian profiles to spectrum

    Parameters
    ----------
    n_peaks : int
        Number of peaks to estimate spectrum fit to data. If n_peaks=-1 then
        the number of peaks is estimated by relative extrema
        (scipy.signal.argrelextrema).
    heights : array-like, (n_peaks,)
        Initial guess for the peak amplitudes
    centers : array-like, (n_peaks,)
        Initial guess for the peak centers
    widths : array-like, (n_peaks,)
        Initial guess for the full width at half maximum
    bounds : sequence, optional
        Bounds for variables (only for L-BFGS-B, TNC and SLSQP). Passed as
        (min, max) pairs for each element in x, defining the bounds on that
        parameter. Use None for one of min or max when there is no bound in
        that direction.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    method : str or callable, optional
        Type of solver. See scipy.optimize.minimize

    Attributes
    ----------
    heights : array-like, (n_peaks,)
        Optmized height parameter
    centers : array-like, (n_peaks,)
        Optmized center parameter
    widths : array-like, (n_peaks,)
        Optmized width parameter

    See Also
    --------
    scipy.signal.argrelextrema, scipy.optimize.minimize
    """
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
        # Initialize Gaussian centers
        if self.centers is None:
            # Estimate locations of peaks
            index = argrelextrema(np.diff(X), np.greater)[0]
            index = index[np.argsort(y[index])]

            if self.n_peaks == -1:
                # Define number of peaks
                self.n_peaks = len(index)
            else:
                # If n_peaks defined, slice indices accordingly
                index = index[-self.n_peaks:]
            self.centers = X[index]

        # Initialize Gaussian heights
        if self.heights is None:
            self.heights = np.ones(self.n_peaks)

        # Initialize Gaussian width
        if self.widths is None:
            self.widths = np.ones(self.n_peaks)

        self._params = np.column_stack([self.heights,
                                        self.centers,
                                        self.widths]).ravel()

    def fit(self, X, y):
        self._init_params(X, y)
        results = minimize(objective, self._params, args=(X, y), bounds=self.bounds,
                           method=self.method, tol=self.tol)
        self._params = results.x
        self.heights, self.centers, self.widths = np.reshape(results.x, (self.n_peaks, 3)).T
        return self

    def predict(self, X):
        return multigaussian(X, self._params)
