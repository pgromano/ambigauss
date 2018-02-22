import numpy as np
import lmfit
from scipy.signal import find_peaks_cwt
from .curves import multigaussian

def residual(params, func, xdata, ydata=None):
    """Residual function."""
    ymodel = func(xdata, params)
    return ydata - ymodel


def fit(xdata, ydata):
    """Identify and fit an arbitrary number of peaks in a 1-d spectrum array.

    Parameters
    ----------
    xdata : 1-d array
        X data.

    ydata : 1-d array
        Y data.

    Returns
    -------
    results : lmfit.MinimizerResults.
        results of the fit. To get parameters, use `results.params`.
    """
    # Identify peaks
    index = find_peaks_cwt(ydata, widths=np.arange(1,100))

    # Number of peaks
    n_peaks = len(index)

    # Construct initial guesses
    parameters = lmfit.Parameters()

    for peak_i in range(n_peaks):
        idx = index[peak_i]

        # Add center parameter
        parameters.add(
            name='peak_{}_center'.format(peak_i),
            value=xdata[idx]
        )

        # Add height parameter
        parameters.add(
            name='peak_{}_height'.format(peak_i),
            value=ydata[idx]
        )

        # Add width parameter
        parameters.add(
            name='peak_{}_width'.format(peak_i),
            value=.1,
        )


    # Minimize the above residual function.
    results = lmfit.minimize(residual, parameters,
                            args=[multigaussian, xdata],
                            kws={'ydata': ydata})

    return results
