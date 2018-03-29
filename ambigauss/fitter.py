import numpy as np
import lmfit
from scipy.signal import find_peaks_cwt
from .curves import multigaussian, multilorentzian


def residual(params, func, xdata, ydata=None):
    """Residual function."""
    ymodel = func(xdata, params)
    return ydata - ymodel


def fit(xdata, ydata, distribution):
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
                            args=[distribution, xdata],
                            kws={'ydata': ydata})

    return results


def bayes_fit(xdata, ydata, distribution, burn=100, steps=1000, thin=20):
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
    ML_results = lmfit.minimize(residual, parameters,
                            args=[distribution, xdata],
                            kws={'ydata': ydata})

    # Add a noise term for the Bayesian fit
    ML_results.params.add('noise', value=1, min=0.001, max=2)

    # Define the log probability expression for the emcee fitter
    def lnprob(params = ML_results.params):
        noise = params['noise']
        return -0.5 * np.sum((residual(params, distribution, xdata, ydata) / noise)**2 + np.log(2 * np.pi * noise**2))

    # Build a minizer object for the emcee search
    mini = lmfit.Minimizer(lnprob, ML_results.params)

    # Use the emcee version of minimizer class to perform MCMC sampling
    bayes_results = mini.emcee(burn=burn, steps=steps, thin=thin, params=ML_results.params)

    return bayes_results
