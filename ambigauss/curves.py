import numpy as np


def gaussian(x, amp, center, width):
    """A Gaussian distribution for fitting peaks.

    Parameters
    ----------
    x : numpy.array
        x values for distribution
    center : float
        the center of the distribution
    width : float
        full width at half maximum.

    Returns
    -------
    distribution : numpy array
        normalized Gaussian distribution
    """
    distribution = amp * np.exp(-(x - center)**2/ (2*width**2))
    return distribution


def multigaussian(x, parameters):
    """Build array of gaussians from multiple gaussian parameters.
    """
    # Initialize y array.
    y = np.zeros(len(x))

    # Count number of peaks
    n_peaks = int(len(parameters) / 3)

    # Simple sanity check.
    if (len(parameters) % 3) != 0:
        raise Exception("Incorrect number of parameters.")

    for peak_i in range(n_peaks):

        # Get parameters for peak i
        center = parameters['peak_{}_center'.format(peak_i)]
        height = parameters['peak_{}_height'.format(peak_i)]
        width = parameters['peak_{}_width'.format(peak_i)]

        # Add peak to y.
        y += gaussian(x, height, center, width)

    return y
