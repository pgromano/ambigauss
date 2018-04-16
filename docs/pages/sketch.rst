API Design
==========

A sketch of model

Gaussian spectrum
-----------------

.. code-block:: python

  # Import gaussian
  from ambigauss import GaussianSpectrum

  # Initialize a model.
  m = GaussianSpectrum(n_peaks=3)

  # Fit data.
  m.fit(x, y)

  # Plot data
  m.plot()

  # Print parameters
  m.print_parameters()


Voigt spectrum
--------------

.. code-block:: python

  # Import gaussian
  from ambigauss import VoigtSpectrum

  # Initialize a model.
  m = VoigtSpectrum(n_peaks=3)

  # Fit data.
  m.fit(x, y)

  # Plot data
  m.plot()

  # Print parameters
  m.print_parameters()
