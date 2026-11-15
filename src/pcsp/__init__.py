"""particle-csp-twin: hybrid physics-AI models and digital twins for particle CSP.

Conventions used throughout the package:

* SI units everywhere. Temperatures are absolute (kelvin) unless a name ends
  in ``_C``. Time is in seconds.
* Every simulation result is an :class:`xarray.Dataset` built with
  :func:`pcsp.data.schema.new_dataset`.
"""

__version__ = "0.1.0"
