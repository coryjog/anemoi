.. anemoi documentation master file, created by
   sphinx-quickstart on Fri Oct  6 13:55:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to anemoi's documentation!
==================================

EDF's pre-alpha Python package for wind data analysis. This package has an MIT lisence but is primarily for internal use at the moment. This means you are free to use the code but there may be EDF specific functionality, such as access to met mast data, that requires a user be connected to our VPN with permission to access our databases. 

Anemoi were Greek wind gods. Each god was ascribed a cardinal direction from which their respective winds came, and were each associated with various seasons and weather conditions.  

**Requirements:** 

* Python 3
* numpy
* pandas
* plotly
* pyodbc
* requests
* scipy
* statsmodels

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   docs_quick-start.rst
   docs_start-from-scratch.rst
   docs_data_model.rst
   docs_tutorial.rst
   docs_plotting.rst
   code_class_MetMast.rst
   code_class_RefMast.rst
   code_io_references.rst
   code_io_database.rst
   code_analysis_shear.rst
   code_analysis_correlate.rst
   code_analysis_freq_dist.rst
   code_plotting_shear.rst
   code_plotting_correlate.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
