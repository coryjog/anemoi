Analysis - Correlate
==============================

This module is for correlation analysis between multiple masts. The typical format is to give the method the site and reference MetMast or RefMast objects. The method then returns a DataFrame of results. Those results can then be applied with a separate method. 

For example, correlations typically require two lines of text. One for running the actual correlation between masts and one for applying that correlation to synthesize missing data:

.. code-block:: python
    :linenos:

    corr_results_10min = an.analysis.correlate.masts_10_minute_by_direction(ref_mast, site_mast)
    syn_10min = an.analysis.correlate.apply_10min_results_by_direction(ref_mast, site_mast, corr_results_10min)

.. automodule:: analysis.correlate
    :members:
