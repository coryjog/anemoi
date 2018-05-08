Plotting - Correlate
==============================

This module is for plotting correlation analysis resutls between multiple masts. The typical format is to give the method the results from a method in an.analysis.correlate. The method then returns a plotly go.Figure object which consists of data and layout Python dictionaries. 

A more robust example will be added here in the future. 

.. code-block:: python
    :linenos:

    results = an.analysis.correlate.masts_10_minute_by_direction(ref_mast, site_mast)
    results_fig = an.plotting.correlate.masts_10_minute_by_direction(results)
    offline.iplot(results_fig)

.. automodule:: plotting.correlate
    :members:
