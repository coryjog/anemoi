Plotting - Shear
==============================

This module is for plotting shear analysis results between multiple masts. The typical format is to give the method the results from a method in an.analysis.shear. The method then returns a plotly go.Figure object which consists of data and layout Python dictionaries. 

A more robust example will be added here in the future. 

.. code-block:: python
    :linenos:

    results = an.analysis.shear.annual_mast(mast)
    results_fig = an.plotting.shear.annual_mast(results)
    offline.iplot(results_fig)

.. automodule:: plotting.shear
    :members:
