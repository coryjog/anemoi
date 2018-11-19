Plotting
========

This will be a quick tutorial for plotting results from Anemoi. Plotting in Anemoi is based on the `Plotly visualization library <https://plot.ly/d3-js-for-python-and-pandas-charts/>`_. If you haven't used Plotly in Jupyter before you can go `here <https://plot.ly/python/ipython-notebook-tutorial/>`_ for a tutorial on how to get setup the first time using Plotly.

Some users will wonder why Plotly was chosen as the default plotting library over the myriad of other Python visualization libraries available such as matplotlib, seaborn, bokeh, altair, and ggplot. While there are benefits to each of these and we expect consolidation in this space in the years to come, right now the authors think Plolty provides the cleanest interactive visuals for the least amount of overhead. Obviously, this can be debated. But given Plotly is built upon the very powerful and prolific D3.js library, it offers quite a few benefits. Including `Mapbox <https://www.mapbox.com/>`_ maps and standalone exports to .html from Jupyter. If that doesn't mean anything to you, just know Plotly was chosen to give us all access to beautiful plots (including maps) which are easy to package and distribute throughout our respective organizations.


Import initialization for plotting
----------------------------------


.. code-block:: python
    :linenos:

    import anemoi as an
    
    import plotly.plotly as py
    import plotly.graph_objs as go
    import plotly.offline as offline
    offline.init_notebook_mode(connected=True)

    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:99% !important; }</style>"))

Introduction
------------

Generally speaking, Plotly needs two Python dictionaries to create a figure: data and layout. Data describes your actual data, such as the frequencies in a wind rose or the wind speeds in a correlation, along with the formatting information for the data traces, like bar or scatter. Layout describes the general formatting of your plot like title, axes, background color, margins, and figure size. 

To produce standardized plots from Anemoi results the template is as follows:

.. code-block:: python
    :linenos:

    # pseaudo code
    analysis_results = an.analysis.shear.annual(mast)
    results_figure = an.plotting.shear.annual_figure(analysis_results)
    offline.iplot(results_figure)

There are a couple things I'd like to highlight before covering what is going on in the back ground. Firstly, returning analysis results will involve calling: an.analysis.analysis_library. Plotting those same results will involve calling the same analysis library from the plotting module: an.plotting.analysis_library. The shear analysis library is being used in the example above. 

Behind the scenes, Anemoi is constructing both the data and layout dictionaries. The data describes the individual traces, such as:

.. code-block:: python
    :linenos:
    
    plot_data = [go.Scattergl(
                    x=data[ref_ws_col],
                    y=data[site_ws_col],
                    mode='markers',
                    name='data',
                    marker=dict(color=EDFBlue))]

The layout describes the general properties of the plot:

.. code-block:: python
    :linenos:

    plot_layout = dict(autosize=False,
                        width=600,
                        height=600,
                        font=dict(color='#CCCCCC'),
                        titlefont=dict(color='#CCCCCC', size='12'),
                        margin=dict(l=35,r=0,b=0,t=25),
                        hovermode="closest",
                        legend=dict(font=dict(size=10), orientation='h'),
                        title='Wind speed correlation',
                        xaxis=dict(title='Reference [m/s]',
                                rangemode='normal'),
                        yaxis=dict(title='Site [m/s]',
                                rangemode='normal')
                        )

The data and layout dictionaries are combined into a plotly figure and displayed in a Jupyter notebook using the following:

.. code-block:: python
    :linenos:

    fig = go.Figure(data=plot_data, layout=plot_layout)
    offline.iplot(fig)

Formatting
----------

EDF branding colors used:

.. code-block:: python
    :linenos:

    # Colors for plotting
    EDFGreen = '#509E2F'
    EDFLightGreen = '#C4D600'
    EDFOrange = '#FE5815'
    EDFLightOrange = '#FFA02F'
    EDFBlue = '#001A70'
    EDFLightBlue = '#005BBB'
    EDFColors = [EDFGreen, EDFBlue, EDFOrange, EDFLightGreen, EDFLightBlue, EDFLightOrange]
