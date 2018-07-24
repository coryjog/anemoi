Start from Nothing Guide
========================

Instructions for installing Python via Anaconda. 
-------------------------------------------------

Assuming you are starting from scratch, you'll need to first install Python 3. It is recommended you use the Anaconda distribution, which automatically includes many of the libraries upon which anemoi depends.

You can download and install the `Anaconda distribution of Python 3 here <https://www.anaconda.com/download/>`_.  

Then from the Anaconda Prompt (launched from the start menu) you can run the following:

.. code-block:: none
    
    conda install statsmodels
    pip install anemoi

All the dependancies and requirements that weren't already installed with Anaconda will be automatically installed.

Finally, you can launch a Jupyter notebook by running the following in the Anaconda Prompt:

.. code-block:: none
    
    jupyter notebook


.. code-block:: python
    :linenos:

    import anemoi as an 
