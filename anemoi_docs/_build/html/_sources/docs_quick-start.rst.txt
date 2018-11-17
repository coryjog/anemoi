Quick Start Guide
=================

Instructions for installing anemoi. 
-----------------------------------

Anemoi only supports Python 3 and requires statsmodels, which is distributed via Anaconda. First, if you have a Windows machine you'll need to install a C++ compiler. You can do this by installing the most recent version of Visual Studio Community edition from `here <https://visualstudio.microsoft.com/downloads/>`_. Then, assuming you've already installed Anaconda, run the following:  

.. code-block:: none
    
    conda install statsmodels
    pip install anemoi

If you can run the following within a Jupyter Notebook without an error you know you're ready to analize some wind data.

.. code-block:: python
    :linenos:

    import anemoi as an 


Alternate instructions for using anemoi. 
-----------------------------------------

You can download the anemoi package from the `master branch <https://github.com/coryjog/anemoi>`_ on GitHub and place the anemoi folder in your working directory. This option requires you to separately install all the needed Python packages listed in the requirements.txt file.