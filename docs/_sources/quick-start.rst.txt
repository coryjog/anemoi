Quick Start Guide
=================

Instructions for installing anemoi. 
-----------------------------------------

Assuming you are starting from scratch, you'll need to first install Python 3 and git. I recommend installing the `Anaconda distribution of Python 3 <https://www.anaconda.com/download/>`_ which automatically includes many of the libraries on which anemoi depends. 

Next you can install Git for your specific operating system `from here <https://git-scm.com/downloads>`_.

Then from the command line you pip install:

.. code-block:: none
    
    conda install statsmodels
    pip install git+git://github.com/coryjog/anemoi

All the dependancies listed in required.txt will also be installed.

Finally, if you can run the following within a Jupyter Notebook without an error you know you're ready to analize some wind data.

.. code-block:: python
    :linenos:

    import anemoi as an 


Alternate instructions for using anemoi. 
-----------------------------------------

You can download the anemoi package from the master branch on GitHub and place the anemoi folder in your working directory. This option requires you to separately install all the needed Python packages listed in the required.txt file.