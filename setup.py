from setuptools import setup, find_packages

required = ['numpy',
            'pandas',
            'plotly',
            'pyodbc',
            'scipy',
            'seaborn',
            'statsmodels']

setup(name='anemoi',
      version='0.0.31',
      description='EDF wind data analysis package',
      url='http://github.com/coryjog/anemoi',
      author='Cory Jog',
      author_email='cory.jog@edf-re.com',
      license='MIT',
      packages=find_packages(),
      install_requires= required,
      zip_safe=False)
