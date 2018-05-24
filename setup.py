from setuptools import setup, find_packages

exec(open('anemoi/_version.py').read())

required = ['pandas',
            'numpy',
            'scipy',
            'matplotlib',
            'plotly',
            'pyodbc',
            'pyathenajdbc',
            'statsmodels']

classifiers = ['Development Status :: 3 - Alpha',
    			'Intended Audience :: Developers',
    			'Topic :: Software Development :: Build Tools',
    			'License :: OSI Approved :: MIT License',
    			'Programming Language :: Python :: 3']

modules = ['analysis',
			'io',
			'plotting',
			'utils']

project_urls={
	'Source': 'https://github.com/coryjog/anemoi',
    'Documentation': 'https://coryjog.github.io/anemoi/',
    'Tracker': 'https://github.com/coryjog/anemoi/issues'}

setup(name='anemoi',
      version=__version__,
      description='EDF wind data analysis package',
      url='http://github.com/coryjog/anemoi',
      author='Cory Jog',
      author_email='cory.jog@edf-re.com',
      license='MIT License',
      packages=find_packages(),
      install_requires= required,
      python_requires='>=3',
      py_modules=modules,
      classifiers=classifiers,
      keywords='wind data analysis EDF',
      project_urls=project_urls,
      zip_safe=False)
