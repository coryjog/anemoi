import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
mpl.style.use('ggplot')
sns.set_context('talk')

# Colors for plotting
EDFGreen = '#509E2F'
EDFOrange = '#FE5815'
EDFBlue = '#001A70'
EDFColors = [EDFGreen, EDFBlue, EDFOrange]
Context = 'talk'

# def monthly_recovery(mast_data):
    