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

def plot_shear_results(shear_results, upper_height=80.0, lower_shear_bound=0.1, upper_shear_bound=0.25, fig_size=7):
    '''
    Returns a shear analysis plot for every mast
    Required inputs: shear results DataFrame from a shear analysis
    Optional inputs: upper height bound, lower shear bound, upper shear bound, square figure size
    '''
    figs = []

    for mast in shear_results.columns.get_level_values('Mast').unique():
        fig = plt.figure(figsize=(fig_size,fig_size))
        ax = fig.add_subplot(111)
        single_mast_shear_results = shear_results[mast].replace('-',np.nan).dropna(how='all').fillna('-')
        h1s = single_mast_shear_results.columns.get_level_values('Ht').tolist()
        h2s = single_mast_shear_results.index.get_level_values('Ht').tolist()
        orients = single_mast_shear_results.index.get_level_values('Orient').unique().tolist()
        for i,orient in enumerate(orients):
            for h1 in h1s:
                for h2 in h2s:
                    alpha = single_mast_shear_results.loc[pd.IndexSlice[orient,h2],h1]
                    if isinstance(alpha, float):
                        ax.plot([alpha, alpha], [h1, h2], color=EDFColors[i], linewidth=3, label=orient)

        ax.legend(loc='best')
        handles, labels = plt.gca().get_legend_handles_labels()
        i = 1
        while i<len(labels):
            if labels[i] in labels[:i]:
                del(labels[i])
                del(handles[i])
            else:
                i +=1

        plt.legend(handles, labels)

        ax.set_title('Mast: {}'.format(mast))
        ax.set_xlim([lower_shear_bound, upper_shear_bound])
        ax.set_ylim([0,upper_height])
        ax.set_ylabel('Sensor heights [m]')
        ax.set_xlabel('Measured alpha')
        figs.append(fig)
    return(figs)