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

def plot_ws_correlation_by_direction(df, site_corr_results, site_mast_name=None, ref_mast_name=None):
    dir_bins = site_corr_results.index.get_level_values('DirBin').unique()

    fig = plt.figure(figsize=(30,2))
    for i, dir_bin in enumerate(dir_bins):
        slope = site_corr_results.loc[pd.IndexSlice[site_mast_name, ref_mast_name, dir_bin], 'slope']
        offset = site_corr_results.loc[pd.IndexSlice[site_mast_name, ref_mast_name, dir_bin], 'offset']
        R2 = site_corr_results.loc[pd.IndexSlice[site_mast_name, ref_mast_name, dir_bin], 'R2']

        if not np.isnan(slope):
            ax = fig.add_subplot(1, len(dir_bins), i+1)
            ws_bin_corr_data = df.loc[df.DirBin == dir_bin, ['RefWS', 'SiteWS']]
            ws_bin_corr_data.plot(kind='scatter', x='RefWS', y='SiteWS', ax=ax)
            ax.plot([0.0,25.0], np.array([0.0,25.0])*slope+offset, '-r')
            ax.set_xlim([0,25])
            ax.set_ylim([0,25])
            if i == 0:
                ax.set_xlabel('{}'.format(ref_mast_name))
                ax.set_ylabel('{}'.format(site_mast_name))
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.set_title('{}'.format(dir_bin))
    plt.show()