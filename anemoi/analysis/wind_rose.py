'''
Frequency distribution analysis tools
_________
A toolbox of wind data frequency distribution tools
'''
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import scipy.optimize as spyopt
from scipy.special import gamma

### WIND AND POWER ROSES ###
def return_dir_bin_information(dir_sectors=16):
    dir_bin_width = 360.0/dir_sectors
    dir_bin_width_rad = np.radians(dir_bin_width)
    dir_bin_edges = np.arange(0.0-dir_bin_width/2.0, 360.0+dir_bin_width, dir_bin_width)
    dir_bin_labels = np.arange(0.0,360.0+dir_bin_width, dir_bin_width)
    return dir_bin_width, dir_bin_width_rad, dir_bin_edges, dir_bin_labels

def return_wind_direction_bins(mast_data, dir_sensor, dir_sectors=16):
    dir_bin_width, dir_bin_width_rad, dir_bin_edges, dir_bin_labels = return_dir_bin_information(dir_sectors=dir_sectors)

    dir_data = mast_data[dir_sensor].dropna().to_frame(dir_sensor)
    dir_data['dir_bin_center'] = pd.cut(mast_data[dir_sensor], dir_bin_edges, labels=dir_bin_labels)
    dir_data.dir_bin_center = dir_data.dir_bin_center.replace(360.0, 0.0)
    dir_data = dir_data.dropna()
    return dir_data

def append_dir_bin(dir_data, dir_sectors=16):
    dir_data = dir_data.dropna()
    dir_bin_width, dir_bin_width_rad, dir_bin_edges, dir_bin_labels = return_dir_bin_information(dir_sectors=dir_sectors)
    bin_width = 360.0/dir_sectors
    dir_bin = np.floor_divide(np.mod(dir_data + (bin_width/2.0),360.0),bin_width)+1
    dir_bin = dir_bin.dropna().astype(int)
    return dir_bin

def return_wind_speed_bins(df, ws_sensor, bin_width=1.0, half_first_bin=False):
    if half_first_bin:
        first_wind_speed_bin_halved = np.arange(0.0,bin_width,bin_width/2.0)
        ws_bin_edges = np.arange(0.0+bin_width,30.0+bin_width, bin_width)
        ws_bin_edges = np.concatenate([first_wind_speed_bin_halved,ws_bin_edges])
    else:
        ws_bin_edges = np.arange(0.0,30.0+bin_width, bin_width)

    ws_bin_labels=ws_bin_edges[1::]

    ws_data = df[ws_sensor].dropna().to_frame(ws_sensor)
    ws_data['ws_bin'] = pd.cut(df[ws_sensor].dropna(), ws_bin_edges, labels=ws_bin_labels)
    ws_data.dropna(inplace=True)
    return ws_data

def return_directional_wind_frequencies(df, dir_sensor, dir_sectors=16):
    dir_bin_width, dir_bin_width_rad, dir_bin_edges, dir_bin_labels = return_dir_bin_information(dir_sectors=dir_sectors)

    dir_data = return_wind_direction_bins(df, dir_sensor, dir_sectors=dir_sectors)
    dir_freqs = dir_data.groupby('dir_bin_center').sum()/dir_data.shape[0]
    dir_freqs.drop(360.0, axis=0, inplace=True)
    dir_freqs.fillna(0, inplace=True)
    return dir_freqs

def return_tab_file(df, ws_sensor, dir_sensor, dir_sectors=16, ws_bin_width=1.0, half_first_bin=False, freq_as_label=False):
    dir_freqs = return_directional_wind_frequencies(df, dir_sensor, dir_sectors=dir_sectors)
    dir_bins = return_wind_direction_bins(df, dir_sensor, dir_sectors=dir_sectors)
    ws_bins = return_wind_speed_bins(df, ws_sensor, bin_width=ws_bin_width, half_first_bin=half_first_bin)
    ws_dir_bins = pd.concat([ws_bins, dir_bins], axis=1)

    tab_df = []
    for dir_bin in dir_freqs.index.values:
        ws_dir_binned = ws_dir_bins.loc[ws_dir_bins.dir_bin_center == dir_bin,[ws_sensor, 'ws_bin']].groupby('ws_bin').count()
        ws_dir_binned = ws_dir_binned/ws_dir_binned.sum()*1000.0
        ws_dir_binned.columns = [dir_bin]
        tab_df.append(ws_dir_binned)
    tab_df = pd.concat(tab_df, axis=1)

    if freq_as_label:
        tab_df.columns = np.round(dir_freqs.T.values,3).tolist()
    else:
        tab_df.columns = dir_freqs.index.values

    tab_df.fillna(0, inplace=True)
    tab_df.index.name = ''
    return tab_df

def return_directional_energy_frequencies(df, ws_sensor, dir_sensor, dir_sectors=16):
    dir_energy_data = return_wind_direction_bins(df, dir_sensor, dir_sectors=dir_sectors)
    dir_energy_data[ws_sensor] = df[ws_sensor]**3
    ws_dir_freqs = dir_energy_data.groupby('dir_bin_center').agg({ws_sensor:['sum'], dir_sensor:['count']})
    ws_dir_freqs = ws_dir_freqs/ws_dir_freqs.sum(axis=0)
    ws_dir_freqs.columns = ['energy', 'dir']
    return ws_dir_freqs

def plot_rose_on_axes(dir_bin_centers, dir_bin_freqs, ax, title='', color='#001A70', show_yticklabels=False):
    dir_bin_width_radians = np.radians(360.0/dir_bin_centers.shape[0])
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')
    ax.bar(np.radians(dir_bin_centers), dir_bin_freqs, width=dir_bin_width_radians, color=color)
    ax.set_title(title)
    ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
    if not show_yticklabels:
        ax.set_yticklabels([])
    return ax

def return_wind_rose_figure(dir_bin_centers, dir_bin_freqs, fig=None, title='Wind speed', color='#001A70'):
    if fig is None:
        fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='polar')
    plot_rose_on_axes(dir_bin_centers, dir_bin_freqs, ax=ax, title=title, color=color)
    return fig

def return_energy_rose_figure(dir_bin_centers, dir_bin_freqs, fig=None, title='Energy', color='#509E2F'):
    if fig is None:
        fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='polar')
    plot_rose_on_axes(dir_bin_centers, dir_bin_freqs, ax=ax, title=title, color=color)
    return fig

def return_wind_energy_rose_figure(dir_bin_centers, dir_bin_freqs_ws, dir_bin_freqs_energy, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122, projection='polar')

    plot_rose_on_axes(dir_bin_centers, dir_bin_freqs_ws, ax=ax1, title='Wind speed', color='#001A70')
    plot_rose_on_axes(dir_bin_centers, dir_bin_freqs_energy, ax=ax2, title='Energy', color='#509E2F')
    return fig

def plot_freq_dist(params, data=None, sensor=None, title='Wind speed frequency distribution'):
    '''
    Plots a wind speed frequency distribution
    Parameters:
    ___________
    params: list of float [A,k]
        Array of Weibull A and k parameters
        Can be caluclated using mast.return_weibull_params()
    data: pandas Series, default None
        Measured wind speed data from a sensor
        Can be called using mast.return_primary_ano_data()
    title: string
        Plot title, default 'Wind speed frequency distribution'
    Returns:
    ________
    Frequency distribution plot
    '''
    A, k = params
    ws_bin_edges = np.arange(0.0,31.0)
    x_smooth = np.linspace(0.0, 31.0, 100)
    weibull_dist = stats.exponweib(1, k, scale=A)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    if data is not None:
        data.dropna().plot(kind='hist', bins=ws_bin_edges, ax=ax, color='gray', normed=True, alpha=0.6, label='Measured')
    ax.plot(x_smooth, weibull_dist.pdf(x_smooth), color='darkblue', label='Weibull fit')
    ax.legend(loc='best')
    ax.set_xlim([0,30])
    ax.set_title(title + ' (A: %.2f; k: %.2f)' %(A,k))
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Wind speed [m/s]')
    return fig
