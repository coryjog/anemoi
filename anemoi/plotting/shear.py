import anemoi as an
import pandas as pd
import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode(connected=True)

# Colors for plotting
EDFGreen = '#509E2F'
EDFOrange = '#FE5815'
EDFBlue = '#001A70'
EDFColors = [EDFGreen, EDFBlue, EDFOrange]

def plotly_data_by_column(df):
    if df.columns.nlevels > 1:
        new_cols = []
        for level in df.columns.names:
            new_cols.append(df.columns.get_level_values(level=level).astype(str))
        new_cols = pd.MultiIndex.from_arrays(new_cols).map('_'.join)
        df.columns = new_cols

    plotting_data = [{'x': df.index, 'y': df[col], 'name': col, 'mode':'lines'} for col in df.columns]
    return plotting_data

def flatten_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

def nest_list(list):
    return [[element] for element in list]

def annual_mast_results(mast_shear_results, lower_shear_bound=0.1, upper_shear_bound=0.3):
    '''
    Returns plotting data and a layout for a single mast shear analysis plot.
    '''
    assert mast_shear_results.index.nlevels == 3, "Expecting three index levels for plotting ['orient','height','sensor']"

    orients = mast_shear_results.index.get_level_values(level='orient').unique().tolist()
    mast_shear_results.columns = mast_shear_results.columns.get_level_values('height')
    mast_shear_results.columns.name = 'ht2'
    mast_shear_results.index = mast_shear_results.index.droplevel(level=['sensor'])
    mast_shear_results.index.names = ['orient','ht1']
    stacked_shear_results = mast_shear_results.stack().to_frame('alpha')

    plotting_data = []
    for i, orient in enumerate(orients):
        stacked_shear_results_orient = stacked_shear_results.loc[pd.IndexSlice[orient,:,:],:]
        for sensor_combo in stacked_shear_results_orient.index:
            h1 = sensor_combo[1]
            h2 = sensor_combo[2]
            alpha = stacked_shear_results.loc[sensor_combo,'alpha']
            plotting_data.append(go.Scatter(x=[alpha,alpha], 
                                            y=[h1,h2],
                                            marker=dict(color=EDFColors[i]),
                                            name='{}: {} - {}'.format(orient, h1, h2)))

    layout = go.Layout(showlegend=True, 
                        autosize=True,
                        font=dict(size=12), 
                        title='Shear results',
                        height=400,
                        width=600,  
                        yaxis=dict(title='Sensor height [m]',
                                   rangemode='tozero',
                                   dtick=10.0),
                        xaxis=dict(title='Alpha',
                                   range=[lower_shear_bound, upper_shear_bound]),
                        margin=dict(l=40,r=20,t=25,b=30))

    return go.Figure(data=plotting_data, layout=layout)

def mast_results_by_dir_and_orient(mast_dir_shear_results):
    '''
    Returns plotting data and a layout for a single mast directional shear analysis plot.
    '''
    plotting_data = plotly_data_by_column(mast_dir_shear_results)

    layout = go.Layout(showlegend=True, 
                    autosize=True,
                    font=dict(size=12), 
                    title='Directional shear results',
                    height=400,
                    yaxis=dict(title='Alpha',
                               rangemode='tozero'),
                    xaxis=dict(title='Direction [deg]',
                               range=[0.0, 360.0],
                               tick0=0.0,
                               dtick=30.0),
                    margin=dict(l=50,r=20,t=25,b=30))
    
    return go.Figure(data=plotting_data, layout=layout)

def mast_results_by_month_and_orient(monthly_mast_shear_results_by_orient):
    '''
    Returns plotting data and a layout for a single mast directional shear analysis plot.
    '''
    plotting_data = plotly_data_by_column(monthly_mast_shear_results_by_orient)

    layout = go.Layout(showlegend=True, 
                    autosize=True,
                    font=dict(size=12), 
                    title='Monthly shear results',
                    height=400,
                    yaxis=dict(title='Alpha',
                               rangemode='tozero'),
                    margin=dict(l=50,r=20,t=25,b=30))
    
    return go.Figure(data=plotting_data, layout=layout)

def mast_annual_profiles_by_orient(annual_alpha_profiles_by_orient):
    '''
    Returns plotting data and a layout for a single mast directional shear analysis plot.
    '''
    plotting_data = an.plotting.shear.plotly_data_by_column(annual_alpha_profiles_by_orient)
    layout = go.Layout(showlegend=True, 
                    autosize=True,
                    font=dict(size=12), 
                    title='Annual shear profiles',
                    height=400,
                    width=800,
                    yaxis=dict(title='Alpha',
                               rangemode='tozero'),
                    xaxis=dict(title='Month',
                               range=[1.0, 12.0],
                               tick0=1.0,
                               dtick=1.0),
                    margin=dict(l=50,r=20,t=25,b=30))
    
    return go.Figure(data=plotting_data, layout=layout)

def row_colors_by_index_level(df, level):
    unique_values = df.index.get_level_values(level=level).unique().tolist()
    row_counts = [df.loc[value,:].shape[0] for value in unique_values]
    row_colors = ['lightgray', 'white', 'lightgray', 'white', 'lightgray', 'white']
    rows = [[row_colors[i]]*row_count for i,row_count in enumerate(row_counts)]
    return [flatten_list(rows)]

def plotly_table_from_df(df, color_by_index_level=None):
    if color_by_index_level is None:
        row_colors = ['white']
    else:
        row_colors = row_colors_by_index_level(df, level=color_by_index_level)
        
    df = df.astype(np.float).round(3).fillna('-').reset_index()
    
    plotly_fig = [go.Table(columnwidth=[5]*df.shape[1],
                            header=dict(values=df.columns,
                                        fill = dict(color='gray'),
                                        font = dict(color='white'),
                                        align = ['left']*df.shape[1]),
                            cells=dict(values=[df.round(3)[col].values for col in df.columns],
                                       fill = dict(color=row_colors),
                                       align = ['left']*df.shape[1]))]
    return plotly_fig
