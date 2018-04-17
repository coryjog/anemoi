import anemoi as an
import pandas as pd
import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode(connected=True)

# Colors for plotting
EDFGreen = '#509E2F'
EDFLightGreen = '#C4D600'
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

def plotly_data_by_column_line(df, kind='line'):
    if df.columns.nlevels > 1:
        new_cols = []
        for level in df.columns.names:
            new_cols.append(df.columns.get_level_values(level=level).astype(str))
        new_cols = pd.MultiIndex.from_arrays(new_cols).map('_'.join)
        df.columns = new_cols

    if kind=='line':
        plotting_data = [{'x': df.index, 'y': df[col], 'name': col, 'mode':'lines'} for col in df.columns]
    elif kind=='bar':
        plotting_data = [go.Bar(x=df.index, y=df[col], name=col) for col in df.columns]
    return plotting_data
