import anemoi as an
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode(connected=True)

# Colors for plotting
EDFGreen = '#509E2F'
EDFLightGreen = '#C4D600'
EDFOrange = '#FE5815'
EDFLightOrange = '#FFA02F'
EDFBlue = '#001A70'
EDFLightBlue = '#005BBB'
EDFColors = [EDFGreen, EDFBlue, EDFOrange, EDFLightGreen, EDFLightBlue, EDFLightOrange]

def plot_layout(xlabel=None, ylabel=None, title=None, non_negative_values=True):
    if xlabel is None:
        xlabel='Ref'

    if ylabel is None:
        ylabel='Site'

    if title is None:
        title='Correlation'

    if non_negative_values:
        rangemode='nonnegative'
    else:
        rangemode='normal'

    layout = dict(autosize=False,
                  width=600,
                  height=600,
                  font=dict(color='#CCCCCC'),
                  titlefont=dict(color='#CCCCCC', size='12'),
                  margin=dict(l=35,r=0,b=0,t=25),
                  hovermode="closest",
                  legend=dict(font=dict(size=10), orientation='h'),
                  title=title,
                  xaxis=dict(title=xlabel,
                            rangemode=rangemode),
                  yaxis=dict(title=ylabel,
                            rangemode=rangemode)
                )
    return layout

def plot_data_from_df(df, ref_ws_col, site_ws_col, color=EDFBlue):

    data=df.dropna()

    plot_data = [go.Scattergl(
                    x=data[ref_ws_col],
                    y=data[site_ws_col],
                    mode='markers',
                    name='data',
                    marker=dict(color=color))]
    return plot_data