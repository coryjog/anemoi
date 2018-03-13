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

def map_layout(lat, lon, zoom=7):
    mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'

    layout = dict(autosize=False,
                  width=600,
                  height=600,
                  font=dict(color='#CCCCCC'),
                  titlefont=dict(color='#CCCCCC', size='12'),
                  margin=dict(l=0,r=0,b=0,t=25),
                  hovermode="closest",
                  legend=dict(font=dict(size=10), orientation='h'),
                  title='Reference stations',
                  mapbox=dict(accesstoken=mapbox_access_token,
                            center=dict(lon=lon,lat=lat),
                            zoom=zoom,
                            style='satellite')
                )
    return layout

def normalized_rolling_monthly_average_layout():
    layout = go.Layout(showlegend=True,
                    autosize=True,
                    font=dict(size=12),
                    title='Normalized monthly rolling average',
                    height=400,
                    yaxis=dict(title='Annual rolling average'),
                    margin=dict(l=50,r=20,t=25,b=30))
    return layout

def normalized_rolling_monthly_average_figure(normalized_rolling_monthly_averages):
    plotly_data = an.plotting.plotting.plotly_data_by_column(normalized_rolling_monthly_averages)
    layout = normalized_rolling_monthly_average_layout()
    return go.Figure(data=plotly_data, layout=layout)

def map_data_from_references(references):

    reference_traces = []
    for i,network in enumerate(references.network.unique().tolist()):
        references_by_network = references.loc[references.network == network,:]
        labels = list(zip(references_by_network.station_name, [' Distance [km]: ']*references_by_network.shape[0], references_by_network.dist.round(0)))

        reference_traces.append(dict(type='scattermapbox',
                          lon=references_by_network.lon.values,
                          lat=references_by_network.lat.values,
                          text=labels,
                          customdata=network,
                          hoverinfo='text',
                          name=network,
                          marker=dict(size=15,
                                      opacity=1,
                                      color=EDFColors[i])
                        )
                    )
    return reference_traces

def map_data_from_site(lat,lon):

    site_trace = dict(type='scattermapbox',
                      lon=[lon],
                      lat=[lat],
                      text='Site',
                      hoverinfo='text',
                      name='Site',
                      marker=dict(size=30,
                                  opacity=1,
                                  color=EDFGreen)
                    )
    return site_trace

def map_figure(lat, lon, references=None):

    if references is None:
        references = an.io.references.get_proximate_reference_stations_north_america(lat, lon, max_dist=120.0, number_reanalysis_cells_to_keep=4)

    layout = map_layout(lat,lon)
    traces = map_data_from_references(references)
    traces.append(map_data_from_site(lat,lon))
    return go.Figure(data=traces,layout=layout)
