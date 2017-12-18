import anemoi as an
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

def read_mast_data_from_windographer_csv(filename, skiprows=13, na_values=-999, sensors=None):
    if sensors is None:
        header = skiprows
        skiprows = 0
    else:
        header = None

    data = pd.read_csv(filename, 
                    index_col=0, 
                    infer_datetime_format=True, 
                    parse_dates=True,
                    skiprows=skiprows,
                    header=header, 
                    na_values=na_values)
    data.index.name = 'Stamp'
    if sensors is not None:
        data.columns = sensors
    return data

def read_mast_data_from_parquet_file(filename):
    table = pq.read_table(filename)
    data = table.to_pandas()
    return data

def read_mast_metadata_from_parquet_file(filename):
    table = pq.read_table(filename)
    data = table.to_pandas().T
    return data

def import_mast_from_parquet_files(filename_data, filename_metadata):
    table = pq.read_table(filename_data)
    data = table.to_pandas()
    table = pq.read_table(filename_metadata)
    metadata = table.to_pandas().T
    name = metadata.columns[0]
    lat = metadata.loc['lat', name]
    lon = metadata.loc['lon', name]
    elev = metadata.loc['elev', name]
    height = metadata.loc['height', name]
    primary_ano = metadata.loc['primary_ano', name]
    primary_vane = metadata.loc['primary_vane', name]
    mast = an.MetMast(data=data, 
                  name=name, 
                  lat=lat, 
                  lon=lon, 
                  elev=elev, 
                  height=height, 
                  primary_ano=primary_ano, 
                  primary_vane=primary_vane)
    return mast
