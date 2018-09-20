import anemoi as an
import pandas as pd
import numpy as np
import os
from datetime import datetime
import requests

def get_reference_stations_north_america():
    '''Return list of North American reference stations'''
    
    filename = 'https://raw.githubusercontent.com/coryjog/anemoi/master/anemoi/io/reference_stations_NA.csv'
    references = pd.read_csv(filename, encoding='windows-1252')
    return references

def distances_to_project(lat_project, lon_project, lats, lons):
    '''Method to calculate distances between a project and an array of lats and lons

    :Parameters:

    lat_project: float
        Project latitude

    lon_project: float
        Project longitude

    lats: np.array
        Latitudes from which to calculate distances

    lons: np.array
        Longitudes from which to calculate distances

    :Returns:

    out: np.array of distances
    '''
    lat_project = np.deg2rad(lat_project)
    lon_project = np.deg2rad(lon_project)
    avg_earth_radius = 6373  # in km
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    lat = lat_project - lats
    lon = lon_project - lons
    d = np.sin(lat * 0.5)**2 + np.cos(lat_project) * np.cos(lats) * np.sin(lon * 0.5)**2
    dist = 2 * avg_earth_radius * np.arcsin(np.sqrt(d))
    return dist

def filter_references_for_top_reanalysis(references, number_reanalysis_cells_to_keep=6):
    reanalysis_networks = ['CSFR','ERAI','ERA5','MERRA2']

    proximate_references = []
    [proximate_references.append(references.loc[references.network == network,:].iloc[0:number_reanalysis_cells_to_keep,:]) for network in reanalysis_networks]
    proximate_references.append(references.loc[~references.network.isin(reanalysis_networks),:])
    proximate_references = pd.concat(proximate_references, axis=0)
    return proximate_references

def get_proximate_reference_stations_north_america(lat_project, lon_project, max_dist=120.0, number_reanalysis_cells_to_keep=None):
    references = get_reference_stations_north_america()
    references['dist'] = distances_to_project(lat_project, lon_project, references.lat, references.lon)
    references = references.loc[references.dist < max_dist, :]
    references = references.sort_values(by=['network', 'dist'])

    if number_reanalysis_cells_to_keep is not None:
        references = filter_references_for_top_reanalysis(references, number_reanalysis_cells_to_keep=number_reanalysis_cells_to_keep)

    return references

### MERRA2 DATA ###
def readslice(bin_filename,records_per_year,cell_id):
    # inputfilename: binary file name
    # records_per_year: number time steps packed for 1-year (8760 or 8784-leap year)
    # cell_id: int
    file = open(bin_filename,'rb')
    file.seek(2*(cell_id-1)*records_per_year)
    field = np.fromfile(file,dtype='int16',count=records_per_year)
    file.close()
    return field

def closest_merra2_cell_id(lat,lon):
    icol = int((lon + 180.0)/0.625 + .5) + 1 # 0.625 - 1st col at -180 (lon), range 1-576.
    irow = int((lat + 90.0 )/0.500 + .5) + 1 # 0.500 - 1st raw at -90  (lat), range 1-361.
    
    cell_id = -999
    if lon <= 180 and icol == 577:           # first cell from [179.6875,-179.6875) with center at -180.0
        icol = 1
    if icol >= 1 and icol <= 576 and irow >=1 and irow <= 361:
        cell_id = (irow - 1) * 576 + icol
    return cell_id

def closest_era5_cell_id(lat,lon):
    cell_id = -999
    if lon < 0: 
        lon = lon + 360.0                    # 0 - 360
    icol = int(lon/0.3 + 0.5) + 1            # 0.3, range 1-1200.
    irow = int((90.0 - lat)/0.3+ 0.5) + 1    # 0.3, range 1-601.
    if lon <= 360 and icol == 1201:          # first cell from [359.85,0.15) with centre at 0.0 
        icol = 1
    if icol >= 1 and icol <= 1200 and irow >=1 and irow <= 601:
        cell_id = (irow - 1) * 1200 + icol
    return cell_id

def closest_cfsr_cell_id(lat,lon):
    if lon < 0:
        lon = lon + 360.0                    # 0 - 360
    icol = int(lon/0.5 + .5) + 1             # 0.5, range 1-720.
    irow = int((lat + 90.0 )/0.5 + .5) + 1   # 0.5, range 1-361.
    if lon <= 360 and icol == 721:           # first cell from [359.75,0.25) with centre at 0.0
        icol = 1  
    if icol >= 1 and icol <= 720 and irow >=1 and irow <= 361:
        cell_id = (irow - 1) * 720 + icol

def create_empty_time_series_to_fill(freq):
    year = datetime.now().year
    dates = pd.date_range('2000-01-01 00:00', '%s-12-31 23:00' %year, freq=freq)
    empty_ref_data = pd.DataFrame(index=dates, columns=['spd', 'dir', 't'])
    empty_ref_data.index.name = 'Stamp'
    return empty_ref_data

def get_local_timezone_from_google(lat, lon):
    Google_URL = 'https://maps.googleapis.com/maps/api/timezone/json?location={0},{1}&timestamp={2}&language=en&esnsor=false'
    timezone_response = requests.get(Google_URL.format(lat,lon,1))
    timezone_response_dict = timezone_response.json()

    if timezone_response_dict['status'] == 'OK':
        timezone_hour = timezone_response_dict['rawOffset'] / 3600.0 # in hours
    else:
        timezone_hour = int((lon + 360.0)*24.0/360.0 + 0.5) - 24.0 # Estimate from longitude
    return timezone_hour

def get_merra2_data_from_cell_id(cell_id, lat=None, lon=None, daily_only=True, local_time=True):
    '''Returns hourly data from MERRA2 cell using EDF's cell_id

    :Parameters:

    cell_id: int
        Unique cell id

    lat: float
        Latitude value - used to infer local time

    lon: float
        Longitude value - used to infer local time

    local_time: boolean, default: True
        Return data in local time, as opposed to UTC

    :Returns:

    out: pd.DataFrame of MERRA2 data
    '''

    if local_time & ((lat is None) | (lon is None)):
        raise ValueError('Trying to convert MERRA2 data to local time without latitude and/or longitude.')

    if daily_only:
        results = create_empty_time_series_to_fill(freq='D')
    else:
        results = create_empty_time_series_to_fill(freq='H')

    if cell_id == -999:
        return results

    sttYr = results.index[0].year    # start year
    endYr = results.index[-1].year

    for year in results.index.year.unique():

        dtEnd=datetime(year,12,31,0,0)             # end of year

        filenames = []
        if daily_only:
            nSize=dtEnd.timetuple().tm_yday        # days in for given year substitute np.sum(results.index.year==year)
            filenames.append('//sdhqragarch01/RAG_Archive/UserArchive/Z_Yiping/MERRA2/BIN/%i_spd50m_dd.bin' %year)
        else:
            nSize=dtEnd.timetuple().tm_yday*24         # hours in given year substitute np.sum(results.index.year==year)
            filenames.append('//sdhqragarch01/RAG_Archive/UserArchive/Z_Yiping/MERRA2/BIN/%i_spd80m.bin' %year)
            filenames.append('//sdhqragarch01/RAG_Archive/UserArchive/Z_Yiping/MERRA2/BIN/%i_dir50m.bin' %year)
            filenames.append('//sdhqragarch01/RAG_Archive/UserArchive/Z_Yiping/MERRA2/BIN/%i_tmp10m.bin' %year)

        for i, filename in enumerate(filenames):
            if os.path.isfile(filename):
                results.iloc[results.index.year == year, i] = readslice(filename,nSize,cell_id)

    results.spd = results.spd * 0.01
    results.t = results.t * 0.1 - 273.15
    results = results.replace(-999, np.nan).dropna()
    results = results.dropna(axis=1, how='all')

    if local_time:
        timezone_hour = get_local_timezone_from_google(lat=lat, lon=lon)
        results.index = results.index + pd.Timedelta(timezone_hour, unit='h')

    return results

def get_era5_data_from_cell_id(cell_id, lat=None, lon=None, local_time=True, daily_only=False):
    '''Returns hourly data from ERA5 cell using EDF's cell_id

    :Parameters:

    cell_id: int
        Unique cell id

    lat: float
        Latitude value - used to infer local time

    lon: float
        Longitude value - used to infer local time

    local_time: boolean, default: True
        Return data in local time, as opposed to UTC

    :Returns:

    out: pd.DataFrame of ERA5 data
    '''

    if local_time & ((lat is None) | (lon is None)):
        raise ValueError('Trying to convert ERA-5 data to local time without latitude and/or longitude.')

    if daily_only:
        results = create_empty_time_series_to_fill(freq='D')
    else:
        results = create_empty_time_series_to_fill(freq='H')

    if cell_id == -999:
        return results

    results = results.loc['2000-01-01':,:]
    sttYr = results.index[0].year    # start year
    endYr = results.index[-1].year

    for year in results.index.year.unique():

        dtEnd=datetime(year,12,31,0,0)             # end of year

        filenames = []
        if daily_only:
            nSize=dtEnd.timetuple().tm_yday        # days in for given year substitute np.sum(results.index.year==year)
            filenames.append('//sdhqfile03.enxco.com/arcgis/MetData/DataLibrary/ERA5/Bin/%i_spd100m.bin' %year)
        else:
            nSize=dtEnd.timetuple().tm_yday*24         # hours in given year substitute np.sum(results.index.year==year)
            filenames.append('//sdhqfile03.enxco.com/arcgis/MetData/DataLibrary/ERA5/Bin/%i_spd100m.bin' %year)
            filenames.append('//sdhqfile03.enxco.com/arcgis/MetData/DataLibrary/ERA5/Bin/%i_dir100m.bin' %year)
            filenames.append('//sdhqfile03.enxco.com/arcgis/MetData/DataLibrary/ERA5/Bin/%i_tmp02m.bin' %year)

        for i, filename in enumerate(filenames):
            if os.path.isfile(filename):
                results.iloc[results.index.year == year, i] = readslice(filename,nSize,cell_id)

    results.spd = results.spd * 0.01
    results.t = results.t * 0.1 - 273.15
    results = results.replace(-999, np.nan).dropna()
    results = results.dropna(axis=1, how='all')

    if local_time:
        timezone_hour = get_local_timezone_from_google(lat=lat, lon=lon)
        results.index = results.index + pd.Timedelta(timezone_hour, unit='h')

    return results

def get_closest_merra2_data(lat, lon, local_time=True, daily_only=False):
    '''Returns hourly data from closest MERRA2 cell

    :Parameters:

    lat: float
        Latitude value

    lon: float
        Longitude value

    local_time: boolean, default: True
        Return data in local time, as opposed to UTC

    :Returns:

    out: pd.DataFrame of MERRA2 data
    '''

    cell_id = closest_merra2_cell_id(lat,lon)
    results = get_merra2_data_from_cell_id(cell_id, lat=lat, lon=lon, daily_only=daily_only, local_time=local_time)
    return results

def get_closest_era5_data(lat, lon, local_time=True, daily_only=False):
    '''Returns hourly data from closest ERA5 cell

    :Parameters:

    lat: float
        Latitude value

    lon: float
        Longitude value

    local_time: boolean, default: True
        Return data in local time, as opposed to UTC

    :Returns:

    out: pd.DataFrame of ERA5 data
    '''

    cell_id = closest_era5_cell_id(lat,lon)
    results = get_era5_data_from_cell_id(cell_id, lat=lat, lon=lon, daily_only=daily_only, local_time=local_time)
    return results