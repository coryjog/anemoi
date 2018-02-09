import pandas as pd
import numpy as np
from datetime import datetime
import os.path
import requests

def readslice(bin_filename,nts,timeslice):
    # inputfilename: binary file name
    # nts: number time steps packed for 1-year (8760 or 8784-leap year)
    # timeslice: cell-id
    file = open(bin_filename,'rb')
    file.seek(2*(timeslice-1)*nts)
    field = np.fromfile(file,dtype='int16',count=nts)
    file.close()
    return field

def closest_merra2_cell_id(lat,lon):
    icol = int((lon + 180.0)/0.625 + .5) + 1 # 0.625 - 1st col at -180 (lon), range 1-576.
    irow = int((lat + 90.0 )/0.500 + .5) + 1 # 0.500 - 1st raw at -90  (lat), range 1-361.
    if lon <= 180 and icol == 577:           # first cell from [179.6875,-179.6875) with center at -180.0
        icol = 1
    cell_id = -999
    if icol >= 1 and icol <= 576 and irow >=1 and irow <= 361:
        cell_id = (irow - 1) * 576 + icol
    return cell_id

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
        timezone_hour = 0.0 # GMT will be used
    return timezone_hour 

def get_closest_merra2_data(lat, lon, daily_only=True, local_time=True):
    
    if daily_only:
        results = create_empty_time_series_to_fill(freq='D')
    else:
        results = create_empty_time_series_to_fill(freq='H')

    sttYr = results.index[0].year    # start year
    endYr = results.index[-1].year    # end year

    cell_id = closest_merra2_cell_id(lat,lon)   
    if cell_id == -999:
        return results

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
    results = results.dropna(axis=1, how='all')
    
    if local_time:
        timezone_hour = get_local_timezone_from_google(lat=lat, lon=lon)
        results.index = results.index + pd.Timedelta(timezone_hour, unit='h')
    
    return results