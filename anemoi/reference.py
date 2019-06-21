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
