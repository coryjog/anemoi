import pandas as pd
import numpy as np
import requests
# import HrReanalysis as HR # Fortran data scrub routine

def create_empty_time_series_to_fill():
    dates = pd.date_range('2000-01-01 00:00', '2017-12-31 23:00', freq='H')
    ref_data = pd.DataFrame(index=dates, columns=['Ref'])
    ref_data.index.name = 'Stamp'
    return ref_data    

def get_local_timezone_from_google(lat, lon):
    Google_URL = 'https://maps.googleapis.com/maps/api/timezone/json?location={0},{1}&timestamp={2}&language=en&esnsor=false'
    timezone_response = requests.get(Google_URL.format(lat,lon,1))
    timezone_response_dict = timezone_response.json()

    if timezone_response_dict['status'] == 'OK':
        timezone_hour = timezone_response_dict['rawOffset'] / 3600.0 # in hours
    else:
        timezone_hour = 0.0 # GMT will be used
    return timezone_hour 

def get_ref_data(lat,lon):
    ref_data = create_empty_time_series_to_fill()
    timezone_hour = get_local_timezone_from_google(lat=lat, lon=lon)
    
    SttYr = ref_data.index[0].year       # start year
    EndYr = ref_data.index[-1].year       # end year
    nCell=1                      # interested reanalysis cell
    iRATYP=1                     # 1/2/3/4/5 merra2/merra/cfsr/era-i/jra55
    nDim1  = ref_data.shape[0] # hourly time step
    nDim2  = nCell*2             # wind speed/direction
    iData = np.zeros(shape=(nDim1,nDim2),dtype=np.int)   # integer for large data passing
    dGeo  = np.zeros(shape=(nDim2,2)    ,dtype=np.float) # float with coordinate

    # Extract data from G: drive
    iData, dGeo = HR.hrreanalysis(lat,lon,SttYr,EndYr,iRATYP,nCell,nDim1,nDim2,timezone_hour)

    # Fill data frame
    ref_data['Ref'] = iData[:,0]/100.0
    
    # Trim trailing zeros from time series
    start = ref_data.index[0]
    end = ref_data.Ref[ref_data.Ref != 0].index[-1]
    ref_data = ref_data.loc[start:end, :]
    ref_data.columns = pd.MultiIndex.from_tuples([(lat, lon)])
    ref_data.columns.names = ['Lat', 'Long']
    return ref_data