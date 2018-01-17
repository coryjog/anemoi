import anemoi as an
import pandas as pd
import numpy as np
import scipy as sp
import scipy.odr.odrpack as odrpack
import itertools

def check_and_return_wind_speed_data_for_annual_shear(mast_data, wind_speed_sensors=None):
    '''Perform checks on wind speed data for shear calculations.'''
    
    if wind_speed_sensors is None:
        mast_data = an.utils.mast_data.return_sensor_type_data(mast_data, sensor_type='SPD')
        mast_data = mast_data.loc[:,pd.IndexSlice[:,:,'Avg']]
    else:
        mast_data = an.utils.mast_data.remove_sensor_levels(mast_data)
        mast_data = mast_data.loc[:,wind_speed_sensors]

    heights = an.utils.mast_data.remove_and_add_sensor_levels(mast_data).columns.get_level_values('Ht')
    wind_speed_sensors = mast_data.columns.get_level_values('Sensors').tolist()
    mast_data.columns = mast_data.columns.get_level_values('Sensors').tolist()
    mast_data.columns.names = ['Sensors']
    mast_data = mast_data.dropna()

    return mast_data, wind_speed_sensors, heights

def check_and_return_wind_dir_data_for_shear(mast_data, wind_dir_sensor):
    '''Perform checks on wind direction data for shear calculations.'''

    if wind_dir_sensor is None:
        raise ValueError('Need to specify a wind vane for directional shear calculations')

    wind_dir_data = an.utils.mast_data.remove_sensor_levels(mast_data).loc[:,wind_dir_sensor].to_frame(wind_dir_sensor)
    return wind_dir_data

### SHEAR METHODS - Single Mast ###
def alpha_time_series(mast_data, wind_speed_sensors=None, heights=None):
    '''Returns a time series of alpha values from a time series of wind speeds.
    
    :Parameters: 
    
    mast data: DataFrame
        Measured data from MetMast.data

    wind_speed_sensors: list
        Specific wind speeds sensors 

    heights: list
        List of the specified sensor heights
    
    :Returns:

    out: DataFrame
        Time series of alpha values with the same index as the input mast data 
    
    '''
    mast_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast_data, wind_speed_sensors)
    
    ln_heights = np.log(heights) - np.mean(np.log(heights))
    ln_heights = pd.DataFrame(index=mast_data.index, columns=wind_speed_sensors, data=np.tile(ln_heights, (mast_data.shape[0],1)))
    ln_heights_avg = ln_heights.mean(axis=1)
    ln_heights = ln_heights.sub(ln_heights_avg,axis=0)
    ln_wind_speeds = mast_data.loc[:,wind_speed_sensors].apply(np.log)
    ln_wind_speeds_avg = ln_wind_speeds.mean(axis=1)
    ln_wind_speeds = ln_wind_speeds.sub(ln_wind_speeds_avg,axis=0)
    shear_alpha = (ln_heights*ln_wind_speeds).sum(axis=1) / (ln_heights**2).sum(axis=1)
    shear_alpha = shear_alpha.to_frame(name='Alpha')
    return shear_alpha

def alpha_mean_from_alpha_time_series(mast_data, wind_speed_sensors=None, heights=None):
    '''Returns the mean of monthly means of the alpha time series from wind speed mast data.
    
    :Parameters: 
    
    mast data: DataFrame
        Measured data from MetMast.data

    wind_speed_sensors: list, default all anemometers
        Specific wind speeds sensors 

    heights: list
        List of the specified sensor heights
    
    :Returns:

    out: DataFrame
        Mean of monthly means of an alpha time series 
    
    '''
    mast_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast_data, wind_speed_sensors)
    
    alpha_ts = alpha_time_series(mast_data, wind_speed_sensors=wind_speed_sensors, heights=heights)
    alpha = an.utils.mast_data.return_momm(alpha_ts)
    return alpha

def alpha_annual_profile_from_wind_speed_time_series(mast_data, wind_speed_sensors=None, heights=None):
    '''Returns monthly mean alpha values from a time series of wind speeds.
    
    :Parameters: 
    
    mast data: DataFrame
        Measured data from MetMast.data

    wind_speed_sensors: list, default all anemometers
        Specific wind speeds sensors 

    heights: list
        List of the specified sensor heights
    
    :Returns:

    out: DataFrame
        Mean alpha values indexed by month (annual shear profile) 
    
    '''
    mast_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast_data, wind_speed_sensors)
    
    alpha_ts = alpha_time_series(mast_data, wind_speed_sensors=wind_speed_sensors, heights=heights)
    alpha_profile = alpha_ts.groupby(alpha_ts.index.month).mean()
    alpha_profile.index.name = 'Month'
    return alpha_profile

def alpha_dir_profile_from_wind_speed_time_series(mast_data, wind_dir_sensor, dir_sectors=16, wind_speed_sensors=None):
    '''Returns mean alpha values by direction bin from a time series of wind speeds.
    
    :Parameters: 
    
    mast data: DataFrame
        Measured data from MetMast.data

    wind_dir_sensors: list
        Specific wind wind vane for directional binning 
    
    dir_sectors: int, default 16
        Number of equally spaced direction sectors in which to bin the mean shear values 

    wind_speed_sensors: list, default all anemometers
        Specific wind speeds sensors 

    heights: list
        List of the specified sensor heights
    
    :Returns:

    out: DataFrame
        Mean alpha values indexed by the specified number of direction bins (directional shear profile)
    
    '''
    wind_speed_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast_data, wind_speed_sensors)
    wind_dir_data = check_and_return_wind_dir_data_for_shear(mast_data, wind_dir_sensor=wind_dir_sensor)

    alpha_ts = alpha_time_series(wind_speed_data, wind_speed_sensors=wind_speed_sensors, heights=heights)
    alpha_ts = pd.concat([alpha_ts, wind_dir_data], axis=1).dropna()
    alpha_ts.columns = ['alpha', 'dir']

    dir_bin_ts = an.analysis.wind_rose.append_dir_bin(alpha_ts.dir, dir_sectors=dir_sectors).to_frame('dir_bin')
    alpha_dir_ts = pd.concat([alpha_ts, dir_bin_ts], axis=1).dropna()
    alpha_by_dir = alpha_dir_ts.loc[:,['alpha', 'dir_bin']].groupby('dir_bin').mean()
    
    return alpha_by_dir

def alpha_mean_from_wind_speed_time_series(mast_data, wind_speed_sensors=None, heights=None):
    '''Returns alpha values from the mean of monthly means of a time series of wind speeds.
    
    :Parameters: 
    
    mast data: DataFrame
        Measured data from MetMast.data

    wind_speed_sensors: list, default all anemometers
        Specific wind speeds sensors 

    heights: list
        List of the specified sensor heights
    
    :Returns:

    out: DataFrame
        Alpha value from the mean of monthly means of a wind speed time series 
    
    '''
    mast_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast_data, wind_speed_sensors)
    
    mast_data_mean = an.utils.mast_data.return_momm(mast_data)
    alpha = alpha_time_series(mast_data_mean, wind_speed_sensors=wind_speed_sensors, heights=heights).values[0][0]
    alpha = pd.DataFrame(index=['MoMM'], columns=['Alpha'], data=alpha)
    return alpha

def mast_annual(mast):
    '''Returns a DataFrame of annual alpha values from a single mast, indexed by sensor orientation and height.
    
    :Parameters: 
    
    mast data: DataFrame
        Measured data from MetMast.data

    :Returns:

    out: DataFrame
        Alpha values from a single mast by sensor orientation and height 
    
    '''
    wind_speed_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast.data, wind_speed_sensors=None)
    
    unique_orients = an.utils.mast_data.get_unique_sensor_orients(wind_speed_data)
    
    shear_analysis_mast = []
    for orient in unique_orients:
        heights = wind_speed_data.loc[:,pd.IndexSlice[:,:,orient]].columns.get_level_values('Ht').unique()

        alphas = pd.DataFrame(index=heights, columns=heights)
        for height_combination in itertools.combinations(heights,2):
            ws_shear_data_height_orient = wind_speed_data.loc[:,pd.IndexSlice[:,height_combination,orient]]
            alpha = alpha_mean_from_wind_speed_time_series(ws_shear_data_height_orient)
            alphas.loc[height_combination[0], height_combination[1]] = alpha.loc['MoMM', 'Alpha']
        
        shear_analysis_mast.append(alphas)
    
    shear_analysis_mast = pd.concat(shear_analysis_mast, axis=0, keys=unique_orients, names=['Orient', 'Ht']).dropna(how='all')
    shear_analysis_mast = shear_analysis_mast.dropna(axis=1, how='all')

    return shear_analysis_mast

def mast_directional(mast, wind_dir_sensor=None, dir_sectors=16, wind_speed_sensors=None):
    '''Returns a DataFrame of annual alpha values from a single mast, indexed by direction bin.
    Alpha only calcualted for time steps with valid measurements from each wind speed sensor. 
    
    :Parameters: 
    
    masts : list
        List of MetMast objects from which all anemometer data is extracted

    wind_dir_sensors: list, default mast.primary_vane
        Specific wind wind vane for directional binning 
    
    dir_sectors: int, default 16
        Number of equally spaced direction sectors in which to bin the mean shear values 

    wind_speed_sensors: list, default all anemometers
        Specific wind speeds sensors 

    :Returns:

    out: DataFrame
        Mean alpha values indexed by the specified number of direction bins (directional shear profile)
    
    '''
    wind_speed_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast.data, wind_speed_sensors=wind_speed_sensors)
    
    if wind_dir_sensor is None:
        wind_dir_sensor = mast.primary_vane
    
    wind_dir_data = check_and_return_wind_dir_data_for_shear(mast.data, wind_dir_sensor=wind_dir_sensor)
    mast_data = pd.concat([wind_speed_data, wind_dir_data], axis=1).dropna()
    
    shear_analysis_mast = alpha_dir_profile_from_wind_speed_time_series(mast_data, wind_dir_sensor, 
                                                                        dir_sectors=dir_sectors, 
                                                                        wind_speed_sensors=wind_speed_sensors)

    return shear_analysis_mast

def site_annual(masts):
    '''Returns a DataFrame of annual alpha values from a multiple site masts, indexed by mast, sensor orientation, and height.
    
    :Parameters: 
    
    masts : list
        List of MetMast objects from which all anemometer data is extracted

    :Returns:

    out: DataFrame
        Alpha values from multiple site masts by mast, sensor orientation, and height 
    
    '''
    shear_analysis_site = []
    mast_names = []
    for mast in masts:
        mast_names.append(mast.name)
        shear_analysis_site.append(mast_annual(mast))
    
    shear_analysis_site = pd.concat(shear_analysis_site, axis=1, keys=mast_names)
    shear_analysis_site.columns.names = ['Mast', 'Ht']
    shear_analysis_site = shear_analysis_site.dropna(axis=1, how='all')
    
    return shear_analysis_site

def site_directional(masts, dir_sectors=16):
    '''Returns a DataFrame of annual alpha values from a single mast, indexed by direction bin.
    Alpha only calcualted for time steps with valid measurements from each wind speed sensor. 
    
    :Parameters: 
    
    masts : list
        List of MetMast objects from which all anemometer data is extracted

    dir_sectors: int, default 16
        Number of equally spaced direction sectors in which to bin the mean shear values 

    :Returns:

    out: DataFrame
        Mean alpha values for each mast indexed by the specified number of direction bins (directional shear profile)
    
    '''
    shear_analysis_site = []
    mast_names = []
    for mast in masts:
        mast_names.append(mast.name)
        shear_analysis_site.append(mast_directional(mast))
    
    shear_analysis_site = pd.concat(shear_analysis_site, axis=1)
    shear_analysis_site.columns = mast_names
    shear_analysis_site.columns.names = ['Mast']
    shear_analysis_site = shear_analysis_site.dropna(axis=1, how='all')
    
    return shear_analysis_site

def site_mean(masts):
    '''Returns a DataFrame of the mean annual alpha value from each site masts. Uses all avaialble anemometer combinations.
    
    :Parameters: 
    
    masts : list
        List of MetMast objects from which all anemometer data is extracted

    :Returns:

    out: DataFrame
        Average annual alpha values from each site mast using all available anemometer combinations 
    
    '''
    shear_results = shear_analysis_site(masts)
    shear_results = shear_results.T.unstack().mean(axis=1).to_frame('Alpha')
    return shear_results

def site_mean_from_results(shear_results):
    '''Returns a DataFrame of the mean annual alpha value from each site mast from a previously run shear analysis.
    This allows the user to choose the heights and oreintations used within the final calculated alpha value.
    
    :Parameters: 
    
    shear results : DataFrame
        DataFrame of shear results from shear.shear_analysis_annual or shear.shear_analysis_site

    :Returns:

    out: DataFrame
        Average annual alpha values from each site mast using all the provided anemometer combinations 
    
    '''
    shear_results = shear_results.T.unstack().replace('-', np.nan).mean(axis=1).to_frame('Alpha')
    return shear_results
