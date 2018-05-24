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
        mast_data = mast_data.loc[:,pd.IndexSlice[:,:,:,'AVG']]
    else:
        mast_data = an.utils.mast_data.remove_sensor_levels(mast_data)
        mast_data = mast_data.loc[:,wind_speed_sensors]

    heights = an.utils.mast_data.remove_and_add_sensor_levels(mast_data).columns.get_level_values('height')
    wind_speed_sensors = mast_data.columns.get_level_values('sensor').tolist()
    mast_data.columns = mast_data.columns.get_level_values('sensor').tolist()
    mast_data.columns.names = ['sensor']
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
    shear_alpha = shear_alpha.to_frame(name='alpha')
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
    
    mast: an.MetMast
        Measured data from MetMast.data

    :Returns:

    out: DataFrame
        Alpha values from a single mast by sensor orientation and height 
    
    '''
    wind_speed_data, wind_speed_sensors, heights = check_and_return_wind_speed_data_for_annual_shear(mast.data, wind_speed_sensors=None)
    
    unique_orients = an.utils.mast_data.get_unique_sensor_orients(wind_speed_data)
    
    shear_analysis_mast = []
    for orient in unique_orients:
        heights = wind_speed_data.loc[:,pd.IndexSlice[:,:,orient]].columns.get_level_values('height').unique()

        alphas = pd.DataFrame(index=heights, columns=heights)
        for height_combination in itertools.combinations(heights,2):
            ws_shear_data_height_orient = wind_speed_data.loc[:,pd.IndexSlice[:,height_combination,orient]]
            alpha = alpha_mean_from_wind_speed_time_series(ws_shear_data_height_orient)
            alphas.loc[height_combination[0], height_combination[1]] = alpha.loc['MoMM', 'Alpha']
        
        shear_analysis_mast.append(alphas)
    
    shear_analysis_mast = pd.concat(shear_analysis_mast, axis=0, keys=unique_orients, names=['orient', 'height']).dropna(how='all')
    shear_analysis_mast = shear_analysis_mast.dropna(axis=1, how='all')

    return shear_analysis_mast

def mast_annual_avg(mast):
    '''Returns a DataFrame of an annual alpha value from a single mast, indexed by mast name.
    
    :Parameters: 
    
    mast: an.MetMast
        Measured data from MetMast.data

    :Returns:

    out: DataFrame
        Average alpha value from a single mast. 
    
    '''
    annual_avg_alpha = mast_annual(mast=mast).stack().mean()
    annual_avg_alpha = pd.DataFrame(index=[mast.name], columns=['Alpha'], data=annual_avg_alpha)
    annual_avg_alpha.index.name = 'Mast'
    return annual_avg_alpha

def mast_directional(mast, wind_dir_sensor=None, dir_sectors=16, wind_speed_sensors=None):
    '''Returns a DataFrame of annual alpha values from a single mast, indexed by direction bin.
    Alpha only calcualted for time steps with valid measurements from each wind speed sensor. 
    
    :Parameters: 

    mast: an.MetMast
        Measured data from MetMast.data

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

    mast.remove_and_add_sensor_levels()
    return shear_analysis_mast

def mast_directional_by_orient(mast, wind_dir_sensor=None, dir_sectors=16):
    '''Returns a DataFrame of annual alpha values from a single mast, indexed by direction bin.
    Alpha only calcualted for time steps with valid measurements from each wind speed sensor. 
    
    :Parameters: 
    
    mast: an.MetMast
        Measured data from MetMast.data

    wind_dir_sensors: list, default mast.primary_vane
        Specific wind wind vane for directional binning 
    
    dir_sectors: int, default 16
        Number of equally spaced direction sectors in which to bin the mean shear values 

    :Returns:

    out: DataFrame
        Mean alpha values indexed by the specified number of direction bins (directional shear profile)
    
    '''
    anemometers = mast.data.loc[:,pd.IndexSlice['SPD',:,:,'AVG',:]].columns.get_level_values(level='sensor').tolist()

    anemometer_data = mast.return_sensor_data(sensors=anemometers)
    anemometer_orients = sorted(anemometer_data.columns.get_level_values(level='orient').unique().tolist())

    alpha_by_dir = []
    for anemometer_orient in anemometer_orients:
        anemometers = anemometer_data.loc[:,pd.IndexSlice[:,:,anemometer_orient]].columns.get_level_values(level='sensor').tolist()
        alpha_by_dir.append(mast_directional(mast=mast, 
                                            wind_dir_sensor=wind_dir_sensor, 
                                            dir_sectors=dir_sectors,
                                            wind_speed_sensors=anemometers))
        
    alpha_by_dir = pd.concat(alpha_by_dir, axis=1, keys=anemometer_orients, names=['orient', 'alpha'])
    alpha_by_dir.columns = alpha_by_dir.columns.droplevel(level='alpha')
    alpha_by_dir.index = alpha_by_dir.index.values * 360.0/dir_sectors
    alpha_by_dir.loc[0.0,:] = alpha_by_dir.loc[360.0,:]
    alpha_by_dir = alpha_by_dir.sort_index()
    return alpha_by_dir

def mast_monthly_by_orient(mast):
    '''Returns a DataFrame of monthly time series of alpha values from a single mast for each sensor orientation.
    Alpha only calcualted for time steps with valid measurements from each wind speed sensor. 
    
    :Parameters: 
    
    mast: an.MetMast
        Measured data from MetMast.data

    :Returns:

    out: DataFrame
        Mean alpha values for each sensor orientation, indexed by month
    
    '''
    anemometers = mast.data.loc[:,pd.IndexSlice['SPD',:,:,'AVG',:]].columns.get_level_values(level='sensor').tolist()
    anemometer_data = mast.return_sensor_data(sensors=anemometers)
    anemometer_orients = sorted(anemometer_data.columns.get_level_values(level='orient').unique().tolist())

    alpha_ts_by_orient = []
    for anemometer_orient in anemometer_orients:
        anemometer_data = an.utils.mast_data.remove_and_add_sensor_levels(anemometer_data)
        anemometers = anemometer_data.loc[:,pd.IndexSlice[:,:,anemometer_orient]].columns.get_level_values(level='sensor').tolist()
        alpha_ts = an.analysis.shear.alpha_time_series(anemometer_data, wind_speed_sensors=anemometers)
        alpha_ts_by_orient.append(alpha_ts)
        
    alpha_ts_by_orient = pd.concat(alpha_ts_by_orient, axis=1, keys=anemometer_orients, names=['orient', 'alpha'])
    alpha_ts_by_orient.columns = alpha_ts_by_orient.columns.droplevel(level='alpha')
    monthly_alpha_ts_by_orient = alpha_ts_by_orient.resample('MS').mean()
    return monthly_alpha_ts_by_orient

def mast_annual_profile_by_orient(mast):
    '''Returns a DataFrame of annual alpha profiles from a single mast for each sensor orientation.
    
    :Parameters: 
    
    mast: an.MetMast
        Measured data from MetMast.data

    :Returns:

    out: DataFrame
        Annual alpha profiles for each sensor orientation, indexed by month
    
    '''
    monthly_alpha_ts_by_orient = mast_monthly_by_orient(mast)
    annual_alpha_profiles_by_orient = monthly_alpha_ts_by_orient.groupby([monthly_alpha_ts_by_orient.index.year, monthly_alpha_ts_by_orient.index.month]).mean()
    annual_alpha_profiles_by_orient.index.names = ['year', 'month']
    annual_alpha_profiles_by_orient = annual_alpha_profiles_by_orient.unstack(level='year')
    return annual_alpha_profiles_by_orient

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
    shear_analysis_site.columns.names = ['Mast', 'height']
    shear_analysis_site = shear_analysis_site.dropna(axis=1, how='all')
    
    return shear_analysis_site

def site_annual_avg(masts):
    '''Returns a DataFrame of annual alpha values from multiple site masts, indexed by mast.
    
    :Parameters: 
    
    masts : list
        List of MetMast objects from which all anemometer data is extracted

    :Returns:

    out: DataFrame
        Alpha values from multiple site masts, indexed by mast 
    
    '''
    annual_avg_alpha = site_annual(masts).stack().mean().to_frame('Alpha')
    return annual_avg_alpha

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
