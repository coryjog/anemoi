import anemoi as an
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import scipy.odr.odrpack as odrpack

import warnings

### TODO: Need to rewrite so that all methods assume a DataFrame of columns: ref, site, dir ###
### TODO: Need to rewrite so that all methods output a DataFrame of columns: slope, offset, R2, uncert, points ###

def compare_sorted_df_columns(cols_1, cols_2):
    return sorted(cols_1) == sorted(cols_2)

def valid_correlation_data(data):
    '''Perform checks on correlation data.''' 
    if not compare_sorted_df_columns(data.columns.tolist(), ['ref', 'site', 'dir']):
        raise ValueError("Error: the correlation data don't match the expected format.")
        return False

    if not data.shape[0] > 6:
        warnings.warn("Warning: you are trying to correalate between less than six points.")
        return False

    if (data.ref == data.site).sum() == data.shape[0]:
        warnings.warn("Warning: it seems you are trying to correalate a single mast against itself.")
        return False

    return True

def return_correlation_results_frame():
    results = pd.DataFrame(columns=['slope', 'offset' , 'R2', 'uncert', 'points'],
                            index=pd.MultiIndex.from_tuples([('ref', 'site')], 
                                names=['ref', 'site'])
                            )
    return results

def return_correlation_data_from_masts(ref_mast, site_mast):
    '''Return a DataFrame of reference and site data for correlations. 
    Will be extracted from each MetMast object using the primary anemometers and wind vanes. 
    
    :Parameters: 
    
    ref_mast: MetMast
        Anemoi MetMast object

    site_mast: MetMast
        Anemoi MetMast object

    :Returns:

    out: DataFrame with columns ref, site, and dir
    
    '''
    ref_data = ref_mast.return_primary_ano_vane_data()
    ref_data.columns = ['ref', 'dir']
    site_data = site_mast.return_primary_ano_vane_data()
    site_data.columns = ['site', 'site_dir']
    data = pd.concat([ref_data, site_data.site], axis=1).dropna()
    data = data.loc[:, ['ref', 'site', 'dir']]
    
    if not valid_correlation_data(data):
        warning_string = "Warning: {} and {} don't seem to have valid concurrent data for a correlation.".format(ref_mast.name, site_mast.name)
        warnings.warn(warning_string)
    return data

### CORRELATION METHODS ###
def calculate_R2(data):
    '''Return a single R2 between two wind speed columns'''
    if not valid_correlation_data(data=data):
        return np.nan
    
    return data.loc[:,['ref', 'site']].corr().iloc[0,1]**2

def calculate_IEC_uncertainty(data):
    '''Calculate the IEC correlation uncertainty between two wind speed columns'''
    if not valid_correlation_data(data):
        return np.nan

    X = data.ref.values
    Y = data.site.values
    uncert = np.std(Y/X)*100/len(X)
    return uncert*100.0
    
def calculate_EDF_uncertainty(data):
    '''Calculate the EDF estimated correaltion uncetianty between two wind speed columns. 
    Assumes a correalation forced through the origin'''
    
    if not valid_correlation_data(data):
        return np.nan

    Sxx = np.sum(data.ref**2)
    Syy = np.sum(data.site**2)
    Sxy = np.sum(data.ref*data.site)
    B = 0.5*(Sxx - Syy)/Sxy
    SU = -B + np.sqrt(B**2 + 1)

    e2 = np.sum((data.site - SU*data.ref)**2)/(1 + SU**2)
    Xsi2 = e2/(data.shape[0] - 1)
    uncert = np.sqrt((Xsi2*SU**2)*(Sxx*Sxy**2 + 0.25*((Sxx - Syy)**2)*Sxx)/((B**2 + 1.0)*Sxy**4))
    return uncert

def ws_correlation_least_squares_model(data, force_through_origin=False):    
    '''Calculate the slope and offset between two wind speed columns using ordinary least squares regression.
    
    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html

    :Parameters: 
    
    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir 

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    results = return_correlation_results_frame()

    if not valid_correlation_data(data):
        return results

    points = data.shape[0]
    R2 = calculate_R2(data)
    uncert = calculate_IEC_uncertainty(data)
    
    if force_through_origin:
        data.loc[:,'offset'] = 0
    else:
        data.loc[:,'offset'] = 1
    
    X = data.loc[:, ['ref','offset']].values
    Y = data.loc[:, 'site'].values
    slope, offset = np.linalg.lstsq(X, Y)[0]
    results.loc[pd.IndexSlice['ref', 'site'],['slope', 'offset' , 'R2', 'uncert', 'points']] = np.array([slope, offset, R2, uncert, points])
    return results

def f_with_offset(B, x):
    return B[0]*x + B[1]

def f_without_offset(B, x):
    return B[0]*x

def ws_correlation_orthoginal_distance_model(data, force_through_origin=False):
    '''Calculate the slope and offset between two wind speed columns using orthoganal distance regression.
    
    https://docs.scipy.org/doc/scipy-0.18.1/reference/odr.html

    :Parameters: 
    
    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir 

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    results = return_correlation_results_frame()

    if not valid_correlation_data(data):
        return results

    points = data.shape[0]
    R2 = calculate_R2(data)
    uncert = calculate_IEC_uncertainty(data)
    
    X = data.ref.values
    Y = data.site.values

    realdata = odrpack.RealData(X, Y)

    if force_through_origin:
        linear = odrpack.Model(f_without_offset)
        odr = odrpack.ODR(realdata, linear, beta0=[1.0])
        slope = odr.run().beta[0]
        offset = 0
    else:
        linear = odrpack.Model(f_with_offset)
        odr = odrpack.ODR(realdata, linear, beta0=[1.0, 0.0])
        slope, offset = odr.run().beta[0], odr.run().beta[1]
    
    results.loc[pd.IndexSlice['ref', 'site'],['slope', 'offset' , 'R2', 'uncert', 'points']] = np.array([slope, offset, R2, uncert, points])
    return results
            
def ws_correlation_robust_linear_model(data, force_through_origin=False):
    '''Calculate the slope and offset between two wind speed columns using robust linear model.
    
    http://www.statsmodels.org/dev/rlm.html

    :Parameters: 
    
    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir 

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    results = return_correlation_results_frame()

    if not valid_correlation_data(data):
        return results    

    points = data.shape[0]
    R2 = calculate_R2(data)
    uncert = calculate_IEC_uncertainty(data)
    
    X = data.ref.values
    Y = data.site.values
    
    if not force_through_origin:
        X = sm.add_constant(X)
    else:
        X = [np.zeros(X.shape[0]), X]
        X = np.column_stack(X)
    
    mod = sm.RLM(Y, X)
    resrlm = mod.fit()
    offset, slope = resrlm.params
    R2 = sm.WLS(mod.endog, mod.exog, weights=mod.fit().weights).fit().rsquared
    results.loc[pd.IndexSlice['ref', 'site'],['slope', 'offset' , 'R2', 'uncert', 'points']] = np.array([slope, offset, R2, uncert, points])
    return results

def ws_correlation_method(data, method='ODR', force_through_origin=False):
    '''Calculate the slope and offset, for a given correlation method, between two wind speed columns.

    :Parameters: 
    
    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir 

    method: string, default 'ODR'
        Correlation method to use. 

        * Orthoginal distance regression: 'ODR'
        * Ordinary least squares: 'OLS'
        * Robust linear models: 'RLM'
    
    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    if method == 'ODR':
        results = ws_correlation_orthoginal_distance_model(data, force_through_origin=force_through_origin)
    elif method == 'OLS': 
        results = ws_correlation_least_squares_model(data, force_through_origin=force_through_origin)
    elif method == 'RLM':
        results = ws_correlation_robust_linear_model(data, force_through_origin=force_through_origin)

    return results

def ws_correlation_binned_by_direction(data, dir_sectors=16, method='ODR', force_through_origin=False):
    '''Calculate the slope and offset, binned by direction, between two wind speed columns.

    :Parameters: 
    
    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir 

    dir_sectors: int, default 16
        Number of equally spaced direction sectors

    method: string, default 'ODR'
        Correlation method to use. 

        * Orthoginal distance regression: 'ODR'
        * Ordinary least squares: 'OLS'
        * Robust linear models: 'RLM'
    
    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    results = return_correlation_results_frame()

    if not valid_correlation_data(data):
        return results

    dir_bins = np.arange(1,dir_sectors+1)
    results = pd.concat([results]*dir_sectors, axis=0)
    results.index = pd.Index(dir_bins, name='dir_bin')

    data['dir_bin'] = an.analysis.wind_rose.append_dir_bin(data.dir, dir_sectors=dir_sectors)
    
    for dir_bin in dir_bins:
        dir_bin_data = data.loc[data['dir_bin']==dir_bin, ['ref','site', 'dir']]
        points = dir_bin_data.shape[0]
        
        if not valid_correlation_data(dir_bin_data):
            results.loc[dir_bin, 'points'] = points

        else:
            uncert = calculate_IEC_uncertainty(dir_bin_data)
    
            dir_bin_results = ws_correlation_method(dir_bin_data, method=method, force_through_origin=force_through_origin)
            results.loc[dir_bin, ['slope', 'offset', 'R2' , 'uncert', 'points']] = dir_bin_results.values

    return results

def ws_correlation_binned_by_month(data, method='ODR', force_through_origin=False):
    '''Calculate the slope and offset, binned by month, between two wind speed columns.

    :Parameters: 
    
    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir 

    method: string, default 'ODR'
        Correlation method to use. 

        * Orthoginal distance regression: 'ODR'
        * Ordinary least squares: 'OLS'
        * Robust linear models: 'RLM'
    
    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    results = return_correlation_results_frame()

    if not valid_correlation_data(data):
        return results

    months = np.arange(1,13)
    results = pd.concat([results]*12, axis=0)
    results.index = pd.Index(months, name='month')

    for month in months:
        monthly_data = data.loc[data.index.month==month, ['ref','site','dir']]
        points = monthly_data.shape[0]
        
        if not valid_correlation_data(monthly_data):
            results.loc[month, 'points'] = points

        else:
            uncert = calculate_IEC_uncertainty(monthly_data)
    
            monthly_results = ws_correlation_method(monthly_data, method=method, force_through_origin=force_through_origin)
            results.loc[month, ['slope', 'offset', 'R2' , 'uncert', 'points']] = monthly_results.values

    return results

### MAST CORRELATIONS ###
''' Basic outline is that for every correlate method you have to pass it
reference and site mast objects along with the needed sensor names
'''
def correlate_masts_10_minute(ref_mast, site_mast, ref_ws_sensor=None, site_ws_sensor=None, method='ODR', force_through_origin=False):
    '''Calculate the slope and offset between two met masts.

    :Parameters: 
    
    ref_mast: MetMast
        MetMast object 

    site_mast: MetMast
        MetMast object 

    ref_ws_sensor: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_sensor: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    method: string, default 'ODR'
        Correlation method to use. 

        * Orthoginal distance regression: 'ODR'
        * Ordinary least squares: 'OLS'
        * Robust linear models: 'RLM'

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    ref_ws_sensor = ref_mast.check_and_return_mast_ws_sensor(ref_ws_sensor)
    ref_dir_sensor = ref_mast.check_and_return_mast_dir_sensor(None)
    site_ws_sensor = site_mast.check_and_return_mast_ws_sensor(site_ws_sensor)
    
    ref_ws_data = ref_mast.return_sensor_data([ref_ws_sensor])
    ref_dir_data = ref_mast.return_sensor_data([ref_dir_sensor])
    site_ws_data = site_mast.return_sensor_data([site_ws_sensor])

    data = pd.concat([ref_ws_data, site_ws_data, ref_dir_data], axis=1, join='inner').dropna()
    data.columns = ['ref', 'site', 'dir']
    
    results_index = pd.MultiIndex.from_tuples([(ref_mast.name, site_mast.name)], names=['ref', 'site'])
    results = pd.DataFrame(index=results_index, columns=['slope', 'offset', 'R2', 'uncert', 'points'])
    valid_results = ws_correlation_method(data, method=method, force_through_origin=force_through_origin)
    results.loc[pd.IndexSlice[ref_mast.name, site_mast.name], ['slope', 'offset', 'R2' , 'uncert', 'points']] = valid_results.values
    return results

def correlate_masts_10_minute_by_direction(ref_mast, site_mast, ref_ws_sensor=None, ref_dir_sensor=None, site_ws_sensor=None, method='ODR', force_through_origin=False, dir_sectors=16):
    '''Calculate the slope and offset, binned by direction, between two met masts.

    :Parameters: 
    
    ref_mast: MetMast
        MetMast object 

    site_mast: MetMast
        MetMast object 

    ref_ws_sensor: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    ref_dir_sensor: string, default None (primary wind vane assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_sensor: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    method: string, default 'ODR'
        Correlation method to use. 

        * Orthoginal distance regression: 'ODR'
        * Ordinary least squares: 'OLS'
        * Robust linear models: 'RLM'

    dir_sectors: int, default 16
        Number of equally spaced direction sectors

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    ref_ws_sensor = ref_mast.check_and_return_mast_ws_sensor(ref_ws_sensor)
    ref_dir_sensor = ref_mast.check_and_return_mast_dir_sensor(None)
    site_ws_sensor = site_mast.check_and_return_mast_ws_sensor(site_ws_sensor)
    
    ref_ws_data = ref_mast.return_sensor_data([ref_ws_sensor])
    ref_dir_data = ref_mast.return_sensor_data([ref_dir_sensor])
    site_ws_data = site_mast.return_sensor_data([site_ws_sensor])

    data = pd.concat([ref_ws_data, site_ws_data, ref_dir_data], axis=1, join='inner').dropna()
    data.columns = ['ref', 'site', 'dir']
    
    results = ws_correlation_binned_by_direction(data, dir_sectors=dir_sectors, method=method, force_through_origin=force_through_origin)
    results = results.reset_index()
    results['ref'] = ref_mast.name
    results['site'] = site_mast.name
    results = results.set_index(['ref', 'site', 'dir_bin'])
    return results

def correlate_masts_daily(ref_mast, site_mast, ref_ws_sensor=None, site_ws_sensor=None, method='ODR', force_through_origin=False, minimum_recovery_rate=0.7):
    '''Calculate the slope and offset for daily data between two met masts.

    :Parameters: 
    
    ref_mast: MetMast
        MetMast object 

    site_mast: MetMast
        MetMast object 

    ref_ws_sensor: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_sensor: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    method: string, default 'ODR'
        Correlation method to use. 

        * Orthoginal distance regression: 'ODR'
        * Ordinary least squares: 'OLS'
        * Robust linear models: 'RLM'

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)

    minimum_recovery_rate: float, default 0.7
        Minimum allowable recovery rate until resampled data are excluded. 
        For example, by defalt, when resampling 10-minute data to daily averages you would need 
        at least 101 valid records to have a valid daily average.
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points
    
    '''
    ref_ws_sensor = ref_mast.check_and_return_mast_ws_sensor(ref_ws_sensor)
    site_ws_sensor = site_mast.check_and_return_mast_ws_sensor(site_ws_sensor)
    
    ref_ws_data = ref_mast.return_sensor_data([ref_ws_sensor])
    site_ws_data = site_mast.return_sensor_data([site_ws_sensor])

    if minimum_recovery_rate > 1:
        minimum_recovery_rate = minimum_recovery_rate/100.0

    ref_data_daily_mean = an.utils.mast_data.resample_mast_data(ref_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    site_data_daily_mean = an.utils.mast_data.resample_mast_data(site_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    data_daily = pd.concat([ref_data_daily_mean, site_data_daily_mean], axis=1).dropna()
    data_daily.columns  = ['ref', 'site']
    data_daily['dir'] = np.nan

    results = ws_correlation_method(data_daily, method=method, force_through_origin=force_through_origin)
    results.index = pd.MultiIndex.from_tuples([(ref_mast.name, site_mast.name)], names=['ref', 'site'])
    return results

def correlate_masts_daily_by_month(ref_mast, site_mast, ref_ws_sensor=None, site_ws_sensor=None, method='ODR', force_through_origin=False, minimum_recovery_rate=0.7):
    '''Calculate the slope and offset for daily data, binned by month, between two met masts.

    :Parameters: 
    
    ref_mast: MetMast
        MetMast object 

    site_mast: MetMast
        MetMast object 

    ref_ws_sensor: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_sensor: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    method: string, default 'ODR'
        Correlation method to use. 

        * Orthoginal distance regression: 'ODR'
        * Ordinary least squares: 'OLS'
        * Robust linear models: 'RLM'

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)

    minimum_recovery_rate: float, default 0.7
        Minimum allowable recovery rate until resampled data are excluded. 
        For example, by defalt, when resampling 10-minute data to daily averages you would need 
        at least 101 valid records to have a valid daily average.
    
    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points for each month
    
    '''
    ref_ws_sensor = ref_mast.check_and_return_mast_ws_sensor(ref_ws_sensor)
    site_ws_sensor = site_mast.check_and_return_mast_ws_sensor(site_ws_sensor)
    
    ref_ws_data = ref_mast.return_sensor_data([ref_ws_sensor])
    site_ws_data = site_mast.return_sensor_data([site_ws_sensor])

    if minimum_recovery_rate > 1:
        minimum_recovery_rate = minimum_recovery_rate/100.0

    ref_data_daily_mean = an.utils.mast_data.resample_mast_data(ref_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    site_data_daily_mean = an.utils.mast_data.resample_mast_data(site_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    data_daily = pd.concat([ref_data_daily_mean, site_data_daily_mean], axis=1).dropna()
    data_daily.columns  = ['ref', 'site']
    data_daily['dir'] = np.nan

    results = ws_correlation_binned_by_month(data_daily, method='ODR', force_through_origin=force_through_origin)
    results = results.reset_index()
    results['ref'] = ref_mast.name
    results['site'] = site_mast.name
    results = results.set_index(['ref', 'site', 'month'])
    return results