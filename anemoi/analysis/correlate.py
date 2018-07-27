import anemoi as an
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import scipy.odr.odrpack as odrpack

import warnings

def compare_sorted_df_columns(cols_1, cols_2):
    return sorted(cols_1) == sorted(cols_2)

def valid_ws_correlation_data(data, ref_ws_col='ref', site_ws_col='site'):
    '''Perform checks on wind speed correlation data.

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref_ws_col and site_ws_col

    ref_ws_col: string, default 'ref'
        Reference anemometer data column to use.

    site_ws_col: string, default 'site'
        Site anemometer data column to use.

    '''
    if ref_ws_col == site_ws_col:
        raise ValueError("Error: Reference and site wind speed columns cannot have the same name.")
        return False

    if not compare_sorted_df_columns(data.columns.tolist(), [ref_ws_col, site_ws_col]):
        raise ValueError("Error: the correlation data don't match the expected format.")
        return False

    if not data.shape[0] > 6:
        warnings.warn("Warning: trying to correalate between less than six points.")
        return False

    if (data.loc[:,ref_ws_col] == data.loc[:,site_ws_col]).sum() == data.shape[0]:
        warnings.warn("Warning: it seems you are trying to correalate a single mast against itself.")
        return False

    return True

def return_correlation_results_frame(ref_label='ref', site_label='site'):
    results = pd.DataFrame(columns=['slope', 'offset' , 'R2', 'uncert', 'points'],
                            index=pd.MultiIndex.from_tuples([(ref_label, site_label)],
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

    if not valid_ws_correlation_data(data=data, ref_ws_col='ref', site_ws_col='site'):
        warning_string = "Warning: {} and {} don't seem to have valid concurrent data for a correlation.".format(ref_mast.name, site_mast.name)
        warnings.warn(warning_string)
    return data

### CORRELATION METHODS ###
def calculate_R2(data, ref_ws_col='ref', site_ws_col='site'):
    '''Return a single R2 between two wind speed columns

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref_ws_col and site_ws_col

    ref_ws_col: string, default 'ref'
        Reference anemometer data column to use.

    site_ws_col: string, default 'site'
        Site anemometer data column to use.

    '''

    data = data.loc[:,[ref_ws_col, site_ws_col]].dropna()
    if not valid_ws_correlation_data(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
        return np.nan

    r2 = data[ref_ws_col].corr(data[site_ws_col])**2
    return r2

def calculate_IEC_uncertainty(data, ref_ws_col='ref', site_ws_col='site'):
    '''Calculate the IEC correlation uncertainty between two wind speed columns

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref_ws_col and site_ws_col

    ref_ws_col: string, default 'ref'
        Reference anemometer data column to use.

    site_ws_col: string, default 'site'
        Site anemometer data column to use.

    '''

    data = data.loc[:,[ref_ws_col, site_ws_col]].dropna()
    if not valid_ws_correlation_data(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
        return np.nan

    X = data.loc[:,ref_ws_col].values
    Y = data.loc[:,site_ws_col].values
    uncert = np.std(Y/X)*100/len(X)
    return uncert*100.0

def calculate_EDF_uncertainty(data, ref_ws_col='ref', site_ws_col='site'):
    '''Calculate the EDF estimated correaltion uncetianty between two wind speed columns.
    Assumes a correalation forced through the origin

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref_ws_col and site_ws_col

    ref_ws_col: string, default 'ref'
        Reference anemometer data column to use.

    site_ws_col: string, default 'site'
        Site anemometer data column to use.

    '''

    data = data.loc[:,[ref_ws_col, site_ws_col]].dropna()
    if not valid_ws_correlation_data(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
        return np.nan

    X = data.loc[:,ref_ws_col].values
    Y = data.loc[:,site_ws_col].values
    Sxx = np.sum(X**2)
    Syy = np.sum(Y**2)
    Sxy = np.sum(X*Y)
    B = 0.5*(Sxx - Syy)/Sxy
    SU = -B + np.sqrt(B**2 + 1)

    e2 = np.sum((Y - SU*X)**2)/(1 + SU**2)
    Xsi2 = e2/(data.shape[0] - 1)
    uncert = np.sqrt((Xsi2*SU**2)*(Sxx*Sxy**2 + 0.25*((Sxx - Syy)**2)*Sxx)/((B**2 + 1.0)*Sxy**4))
    return uncert*100.0

def ws_correlation_least_squares_model(data, ref_ws_col='ref', site_ws_col='site', force_through_origin=False):
    '''Calculate the slope and offset between two wind speed columns using ordinary least squares regression.

    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)

    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points

    '''
    data = data.loc[:, [ref_ws_col, site_ws_col]].dropna()
    results = return_correlation_results_frame(ref_label=ref_ws_col, site_label=site_ws_col)

    if not valid_ws_correlation_data(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
        return results

    points = data.shape[0]
    R2 = calculate_R2(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)
    uncert = calculate_IEC_uncertainty(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)

    if force_through_origin:
        data.loc[:,'offset'] = 0
    else:
        data.loc[:,'offset'] = 1

    X = data.loc[:, [ref_ws_col,'offset']].values
    Y = data.loc[:, site_ws_col].values
    slope, offset = np.linalg.lstsq(X, Y)[0]
    results.loc[pd.IndexSlice[ref_ws_col, site_ws_col],['slope', 'offset' , 'R2', 'uncert', 'points']] = np.array([slope, offset, R2, uncert, points])
    return results

def f_with_offset(B, x):
    return B[0]*x + B[1]

def f_without_offset(B, x):
    return B[0]*x

def ws_correlation_orthoginal_distance_model(data, ref_ws_col='ref', site_ws_col='site', force_through_origin=False):
    '''Calculate the slope and offset between two wind speed columns using orthoganal distance regression.

    https://docs.scipy.org/doc/scipy-0.18.1/reference/odr.html

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)

    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points

    '''
    data = data.loc[:, [ref_ws_col, site_ws_col]].dropna().astype(np.float)
    results = return_correlation_results_frame(ref_label=ref_ws_col, site_label=site_ws_col)

    if not valid_ws_correlation_data(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
        return results

    points = data.shape[0]
    R2 = calculate_R2(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)
    uncert = calculate_IEC_uncertainty(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)

    X = data.loc[:, ref_ws_col].values
    Y = data.loc[:, site_ws_col].values

    data_mean = data.mean()
    slope_estimate_via_ratio = data_mean[site_ws_col]/data_mean[ref_ws_col]
    
    realdata = odrpack.RealData(X, Y)

    if force_through_origin:
        linear = odrpack.Model(f_without_offset)
        odr = odrpack.ODR(realdata, linear, beta0=[slope_estimate_via_ratio])
        slope = odr.run().beta[0]
        offset = 0
    else:
        linear = odrpack.Model(f_with_offset)
        odr = odrpack.ODR(realdata, linear, beta0=[slope_estimate_via_ratio, 0.0])
        slope, offset = odr.run().beta[0], odr.run().beta[1]

    results.loc[pd.IndexSlice[ref_ws_col, site_ws_col],['slope', 'offset' , 'R2', 'uncert', 'points']] = np.array([slope, offset, R2, uncert, points])
    return results

def ws_correlation_robust_linear_model(data, ref_ws_col='ref', site_ws_col='site', force_through_origin=False):
    '''Calculate the slope and offset between two wind speed columns using robust linear model.

    http://www.statsmodels.org/dev/rlm.html

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    force_through_origin: boolean, default False
        Force the correlation through the origin (offset equal to zero)

    :Returns:

    out: DataFrame
        slope, offset, R2, uncert, points

    '''
    data = data.loc[:, [ref_ws_col, site_ws_col]].dropna().astype(np.float)
    results = return_correlation_results_frame(ref_label=ref_ws_col, site_label=site_ws_col)

    if not valid_ws_correlation_data(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
        return results

    points = data.shape[0]
    R2 = calculate_R2(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)
    uncert = calculate_IEC_uncertainty(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)

    X = data.loc[:, ref_ws_col].values
    Y = data.loc[:, site_ws_col].values

    if not force_through_origin:
        X = sm.add_constant(X)
    else:
        X = [np.zeros(X.shape[0]), X]
        X = np.column_stack(X)

    mod = sm.RLM(Y, X)
    resrlm = mod.fit()
    offset, slope = resrlm.params
    R2 = sm.WLS(mod.endog, mod.exog, weights=mod.fit().weights).fit().rsquared
    results.loc[pd.IndexSlice[ref_ws_col, site_ws_col],['slope', 'offset' , 'R2', 'uncert', 'points']] = np.array([slope, offset, R2, uncert, points])
    return results

def ws_correlation_method(data, ref_ws_col='ref', site_ws_col='site', method='ODR', force_through_origin=False):
    '''Calculate the slope and offset, for a given correlation method, between two wind speed columns.

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
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
    if method == 'ODR':
        results = ws_correlation_orthoginal_distance_model(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col, force_through_origin=force_through_origin)
    elif method == 'OLS':
        results = ws_correlation_least_squares_model(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col, force_through_origin=force_through_origin)
    elif method == 'RLM':
        results = ws_correlation_robust_linear_model(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col, force_through_origin=force_through_origin)

    return results

def ws_correlation_binned_by_direction(data, ref_ws_col='ref', site_ws_col='site', ref_dir_col='dir', dir_sectors=16, method='ODR', force_through_origin=False):
    '''Calculate the slope and offset, binned by direction, between two wind speed columns.

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    ref_dir_col: string, default None (primary wind vane assumed)
        Reference wind vane data to use. Extracted from MetMast.data

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
    data = data.loc[:,[ref_ws_col, site_ws_col, ref_dir_col]].dropna().astype(np.float)
    results = return_correlation_results_frame(ref_label=ref_ws_col, site_label=site_ws_col)

    dir_bins = np.arange(1,dir_sectors+1)
    results = pd.concat([results]*dir_sectors, axis=0)
    results.index = pd.Index(dir_bins, name='dir_bin')

    data['dir_bin'] = an.analysis.wind_rose.append_dir_bin(data[ref_dir_col], dir_sectors=dir_sectors)

    for dir_bin in dir_bins:
        dir_bin_data = data.loc[data['dir_bin']==dir_bin, [ref_ws_col, site_ws_col]]
        points = dir_bin_data.shape[0]

        if not valid_ws_correlation_data(data=dir_bin_data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
            results.loc[dir_bin, 'points'] = points

        else:
            uncert = calculate_IEC_uncertainty(data=dir_bin_data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)

            dir_bin_results = ws_correlation_method(data=dir_bin_data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col, method=method, force_through_origin=force_through_origin)
            results.loc[dir_bin, ['slope', 'offset', 'R2' , 'uncert', 'points']] = dir_bin_results.values

    return results

def ws_correlation_binned_by_month(data, ref_ws_col='ref', site_ws_col='site', method='ODR', force_through_origin=False):
    '''Calculate the slope and offset, binned by month, between two wind speed columns.

    :Parameters:

    data: DataFrame
        DataFrame with wind speed columns ref and site, and direction data dir

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
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
    data = data.loc[:, [ref_ws_col, site_ws_col]].dropna().astype(np.float)
    results = return_correlation_results_frame(ref_label=ref_ws_col, site_label=site_ws_col)

    if not valid_ws_correlation_data(data=data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
        return results

    months = np.arange(1,13)
    results = pd.concat([results]*12, axis=0)
    results.index = pd.Index(months, name='month')

    for month in months:
        monthly_data = data.loc[data.index.month==month, [ref_ws_col, site_ws_col]]
        points = monthly_data.shape[0]

        if not valid_ws_correlation_data(data=monthly_data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col):
            results.loc[month, 'points'] = points

        else:
            uncert = calculate_IEC_uncertainty(data=monthly_data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col)

            monthly_results = ws_correlation_method(data=monthly_data, ref_ws_col=ref_ws_col, site_ws_col=site_ws_col, method=method, force_through_origin=force_through_origin)
            results.loc[month, ['slope', 'offset', 'R2' , 'uncert', 'points']] = monthly_results.values

    return results

### MAST CORRELATIONS ###
''' Basic outline is that for every correlate method you have to pass it
reference and site mast objects along with the needed sensor names
'''
def masts_10_minute(ref_mast, site_mast, ref_ws_col=None, site_ws_col=None, method='ODR', force_through_origin=False):
    '''Calculate the slope and offset between two met masts.

    :Parameters:

    ref_mast: MetMast
        MetMast object

    site_mast: MetMast
        MetMast object

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
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
    ref_ws_col = ref_mast.check_and_return_mast_ws_sensor(ref_ws_col)
    site_ws_col = site_mast.check_and_return_mast_ws_sensor(site_ws_col)

    ref_ws_data = ref_mast.return_sensor_data([ref_ws_col])
    site_ws_data = site_mast.return_sensor_data([site_ws_col])

    data = pd.concat([ref_ws_data, site_ws_data], axis=1, join='inner').dropna().astype(np.float)
    data.columns = ['ref', 'site']

    results = return_correlation_results_frame(ref_label=ref_mast.name, site_label=site_mast.name)
    valid_results = ws_correlation_method(data=data, ref_ws_col='ref', site_ws_col='site', method=method, force_through_origin=force_through_origin)
    results.loc[pd.IndexSlice[ref_mast.name, site_mast.name], ['slope', 'offset', 'R2' , 'uncert', 'points']] = valid_results.values
    return results

def masts_10_minute_by_direction(ref_mast, site_mast, ref_ws_col=None, ref_dir_col=None, site_ws_col=None, site_dir_col=None, method='ODR', force_through_origin=False, dir_sectors=16):
    '''Calculate the slope and offset, binned by direction, between two met masts.

    :Parameters:

    ref_mast: MetMast
        MetMast object

    site_mast: MetMast
        MetMast object

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    ref_dir_col: string, default None (primary wind vane assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_dir_col: string, default None (primary anemometer assumed)
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
    ref_ws_col = ref_mast.check_and_return_mast_ws_sensor(ref_ws_col)
    ref_dir_col = ref_mast.check_and_return_mast_dir_sensor(ref_dir_col)
    site_ws_col = site_mast.check_and_return_mast_ws_sensor(site_ws_col)
    site_dir_col = site_mast.check_and_return_mast_dir_sensor(site_dir_col)

    ref_ws_data = ref_mast.return_sensor_data([ref_ws_col])
    ref_dir_data = ref_mast.return_sensor_data([ref_dir_col])
    site_ws_data = site_mast.return_sensor_data([site_ws_col])

    data = pd.concat([ref_ws_data, site_ws_data, ref_dir_data], axis=1, join='inner').dropna().astype(np.float)
    data.columns = ['ref', 'site', 'dir']

    results = ws_correlation_binned_by_direction(data, dir_sectors=dir_sectors, method=method, force_through_origin=force_through_origin)
    results = results.reset_index()
    results['ref'] = ref_mast.name
    results['site'] = site_mast.name
    results = results.set_index(['ref', 'site', 'dir_bin'])
    return results

def masts_daily(ref_mast, site_mast, ref_ws_col=None, site_ws_col=None, method='ODR', force_through_origin=False, minimum_recovery_rate=0.7):
    '''Calculate the slope and offset for daily data between two met masts.

    :Parameters:

    ref_mast: MetMast
        MetMast object

    site_mast: MetMast
        MetMast object

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
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
    ref_ws_col = ref_mast.check_and_return_mast_ws_sensor(ref_ws_col)
    site_ws_col = site_mast.check_and_return_mast_ws_sensor(site_ws_col)

    ref_ws_data = ref_mast.return_sensor_data([ref_ws_col])
    site_ws_data = site_mast.return_sensor_data([site_ws_col])

    if minimum_recovery_rate > 1:
        minimum_recovery_rate = minimum_recovery_rate/100.0

    ref_data_daily_mean = an.utils.mast_data.resample_mast_data(ref_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    site_data_daily_mean = an.utils.mast_data.resample_mast_data(site_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    data_daily = pd.concat([ref_data_daily_mean, site_data_daily_mean], axis=1).dropna().astype(np.float)
    data_daily.columns  = ['ref', 'site']
    data_daily['dir'] = np.nan

    results = ws_correlation_method(data_daily, method=method, force_through_origin=force_through_origin)
    results.index = pd.MultiIndex.from_tuples([(ref_mast.name, site_mast.name)], names=['ref', 'site'])
    return results

def masts_daily_by_month(ref_mast, site_mast, ref_ws_col=None, site_ws_col=None, method='ODR', force_through_origin=False, minimum_recovery_rate=0.7):
    '''Calculate the slope and offset for daily data, binned by month, between two met masts.

    :Parameters:

    ref_mast: MetMast
        MetMast object

    site_mast: MetMast
        MetMast object

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
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
    ref_ws_col = ref_mast.check_and_return_mast_ws_sensor(ref_ws_col)
    site_ws_col = site_mast.check_and_return_mast_ws_sensor(site_ws_col)

    ref_ws_data = ref_mast.return_sensor_data([ref_ws_col])
    site_ws_data = site_mast.return_sensor_data([site_ws_col])

    if minimum_recovery_rate > 1:
        minimum_recovery_rate = minimum_recovery_rate/100.0

    ref_data_daily_mean = an.utils.mast_data.resample_mast_data(ref_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    site_data_daily_mean = an.utils.mast_data.resample_mast_data(site_ws_data, freq='daily', minimum_recovery_rate=minimum_recovery_rate)
    data_daily = pd.concat([ref_data_daily_mean, site_data_daily_mean], axis=1).dropna().astype(np.float)
    data_daily.columns  = ['ref', 'site']
    data_daily['dir'] = np.nan

    results = ws_correlation_binned_by_month(data_daily, method='ODR', force_through_origin=force_through_origin)
    results = results.reset_index()
    results['ref'] = ref_mast.name
    results['site'] = site_mast.name
    results = results.set_index(['ref', 'site', 'month'])
    return results

def apply_10min_results_by_direction(ref_mast, site_mast, corr_results, ref_ws_col=None, ref_dir_col=None, site_ws_col=None, splice=True):
    '''Applies the slopes and offsets from a 10-minute correaltion, binned by direction, between two met masts.

    :Parameters:

    ref_mast: MetMast
        MetMast object

    site_mast: MetMast
        MetMast object

    corr_results: DataFrame
        slope, offset, R2, uncert, points for each direction sector

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    ref_dir_col: string, default None (primary vane assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    splice: Boolean, default True
        Returns site data where available and gap-fills any missing periods between the site mast
        and the reference mast's measurement period. Otherwise, returns purely sythesized data without
        taking into account the measured wind speeds.

    :Returns:

    out: time series DataFrame
        predicted wind speeds at the site

    '''
    ref_ws_col = ref_mast.check_and_return_mast_ws_sensor(ref_ws_col)
    ref_dir_col = ref_mast.check_and_return_mast_dir_sensor(ref_dir_col)
    site_ws_col = site_mast.check_and_return_mast_ws_sensor(site_ws_col)

    ref_ws_data = ref_mast.return_sensor_data([ref_ws_col])
    ref_dir_data = ref_mast.return_sensor_data([ref_dir_col])
    site_ws_data = site_mast.return_sensor_data([site_ws_col])

    data = pd.concat([ref_ws_data, site_ws_data, ref_dir_data], axis=1, join='inner').dropna().astype(np.float)
    data.columns = ['ref', 'site', 'dir']

    ref_data = pd.concat([ref_ws_data, ref_dir_data], axis=1, join='inner').dropna().astype(np.float)
    ref_data.columns = ['ref', 'dir']

    ref_mast_name = ref_mast.name
    site_mast_name = site_mast.name

    corr_to_apply = corr_results.loc[pd.IndexSlice[ref_mast_name,site_mast_name],:]

    ref_data['dir_bin'] = an.analysis.wind_rose.append_dir_bin(ref_data.dir, dir_sectors=corr_to_apply.shape[0])
    ref_data['slope'] = ref_data.dir_bin.map(corr_to_apply.slope)
    ref_data['offset'] = ref_data.dir_bin.map(corr_to_apply.offset)
    syn = (ref_data.ref * ref_data.slope + ref_data.offset).to_frame('syn')

    syn_data = pd.concat([data.site.to_frame('site'), syn], axis=1)
    syn_data['syn_splice'] = syn_data.syn
    fill_index = syn_data.site.notnull()
    syn_data.loc[fill_index, 'syn_splice'] = syn_data.loc[fill_index, 'site']

    if splice:
        syn_data = syn_data.syn_splice.to_frame('syn')
    else:
        syn_data = syn_data.syn.to_frame('syn')

    return syn_data

def apply_daily_results_by_month_to_mast_data(mast_data, corr_results, ref_ws_col='ref', site_ws_col='site', splice=True):
    '''Applies the slopes and offsets from a daily correaltion, binned by month, to a DataFrame of wind speed data.

    :Parameters:

    mast_data: DataFrame
        timeseries of wind speed data

    corr_results: DataFrame
        slope, offset, R2, uncert, points for each month

    ref_ws_col: string, default 'ref'
        Reference anemometer data to use. Extracted from mast_data DataFrame.

    site_ws_col: string, default 'site'
        Site anemometer data to use. Extracted from mast_data DataFrame

    splice: Boolean, default True
        Returns site data where available and gap-fills any missing periods between the site mast
        and the reference mast's measurement period. Otherwise, returns purely sythesized data without
        taking into account the measured wind speeds.

    :Returns:

    out: time series DataFrame
        predicted wind speeds at the site

    '''

    if corr_results.index.nlevels > 1:
        corr_results = corr_results.loc[ref_ws_col,:]

    corr_data = mast_data.loc[:,[ref_ws_col, site_ws_col]].dropna(how='all')
    corr_data['month'] = corr_data.index.month
    corr_data['slope'] = corr_data.month.map(corr_results.slope)
    corr_data['offset'] = corr_data.month.map(corr_results.offset)
    corr_data['syn'] = corr_data[ref_ws_col]*corr_data.slope + corr_data.offset
    corr_data['syn_splice'] = corr_data[site_ws_col]
    corr_data.loc[corr_data.syn_splice.isnull(),'syn_splice'] = corr_data.loc[corr_data.syn_splice.isnull(),'syn']

    if splice:
         syn_data = corr_data.syn_splice.to_frame('syn')
    else:
        syn_data = corr_data.syn.to_frame('syn')

    return syn_data


def apply_daily_results_by_month(ref_mast, site_mast, corr_results, ref_ws_col=None, site_ws_col=None, splice=True):
    '''Applies the slopes and offsets from a daily correaltion, binned by month, between two met masts.
    If the reference or site masts don't have daily time series the method resamples to daily frequency,
    requiring 70% data coverage within each day to be a valid day.

    :Parameters:

    ref_mast: MetMast
        MetMast object

    site_mast: MetMast
        MetMast object

    corr_results: DataFrame
        slope, offset, R2, uncert, points for each direction sector

    ref_ws_col: string, default None (primary anemometer assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    ref_dir_col: string, default None (primary vane assumed)
        Reference anemometer data to use. Extracted from MetMast.data

    site_ws_col: string, default None (primary anemometer assumed)
        Site anemometer data to use. Extracted from MetMast.data

    splice: Boolean, default True
        Returns site data where available and gap-fills any missing periods between the site mast
        and the reference mast's measurement period. Otherwise, returns purely sythesized data without
        taking into account the measured wind speeds.

    :Returns:

    out: time series DataFrame
        predicted wind speeds at the site

    '''
    ref_ws_col = ref_mast.check_and_return_mast_ws_sensor(ref_ws_col)
    site_ws_col = site_mast.check_and_return_mast_ws_sensor(site_ws_col)
    ref_ws_data = ref_mast.return_sensor_data([ref_ws_col])
    site_ws_data = site_mast.return_sensor_data([site_ws_col])

    ref_mast_name = ref_mast.name
    site_mast_name = site_mast.name

    data = pd.concat([ref_ws_data, site_ws_data], axis=1, join='inner').dropna().astype(np.float)
    data.columns = ['ref', 'site']

    if site_mast.infer_time_step() != 'daily':
        site_data = an.utils.mast_data.resample_mast_data(data.site.to_frame('site'), freq='daily').dropna()

    if ref_mast.infer_time_step() != 'daily':
        ref_data = an.utils.mast_data.resample_mast_data(data.ref.to_frame('ref'), freq='daily').dropna()

    corr_to_apply = corr_results.loc[pd.IndexSlice[ref_mast_name,site_mast_name],:]

    ref_data['month'] = ref_data.index.month
    ref_data['slope'] = ref_data.month.map(corr_to_apply.slope)
    ref_data['offset'] = ref_data.month.map(corr_to_apply.offset)
    syn = (ref_data.ref * ref_data.slope + ref_data.offset).to_frame('syn')

    syn_data = pd.concat([site_data, syn], axis=1)
    syn_data['syn_splice'] = syn_data.syn
    fill_index = syn_data.site.notnull()
    syn_data.loc[fill_index, 'syn_splice'] = syn_data.loc[fill_index, 'site']

    if splice:
        syn_data = syn_data.syn_splice.to_frame('syn')
    else:
        syn_data = syn_data.syn.to_frame('syn')

    return syn_data
