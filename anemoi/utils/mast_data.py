import anemoi as an
import pandas as pd
import numpy as np

### MAST DATA CHECKS AND VALIDATIONS ###
def check_mast_data_not_empty(mast_data):
    assert not mast_data.empty, 'No data in mast_data DataFrame.'

def check_sensor_names_in_mast_data_columns(mast_data_columns, sensor_names):
    if not isinstance(sensor_names, list):
        sensor_names = [sensor_names]
    
    mast_data_sensor_names = mast_data_columns.get_level_values('sensor').unique().tolist()
    assert all(sensor_name in mast_data_sensor_names for sensor_name in sensor_names)

def check_sensor_types_in_mast_data_columns(mast_data_columns, sensor_types):
    if not isinstance(sensor_types, list):
        sensor_types = [sensor_types]
    
    mast_data_sensor_types = mast_data_columns.get_level_values('type').unique().tolist()
    assert all(sensor_type in mast_data_sensor_types for sensor_type in sensor_types)

def check_sensor_orients_in_mast_data_columns(mast_data_columns, sensor_orients):
    if not isinstance(sensor_orients, list):
        sensor_orients = [sensor_orients]
    
    mast_data_sensor_orients = mast_data_columns.get_level_values('orient').unique().tolist()
    assert all(sensor_orient in mast_data_sensor_orients for sensor_orient in sensor_orients)

def list_of_sensor_names_from_mast_data_columns(mast_data_columns):
    assert 'sensor' in mast_data_columns.names, 'Sensor names not in mast_data column levels'
    return sorted(mast_data_columns.get_level_values('sensor').tolist())

def remove_sensor_levels_from_mast_data_columns(mast_data_columns):
    '''
    Removes kind, type, height, and orient levels from sensor columns in mast data DataFrame
    '''
    assert 'sensor' in mast_data_columns.names, '"sensor" level not in mast_data columns'
    mast_data_columns = mast_data_columns.get_level_values('sensor')
    return mast_data_columns

def add_sensor_levels_to_from_mast_data_columns(mast_data_columns):
    '''
    Add kind, type, height, and orient levels to sensor columns in mast data DataFrame
    '''
    if mast_data_columns.nlevels > 1:
        mast_data_columns = remove_sensor_levels_from_mast_data_columns(mast_data_columns)

    sensors = pd.DataFrame(mast_data_columns.get_level_values('sensor'))
    details = pd.Series(sensors.sensor).str.split('_', expand=True)
    details.columns = ['type', 'height', 'orient', 'signal']
    details = pd.concat([details,sensors], axis=1)
    null_signal = details.signal[details.signal.isnull()].index
    details.signal[null_signal] = details.orient[null_signal]
    details.orient[null_signal] = '-'
    details.signal = details.signal.fillna('-')
    details.height[~details.height.str.isnumeric()] = 0.0
    details.height = details.height.astype(np.float)
    mast_data_columns = pd.MultiIndex.from_arrays(details.T.values, names=details.columns)

    return mast_data_columns

def remove_and_add_sensor_levels_to_mast_data_columns(mast_data_columns):
    mast_data_columns = remove_sensor_levels_from_mast_data_columns(mast_data_columns)
    mast_data_columns = add_sensor_levels_to_from_mast_data_columns(mast_data_columns)
    return mast_data_columns

def infer_time_step(mast_data):
    '''Returns the frequency of the mast data time step; 10-min, hourly, daily, or monthly
    '''
    check_mast_data_not_empty(mast_data)
    time_delta_days = np.diff(mast_data.index.values)/np.timedelta64(1,'D')
    avg_time_delta = np.mean(time_delta_days)

    if avg_time_delta < 0.01:
        freq = '10min'
    elif avg_time_delta < 0.05:
        freq = '1hour'
    elif avg_time_delta < 1.1:
        freq = '1day'
    elif (avg_time_delta > 28.0) and (avg_time_delta < 32):
        freq = 'month'
    return freq

def sensor_details(mast_data, level, sensors=None):
        '''Returns a list of sensor details for a given column level in mast data

        :Parameters:

        level: string
            Level from which to return details ('type', 'height', 'orient', 'signal', 'sensor')

        sensors: list, default None
            List of specific sensors from which to return details
        '''
        check_mast_data_not_empty(mast_data)
        mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
        if (sensors is not None) and check_sensor_names_in_mast_data_columns(mast_data.columns, sensors):
            details_data = mast_data.loc[:, pd.IndexSlice[:,:,:,:,sensors]]
            details = details_data.columns.get_level_values(level)
        else:
            details = mast_data.columns.get_level_values(level)
        return details.tolist()

def sensor_details_unique(mast_data, level, sensors=None):
        '''Returns a list of unique sensor details for a given column level in mast data

        :Parameters:

        level: string
            Level from which to return details ('type', 'height', 'orient', 'signal', 'sensor')

        sensors: list, default None
            List of specific sensors from which to return details
        '''
        mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
        if (sensors is not None) and check_sensor_names_in_mast_data_columns(mast_data, sensors):
            details_data = mast_data.loc[:, pd.IndexSlice[:,:,:,:,sensors]]
            details = details_data.columns.get_level_values(level)
        else:
            details = mast_data.columns.get_level_values(level)
        return sorted(details.unique().tolist())

def sensor_types(mast_data, sensors=None):
        '''Returns a list of sensor types for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        types = sensor_details(mast_data, level='type', sensors=sensors)
        return types

def sensor_types_unique(mast_data, sensors=None):
        '''Returns a list of unique sensor types for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        types = sensor_details_unique(mast_data, level='type', sensors=sensors)
        return types

def sensor_signals(mast_data, sensors=None):
        '''Returns a list of sensor signals for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        signals = sensor_details(mast_data, level='signal', sensors=sensors)
        return signals

def sensor_signals_unique(mast_data, sensors=None):
        '''Returns a list of unique sensor signals for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        signals = sensor_details_unique(mast_data, level='signal', sensors=sensors)
        return signals

def sensor_orients(mast_data, sensors=None):
        '''Returns a list of sensor orientations for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        orients = sensor_details(mast_data, level='orient', sensors=sensors)
        return orients

def sensor_orients_unique(mast_data, sensors=None):
        '''Returns a list of unique sensor orientations for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        orients = sensor_details_unique(mast_data, level='orient', sensors=sensors)
        return orients

def sensor_heights(mast_data, sensors=None):
        '''Returns a list of sensor heights for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        heights = sensor_details(mast_data, level='height', sensors=sensors)
        return heights

def sensor_heights_unique(mast_data, sensors=None):
        '''Returns a list of unique sensor heights for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        heights = sensor_details_unique(mast_data, level='height', sensors=sensors)
        return heights

def sensor_names(mast_data, sensors=None):
        '''Returns a list of sensor names for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        names = sensor_details(mast_data, level='sensor', sensors=sensors)
        return names

def return_data_from_sensors_by_name(mast_data, sensors):
    check_mast_data_not_empty(mast_data)
    check_sensor_names_in_mast_data_columns(mast_data.columns, sensors)
    
    if not isinstance(sensors, list):
        sensors = [sensors]

    mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
    sensor_data = mast_data.loc[:,pd.IndexSlice[:,:,:,:,sensors]]
    # sensor_data = remove_and_add_sensor_levels_to_mast_data_columns(sensor_data)
    return sensor_data

def return_data_from_sensors_by_type(mast_data, sensor_type=None, sensor_signal='AVG'):
    check_mast_data_not_empty(mast_data)
    mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
    return mast_data.loc[:,pd.IndexSlice[sensor_type,:,:,sensor_signal]]

def return_data_from_sensors_by_orient(mast_data, sensor_orient=None, sensor_signal='AVG'):
    check_mast_data_not_empty(mast_data)
    valid_orients = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'COMB', 'EXT']
    assert sensor_orient in valid_orients, 'Must be a valid orient to extract sensor data by orientation'
    mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
    return mast_data.loc[:,pd.IndexSlice[:,:,sensor_orient,sensor_signal]]

def return_data_from_sensors_by_signal(mast_data, signal_type='AVG'):
    check_mast_data_not_empty(mast_data)
    mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
    return mast_data.loc[:,pd.IndexSlice[:,:,:,signal_type]]

def return_data_from_anemometers(mast_data):
    check_mast_data_not_empty(mast_data)
    mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
    return mast_data.loc[:,pd.IndexSlice['SPD',:,:,'AVG']]

def return_data_from_vanes(mast_data):
    check_mast_data_not_empty(mast_data)
    mast_data.columns = remove_and_add_sensor_levels_to_mast_data_columns(mast_data.columns)
    return mast_data.loc[:,pd.IndexSlice['DIR',:,:,'AVG']]

def monthly_days():
    days = pd.DataFrame(index=np.arange(1,13),
                        data=[31.0,28.25,31.0,30.0,31.0,30.0,31.0,31.0,30.0,31.0,30.0,31.0],
                        columns=['Days'])
    days.index.name = 'Month'
    return days

def return_momm(mast_data):
    check_mast_data_not_empty(mast_data)
    if isinstance(mast_data.index, pd.DatetimeIndex):
        mast_data = mast_data.groupby(mast_data.index.month).apply(np.mean)
    elif 'month' in mast_data.index.names:
        mast_data = mast_data.groupby(level='month').apply(np.mean)
    else: 
        raise ValueError('DatetimeIndex or index level "month" needed for MoMM calculation.')

    mast_data.index.name = 'month'
    days = pd.concat([monthly_days()]*mast_data.shape[1], axis=1)
    days.columns = mast_data.columns
    momm = (mast_data*days).sum()/365.25
    momm = momm.to_frame(name='MoMM')
    return momm

def resample_mast_data(mast_data, freq, agg='mean', minimum_recovery_rate=0.7):
    '''Returns a DataFrame of measured data resampled to the specified frequency

    :Parameters:

    freq: string; ('hourly', 'daily', 'weekly', 'monthly', 'yearly')
        Frequency to resample.

        Accepts Python offset aliases.

        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    agg: string; default 'mean'
        Aggregator ('mean', 'std', 'max', 'min', 'count', 'first', 'last')

    minimum_recovery_rate: float, default 0.7
        Minimum allowable recovery rate until resampled data are excluded.
        For example, by defalt, when resampling 10-minute data to daily averages you would need
        at least 101 valid records to have a valid daily average.
    '''
    check_mast_data_not_empty(mast_data)

    start_time = mast_data.index[0]

    if freq == 'hourly':
        freq = 'H'
        start_time = pd.to_datetime('{}/{}/{} {}:00'.format(start_time.year, start_time.month, start_time.day, start_time.hour))
    elif freq == 'daily':
        freq = 'D'
        start_time = pd.to_datetime('{}/{}/{}'.format(start_time.year, start_time.month, start_time.day))
    elif freq == 'weekly':
        freq = 'W'
    elif freq == 'monthly':
        freq = 'MS'
        start_time = pd.to_datetime('{}/{}/01'.format(start_time.year, start_time.month))
    elif freq == 'yearly':
        freq = 'AS'
        start_time = pd.to_datetime('{}/01/01'.format(start_time.year))

    if minimum_recovery_rate > 1:
        minimum_recovery_rate = minimum_recovery_rate/100.0

    data_resampled = mast_data.resample(freq).agg(agg)
    data_count = mast_data.resample(freq).agg('count')
    date_range = pd.date_range(start_time, mast_data.index[-1], freq=mast_data.index[1]-mast_data.index[0])
    max_counts = pd.DataFrame(index=date_range, columns=data_count.columns)
    max_counts.fillna(0, inplace=True)
    max_counts = max_counts.resample(freq).count()
    data_recovery_rate = data_count/max_counts
    data_resampled = data_resampled[data_recovery_rate > minimum_recovery_rate].dropna()
    return data_resampled

### Mast Statistics ###
def monthly_data_recovery(mast_data):
    check_mast_data_not_empty(mast_data)
    data = mast_data.copy()

    # Calculate monthly data recovery by dividing the number of valid records by the number possible
    data.index = pd.MultiIndex.from_arrays([data.index.year, data.index.month, data.index], names=['Year', 'Month', 'Stamp'])
    monthly_data_count = data.groupby(level=['Year', 'Month']).count()
    monthly_max = pd.DataFrame(data=pd.concat([data.groupby(level=['Year', 'Month']).size()]*monthly_data_count.shape[1], axis=1).values,
                              index=monthly_data_count.index,
                              columns=monthly_data_count.columns)
    monthly_data_recovery = monthly_data_count/monthly_max*100
    return monthly_data_recovery

def normalized_rolling_monthly_average(mast_data, min_months=11):
    '''Returns a DataFrame of annual rolling averages of monthly wind speeds normalized by individual monthly means

    :Parameters:

    min_months: int, default 11
        Minimum number of months to be considered a valid year in the rolling average
    '''

    check_mast_data_not_empty(mast_data)
    data = mast_data.copy().astype(np.float)

    yearly_monthly_avg = data.groupby([data.index.year, data.index.month]).mean()
    yearly_monthly_avg.index.names = ['year', 'month']

    monthly_avg = data.groupby(data.index.month).mean()
    monthly_avg.index.name = 'month'
    monthly_avg = monthly_avg.loc[yearly_monthly_avg.index.get_level_values('month'),yearly_monthly_avg.columns.values]
    monthly_avg.index = yearly_monthly_avg.index

    yearly_monthly_avg_normalized = yearly_monthly_avg / monthly_avg
    yearly_monthly_avg_normalized = yearly_monthly_avg_normalized.reset_index()
    yearly_monthly_avg_normalized['day'] = 1
    yearly_monthly_avg_normalized.index = pd.to_datetime(yearly_monthly_avg_normalized.loc[:,['year', 'month', 'day']])
    yearly_monthly_avg_normalized = yearly_monthly_avg_normalized.drop(['year', 'month', 'day'], axis=1)
    yearly_monthly_avg_normalized_rolling = yearly_monthly_avg_normalized.rolling(window=12, min_periods=min_months, center=True).mean() - 1

    return yearly_monthly_avg_normalized_rolling
