import anemoi as an
import pandas as pd
import numpy as np

### MAST DATA CHECKS AND VALIDATIONS ###
def is_mast_data_size_greater_than_zero(mast_data):
    if (mast_data.shape[0] > 0) and (mast_data.shape[1] > 0):
        return None
    raise ValueError('No data associated within mast_data DataFrame.')

def infer_time_step(mast_data):
    '''Returns the frequency of the mast data time step; 10-min, hourly, daily, or monthly
    '''
    is_mast_data_size_greater_than_zero(mast_data)
    time_delta_days = np.diff(mast_data.index.values)/np.timedelta64(1,'D')
    avg_time_delta = np.mean(time_delta_days)

    if avg_time_delta < 0.01:
        freq = '10-min'
    elif avg_time_delta < 0.05:
        freq = 'hourly'
    elif avg_time_delta < 1.1:
        freq = 'daily'
    elif (avg_time_delta > 28.0) and (avg_time_delta < 32):
        freq = 'monthly'
    return freq

def is_sensor_name_included(mast_data, sensor_name=None):
    if sensor_name is None:
        raise ValueError('Need to define a sensor name to verify it is installed on the mast...')

    sensor_names = mast_data.columns.get_level_values('Sensors').unique().tolist()
    return sensor_name in sensor_names

def is_sensor_names_included(mast_data, sensors=None):
    if sensors is None:
        raise ValueError('Need to define a sensor name to verify it is installed on the mast...')

    sensor_names = mast_data.columns.get_level_values('Sensors').unique().tolist()
    return all(sensor in sensor_names for sensor in sensors)

def is_sensor_type_included(mast_data, sensor_type=None):
    if sensor_type is None:
        raise ValueError('Need to define a sensor type to verify it is installed on the mast...')

    sensor_types = mast_data.columns.get_level_values('Type').unique().tolist()
    return sensor_type in sensor_types

def is_mast_data_size_greater_than_zero(mast_data):
        if mast_data.shape[0] < 1:
            raise ValueError('No data associated with mast.')
        if mast_data.shape[1] < 0:
            raise ValueError('No data associated with mast.')
        return None

def remove_sensor_levels(mast_data):
    '''
    Removes kind, type, height, and orient levels from sensor columns in mast data DataFrame
    '''
    is_mast_data_size_greater_than_zero(mast_data)
    if 'Sensors' in mast_data.columns.names: 
        mast_data.columns = mast_data.columns.get_level_values('Sensors')
    else:
        mast_data.columns = mast_data.columns.get_level_values(-1)
    mast_data.columns.names = ['Sensors']
    return mast_data

def add_sensor_levels(mast_data):
    '''
    Add kind, type, height, and orient levels to sensor columns in mast data DataFrame
    '''
    if mast_data.columns.nlevels > 1:
        mast_data = remove_sensor_levels(mast_data)

    is_mast_data_size_greater_than_zero(mast_data)
    sensor_details = pd.Series(mast_data.columns).str.split('_', expand=True)
    sensor_cols = ['kind', 'height', 'orient', 'signal']
    sensors = pd.DataFrame(index=sensor_details.index, columns=sensor_cols)
    sensor_details.columns = sensors.columns[0: sensor_details.shape[1]]
    sensors = sensors.merge(sensor_details, how='right')
    sensors.signal.fillna('Avg', inplace=True)
    sensors.height.fillna('0', inplace=True)
    kind = sensors.kind.astype(str)
    height = sensors.height.values
    height = list(map(lambda hts: ''.join([ht for ht in hts if ht in '1234567890.']), height))
    height = [float(ht) for ht in height]
    orient = sensors.orient.astype(str)
    signal = sensors.signal.astype(str)
    cols = pd.MultiIndex.from_arrays([kind, height, orient, signal, mast_data.columns], 
        names=['Type', 'Ht', 'Orient', 'Signal', 'Sensors'])
    mast_data.columns = cols
    return mast_data

def remove_and_add_sensor_levels(mast_data):
    mast_data = remove_sensor_levels(mast_data)
    mast_data = add_sensor_levels(mast_data)
    return mast_data

def get_sensor_details(mast_data, level, sensors=None):
        '''Returns a list of sensor details for a given column level in mast data
        
        :Parameters:

        level: string
            Level from which to return details ('Type', 'Ht', 'Orient', 'Signal', 'Sensors')
        
        sensors: list, default None
            List of specific sensors from which to return details
        '''
        mast_data = remove_and_add_sensor_levels(mast_data)
        is_mast_data_size_greater_than_zero(mast_data)
        if (sensors is not None) and is_sensor_names_included(mast_data, sensors):
            details_data = mast_data.loc[:, pd.IndexSlice[:,:,:,:,sensors]]
            details = details_data.columns.get_level_values(level)
        else:
            details = mast_data.columns.get_level_values(level)
        return details.tolist()

def get_unique_sensor_details(mast_data, level, sensors=None):
        '''Returns a list of unique sensor details for a given column level in mast data
        
        :Parameters:

        level: string
            Level from which to return details ('Type', 'Ht', 'Orient', 'Signal', 'Sensors')
        
        sensors: list, default None
            List of specific sensors from which to return details
        '''
        mast_data = remove_and_add_sensor_levels(mast_data)
        is_mast_data_size_greater_than_zero(mast_data)
        if (sensors is not None) and is_sensor_names_included(mast_data, sensors):
            details_data = mast_data.loc[:, pd.IndexSlice[:,:,:,:,sensors]]
            details = details_data.columns.get_level_values(level)
        else:
            details = mast_data.columns.get_level_values(level)
        return sorted(details.unique().tolist())

def get_sensor_types(mast_data, sensors=None):
        '''Returns a list of sensor types for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        types = get_sensor_details(mast_data, level='Type', sensors=sensors)
        return types

def get_unique_sensor_types(mast_data, sensors=None):
        '''Returns a list of unique sensor types for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        types = get_unique_sensor_details(mast_data, level='Type', sensors=sensors)
        return types

def get_sensor_signals(mast_data, sensors=None):
        '''Returns a list of sensor signals for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        signals = get_sensor_details(mast_data, level='Signal', sensors=sensors)
        return signals

def get_unique_sensor_signals(mast_data, sensors=None):
        '''Returns a list of unique sensor signals for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        signals = get_unique_sensor_details(mast_data, level='Signal', sensors=sensors)
        return signals

def get_sensor_orients(mast_data, sensors=None):
        '''Returns a list of sensor orientations for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        orients = get_sensor_details(mast_data, level='Orient', sensors=sensors)
        return orients

def get_unique_sensor_orients(mast_data, sensors=None):
        '''Returns a list of unique sensor orientations for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        orients = get_unique_sensor_details(mast_data, level='Orient', sensors=sensors)
        return orients

def get_sensor_heights(mast_data, sensors=None):
        '''Returns a list of sensor heights for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        heights = get_sensor_details(mast_data, level='Ht', sensors=sensors)
        return heights

def get_unique_sensor_heights(mast_data, sensors=None):
        '''Returns a list of unique sensor heights for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        heights = get_unique_sensor_details(mast_data, level='Ht', sensors=sensors)
        return heights

def get_sensor_names(mast_data, sensors=None):
        '''Returns a list of sensor names for columns in mast data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        names = get_sensor_details(mast_data, level='Sensors', sensors=sensors)
        return names

def return_sensor_data(mast_data, sensors):
    is_mast_data_size_greater_than_zero(mast_data)
    if not isinstance(sensors, list):
        sensors = [sensors]
    if not is_sensor_names_included(mast_data, sensors):
        raise ValueError('Trying to return sensor data from a sensor not in the mast data')
                
    sensor_data = remove_sensor_levels(mast_data).loc[:,sensors]
    mast_data = remove_and_add_sensor_levels(mast_data)
    sensor_data = remove_and_add_sensor_levels(sensor_data)
    return sensor_data

def return_sensor_type_data(mast_data, sensor_type=None, sensor_signal='Avg'):
    is_mast_data_size_greater_than_zero(mast_data)
    mast_data = remove_and_add_sensor_levels(mast_data)
    return mast_data.loc[:,pd.IndexSlice[sensor_type,:,:,sensor_signal]]

def return_signal_type_data(mast_data, signal_type='Avg'):
    is_mast_data_size_greater_than_zero(mast_data)
    mast_data = remove_and_add_sensor_levels(mast_data)
    return mast_data.loc[:,pd.IndexSlice[:,:,:,signal_type]]

def return_anemometer_data(mast_data):
    mast_data = an.utils.mast_data.return_sensor_type_data(mast_data, sensor_type='SPD')
    mast_data = an.utils.mast_data.return_signal_type_data(mast_data, signal_type='Avg')
    mast_data = an.utils.mast_data.remove_sensor_levels(mast_data)
    return mast_data
        
def return_monthly_days():
    days = pd.DataFrame(index=np.arange(1,13), 
                        data=[31.0,28.25,31.0,30.0,31.0,30.0,31.0,31.0,30.0,31.0,30.0,31.0], 
                        columns=['Days'])
    days.index.name = 'Month'
    return days

def return_momm(mast_data):
    mast_data = mast_data.groupby(mast_data.index.month).apply(np.mean)
    mast_data.index.name = 'Month'
    days = pd.concat([return_monthly_days()]*mast_data.shape[1], axis=1)
    days.columns = mast_data.columns
    momm = (mast_data*days).sum()/365.25
    momm = momm.to_frame(name='MoMM').T
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
    is_mast_data_size_greater_than_zero(mast_data)

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
    is_mast_data_size_greater_than_zero(mast_data)
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

    is_mast_data_size_greater_than_zero(mast_data)
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