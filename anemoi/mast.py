# -*- coding: utf-8 -*-
import anemoi as an
import pandas as pd
import numpy as np

import scipy as sp
from scipy import stats
import scipy.optimize as spyopt
from scipy.special import gamma

class MetMast(object):
    '''Primary Anemoi object. Data structure made up of two components:

    * Metadata (mast coordinates, mast height, primary anemometer, primary wind vane)
    * Pandas DataFrame of time series wind measurements which assumes EDF's standard sensor naming conventions.

    :Metadata:

    lat: float, default None
        Latitude of met mast

    long: float, default None
        Longitude of met mast

    height: float or int, default None
        Height of met mast in meters

    primary_ano: string
        Column label of the primary anemometer

    primary_vane: string
        Column label of the primary wind vane

    shear_sensors: list of strings
        List of anemometer columns for use in shear analysis

    :Data:

    data: DataFrame
        Pandas DataFrame of a time series of measured wind data. The column labels assume EDF's standard naming conventions.
        See ECRM for more on the format of the naming convensions:

        https://my.ecrm.edf-re.com/personal/benjamin_kandel/WRAMethod/WRA%20Wiki%20page/Definitions%20and%20conventions.aspx
    '''

    def __init__(self,
                data=None,
                name=None,
                lat=None,
                lon=None,
                elev=None,
                height=None,
                primary_ano=None,
                primary_vane=None,
                shear_sensors=None):

        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.height = height
        self.primary_ano = primary_ano
        self.primary_vane = primary_vane
        self.name = name
        self.shear_sensors = shear_sensors
        self.metadata = pd.DataFrame(index=['height', 'elev', 'lat', 'lon', 'primary_ano', 'primary_vane'],
                                    columns=[self.name],
                                    data=[height, elev, lat, lon, primary_ano, primary_vane])

        if data is not None:
            if data.columns.nlevels > 1:
                data.columns = an.utils.mast_data.remove_sensor_levels_from_mast_data_columns(data.columns)
            
            data.columns = an.utils.mast_data.add_sensor_levels_to_from_mast_data_columns(data.columns)
            data = data.sort_index(axis=1)

        self.data = data

    def __repr__(self):
        name = self.name
        if self.data is not None:
            sensors = self.data.columns.get_level_values(level='sensor').tolist()
        else:
            sensors = []
        repr_string = '''\n\nMast {name}
# of sensors: {sensors}
Coords: {lat:.3f}, {lon:.3f}
Primary ano: {ano}
Primary vane: {vane}'''.format(name=name,
                            sensors=len(sensors),
                            lat=self.lat,
                            lon=self.lon,
                            ano=self.primary_ano,
                            vane=self.primary_vane)
        return repr_string

    def print_mast_data_summary(self):
        measured_period = (self.data.index[-1] - self.data.index[0]).total_seconds()
        measured_period = measured_period/(60*60*24*365.25) #convert to years
        print('Mast {}'.format(self.name))
        print('Elevation: {} m'.format(self.elev))
        print('Start date: {}'.format(self.data.index[0]))
        print('End date: {}'.format(self.data.index[-1]))
        print('Measured period: {:.2f} years'.format(measured_period))

    #### MAST CHECKS ####
    def is_sensor_type_included(self, sensor_type=None):
        return an.utils.mast_data.is_sensor_type_included(self.data, sensor_type=sensor_type)

    def is_sensor_name_included(self, sensor_name=None):
        return an.utils.mast_data.is_sensor_name_included(self.data, sensors=sensors)

    def is_sensor_names_included(self, sensors=None):
        return an.utils.mast_data.is_sensor_names_included(self.data, sensors=sensors)

    def is_mast_data_size_greater_than_zero(self):
        an.utils.mast_data.is_mast_data_size_greater_than_zero(self.data)

    def check_and_return_mast_ws_sensor(self, ano):
        if (ano is None) and (self.primary_ano is None):
            raise ValueError('Unclear which anemometer to use.')
        if ano is None:
            ano = self.primary_ano
        if ano not in self.get_sensor_names():
            raise ValueError('Anemometer not installed on mast.')

        return ano

    def check_and_return_mast_dir_sensor(self, vane):
        if (vane is None) and (self.primary_vane is None):
            raise ValueError('Unclear which wind vane to use.')
        if vane is None:
            vane = self.primary_vane
        if vane not in self.get_sensor_names():
            raise ValueError('Wind vane not installed on mast.')

        return vane

    def infer_time_step(self):
        freq = an.utils.mast_data.infer_time_step(self.data)
        return freq

    #### MAST DATAFRAME MANIPULATION ####
    def remove_sensor_levels_from_mast_data_columns(self):
        self.data = an.utils.mast_data.remove_sensor_levels_from_mast_data_columns(self.data)
        return self

    def add_sensor_levels_to_from_mast_data_columns(self):
        self.data = an.utils.mast_data.add_sensor_levels_to_from_mast_data_columns(self.data)
        return self

    def remove_and_add_sensor_levels_to_from_mast_data_columns(self):
        self.data = an.utils.mast_data.remove_and_add_sensor_levels_to_from_mast_data_columns(self.data)
        return self

    def sensor_names(self):
        '''Returns a list of sensor columns from the MetMast.data DataFrame
        '''
        return an.utils.mast_data.sensor_details(self.data, level='sensor')

    def sensor_details(self, level, sensors=None):
        '''Returns a list of sensor details for a given column level in MetMast.data.
        See also: an.utils.mast_data.get_sensor_details

        :Parameters:

        level: string, default None
            Level from which to return details ('type', 'height', 'orient', 'signal', 'sensor')

        sensors: list, default None
            List of specific sensors from which to return details
        '''
        details = an.utils.mast_data.sensor_details(self.data, level=level, sensors=sensors)
        return details

    def sensor_details_unique(self, level, sensors=None):
        '''Returns a list of sensor details for a given column level in an.MetMast.data.
        See also: an.utils.mast_data.get_unique_sensor_details

        :Parameters:

        level: string, default None
            Level from which to return details ('type', 'height', 'orient', 'signal', 'sensor')

        sensors: list, default None
            List of specific sensors from which to return details
        '''
        details = an.utils.mast_data.sensor_details_unique(self.data, level=level, sensors=sensors)
        return details

    def sensor_types(self, sensors=None):
        '''Returns a list of sensor types for columns in an.MetMast.data

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return details, otherwise all columns assumed
        '''
        types = an.utils.mast_data.sensor_types(self.data, sensors=sensors)
        return types

    def sensor_types_unique(self, sensors=None):
            '''Returns a list of unique sensor types for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            types = an.utils.mast_data.sensor_types_unique(self.data, sensors=sensors)
            return types

    def sensor_signals(self, sensors=None):
            '''Returns a list of sensor signals for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            signals = an.utils.mast_data.sensor_signals(self.data, sensors=sensors)
            return signals

    def sensor_signals_unique(self, sensors=None):
            '''Returns a list of unique sensor signals for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            signals = an.utils.mast_data.sensor_signals_unique(self.data, sensors=sensors)
            return signals

    def sensor_orients(self, sensors=None):
            '''Returns a list of sensor orientations for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            orients = an.utils.mast_data.sensor_orients(self.data, sensors=sensors)
            return orients

    def sensor_orients_unique(self, sensors=None):
            '''Returns a list of unique sensor orientations for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            orients = an.utils.mast_data.sensor_orients_unique(self.data, sensors=sensors)
            return orients

    def sensor_heights(self, sensors=None):
            '''Returns a list of sensor heights for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            heights = an.utils.mast_data.sensor_heights(self.data, sensors=sensors)
            return heights

    def sensor_heights_unique(self, sensors=None):
            '''Returns a list of unique sensor heights for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            heights = an.utils.mast_data.sensor_heights_unique(self.data, sensors=sensors)
            return heights

    def get_sensor_names(self, sensors=None):
            '''Returns a list of sensor names for columns in an.MetMast.data

            :Parameters:

            sensors: list, default None
                List of specific sensors from which to return details, otherwise all columns assumed
            '''
            names = self.get_sensor_details(level='sensor', sensors=sensors)
            return names

    def return_sensor_data(self, sensors=None):
        '''Returns a DataFrame of measured data from specified sensors

        :Parameters:

        sensors: list, default None
            List of specific sensors from which to return data
        '''
        return an.utils.mast_data.return_sensor_data(self.data, sensors=sensors)

    def return_primary_ano_data(self):
        '''Returns a DataFrame of measured data from the primary anemomter
        '''
        return self.return_sensor_data(sensors=self.primary_ano)

    def return_primary_vane_data(self):
        '''Returns a DataFrame of measured data from the primary wind vane
        '''
        return self.return_sensor_data(sensors=self.primary_vane)

    def return_primary_ano_vane_data(self):
        '''Returns a DataFrame of measured data from the primary anemometer and primary wind vane
        '''
        ano_data = self.return_primary_ano_data()
        vane_data = self.return_primary_vane_data()
        return pd.concat([ano_data, vane_data], axis=1)

    def return_sensor_type_data(self, sensor_type=None, sensor_signal='AVG'):
        '''Returns a DataFrame of measured data from a specified sensor type

        :Parameters:

        sensor_type: string, default None
            Sensor type ('SPD', 'DIR', 'T', 'RH')

        see ECRM for more on naming convensions

        https://my.ecrm.edf-re.com/personal/benjamin_kandel/WRAMethod/WRA%20Wiki%20page/Definitions%20and%20conventions.aspx
        '''
        return an.utils.mast_data.return_sensor_type_data(self.data, sensor_type=sensor_type, sensor_signal=sensor_signal)

    def resample_sensor_data(self, sensors, freq, agg='mean', minimum_recovery_rate=0.7):
        '''Returns a DataFrame of measured data resampled to the specified frequency

        :Parameters:

        sensors: list of sensors
            List of specific sensor columns to resample

        freq: string; ('hourly', 'daily', 'weekly', 'monthly', 'yearly')
            Frequency to resample.

            Accepts Python offset aliases.

            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

        agg: string; default 'mean'
            Aggregator ('mean', 'std', 'max', 'min', 'count', 'first', 'last')
        '''
        self.is_mast_data_size_greater_than_zero()

        if not isinstance(sensors, list):
            sensors = [sensors]

        start_time = self.data.index[0]
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

        data = self.return_sensor_data(sensors)
        data_agg = data.resample(freq).agg(agg)
        data_count = data.resample(freq).agg('count')
        date_range = pd.date_range(start_time, data.index[-1], freq=data.index[1]-data.index[0])
        max_counts = pd.DataFrame(index=date_range, columns=data_count.columns)
        max_counts.fillna(0, inplace=True)
        max_counts = max_counts.resample(freq).count()
        data_recovery_rate = data_count/max_counts
        data_agg = data_agg[data_recovery_rate > minimum_recovery_rate].dropna()
        return data_agg

    def return_self_corr_result_dataframe(self):
        self = self.remove_and_add_sensor_levels_to_from_mast_data_columns()
        df = self.data.loc[:,'SPD']
        df.columns = df.columns.droplevel(level='height')
        orients = df.columns.get_level_values(level='orient').unique().tolist()
        result_dataframe = []
        for orient in orients:
            oriented_sensors = df.loc[:,orient].columns.get_level_values(level='sensor').unique().tolist()
            orientated_result = pd.DataFrame(index=pd.MultiIndex.from_product([oriented_sensors]*2, names=['Ref', 'Site']),
                                             columns=['Slope', 'R2', 'Uncert'])
            result_dataframe.append(orientated_result)

        result_dataframe = pd.concat(result_dataframe, axis=0)
        result_dataframe = result_dataframe.loc[result_dataframe.index.get_level_values(0) != result_dataframe.index.get_level_values(1),:]
        return result_dataframe

    #### IO ####
    def from_M2D2(wmm_id, primary_ano=None, primary_vane=None, start_date=None, end_date=None):
        '''Download mast data and metadata from the met data database M2D2 (EDF specific). Currently
        this is only the 10-minute average of each installed sensor. Will add standard deviation, maximum, and
        minimum in the future.

        :Parameters:

        wmm_id: int
            Wind met mast id number in M2D2.
            To get a list of masts in M2D2 run the following::

                import anemoi as an
                m2d2 = an.io.database.M2D2()
                m2d2.get_masts()

        primary_ano: string; default None
            Primary anemomter sensor name

        primary_vane: string; default None
            Primary wind vane sensor name

        start_date: string; default None
            Begining of the desired data period.
            If None, first available day is assumed.
            Date format: 'yyyy-mm-dd' ex:'2017-01-31'

        end_date: string; default None
            End of the desired data period.
            If None, present day is assumed from you computers clock.
            Date format: 'yyyy-mm-dd' ex:'2017-01-31'

            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

        :Returns:

        out: an.MetMast object
        '''

        M2D2 = an.io.database.M2D2()

        mast_metadata = M2D2.get_wmm_id_metadata(wmm_id)
        lat = mast_metadata.loc[0,'WMM_Latitude']
        lon = mast_metadata.loc[0,'WMM_Longitude']
        elev = mast_metadata.loc[0,'WMM_Elevation']

        mast_data = M2D2.get_wmm_id_data(wmm_id, start_date=start_date, end_date=end_date)

        mast = an.MetMast(data=mast_data, elev=elev, lat=lat, lon=lon, name=wmm_id)
        return mast

    #### MAST STATS ####
    def return_monthly_data_recovery(self):
        self.is_mast_data_size_greater_than_zero()
        data = self.data.copy()

        # Calculate monthly data recovery by dividing the number of valid records by the number possible
        data.index = pd.MultiIndex.from_arrays([data.index.year, data.index.month, data.index], names=['Year', 'Month', 'Stamp'])
        monthly_data_count = data.groupby(level=['Year', 'Month']).count()
        monthly_max = pd.DataFrame(data=pd.concat([data.groupby(level=['Year', 'Month']).size()]*monthly_data_count.shape[1], axis=1).values,
                                  index=monthly_data_count.index,
                                  columns=monthly_data_count.columns)
        monthly_data_recovery = monthly_data_count/monthly_max*100
        return monthly_data_recovery

    def return_momm(self, sensors=None, sensor_type=None):
        self.is_mast_data_size_greater_than_zero()
        if sensors is not None:
            df = self.return_sensor_data(sensors=sensors)
        else:
            df = self.return_sensor_type_data(sensor_type=sensor_type)

        df = df.groupby(df.index.month).mean()
        df.index.name = 'Month'
        df.columns = df.columns.get_level_values(-1)
        days = pd.concat([shear.return_monthly_days()]*df.shape[1], axis=1)
        days.columns = df.columns
        MoMM = (df*days).sum()/365.25
        MoMM = MoMM.to_frame(name='MoMM')
        return MoMM

    def return_directional_energy_frequencies(self, dir_sensor=None, ws_sensor=None, dir_sectors=16):
        self.is_mast_data_size_greater_than_zero()
        if (dir_sensor is None) and (self.primary_vane is None):
            return None
        elif (ws_sensor is None) and (self.primary_ano is None):
            return None
        if (dir_sensor is None) and (self.primary_vane is not None):
            dir_sensor = self.primary_vane
        if (ws_sensor is None) and (self.primary_ano is not None):
            ws_sensor = self.primary_ano

        data = self.return_sensor_data([dir_sensor, ws_sensor])
        freqs = an.analysis.wind_rose.return_directional_energy_frequencies(df=data, 
                                                                  dir_sensor=dir_sensor,
                                                                  ws_sensor=ws_sensor,
                                                                  dir_sectors=dir_sectors)
        return freqs

    # #### ANALYSIS - LONG-TERM ####
    # def correlate_to_reference(self, reference_mast, site_ano=None, reference_ano=None, sensor=None, method='ODR'):
    #     self.is_mast_data_size_greater_than_zero()
    #     if site_ano is None:
    #         site_ano = self.primary_ano
    #     if reference_ano is None:
    #         reference_ano = reference_mast.primary_ano

    #     site_data = self.return_sensor_data(site_ano)
    #     reference_data = reference_mast.return_sensor_data(reference_ano)
    #     data = pd.concat(
    #         [reference_data, site_data],
    #         axis=1,
    #         join='inner').dropna()
    #     data.columns = ['Ref', 'Site']

    #     slope, offset = corr.correlate_orthoginal_distance(
    #         df=data,
    #         ref='Ref',
    #         site='Site',
    #         force_through_origin=False)
    #     uncertainty = corr.calculate_IEC_uncertainty(data)
    #     R2 = data.corr().loc['Ref', 'Site']**2
    #     return slope, offset, uncertainty, R2

    # def calculate_self_corr_results(self, sensor_type='SPD'):
    #     self.is_mast_data_size_greater_than_zero()
    #     results_dataframe = self.return_self_corr_result_dataframe()
    #     for i in results_dataframe.index:
    #         ref = i[0]
    #         site = i[1]
    #         df = self.return_sensor_data(sensors=[ref, site]).dropna()
    #         df.columns = df.columns.get_level_values(level='sensor')
    #         slope = corr.correlate_principal_component(df=df,
    #                                                   ref=ref,
    #                                                   site=site)
    #         results_dataframe.loc[pd.IndexSlice[ref, site], 'Slope'] = slope
    #         results_dataframe.loc[pd.IndexSlice[ref, site], 'R2'] = corr.calculate_R2(df=df, ref=ref, site=site)
    #         results_dataframe.loc[pd.IndexSlice[ref, site], 'Uncert'] = corr.calculate_IEC_uncertainty(df=df, ref=ref, site=site)
    #     return results_dataframe.dropna()

    # #### ANALYSIS - SHEAR ####
    # def calculate_time_series_alpha(self, wind_speed_sensors=None):
    #     self.is_mast_data_size_greater_than_zero()
    #     if wind_speed_sensors is None:
    #         df = self.return_sensor_type_data(sensor_type='SPD')
    #         wind_speed_sensors = df.columns.get_level_values(level='sensor')
    #     else:
    #         df = self.return_sensor_data(sensors=wind_speed_sensors)

    #     heights = df.columns.get_level_values(level='height').values.astype(np.float)

    #     df.columns = df.columns.get_level_values(level='sensor')
    #     shear_time_series = shear.shear_alpha_time_series(df=df,
    #                                                     wind_speed_columns=wind_speed_sensors,
    #                                                     heights=heights)
    #     return shear_time_series

    # def calculate_monthly_alpha(self, wind_speed_sensors=None):
    #     self.is_mast_data_size_greater_than_zero()
    #     if wind_speed_sensors is None:
    #         df = self.return_sensor_type_data(sensor_type='SPD')
    #         wind_speed_sensors = df.columns.get_level_values(level='sensor')
    #     else:
    #         df = self.return_sensor_data(sensors=wind_speed_sensors)

    #     heights = df.columns.get_level_values(level='height').values.astype(np.float)

    #     df.columns = df.columns.get_level_values(level='sensor')
    #     df = df.groupby(df.index.month).mean()
    #     df.index.name = 'Month'

    #     monthly_shear = shear.shear_alpha_time_series(df=df,
    #                                                     wind_speed_columns=wind_speed_sensors,
    #                                                     heights=heights)
    #     return monthly_shear

    # def calculate_annual_alpha(self, wind_speed_sensors=None):
    #     self.is_mast_data_size_greater_than_zero()
    #     if wind_speed_sensors is None:
    #         df = self.return_sensor_type_data(sensor_type='SPD')
    #         wind_speed_sensors = df.columns.get_level_values(level='sensor')
    #     else:
    #         df = self.return_sensor_data(sensors=wind_speed_sensors)

    #     heights = df.columns.get_level_values(level='height').values.astype(np.float)

    #     df.columns = df.columns.get_level_values(level='sensor')
    #     df = df.groupby(df.index.year).mean()
    #     df.index.name = 'Year'

    #     annual_shear = shear.shear_alpha_time_series(df=df,
    #                                                 wind_speed_columns=wind_speed_sensors,
    #                                                 heights=heights)
    #     return annual_shear

    # def calculate_long_term_alpha(self,  wind_speed_sensors=None):
    #     '''
    #     **Returns:**
    #     If data in df
    #     '''

    #     self.is_mast_data_size_greater_than_zero()
    #     if wind_speed_sensors is None:
    #         wind_speed_sensors = self.shear_sensors

    #     if self.is_sensor_names_included(sensors=wind_speed_sensors):
    #         momm = self.return_momm(sensors=wind_speed_sensors).T
    #         heights = np.array(map(np.float, self.get_sensor_heights(sensors=wind_speed_sensors)))
    #         alpha = shear.shear_alpha_time_series(df=momm,
    #                                             wind_speed_columns=momm.columns,
    #                                             heights=heights)
    #         return alpha.iloc[0,0]

    # #### PLOTTING ####
    # def plot_monthly_valid_recovery(self, valid_recovery=70, color='#001A70'):
    #     '''Plots valid months with data recovery above a threshhold

    #     *Parameters:*
    #     valid_recovery: int, default 70
    #     Threshold to consider valid month

    #     color: str, default #001A70
    #     Color to plot the months
    #     Default is EDF Dark Blue

    #     *Returns:*
    #     Valid recovery plot
    #     '''

    #     self.is_mast_data_size_greater_than_zero()
    #     sensors = self.get_sensor_names()
    #     no_of_sensors = len(sensors)

    #     monthly_data_recovery = self.return_monthly_data_recovery()
    #     monthly_data_recovery.columns = monthly_data_recovery.columns.get_level_values('sensor')
    #     monthly_data_recovery[monthly_data_recovery<=valid_recovery] = np.nan

    #     #Set y-axis height for each valid month from each sensor
    #     for i, sensor in enumerate(sensors):
    #         monthly_data_recovery.loc[monthly_data_recovery[sensor].notnull(), sensor] = i*+1

    #     # Plot wind speed data recovery rate
    #     plot_width = len(monthly_data_recovery)/1.75
    #     if plot_width < 5:
    #         plot_width = 5

    #     fig = plt.figure(figsize=(plot_width,0.4*no_of_sensors))
    #     ax = fig.add_subplot(111)
    #     if self.name is None:
    #       title = 'Valid months of data (>{}% recovery)'.format(valid_recovery)
    #     else:
    #       title = 'Mast {} - Valid months of data (>{}% recovery)'.format(self.name,
    #                                                                       valid_recovery)
    #     for sensor in sensors:
    #         ax.scatter(pd.date_range('%s-%s-01' %(str(monthly_data_recovery.index.get_level_values('Year')[0]), str(monthly_data_recovery.index.get_level_values('Month')[0])),
    #                                  periods=len(monthly_data_recovery),
    #                                  freq='MS'),
    #                    monthly_data_recovery[sensor].values,
    #                    s=140,
    #                    marker='s',
    #                    color=color)
    #     ax.xaxis.set_minor_locator(mdates.MonthLocator())
    #     ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    #     ax.xaxis.set_major_locator(mdates.YearLocator())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
    #     ax.set_yticks(np.arange(no_of_sensors))
    #     ax.set_yticklabels(sensors)
    #     ax.set_ylim([-0.5,no_of_sensors-0.5])
    #     ax.fmt_xdata = mdates.DateFormatter('%Y-%m')
    #     ax.set_title(title)
    #     return fig

    # def plot_sensor_type(self, sensor_type='SPD'):
    #     '''
    #     Plots valid months with data recovery above a threshhold

    #     **Parameters:**

    #     sensor_type: string, default 'SPD'
    #     Type of sensor to plot

    #     **Returns:**
    #     Time series plot
    #     '''

    #     self.is_mast_data_size_greater_than_zero()
    #     if not self.check_sensor_type_included(sensor_type):
    #         data = pd.DataFrame(index=self.data.index.values, columns=['Empty'])
    #     else:
    #         data = self.data[sensor_type]
    #         data.columns = data.columns.get_level_values('sensor')

    #     if sensor_type == 'SPD':
    #         fig_size = (30,6)
    #         kwargs = {'ylim':[0,30],
    #                  'rot':0,
    #                  'linewidth':2,
    #                  'markersize':0,
    #                  'marker':'.',
    #                  'use_index':True}
    #         ylabel = 'Wind speed [m/s]'
    #     elif sensor_type == 'DIR':
    #         fig_size = (30,5)
    #         kwargs = {'ylim':[0,360],
    #                  'rot':0,
    #                  'linewidth':0,
    #                  'markersize':3,
    #                  'marker':'.',
    #                  'use_index':True}
    #         ylabel = 'Wind direction [deg]'
    #     elif sensor_type == 'T':
    #         fig_size = (30,4)
    #         kwargs = {'ylim':[-40,40],
    #                  'rot':0,
    #                  'linewidth':3,
    #                  'markersize':0,
    #                  'marker':'.',
    #                  'use_index':True}
    #         ylabel = 'Temperature [C]'
    #     else:
    #         fig_size = (30,2)
    #         kwargs = {'ylim':[-1,1],
    #                  'rot':0,
    #                  'use_index':True}
    #         ylabel = 'No data: %s' %sensor_type

    #     fig = plt.figure(figsize=fig_size)
    #     ax = fig.add_subplot(111)
    #     data.plot(kind='line', ax=ax,**kwargs)
    #     ax.xaxis.set_minor_locator(mdates.MonthLocator())
    #     ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    #     ax.xaxis.set_major_locator(mdates.YearLocator())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
    #     ax.fmt_xdata = mdates.DateFormatter('%Y-%m')
    #     ax.set_xlabel('')
    #     ax.legend(loc='best')
    #     ax.set_ylabel(ylabel)

    # def plot_freq_dist(self, wind_speed_sensor=None):
    #     '''
    #     Plots wind speed frequency distribution and weibull fit

    #     **Parameters:**

    #     wind_speed_sensor: string, default None
    #     Wind speed sensor from which to get the measured data

    #     **Returns:**
    #     Frequency distribution plot
    #     '''

    #     if wind_speed_sensor is None:
    #         wind_speed_sensor = self.primary_ano

    #     self.is_mast_data_size_greater_than_zero()
    #     A, k = self.return_weibull_params(sensor=wind_speed_sensor)
    #     freq_dist.plot_freq_dist(params=(A,k),
    #                             data=self.return_sensor_data([wind_speed_sensor]),
    #                             title='Mast {}: Wind speed frequency distribution'.format(self.name))

    # def plot_wind_rose(self, wind_vane=None, bins=16):
    #     '''
    #     Plots wind rose

    #     **Parameters:**

    #     wind_vane: string, default None
    #     Wind speed sensor from which to get the measured data
    #     bins: int, default 16
    #     Number of wind direction bins to plot

    #     **Returns:**
    #     Wind rose plot
    #     '''

    #     self.is_mast_data_size_greater_than_zero()
    #     if wind_vane is None:
    #         wind_vane = self.primary_vane

    #     data = self.return_sensor_data(wind_vane).dropna().values
    #     freqs = freq_dist.return_wind_direction_frequencies(data=data, bins=bins)
    #     freq_dist.plot_wind_rose(freqs, title='Mast {}: Wind rose'.format(self.name))

    # def plot_wind_energy_roses(self, dir_sensor=None, ws_sensor=None, dir_sectors=16):
    #     self.is_mast_data_size_greater_than_zero()
    #     if (dir_sensor is None) and (self.primary_vane is None):
    #         return None
    #     elif (ws_sensor is None) and (self.primary_ano is None):
    #         return None
    #     if (dir_sensor is None) and (self.primary_vane is not None):
    #         dir_sensor = self.primary_vane
    #     if (ws_sensor is None) and (self.primary_ano is not None):
    #         ws_sensor = self.primary_ano

    #     freqs = self.return_directional_energy_frequencies(dir_sensor=dir_sensor,
    #                                                       ws_sensor=ws_sensor,
    #                                                       dir_sectors=dir_sectors)
    #     fig = an.freq_dist.return_wind_energy_rose_figure(dir_bin_centers=freqs.index.values,
    #                                                       dir_bin_freqs_ws=freqs.dir.values,
    #                                                       dir_bin_freqs_energy=freqs.energy.values)
    #     return fig



    # def plot_self_corrs(self, pdf_filename='_self_corrs.pdf', save_pdf=True):
    #     '''
    #     Plots wind speed correlations up and down the mast

    #     **Returns:**
    #     A list of wind speed correlation plots, will also save as a .pdf file
    #     '''

    #     self.is_mast_data_size_greater_than_zero()
    #     self = self.remove_sensor_levels_from_mast_data_columns()
    #     self = self.add_sensor_levels_to_from_mast_data_columns()
    #     self_corr_results = self.calculate_self_corr_results()

    #     df = self.data
    #     df.columns = df.columns.get_level_values(level='sensor')

    #     self_corr_plots = []
    #     for mast_pair in self_corr_results.index:
    #         ref = mast_pair[0]
    #         site = mast_pair[1]

    #         self_corr_plot = corr.plot_wind_speed_correlation(df=df,
    #                                      ref=ref,
    #                                      site=site,
    #                                      title= 'Mast {}'.format(self.name),
    #                                      slope=self_corr_results.loc[pd.IndexSlice[ref, site], 'Slope'],
    #                                      R2=self_corr_results.loc[pd.IndexSlice[ref, site], 'R2'])
    #         self_corr_plots.append(self_corr_plot)

    #     if save_pdf:
    #         print('Will save pdf for Mast {} here'.format(self.name))
    #         ## Save pdf code here

    #     return self_corr_plots
