# Import libraries
import os
import pandas as pd
import pyodbc
from datetime import datetime

def return_between_date_query_string(start_date, end_date):
        if start_date != None and end_date != None:
            start_end_str = '''AND [TimeStampLocal] >= '%s' AND [TimeStampLocal] < '%s' ''' %(start_date, end_date)
        elif start_date != None and end_date == None:
            start_end_str = '''AND [TimeStampLocal] >= '%s' ''' %(start_date)
        elif start_date == None and end_date != None:
            start_end_str = '''AND [TimeStampLocal] < '%s' ''' %(end_date)
        else:
            start_end_str = ''
        
        return start_end_str

# Define DataBase class
class M2D2(object):
    '''Class to connect to standard RAG databases
    '''

    def __init__(self):
        '''Data structure for connecting to and downloading data from M2D2. Convention is::

            import anemoi as an
            m2d2 = an.io.database.M2D2()
        
        :Parameters:
        

        :Returns:

        out: an.M2D2 object connected to M2D2
        '''
        
        self.database = 'M2D2'
        server = '10.1.15.53'
        db = 'M2D2_DB_BE'
        
        conn_str = 'DRIVER={SQL Server}; SERVER=%s; DATABASE=%s; Trusted_Connection=yes' %(server, db)
        self.conn_str = conn_str #Assign connection string
        
        try:
            self.conn = pyodbc.connect(self.conn_str) #Apply connection string to connect to database
        except:
            print('Database connection error: you either don\'t have permission to the database or aren\'t signed onto the VPN')
        
    def connection_check(self, database):
        return self.database == database

    def get_masts(self):
        '''
        :Returns:

        out: DataFrame of all met masts with measured data in M2D2

        Example::

            import anemoi as an
            m2d2 = an.io.database.M2D2()
            m2d2.get_masts()

        ''' 
        
        if not self.connection_check('M2D2'):
            raise ValueError('Need to connect to M2D2 to retrieve met masts. Use anemoi.DataBase(database="M2D2")')
        
        sql_query_masts = '''
        SELECT [Project]
              ,[WMM_ID]
              ,[MVS_ID]
              ,[Name]
              ,[Type]
              ,[StartDate]
              ,[StopDate]
          FROM [M2D2_DB_BE].[dbo].[ViewProjectAssetSensors] WITH (NOLOCK)
        '''
        masts = pd.read_sql(sql_query_masts, self.conn)
        masts.set_index(['Project', 'WMM_ID', 'Type'], inplace=True)
        masts['StartDate'] = pd.to_datetime(masts['StartDate'], infer_datetime_format=True)
        masts['StopDate'] = pd.to_datetime(masts['StopDate'], infer_datetime_format=True)
        masts.sort_index(inplace=True)
        return masts


    def get_windog_raw_data(self, wmm_id, start_date, end_date):
        # Download raw windog data from M2D2 for an identifed period
        
        sql_query_data = '''
        DECLARE @RC int
        DECLARE @WMM_ID int
        DECLARE @startDate datetime
        DECLARE @endDate datetime
        DECLARE @Windog_format bit

        -- TODO: Set parameter values here.
        Set @WMM_id = {}
        Set @startDate = '{}'
        set @endDate = '{}'
        Set @Windog_format = 0

        EXECUTE @RC = [dbo].[proc_GetWindogRawData]
           @WMM_ID
          ,@startDate
          ,@endDate
          ,@Windog_format

        '''.format(wmm_id, start_date, end_date)
        
        mast_data = pd.read_sql(sql_query_data, self.conn, parse_dates=['MRD_CorrectedTimestamp'])
        
        mast_data = mast_data.loc[mast_data.MDVT_ID == 1, ['MRD_CorrectedTimestamp', 'MVS_ID', 'CalDataValue']]
        mast_data = mast_data.pivot_table(index='MRD_CorrectedTimestamp', columns='MVS_ID', values='CalDataValue', aggfunc='first')    
        
        return mast_data

    def get_wmm_id_data(self, wmm_id, start_date=None, end_date=None):
        '''Download all sensor average signals from M2D2 for a given wind met mast id
        
        :Parameters:
        
        wmm_id: int, default None
            Wind met mast ID
        
        start_date: str, default None (first measured record)
            Date at which to start the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will begin at the begining of the measured period
        
        end_date: str, default None (present day)
            Date at which to end the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will stop at the present day
        
        :Returns:
        
        out: DataFrame of measured virtual sensor data with sensor names as the column labels
        '''

        sensors_metadata = self.get_masts().loc[pd.IndexSlice[:,wmm_id],:]
        sensors = sensors_metadata.loc[:,['MVS_ID', 'Name']].set_index('MVS_ID')

        if start_date is None:
            start_date = pd.to_datetime(sensors_metadata.StartDate.min()).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = end_date = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d')

        mast_data = self.get_windog_raw_data(wmm_id, start_date, end_date)
        
        mast_data = mast_data.rename(columns=sensors.Name.to_dict())
        mast_data.index.name = 'Stamp'
        mast_data.columns.name = 'Sensor'
        
        return mast_data

    def get_wmm_id_metadata(self, wmm_id):
        # Download raw windog data from M2D2 for an identifed period
        
        sql_query= '''
        SELECT [WMM_Latitude]
            ,[WMM_Longitude]
            ,[WMM_Elevation]
        FROM [M2D2_DB_BE].[dbo].[ViewWindDataSet]
        WHERE WMM_ID = {}
        '''.format(wmm_id)
        
        mast_metadata = pd.read_sql(sql_query, self.conn)
        
        return mast_metadata 

    def get_sensor_data(self, MVS_ID=None, sensor_name=None, start_date=None, end_date=None, signal=1):
        '''Download sensor data from M2D2
        
        :Parameters:
        
        vs: int, default None
            Virtual sensor ID (MVS_ID)
        
        sensor_name: str, default None
            Sensor name to be used for the column
            Good practice to use get_masts.loc[MVS_ID == vs, Name]
        
        start_date: str, default None
            Date at which to start the data
            Assumed to be ISO format 'yyyy-mm-dd' example: '2017-01-31'
            If None, will begin at the begining of the measured period
        
        end_date: str, default None
            Date at which to end the data
            Assumed to be ISO format 'yyyy-mm-dd' example: '2017-01-31'
            If None, will stop at the end of the measured period
        
        signal: int, default=1
            Signal type to download; 1=average
        
        :Returns:
        
        out: DataFrame with signal data from virtual sensor
        ''' 
                
        if not self.connection_check('M2D2'):
            raise ValueError('Need to connect to M2D2 to retrieve met masts. Use anemoi.DataBase(database="M2D2")')
        
        #Four date conditions to take into account
        if start_date != None and end_date != None:
            start_end_str = '''AND MRD_CorrectedTimestamp BETWEEN '%s' and '%s' ''' %(start_date, end_date)
        elif start_date != None and end_date == None:
            start_end_str = '''AND MRD_CorrectedTimestamp >= '%s' ''' %(start_date)
        elif start_date == None and end_date != None:
            start_end_str = '''AND MRD_CorrectedTimestamp <= '%s' ''' %(end_date)
        else:
            start_end_str = ''
                
        sql_query_data = '''
        SELECT [MRD_CorrectedTimestamp]
              ,[CalDataValue] AS {}
        FROM [M2D2_DB_BE].[dbo].[ViewWindogRawDataDefault] WITH (NOLOCK)
        WHERE MDVT_ID = {} and MVS_ID = {} 
        {}
        ORDER BY MRD_CorrectedTimestamp'''.format(sensor_name, signal, MVS_ID, start_end_str)
        print(sql_query_data)

        sensor_data = pd.read_sql(sql_query_data, self.conn)
        sensor_data.MRD_CorrectedTimestamp = pd.to_datetime(sensor_data.MRD_CorrectedTimestamp, infer_datetime_format=True)
        sensor_data.set_index('MRD_CorrectedTimestamp', inplace=True)
        sensor_data.index.name = 'Stamp'
        return sensor_data
    
    def get_mast_data(self, WMM_ID, start_date=None, end_date=None):
        '''Download all sensor data from a given mast in M2D2
        
        Parameters:
        
        project: string, default None
            Name of the project within M2D2
        mast: int, default None
            WMM_ID for the mast to be downloaded
        start_date: str, default None
            Date at which to start the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will begin at the begining of the measured period
        end_date: str, default None
            Date at which to end the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will stop at the end of the measured period
        
        Returns:
        DataFrame with signal data from every sensor on a mast
        '''
        
        if not self.connection_check('M2D2'):
            raise ValueError('Need to connect to M2D2 to retrieve met masts. Use anemoi.DataBase(database=\'M2D2\')')
        
        masts = self.get_masts()
        sensors = masts.loc[pd.IndexSlice[:,:,WMM_ID],'MVS_ID'].values
        sensor_names = masts.loc[pd.IndexSlice[:,:,WMM_ID],'Name'].values

        mast_data = []
        for i,sensor in enumerate(sensors):
            name = sensor_names[i]
            data = self.get_sensor_data(sensor, name, start_date=start_date, end_date=end_date)
            mast_data.append(data)

        mast_data = pd.concat(mast_data, axis=1)
        return mast_data

    def get_site_data(self, project=None, start_date=None, end_date=None):
        '''Download all mast data from a given site in M2D2
        
        Parameters:
        ___________
        
        project: string, default None
            Name of the project within M2D2
        start_date: str, default None
            Date at which to start the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will begin at the begining of the measured period
        end_date: str, default None
            Date at which to end the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will stop at the end of the measured period
        
        Returns:
        ________
        DataFrame with signal data from every sensor on every mast at a site
        '''

        if not self.connection_check('M2D2'):
            raise ValueError('Need to connect to M2D2 to retrieve met masts. Use anemoi.DataBase(database="M2D2")')

        masts = self.get_masts().loc[pd.IndexSlice[:,project],:].index.get_level_values('AssetID').unique().tolist()
        mast_data = []
        
        for mast in masts:
            mast_data.append(self.get_mast_data(project=project, mast=mast, start_date=start_date, end_date=end_date))
        mast_data = pd.concat(mast_data, axis=1, keys=map(int, masts), names=['Mast', 'Sensors'])
        
        return mast_data

# Define Padre class
class Padre(object):
    '''Class to connect to standard RAG databases
    '''

    def __init__(self, database='PADREScada', conn_str=None, conn=None, domino=False):
        '''Data structure with both database name and connection string.
        Parameters
        ----------
        database: string, default None
          Name of the padre database to connect to
        conn_str: string, default None
          SQL connection string needed to connect to the database
        conn: object, default None
          SQL connection object to database
        '''
        self.database = database
        
        if self.database == 'PADREScada':
            server = '10.1.106.44'
            db = 'PADREScada'
        elif self.database == 'PadrePI':
            server = '10.1.106.44'
            db = 'PADREScada'
        
        conn_str = 'DRIVER={SQL Server}; SERVER=%s; DATABASE=%s; Trusted_Connection=yes' %(server, db)
        self.conn_str = conn_str
        
        try:
            self.conn = pyodbc.connect(self.conn_str)
        except:
            print('Database connection error: you either don\'t have permission to the database or aren\'t signed onto the VPN')
        
    def is_connected(self, database):
        return self.database == database

    def get_assets(self, project=None, turbines_only=False):
        '''Returns:
        DataFrame of all turbines within Padre
        '''
        
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve turbines. Use anemoi.DataBase(database="Padre")')
        
        sql_query_assets = '''
        SELECT [AssetKey]
          ,Projects.[ProjectName] 
          ,[AssetType]
          ,[AssetName]
          ,Turbines.[Latitude]
          ,Turbines.[Longitude]
          ,[elevation_mt]
        FROM [PADREScada].[dbo].[Asset] as Turbines
        WITH (NOLOCK)
        INNER JOIN [PADREScada].[dbo].[Project] as Projects on Turbines.ProjectKey = Projects.ProjectKey
        '''
        assets = pd.read_sql(sql_query_assets, self.conn)
        assets.set_index(['ProjectName', 'AssetName'], inplace=True)
        assets.sort_index(axis=0, inplace=True)
        
        if turbines_only:
            assets = assets.loc[assets.AssetType == 'Turbine', :]
            assets.drop('AssetType', axis=1, inplace=True)
        
        if project is not None:
            assets = assets.loc[project, :]

        return assets.loc[project, :]

    def get_operational_projects(self):
        '''Returns:
        List of all projects within Padre
        '''
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve projects. Use anemoi.DataBase(database="Padre")')
        
        padre_project_query = """
          SELECT [ProjectKey]
            ,[ProjectName]
            ,[State]
            ,[NamePlateCapacity]
            ,[NumGenerators]
            ,[latitude]
            ,[longitude]
            ,[DateCOD]
          FROM [PADREScada].[dbo].[Project]
          WHERE technology = 'Wind'"""

        projects = pd.read_sql(padre_project_query, self.conn).dropna()
        projects.set_index('ProjectName', inplace=True)
        return projects

    def get_turbine_categorizations(self, category_type='EDF'):

        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve turbines. Use anemoi.DataBase(database="Padre")')
        
        padre_cetegory_query = """
          SELECT [CategoryKey]
                ,[StringName]
          FROM [PADREScada].[dbo].[Categories]
          WHERE CategoryType = '%s'""" %category_type

        categories = pd.read_sql(padre_cetegory_query, self.conn)
        categories.set_index('CategoryKey', inplace=True)
        return categories

    def get_QCd_turbine_data(self, asset_key):
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')

        turbine_data_query = '''
        SELECT [TimeStampLocal]
          ,[Average_Nacelle_Wdspd]
          ,[Average_Active_Power]
          ,[Average_Ambient_Temperature]
          ,[IEC Category]
          ,[EDF Category]
          ,[Expected Power (kW)]
          ,[Expected Energy (kWh)]
          ,[EnergyDelta (kWh)]
          ,[EnergyDelta (MWh)]
        FROM [PADREScada].[dbo].[vw_10mDataBI]
        WITH (NOLOCK)
        WHERE [assetkey] = %i''' %asset_key

        turbine_data = pd.read_sql(turbine_data_query, self.conn)
        turbine_data['TimeStampLocal'] = pd.to_datetime(turbine_data['TimeStampLocal'], format='%Y-%m-%d %H:%M:%S')
        turbine_data.set_index('TimeStampLocal', inplace=True)
        turbine_data.sort_index(axis=0, inplace=True)
        turbine_data = turbine_data.groupby(turbine_data.index).first()
        return turbine_data

    def get_raw_turbine_data(self, asset_key, start_date=None, end_date=None):
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')

        turbine_data_query = '''
        SELECT
          [TimeStampLocal]
          ,[Average_Nacelle_Wdspd]
          ,[Average_Active_Power]
          ,[Average_Blade_Pitch]
          ,[Minimum_Blade_Pitch]
          ,[Maximum_Blade_Pitch]
          ,[Average_Rotor_Speed]
          ,[Minimum_Rotor_Speed]
          ,[Maximum_Rotor_Speed]
          ,[Average_Ambient_Temperature]
          ,coalesce([IECStringKey_Manual]
                    ,[IECStringKey_FF]
                    ,[IECStringKey_Default]) IECKey
          ,coalesce([EDFStringKey_Manual]
                    ,[EDFStringKey_FF]
                    ,[EDFStringKey_Default]) EDFKey
          ,coalesce([State_and_Fault_Manual]
                    ,[State_and_Fault_FF]
                    ,[State_and_Fault]) State_and_Fault
        FROM [PADREScada].[dbo].[WTGCalcData10m]
        WITH (NOLOCK)
        WHERE [assetkey] = {} {}'''.format(asset_key, return_between_date_query_string(start_date, end_date))

        turbine_data = pd.read_sql(turbine_data_query, self.conn)
        turbine_data['TimeStampLocal'] = pd.to_datetime(turbine_data['TimeStampLocal'], format='%Y-%m-%d %H:%M:%S')
        turbine_data.set_index('TimeStampLocal', inplace=True)
        turbine_data.sort_index(axis=0, inplace=True)
        turbine_data = turbine_data.groupby(turbine_data.index).first()
        return turbine_data

    def get_raw_turbine_expected_energy(self, asset_key):
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')

        turbine_data_query = '''
        SELECT
          [TimeStampLocal]
          ,[Expected_Power_NTF]
          ,[Expected_Energy_NTF]
          ,[Expected_Power_RefMet]
          ,[Expected_Energy_RefMet]
          ,[Expected_Power_Uncorr]
          ,[Expected_Energy_Uncorr]
          ,[Expected_Power_DensCorr]
          ,[Expected_Energy_DensCorr]
          ,[Expected_Power_AvgMet]
          ,[Expected_Energy_AvgMet]
          ,[Expected_Power_ProxyWTGs]
          ,[Expected_Energy_ProxyWTGs]
          ,[Expected_Power_MPC]
          ,[Expected_Energy_MPC]
        FROM [PADREScada].[dbo].[WTGCalcData10m]
        WITH (NOLOCK)
        WHERE [assetkey] = {}'''.format(asset_key)

        turbine_data = pd.read_sql(turbine_data_query, self.conn)
        turbine_data['TimeStampLocal'] = pd.to_datetime(turbine_data['TimeStampLocal'], format='%Y-%m-%d %H:%M:%S')
        turbine_data.set_index('TimeStampLocal', inplace=True)
        turbine_data.sort_index(axis=0, inplace=True)
        turbine_data = turbine_data.groupby(turbine_data.index).first()
        return turbine_data

    def get_senvion_event_logs(self, project_id):
        if not self.is_connected('PADREScada'):
                raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')

        sql_query = '''
        SELECT [assetkey]
              ,[TimeStamp]
              ,[statuscode]
              ,[incomingphasingoutreset]
        FROM [PADREScada].[dbo].[SenvionEventLog]
        WHERE projectkey = {} and incomingphasingoutreset != 'Reset'
        ORDER BY assetkey, TimeStamp
        '''.format(project_id)
        
        event_log = pd.read_sql(sql_query, self.conn)
        return event_log

    def get_10min_energy_by_status_code(self, project_id, start_date, end_date, padre_NTF=True):
        if not self.is_connected('PADREScada'):
                raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')

        if padre_NTF:
            padre_power_col = 'Expected_Power_NTF'
        else:
            padre_power_col = 'Expected_Power_DensCorr'

        padre_project_query = '''
        SELECT [TimeStampLocal]
            ,[AssetKey]
            ,[Average_Active_Power]
            ,[{}]
        FROM [PADREScada].[dbo].[WTGCalcData10m]
        WITH (NOLOCK)
        WHERE [projectkey] = {} {}
        ORDER BY TimeStampLocal, AssetKey'''.format(padre_power_col, project_id, return_between_date_query_string(start_date, end_date))
        
        data_10min = pd.read_sql(padre_project_query, self.conn).set_index(['TimeStampLocal', 'AssetKey'])
        data_10min.columns = ['power_active','power_expected']
        data_10min = data_10min.groupby(data_10min.index).first()
        data_10min.index = pd.MultiIndex.from_tuples(data_10min.index)
        data_10min.index.names = ['Stamp', 'AssetKey']
        return data_10min

    def get_senvion_10min_energy_by_status_code(self, project_id, status_codes=[6680.0, 6690.0, 6697.0, 15000.0]):
        if not self.is_connected('PADREScada'):
                raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')
        
        projects = self.get_operational_projects()
        project = projects.loc[projects.ProjectKey == project_id].index.values[0]
        
        if project in ['Lac Alfred','Massif du Sud','St. Robert Bellarmin']:
            padre_NTF = False
        else:
            padre_NTF = True

        event_log = self.get_senvion_event_logs(project_id=project_id)
        event_log_icing = event_log.loc[event_log.statuscode.isin(status_codes), :]
        incoming = event_log_icing.loc[event_log_icing.incomingphasingoutreset == 'incoming', ['assetkey', 'statuscode', 'TimeStamp']].reset_index(drop=True)
        outgoing = event_log_icing.loc[event_log_icing.incomingphasingoutreset == 'phasing out', 'TimeStamp'].reset_index(drop=True)
        status = pd.concat([incoming, outgoing], axis=1).dropna()
        status.columns = ['asset_key', 'status_code', 'start', 'end']

        status['start_10min'] = status.start.apply(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour,10*(dt.minute // 10)))
        status['end_10min'] = status.end.apply(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour,10*(dt.minute // 10)))
        
        status_start_date = status.loc[:,['start_10min','end_10min']].min().min()
        status_end_date = status.loc[:,['start_10min','end_10min']].max().max()

        stamp = pd.date_range(start=status_start_date, end=status_end_date, freq='10T')
        icing_flags_cols = pd.MultiIndex.from_product([status.asset_key.unique(), status_codes], names=['AssetKey', 'Flag'])
        icing_flags = pd.DataFrame(index=stamp, columns=icing_flags_cols)
        for col in icing_flags.columns:
            asset_key = col[0]
            icing_flag = col[1]
            icing_flags.loc[status.loc[(status.asset_key==asset_key)&(status.status_code==icing_flag),'start_10min'],pd.IndexSlice[asset_key,icing_flag]] = 1.0
            icing_flags.loc[status.loc[(status.asset_key==asset_key)&(status.status_code==icing_flag), 'end_10min'],pd.IndexSlice[asset_key,icing_flag]] = 0.0
        icing_flags.fillna(method='ffill', inplace=True)
        icing_flags.fillna(0, inplace=True)
        icing_flags.index.name = 'Stamp'
        
        data_power = self.get_10min_energy_by_status_code(project_id=project_id, start_date=status_start_date, end_date=status_end_date, padre_NTF=padre_NTF)
        data_power = data_power.reset_index().pivot(index='Stamp', columns='AssetKey')
        data_power.columns = data_power.columns.swaplevel()
        data_10min = pd.concat([data_power, icing_flags], axis=1).sort_index(axis=0).dropna()
        return data_10min

    def get_monthly_energy_by_status_code(self, project_id, start_date, end_date, padre_NTF=True):
        if not self.is_connected('PADREScada'):
                raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')

        if padre_NTF:
            padre_power_col = 'Expected_Power_NTF'
        else:
            padre_power_col = 'Expected_Power_DensCorr'

        padre_project_query = '''
        SELECT [TimeStampLocal]
            ,[AssetKey]
            ,[Average_Nacelle_Wdspd]
            ,[Average_Ambient_Temperature]
            ,[Average_Active_Power]
            ,[State_and_Fault] as Fault_Code
            ,[{}]
        FROM [PADREScada].[dbo].[WTGCalcData10m]
        WITH (NOLOCK)
        WHERE [projectkey] = {} {}
        ORDER BY TimeStampLocal, AssetKey, Fault_Code'''.format(padre_power_col, project_id, return_between_date_query_string(start_date, end_date))
        
        columns_sum = ['Average_Active_Power','Expected_Power_NTF','AssetKey', 'Fault_Code']
        columns_count = ['Average_Active_Power', 'AssetKey', 'Fault_Code']
        monthly_data = pd.read_sql(padre_project_query, self.conn).set_index('TimeStampLocal').fillna(method='ffill')
        monthly_data_sum = monthly_data.loc[:,columns_sum].groupby([monthly_data.index.year, monthly_data.index.month, 'AssetKey', 'Fault_Code']).sum()
        monthly_data_count = monthly_data.loc[:,columns_count].groupby([monthly_data.index.year, monthly_data.index.month, 'AssetKey', 'Fault_Code']).count()
        monthly_data = pd.concat([monthly_data_sum, monthly_data_count], axis=1)/6.0
        monthly_data.columns = ['Average_Energy','Expected_Energy_NTF','Hours']
        monthly_data.index.names = ['Year', 'Month', 'AssetKey', 'FaultCode']
        return monthly_data

    def get_site_production_data(self, project):
        site_data = []
        turbines = self.get_turbines(project).loc[:, 'AssetKey'].values
        for i, turbine in enumerate(turbines):
            print('{} of {} masts downloaded'.format(i+1, len(turbines)))
            turbine_data = self.get_turbine_data(turbine)
            site_data.append(turbine_data)

        site_data = pd.concat(site_data, axis=1, keys=turbines)
        site_data.columns.names = ['Turbine', 'Signal']
        site_data.sort_index(axis=1, inplace=True)
        return site_data

    def get_meter_data(self, project):
        if not self.is_connected('PadrePI'):
            raise ValueError('Need to connect to PadrePI to retrieve met masts. Use anemoi.DataBase(database="PadrePI")')

        meter_data_query = """
        SELECT
        p.NamePlateCapacity/1000.0   AS NamePlateCapcity,
        p.NumGenerators,
        bopa.bop_asset_type          AS [BoPAssetType],
        bct.Time,
        bct.Average_Power,
        bct.Average_Reactive_Power,
        bct.Range_Produced_Energy,
        bct.Snapshot_Produced_Energy,
        bct.Range_Consumed_Energy,
        bct.Snapshot_Consumed_Energy 

        FROM dbo.BoPCriticalTag bct
            INNER JOIN dbo.BoPAsset bopa ON bopa.bopassetkey = bct.BoPAssetKey
            INNER JOIN dbo.Project  p    ON p.ProjectKey   = bopa.projectkey

        WHERE bopa.BOP_Asset_Type LIKE '%meter%' and p.ProjectName = '{}'""".format(project)
        
        meter_data = pd.read_sql(meter_data_query, self.conn)
        meter_data['Time'] = pd.to_datetime(meter_data['Time'], format='%Y-%m-%d %H:%M:%S')
        meter_data.set_index('Time', inplace=True)
        meter_data.index.name = 'Stamp'
        meter_data.sort_index(axis=0, inplace=True)
        return meter_data