# Import libraries
import os
import sys
import anemoi as an
import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime

import requests
import collections
import json
import urllib3

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
    
def sql_or_string_from_mvs_ids(mvs_ids):
    or_string = ' OR '.join(['mvs_id = {}'.format(mvs_id) for mvs_id in mvs_ids])
    return or_string

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

    def masts(self):
        '''
        :Returns:

        out: DataFrame of all met masts with measured data in M2D2

        Example::

            import anemoi as an
            m2d2 = an.io.database.M2D2()
            m2d2.masts()

        ''' 
        
        if not self.connection_check('M2D2'):
            raise ValueError('Need to connect to M2D2 to retrieve met masts. Use anemoi.DataBase(database="M2D2")')
        
        sql_query_masts = '''
        SELECT [Project]
            ,[AssetID]
            ,[wmm_id]
            ,[mvs_id]
            ,[Name]
            ,[Type]
            ,[StartDate]
            ,[StopDate]
        FROM [M2D2_DB_BE].[dbo].[ViewProjectAssetSensors] WITH (NOLOCK)
        '''
        
        sql_query_coordinates='''
        SELECT [wmm_id]
            ,[WMM_Latitude]
            ,[WMM_Longitude]
            ,[WMM_Elevation]
        FROM [M2D2_DB_BE].[dbo].[ViewWindDataSet]'''
        
        masts = pd.read_sql(sql_query_masts, self.conn, parse_dates=['StartDate', 'StopDate'])
        coordinates = pd.read_sql(sql_query_coordinates, self.conn)
        masts = masts.merge(coordinates, left_on='wmm_id', right_on='wmm_id')

        masts.set_index(['Project', 'wmm_id', 'WMM_Latitude', 'WMM_Longitude', 'Type'], inplace=True)
        masts.sort_index(inplace=True)
        return masts

    def valid_signal_labels(self):
        signal_type_query = '''
        SELECT [MDVT_ID]
              ,[MDVT_Name]
        FROM [M2D2_DB_BE].[dbo].[MDataValueType]'''

        signal_types = pd.read_sql(signal_type_query, self.conn, index_col='MDVT_Name').MDVT_ID
        return signal_types

    def column_labels_for_masts(self):
        masts = self.masts()
        mvs_ids = masts.mvs_id.unique().tolist()
        or_string = ' OR '.join(['mvs_id = {}'.format(mvs_id) for mvs_id in mvs_ids])
        
        column_label_sql_query = '''
        SELECT [column_id]
            ,[label]
        FROM [M2D2_DB_BE].[dbo].[ViewWindogMetaData]
        WITH (NOLOCK)
        WHERE {}'''.format(or_string)

        column_labels = pd.read_sql(column_label_sql_query, self.conn)
        column_labels = column_labels.set_index('column_id')
        return column_labels

    def column_labels_for_wmm_id(self, wmm_id):
        masts = self.masts()
        mvs_ids = masts.loc[pd.IndexSlice[:,wmm_id],:].mvs_id.unique().tolist()
        or_string = ' OR '.join(['mvs_id = {}'.format(mvs_id) for mvs_id in mvs_ids])
        
        column_label_sql_query = '''
        SELECT [column_id]
            ,[label]
        FROM [M2D2_DB_BE].[dbo].[ViewWindogMetaData]
        WITH (NOLOCK)
        WHERE {}'''.format(or_string)

        column_labels = pd.read_sql(column_label_sql_query, self.conn)
        column_labels = column_labels.set_index('column_id')
        return column_labels        
    
    def sensor_data_from_mvs_id(self, mvs_id, signal_type='AVG'):
        '''Download sensor data from M2D2
        
        :Parameters:
        
        mvs_id: int
            Virtual sensor ID (mvs_id) in M2D2
        
        signal_type: str, default 'AVG'
            Signal type for download
            For example: 'AVG', 'SD', 'MIN', 'MAX', 'GUST'
        
        :Returns:
        
        out: DataFrame with signal data from virtual sensor
        '''

        signal_types = self.valid_signal_labels()
        assert signal_type in signal_types.index, 'Tried to look up "{}" but only valid options are: {}'.format(signal_type, signal_types.index.values) 
        
        signal_id = signal_types[signal_type]

        data_by_column_inputs = dict(mvs_id=mvs_id, signal_id=signal_id)
        column_id = '{mvs_id}_{signal_id}'.format(**data_by_column_inputs)

        sql_query= """
        DECLARE  @column_id   NVARCHAR(50)  
                ,@startDate   DATETIME     
                ,@endDate     DATETIME     
               
        SET NOCOUNT ON;
        SET @column_id = '{}'
        SET @startDate = NULL
        SET @endDate   = NULL 

        EXEC dbo.proc_DataExport_GetDataByColumn   
         @column_id   = @column_id               
        ,@startDate   = @startDate 
        ,@endDate     = @endDate     
        """.format(column_id)
        
        sensor_data = pd.read_sql(sql_query, self.conn, index_col='MRD_CorrectedTimestamp')
        sensor_data.index.name = 'stamp'
        sensor_data.columns = [column_id, 'flag']
        return sensor_data[column_id].to_frame(column_id)

    def sensor_flags_from_mvs_id(self, mvs_id):
        '''Download sensor flags from M2D2
        
        :Parameters:
        
        mvs_id: int
            Virtual sensor ID (mvs_id) in M2D2, assume flags associated with average signal
        
        :Returns:
        
        out: DataFrame with signal data from virtual sensor
        '''

        signal_type='AVG'
        signal_types = self.valid_signal_labels()
        assert signal_type in signal_types.index, 'Tried to look up "{}" but only valid options are: {}'.format(signal_type, signal_types.index.values) 
        
        signal_id = signal_types[signal_type]

        data_by_column_inputs = dict(mvs_id=mvs_id, signal_id=signal_id)
        column_id = '{mvs_id}_{signal_id}'.format(**data_by_column_inputs)

        sql_query= """
        DECLARE  @column_id   NVARCHAR(50)  
                ,@startDate   DATETIME     
                ,@endDate     DATETIME     
               
        SET NOCOUNT ON;
        SET @column_id = '{}'
        SET @startDate = NULL
        SET @endDate   = NULL 

        EXEC dbo.proc_DataExport_GetDataByColumn   
         @column_id   = @column_id               
        ,@startDate   = @startDate 
        ,@endDate     = @endDate     
        """.format(column_id)
        
        sensor_data = pd.read_sql(sql_query, self.conn, index_col='MRD_CorrectedTimestamp')
        sensor_data.index.name = 'stamp'
        sensor_data.columns = [column_id, 'flag']
        return sensor_data['flag'].to_frame('flag')

    def mast_data_from_wmm_id(self, wmm_id):
        '''Download data from all sensors on a mast from M2D2
        
        :Parameters:
        
        wmm_id: int
            Mast ID (wmm_id) in M2D2
        
        :Returns:
        
        out: DataFrame with signal data from each virtual sensor
        '''

        masts = self.masts()
        wmm_ids = masts.index.get_level_values('wmm_id').unique().tolist()
        assert wmm_id in wmm_ids, 'Tried to look up "{} but this wmm_id is not found in M2D2'.format(wmm_id)

        labels = self.column_labels_for_wmm_id(wmm_id).label
        mvs_ids = masts.loc[pd.IndexSlice[:,wmm_id],'mvs_id'].unique().tolist()
        data = [self.sensor_data_from_mvs_id(mvs_id) for mvs_id in mvs_ids]
        data = pd.concat(data, axis=1, sort=False)
        data = data.rename(columns=labels.to_dict())
        data.columns.name = 'sensor'
        return data

    def mast_metadata_from_wmm_id(self, wmm_id):
        '''Download mast metadata from M2D2
        
        :Parameters:
        
        wmm_id: int
            Mast ID (wmm_id) in M2D2
        
        :Returns:
        
        out: DataFrame with mast metadata
        '''
        
        sql_query= '''
        SELECT [WMM_Latitude]
            ,[WMM_Longitude]
            ,[WMM_Elevation]
        FROM [M2D2_DB_BE].[dbo].[ViewWindDataSet]
        WHERE wmm_id = {}
        '''.format(wmm_id)
        
        mast_metadata = pd.read_sql(sql_query, self.conn)
        
        return mast_metadata 

    def mast_from_wmm_id(self, wmm_id):
        '''Download an.MetMast from M2D2
        
        :Parameters:
        
        wmm_id: int
            Mast ID (wmm_id) in M2D2
        
        :Returns:
        
        out: an.MetMast with data and metadata from M2D2
        '''
        print('Downloading Mast {} from M2D2'.format(wmm_id))

        masts = self.masts()
        wmm_ids = masts.index.get_level_values('wmm_id').unique().tolist()
        assert wmm_id in wmm_ids, 'Tried to look up "{} but this wmm_id is not found in M2D2'.format(wmm_id)

        data = self.mast_data_from_wmm_id(wmm_id)
        metadata = self.mast_metadata_from_wmm_id(wmm_id)
        mast = an.MetMast(data=data, 
                          name=wmm_id, 
                          lat=metadata.WMM_Latitude[0], 
                          lon=metadata.WMM_Longitude[0], 
                          elev=metadata.WMM_Elevation[0])
        return mast

    def masts_from_project(self, project):
        '''Download an.MetMasts from M2D2 for a given project
        
        :Parameters:
        
        project_name: str
            Project name in M2D2
        
        :Returns:
        
        out: an.MetMasts with data and metadata from M2D2 for a project
        '''

        masts = self.masts()
        projects = masts.index.get_level_values('Project').unique().tolist()
        assert project in projects, 'Tried to look up "{} but this project is not found in M2D2'.format(project)

        wmm_ids = masts.loc[project,:].index.get_level_values('wmm_id').unique().tolist()
        masts = [self.mast_from_wmm_id(wmm_id) for wmm_id in wmm_ids]
        return masts

# Define Turbine class
class Turbine(object):
    '''Class to connect to standard RAG databases
    '''

    def __init__(self):
        '''Data structure for connecting to and downloading data from M2D2. Convention is:

            import anemoi as an
            turb_db = an.io.database.Turbine()
        
        :Parameters:
        

        :Returns:

        out: an.Turbine object connected to Turbine database
        '''
        
        self.database = 'Turbine'
        server = '10.1.15.53'
        db = 'Turbine_DB_BE'
        
        conn_str = 'DRIVER={SQL Server}; SERVER=%s; DATABASE=%s; Trusted_Connection=yes' %(server, db)
        self.conn_str = conn_str #Assign connection string
        
        try:
            self.conn = pyodbc.connect(self.conn_str) #Apply connection string to connect to database
        except:
            print('Database connection error: you either don\'t have permission to the database or aren\'t signed onto the VPN')

    def is_connected(self, database):
        return self.database == database

    def metadata(self):
        '''Get turbine model metadata'''

        assert self.is_connected('Turbine'), 'Trying to query the Turbine DB without being connected.'

        sql_query_turbines = '''
            SELECT [TUR_Manufacturer]
            ,[TUR_RatedOutputkW]
            ,[TPC_MaxOutput]
            ,[TUR_RotorDia]
            ,[TUR_Model]
            ,[AllHubHeights]
            ,[TPC_DocumentDate]
            ,[TUR_ID]
            ,[IECClass]
            ,[TPG_ID]
            ,[TPG_Name]
            ,[TPC_ID]
            ,[TVR_VersionName]
            ,[TPC_dbalevel]
            ,[TPC_TIScenario]
            ,[TPC_BinType]
            ,[TTC_ID]
            ,[TRPMC_ID]
            ,[P_ID]
            ,[P_Name]
            FROM [Turbine_DB_BE].[NodeEstimate].[AllPowerCurves]
            WHERE TPC_Type = 'Manufacturer General Spec'
            '''
        
        turbines = pd.read_sql(sql_query_turbines, self.conn)
        return turbines

# Define Padre class
class Padre(object):
    '''Class to connect to standard RAG databases
    '''

    def __init__(self, database='PADREScada', conn_str=None, conn=None, domino=False):
        '''Data structure with both database name and connection string.
        
        :Parameters:

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

    def assets(self, project=None, turbines_only=False):
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

        return assets

    def operational_projects(self):
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

    def turbine_categorizations(self, category_type='EDF'):

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

    def QCd_turbine_data(self, asset_key):
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

    def raw_turbine_data(self, asset_key, start_date=None, end_date=None):
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')

        turbine_data_query = '''
        SELECT
          [TimeStampLocal]
          ,[Average_Nacelle_Wdspd]
          ,[Average_Active_Power]
          ,[Average_Nacelle_Direction]
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

    def raw_turbine_expected_energy(self, asset_key):
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

    def senvion_event_logs(self, project_id):
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

    def ten_min_energy_by_status_code(self, project_id, start_date, end_date, padre_NTF=True):
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
        
        data_ten_min = pd.read_sql(padre_project_query, self.conn).set_index(['TimeStampLocal', 'AssetKey'])
        data_ten_min.columns = ['power_active','power_expected']
        data_ten_min = data_ten_min.groupby(data_ten_min.index).first()
        data_ten_min.index = pd.MultiIndex.from_tuples(data_ten_min.index)
        data_ten_min.index.names = ['Stamp', 'AssetKey']
        return data_ten_min

    def senvion_ten_min_energy_by_status_code(self, project_id, status_codes=[6680.0, 6690.0, 6697.0, 15000.0]):
        if not self.is_connected('PADREScada'):
                raise ValueError('Need to connect to Padre to retrieve met masts. Use anemoi.DataBase(database="Padre")')
        
        projects = self.operational_projects()
        project = projects.loc[projects.ProjectKey == project_id].index.values[0]
        
        if project in ['Lac Alfred','Massif du Sud','St. Robert Bellarmin']:
            padre_NTF = False
        else:
            padre_NTF = True

        event_log = self.senvion_event_logs(project_id=project_id)
        event_log_icing = event_log.loc[event_log.statuscode.isin(status_codes), :]
        incoming = event_log_icing.loc[event_log_icing.incomingphasingoutreset == 'incoming', ['assetkey', 'statuscode', 'TimeStamp']].reset_index(drop=True)
        outgoing = event_log_icing.loc[event_log_icing.incomingphasingoutreset == 'phasing out', 'TimeStamp'].reset_index(drop=True)
        status = pd.concat([incoming, outgoing], axis=1).dropna()
        status.columns = ['asset_key', 'status_code', 'start', 'end']

        status['start_ten_min'] = status.start.apply(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour,10*(dt.minute // 10)))
        status['end_ten_min'] = status.end.apply(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour,10*(dt.minute // 10)))
        
        status_start_date = status.loc[:,['start_ten_min','end_ten_min']].min().min()
        status_end_date = status.loc[:,['start_ten_min','end_ten_min']].max().max()

        stamp = pd.date_range(start=status_start_date, end=status_end_date, freq='10T')
        icing_flags_cols = pd.MultiIndex.from_product([status.asset_key.unique(), status_codes], names=['AssetKey', 'Flag'])
        icing_flags = pd.DataFrame(index=stamp, columns=icing_flags_cols)
        for col in icing_flags.columns:
            asset_key = col[0]
            icing_flag = col[1]
            icing_flags.loc[status.loc[(status.asset_key==asset_key)&(status.status_code==icing_flag),'start_ten_min'],pd.IndexSlice[asset_key,icing_flag]] = 1.0
            icing_flags.loc[status.loc[(status.asset_key==asset_key)&(status.status_code==icing_flag), 'end_ten_min'],pd.IndexSlice[asset_key,icing_flag]] = 0.0
        icing_flags.fillna(method='ffill', inplace=True)
        icing_flags.fillna(0, inplace=True)
        icing_flags.index.name = 'Stamp'
        
        data_power = self.ten_min_energy_by_status_code(project_id=project_id, start_date=status_start_date, end_date=status_end_date, padre_NTF=padre_NTF)
        data_power = data_power.reset_index().pivot(index='Stamp', columns='AssetKey')
        data_power.columns = data_power.columns.swaplevel()
        data_ten_min = pd.concat([data_power, icing_flags], axis=1).sort_index(axis=0).dropna()
        return data_ten_min

    def monthly_energy_by_status_code(self, project_id, start_date, end_date, padre_NTF=True):
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

    def site_production_data(self, project):
        site_data = []
        turbines = self.turbines(project).loc[:, 'AssetKey'].values
        for i, turbine in enumerate(turbines):
            print('{} of {} masts downloaded'.format(i+1, len(turbines)))
            turbine_data = self.turbine_data(turbine)
            site_data.append(turbine_data)

        site_data = pd.concat(site_data, axis=1, keys=turbines)
        site_data.columns.names = ['Turbine', 'Signal']
        site_data.sort_index(axis=1, inplace=True)
        return site_data

    def meter_data(self, project):
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

    def operational_analysis_metadata(self):
        '''Returns:
        Project metadata for operational analysis
        '''
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve projects. Use anemoi.DataBase(database="Padre")')
        
        metadata_query = """
        SELECT [ProjectName]
              ,[mfg]
              ,[TimeZone]
              ,[NamePlateCapacity]
              ,[NumGenerators]
              ,[latitude]
              ,[longitude]
              ,[DateCOD]
        FROM [PADREScada].[dbo].[Project] with(nolock)
        WHERE [PADREScada].[dbo].[Project].[technology] = 'Wind' and 
        [PADREScada].[dbo].[Project].[active] = 1
        """

        metadata = pd.read_sql(metadata_query, self.conn)
        metadata.set_index('ProjectName', inplace=True)
        return metadata

    def operational_analysis_monthly_invoiced_production(self):
        '''Returns:
        Project monthly invoiced production for operational analysis
        '''
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve projects. Use anemoi.DataBase(database="Padre")')
        
        invoiced_production_query = '''
        SELECT P.[ProjectName]
              ,[Year]
              ,[Month]
              ,[InvoicedProduction_kWh]
        FROM [PADREScada].[dbo].[ProjectInvoicedProduction] PIP with(nolock)
        INNER JOIN [PADREScada].[dbo].[Project] P ON P.projectKey = PIP.ProjectKey
        '''

        invoiced_production = pd.read_sql(invoiced_production_query, self.conn)
        invoiced_production['Day'] = 1
        invoiced_production['Stamp'] = pd.to_datetime(invoiced_production[['Year','Month','Day']])
        invoiced_production = invoiced_production.set_index(['ProjectName','Stamp'])
        invoiced_production = invoiced_production.drop(['Year','Month','Day'], axis=1)
        return invoiced_production

    def operational_analysis_scada_production(self):
        '''Returns:
        Project monthly scada production for operational analysis
        '''
        if not self.is_connected('PADREScada'):
            raise ValueError('Need to connect to Padre to retrieve projects. Use anemoi.DataBase(database="Padre")')
        
        scada_production_query = '''
        SELECT P.[ProjectName]
              ,[Date] Stamp
              ,[WTGProduction_MWh]
              ,[TotalExpectedProduction_MWh]
              ,[TotalEnergyDelta_MWh]
              ,[FullPerformanceEnergyDelta_MWH]
              ,[PartialPerformanceDegradedEnergyDelta_MWH]
              ,[PartialPerformanceDeratedEnergyDelta_MWH]
              ,[PartialPerformanceExtCurtailment_MWH]
              ,[PartialPerformanceEnvironmentEnergyDelta_MWH]
              ,[ForcedOutageEnergyDelta_MWH]
              ,[SchedMaintenanceEnergyDelta_MWH]
              ,[CorrectiveActionsEnergyDelta_MWH]
              ,[TechnicalStandbyEnergyDelta_MWH]
              ,[RequestedShutdownEnergyDelta_MWH]
              ,[RequestedShutdownExtCurtailmentEnergyDelta_MWH]
              ,[OutOfElectricalSpecEnergyDelta_MWH]
              ,[OutOfEnvironmentalSpecEnergyDelta_MWH]
              ,[CalmWindsEnergyDelta_MWH]
              ,[HighWindsEnergyDelta_MWH]
              ,[ForceMajeureEnergyDelta_MWH]
              ,[UnclassifiedDowntimeEnergyDelta_MWH]
        FROM [PADREScada].[dbo].[ProjectCalcDataDaily] PCDD with(nolock)
        INNER JOIN [PADREScada].[dbo].[Project] P ON P.projectKey = PCDD.ProjectKey
        '''

        scada_production = pd.read_sql(scada_production_query, self.conn, parse_dates=['Stamp'])
        scada_production = scada_production.set_index(['ProjectName','Stamp'])
        return scada_production

class EIA(object):
    '''Class to connect to EIA database via HTTP
    '''

    def __init__(self):
        '''Data structure for connecting to and downloading data from EIA. Convention is::

            import anemoi as an
            eia = an.io.database.EIA()
        
        :Parameters:
        

        :Returns:

        out: an.EIA object connected to EIA.gov
        '''
        
        self.database = 'EIA'
        self.api_key = '9B2EDFF62577B236B5D66044ACECA2EF'

    def eia_data_from_id(self, eia_id):
        
        url = 'http://api.eia.gov/series/?api_key={}&series_id=ELEC.PLANT.GEN.{}-WND-WT.M'.format(self.api_key, eia_id)
        
        http = urllib3.PoolManager()
        r = http.request('GET', url)

        if r.status != 200:
            print('No EIA data for station: {}'.format(eia_id))
            return pd.DataFrame(columns=[eia_id])
        
        try:
            data = json.loads(r.data.decode('utf-8'))['series'][0]['data']
            data = pd.DataFrame(data, columns=['Stamp', eia_id])
            data.Stamp = pd.to_datetime(data.Stamp, format='%Y%m')
            data = data.set_index('Stamp')
            data = data.sort_index().astype(int)
            return data
        except:
            return pd.DataFrame(columns=[eia_id])

    def eia_data_from_ids(self, eia_ids):
        
        data = [eia.eia_data(project) for project in projects]
        data = pd.concat(data, axis=1)
        return data

    def eia_project_metadata(self):
        
        filename = 'https://raw.githubusercontent.com/coryjog/wind_data/master/data/AWEA_database_metadata_multiple.parquet'
        metadata = pd.read_parquet(filename)
        metadata.index = metadata.index.astype(np.int)
        metadata.index.name = 'eia_id'
        return metadata

    def eia_turbine_metadata(self):
        
        filename = 'https://raw.githubusercontent.com/coryjog/wind_data/master/data/AWEA_Turb_Report_20171207.parquet'
        metadata = pd.read_parquet(filename)
        return metadata

    def project_centroids(self):

        metadata = self.eia_turbine_metadata()
        centroids = metadata.loc[:,['Turbine Latitude','Turbine Longitude']].groupby(metadata.index).mean()
        return centroids