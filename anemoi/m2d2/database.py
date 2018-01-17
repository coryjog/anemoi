# Import libraries
import os
import pandas as pd
import pyodbc
import datetime

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
class DataBase(object):
    '''Class to connect to standard RAG databases
    '''

    def __init__(self, database=None, conn_str=None, conn=None, domino=False):
        '''Data structure with both database name and connection string.
        Parameters
        ----------
        database: string, default None
          Name of the EDF database to connect to
        conn_str: string, default None
          SQL connection string needed to connect to the database
        conn: object, default None
          SQL connection object to database
        '''
        self.database = database
        
        #Get connection string for database
        if self.database == 'M2D2':
            server = '10.1.15.53'
            db = 'M2D2_DB_BE'
        elif self.database == 'Padre':
            server = '10.1.106.44'
            db = 'PADREScada'
        elif self.database == 'PadrePI':
            server = '10.1.106.44'
            db = 'PADRE'
        elif self.database == 'Turbine':
            server = '10.1.15.53'
            db = 'Turbine_DB_BE'
        else:
            conn_str = conn_str
        
        conn_str = 'DRIVER={SQL Server}; SERVER=%s; DATABASE=%s; Trusted_Connection=yes' %(server, db)
        self.conn_str = conn_str #Assign connection string
        
        try:
            self.conn = pyodbc.connect(self.conn_str) #Apply connection string to connect to database
        except:
            print('Database connection error: you either don\'t have permission to the database or aren\'t signed onto the VPN')
        
    def connection_check(self, database):
        return self.database == database

    def get_masts(self):
        '''Returns:
        DataFrame of all masts within M2D2
        ''' 
        
        if not self.connection_check('M2D2'):
            raise ValueError('Need to connect to M2D2 to retrieve met masts. Use anemoi.DataBase(database="M2D2")')
        
        sql_query_masts = '''
        SELECT [R_Name] as Region
              ,[Project]
              ,[WMM_ID]
              ,[MVS_ID]
              ,[Name]
              ,[Type]
              ,[StartDate]
              ,[StopDate]
          FROM [M2D2_DB_BE].[dbo].[ViewProjectAssetSensors] WITH (NOLOCK)
        '''
        masts = pd.read_sql(sql_query_masts, self.conn)
        masts.set_index(['Region', 'Project', 'WMM_ID', 'Type'], inplace=True)
        masts['StartDate'] = pd.to_datetime(masts['StartDate'], infer_datetime_format=True)
        masts['StopDate'] = pd.to_datetime(masts['StopDate'], infer_datetime_format=True)
        masts.sort_index(inplace=True)
        return masts

    def get_sensor_data(self, MVS_ID=None, sensor_name=None, start_date=None, end_date=None, signal=1):
        '''Download sensor data from M2D2
        
        Parameters:
        ___________
        
        vs: int, default None
            Virtual sensor ID (MVS_ID)
        sensor_name: str, default None
            Sensor name to be used for the column
            Good practice to use get_masts.loc[MVS_ID == vs, Name]
        start_date: str, default None
            Date at which to start the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will begin at the begining of the measured period
        end_date: str, default None
            Date at which to end the data
            Assumed to be ISO format 'yyyy-mm-dd'
            If None, will stop at the end of the measured period
        signal: int, default=1
            Signal type to download; 1=average
        
        Returns:
        ________
        DataFrame with signal data from virtual sensor
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