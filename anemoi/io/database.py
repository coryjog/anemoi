import pandas as pd
import numpy as np
import json
import urllib3


def return_between_date_query_string(start_date, end_date):
    if start_date != None and end_date != None:
        start_end_str = '''AND [TimeStampLocal] >= '%s' AND [TimeStampLocal] < '%s' ''' % (start_date, end_date)
    elif start_date != None and end_date == None:
        start_end_str = '''AND [TimeStampLocal] >= '%s' ''' % (start_date)
    elif start_date == None and end_date != None:
        start_end_str = '''AND [TimeStampLocal] < '%s' ''' % (end_date)
    else:
        start_end_str = ''

    return start_end_str


def sql_or_string_from_mvs_ids(mvs_ids):
    or_string = ' OR '.join(['mvs_id = {}'.format(mvs_id) for mvs_id in mvs_ids])
    return or_string


def sql_list_from_mvs_ids(mvs_ids):
    if not isinstance(mvs_ids, list):
        mvs_ids = [mvs_ids]
    mvs_ids_list = ','.join(['({}_1)'.format(mvs_id) for mvs_id in mvs_ids])
    return mvs_ids_list


def rename_mvs_id_column(col, names, types):
    name = names[int(col.split('_')[0])]
    data_type = types[col.split('_')[1]]
    return '{}_{}'.format(name, data_type)


class EIA(object):
    """Class to connect to EIA database via HTTP
    """

    def __init__(self):
        """Data structure for connecting to and downloading data from EIA. Convention is::

            import anemoi as an
            eia = an.io.database.EIA()

        :Parameters:


        :Returns:

        out: an.EIA object connected to EIA.gov
        """

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

        data = [self.eia_data_from_id(project) for project in eia_ids]
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
        centroids = metadata.loc[:, ['Turbine Latitude', 'Turbine Longitude']].groupby(metadata.index).mean()
        return centroids
