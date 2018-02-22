import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

### MAST I/O ###
def write_mast_data_to_parquet(mast, filename):
    mast.is_mast_data_size_greater_than_zero()
    data = mast.remove_sensor_levels().data
    table = pa.Table.from_pandas(data)
    pq.write_table(table, filename)

def write_mast_metadata_to_parquet(mast, filename):
    if mast.metadata is not None:
        table = pa.Table.from_pandas(mast.metadata.T)
        pq.write_table(table, filename)

def mast_to_parquet(mast, folder_path=None):
    data_file_name = 'm{}_data.parquet'.format(mast.name)
    metadata_file_name = 'm{}_metadata.parquet'.format(mast.name)

    if folder_path is not None:
        data_file_name = folder_path+data_file_name
        metadata_file_name = folder_path+metadata_file_name

    write_mast_data_to_parquet(mast,data_file_name)
    write_mast_metadata_to_parquet(mast,metadata_file_name)
