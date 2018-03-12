import pandas as pd

### MAST I/O ###
def mast_to_parquet(mast, folder_path=None):
    data_file_name = 'm{}_data.parquet'.format(mast.name)
    metadata_file_name = 'm{}_metadata.parquet'.format(mast.name)

    if folder_path is not None:
        data_file_name = folder_path+data_file_name
        metadata_file_name = folder_path+metadata_file_name

    mast.remove_sensor_levels()

    mast.data.to_parquet(data_file_name)
    mast.metadata.T.to_parquet(metadata_file_name)
