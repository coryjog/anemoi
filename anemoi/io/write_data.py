import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

### MAST I/O ###
def write_mast_data_to_parquet(mast, filename):
    mast.is_mast_data_size_greater_than_zero()
    data = mast.remove_sensor_levels().data
    table = pa.Table.from_pandas(data, timestamps_to_ms=True)
    pq.write_table(table, filename)

def write_mast_metadata_to_parquet(mast, filename):
    if mast.metadata is not None:
        table = pa.Table.from_pandas(mast.metadata.T)
        pq.write_table(table, filename)