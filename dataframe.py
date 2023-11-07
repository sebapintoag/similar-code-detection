import pandas as pd

# Creates a dataframe from a parquet file
def from_parquet(filepath, columns = None):
    df = pd.read_parquet(filepath, engine='pyarrow')
    if columns != None:
        return df[columns]

    return df

# Returns only the first rows deppending on the entered param
def cut(dataframe, rows):
    return dataframe.iloc[:rows,:]