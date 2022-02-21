import os
import pandas as pd

_airflow_root = os.path.join(os.environ['HOME'], 'airflow')
_data_root = os.path.join(_airflow_root, 'data', 'covid')

def _get_csv_as_dataframe(input_base, name):
    return pd.read_csv(os.path.join(input_base, name + '.csv'), low_memory=False)

def merge():
    oxford = _get_csv_as_dataframe(_data_root, 'oxford')
    gmobility = _get_csv_as_dataframe(_data_root, 'gmobility')
    demographic = _get_csv_as_dataframe(_data_root, 'demographic')
    rt_estimation = _get_csv_as_dataframe(_data_root, 'rt_estimation')
    country_metrics = _get_csv_as_dataframe(_data_root, 'country_metrics')
    country_metrics = pd.get_dummies(country_metrics, columns=['region'])

    out = pd.merge(oxford, gmobility, how="inner", on=["CountryName", "Date"])
    out = pd.merge(out, rt_estimation, how="inner", on=["CountryName", "Date"])
    out = pd.merge(out, demographic, how="inner", on=["CountryName"])
    out = pd.merge(out, country_metrics, how="left", on=["CountryName"])
    out = out.drop(columns=["R_max", "R_min"])
    out = out.dropna()

    merged_data_dir = os.path.join(_data_root, 'merged')

    if not os.path.exists(merged_data_dir):
        os.mkdir(merged_data_dir)

    out.to_csv(os.path.join(merged_data_dir, 'all.csv'), index=False)

    return merged_data_dir