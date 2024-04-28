import json
import os
import pandas as pd
import numpy as np

def load_credentials(path = "credentials.json"):
    with open(path, 'r') as file:
        config = json.load(file)

    # set up credentials
    for key in config.keys():
        os.environ[key] = config[key]

    return

def downcast(dataframe):
    # reduce memory expenditure during training
    float_cols = [c for c in dataframe if dataframe[c].dtype == "float64"]
    int_cols = [c for c in dataframe if dataframe[c].dtype == "int64"]
    dataframe[float_cols] = dataframe[float_cols].astype(np.float16)
    dataframe[int_cols] = dataframe[int_cols].astype(np.int8)
    print(dataframe.info())
    return dataframe

def pivot_data(data, col):
    top_parent = pd.read_csv("top_parent.csv").rename(columns = {"id": "family_id"})
    data = data.merge(top_parent[["family_id", "nome_top_parent"]], 
                                on = "family_id")
    data = data.pivot_table(values=col, 
                            index=['nome_top_parent', 'family_id', 'product_id'], 
                            columns='date', 
                            aggfunc='sum')
    data = data.fillna(0)
    return (data)

def aggregation(df, agg_levels):
    df_agg = pd.DataFrame([])
    for v in agg_levels.values():
        if v is None:
            agg = df.sum(numeric_only=True)
            agg = pd.DataFrame([agg], 
                               index=['Total'])
        else:
            agg = df.groupby(by=v).sum()
        df_agg = pd.concat([df_agg, agg])
    return df_agg

def reconcile_date(daily_data):
    daily_data["date"] = daily_data.index
    daily_data["date"] = pd.to_datetime(daily_data["date"])
    date_range = pd.date_range(start=daily_data['date'].min().strftime('%Y-%m-%d'), 
                           end=daily_data['date'].max().strftime('%Y-%m-%d'), 
                           freq='D')
    
    new_index_df = pd.DataFrame(index=date_range).reset_index().rename(columns={'index': 'date'})
    daily_data["date"] = pd.to_datetime(daily_data['date'])
    new_index_df["date"] = pd.to_datetime(new_index_df['date'])
    daily_data = daily_data.reset_index(drop = True)
    daily_data = pd.merge(new_index_df, 
            daily_data, 
            on='date', 
            how='left')
    daily_data = daily_data.fillna(0)
    daily_data.index = daily_data["date"]
    daily_data = daily_data.drop("date", axis = 1)
    return (daily_data)