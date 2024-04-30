import pandas as pd

def create_autoregressive_periods(lag, df, column) -> pd.DataFrame:
    df[f"AR_{column}_{str(lag)}"] = df[column].shift(lag)

    return (df)

def create_moving_average(period, df, column) -> pd.DataFrame:
    #need to shift 1 before rolling otherwise it will consider the value 0 (leakage of target)
    df[f"MA_{column}_{str(period)}"] = df[column].shift(1).rolling(window = period+1).mean()

    return (df)

def create_rolling_max(period, df, column) -> pd.DataFrame:
    #need to shift 1 before rolling otherwise it will consider the value 0 (leakage of target)
    df[f"MA_{column}_{str(period)}"] = df[column].shift(1).rolling(window = period+1).max()

    return (df)

def create_rolling_min(period, df, column) -> pd.DataFrame:
    #need to shift 1 before rolling otherwise it will consider the value 0 (leakage of target)
    df[f"MA_{column}_{str(period)}"] = df[column].shift(1).rolling(window = period+1).min()

    return (df)

def create_rolling_std(period, df, column) -> pd.DataFrame:
    #need to shift 1 before rolling otherwise it will consider the value 0 (leakage of target)
    df[f"MA_{column}_{str(period)}_std"] = df[column].shift(1).rolling(window = period+1).std()

    return (df)

def create_calendar_features(dataframe):
    # this allows me to regroup if i must afterwards
    grouped_data = dataframe.groupby(pd.Grouper(key='date', 
                                                    freq='1D')).sum() 
    #grouped_data = grouped_data.set_index("date")
    grouped_data['week'] = grouped_data.index.isocalendar().week.astype("category")
    grouped_data['month'] = grouped_data.index.month.astype("category")
    grouped_data['quarter'] = grouped_data.index.quarter.astype("category")
    grouped_data['year'] = grouped_data.index.year.astype("category")

    grouped_data['day'] = grouped_data.index.day
    grouped_data['weekday'] = grouped_data.index.weekday.astype("category")  # Monday=0, Sunday=6
    grouped_data['dayofyear'] = grouped_data['date'].dt.dayofyear


    grouped_data["is_christmas"] = (grouped_data.dayofyear <= 355) & (grouped_data.dayofyear >= 270) 
    grouped_data["is_christmas"] = grouped_data["is_christmas"].astype(int)

    grouped_data["is_weekend"] = grouped_data["weekday"].apply(lambda x: 1 if (x == 5 or x == 6) else 0)  
    return (grouped_data)