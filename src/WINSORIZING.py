import pandas as pd
# from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
# import pickle

# from pandasgui import show

def weekend_and_winsorize(df:pd.DataFrame, min_limit:float = 0.002, max_limit:float = 0.002 ) -> pd.DataFrame:
    df = df.copy()
    df["weekday"] = pd.to_datetime(df.index).day_name()
    df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)
    price_winsorized = winsorize(df['price'].dropna(), limits=[min_limit, max_limit])
    df.loc[df['price'].notnull(), 'price_winsorized'] = price_winsorized
    return df


#### hier das gleiche datum wie im modell später...
# stichtag = "2025-04-07 00:00:00"

# pfad = 'df_for_model.pkl'
# #pfad = 'all_features_df.pkl' ###nur zum testen

# with open(pfad, 'rb') as file:
#     df = pickle.load(file)

# df = df.copy()

# ## zum testen
# print(df.info())
# print("-"*50)
# print(df.describe())
# print("-"*50)
# print(df)
# print("-"*50)
# print("-"*50)

# df.dropna(inplace=True) ### gefährlich so blind... vllt voher prüfen

# df["weekday"] = pd.to_datetime(df.index).day_name()
# df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)


### eigentlich unnötig
# df['hour'] = pd.to_datetime(df.index).hour
# df['month'] = pd.to_datetime(df.index).month
# df['quarter'] = pd.to_datetime(df.index).quarter
# df['year'] = pd.to_datetime(df.index).year

# price_winsorized = winsorize(df['price'].dropna(), limits=[0.002, 0.002])
# df.loc[df['price'].notnull(), 'price_winsorized'] = price_winsorized

# pd.to_pickle(df, "ready_for_MODEL.pkl")


# ## nur zum testen
# print(df.head())
# print("-"*50)
# print(df.tail())
# print("-"*50)
# print(df.info())
# print("-"*50)
# print(df.describe())
# show(df)
