import wetterapi_forecast
import wettermodul_history
from pandasgui import show
import pandas as pd
import pickle
import eci_module_new_4

subpath = "data/"

df_history = wettermodul_history.func_all_funcs_de()
df_forecast = wetterapi_forecast.func_all_funcs_forecast()
test_eci = eci_module_new_4.EnergyChartsInfo()
price_df = test_eci.get_price_series()



def combine_all_df(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame) -> pd.DataFrame:
    df_gesamt = pd.concat([dataframe1, dataframe2])
    return df_gesamt

def concat_df(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame) -> pd.DataFrame:
    df_concat = pd.concat([dataframe1, dataframe2], axis=1)
    return df_concat

def set_timezone_to_berlin(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("Europe/Berlin", nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert("Europe/Berlin")
    return df

def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="first")]

df_wetter = combine_all_df(df_history, df_forecast)
# show(df_wetter, block=True)

pd.to_pickle(df_wetter, subpath + "df_wetter.pkl")
pd.to_pickle(price_df, subpath + "price_df.pkl")

#with open("df_wetter.pkl", "rb") as f:
#    df_wetter = pickle.load(f)
#with open("price_df.pkl", "rb") as f:
#    price_df = pickle.load(f)


df_wetter = set_timezone_to_berlin(df_wetter)
df_wetter = ensure_unique_index(df_wetter)

price_df = set_timezone_to_berlin(price_df)
price_df = ensure_unique_index(price_df)


df = concat_df(df_wetter, price_df)

df_for_model = df.iloc[:-1]

expected_index = pd.date_range(start=df_for_model.index.min(), end=df_for_model.index.max(), freq="h", tz=df_for_model.index.tz)
is_complete = df_for_model.index.equals(expected_index)
print("✅ Index vollständig & stündlich:", is_complete)

has_duplicates = df_for_model.index.duplicated().any()
print("❗ Doppelte Zeitstempel vorhanden:", has_duplicates)

missing = expected_index.difference(df_for_model.index)
print(f"⛔ Fehlende Zeitstempel ({len(missing)}):")
print(missing)

######## pd.read_pickle("df_for_model.pkl")

#with open("df_for_model.pkl", "rb") as f:
#    df_for_model = pickle.load(f)

#show(df_for_model, block=True)

df_for_model.index = df_for_model.index.tz_convert("UTC")
df_for_model.index = df_for_model.index.tz_localize(None)

pd.to_pickle(df_for_model, subpath + "df_for_model.pkl")