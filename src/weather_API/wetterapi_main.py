from pathlib import Path
import pickle

import pandas as pd
from pandasgui import show #nur für debuggingzwecke!

import wetterapi_forecast
import wetterapi_history
import eci_api

data_dir = "data/"

def combine_df(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([dataframe1, dataframe2]) #untereinander

def concat_df(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([dataframe1, dataframe2], axis=1) #nebeneinander


def check_for_unique_index(df: pd.DataFrame, name:str ="nicht angegeben") -> bool:
    if df.index.duplicated().any():
        print (f"ES GIBT DUPLIKATE IM INDEX!!!!!! ({name})")
        return True
    else:
        print(f"keine duplikate im Index von {name}")
        return False
    
def check_index_timezone(df: pd.DataFrame, name: str = "nicht angegeben") -> bool: 
    tz = df.index.tz
    if tz is None:
        print(f"[{name}] Index hat KEINE Zeitzoneninfo (tz-naiv).")
        return False
    else:
        print(f"[{name}] Index-Zeitzone: {tz}")
        return True
    
def to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:  ### oder ggf. UTC lassen
    """Bringt den Index auf UTC und macht ihn anschließend tz-naiv."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index = df.index.tz_localize(None)
    return df

def check_hourly_index_gaps(df: pd.DataFrame, name: str = "nicht angegeben") -> bool:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        print(f"[{name}] Hat keinen DatetimeIndex!!! keine Prüfung möglich!")
        return True
    expected = pd.date_range(start=idx.min(), end=idx.max(), freq="h", tz=idx.tz)
    missing = expected.difference(idx)
    if len(missing) > 0:
        print(f"[{name}] Es fehlen {len(missing)} Stunden im Index.")
        return True
    else:
        print(f"[{name}] Keine Lücken im stündlichen Index gefunden.")
        return False

### zu Riskant einfach blind Zeilen zu löschen? --> ggf elegantere Lösung später
def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt doppelte Indizes aus dem DataFrame (keep first) und gibt die Anzahl der entfernten Duplikate aus."""
    zeilen_entfernt = len(df[df.index.duplicated(keep="first")])
    print(f"es wurden {zeilen_entfernt} Duplikate entfernt")
    return df[~df.index.duplicated(keep="first")]

def main():
    # 1. Daten abrufen
    df_history = wetterapi_history.func_all_funcs_history()
    df_forecast = wetterapi_forecast.func_all_funcs_forecast()
    eci = eci_api.EnergyChartsInfo()
    df_price = eci.get_price_series()
    # show(df_history, block=False)
    # show(df_forecast, block=False)
    # show(df_price, block=True)

    # 1.1. Prüfungen... (debugging)
    print("."*30)
    check_index_timezone(df_history, "df_history")
    check_index_timezone(df_forecast, "df_forecast")
    check_index_timezone(df_price, "df_price")
    print(". "*15)
    check_for_unique_index(df_history, "df_history")
    check_for_unique_index(df_forecast, "df_forecast")
    check_for_unique_index(df_price, "df_price")
    print(". "*15)
    check_hourly_index_gaps(df_history, "df_history")
    check_hourly_index_gaps(df_forecast, "df_forecast")
    check_hourly_index_gaps(df_price, "df_price")
    print("."*30)

    # 2. Wetterdaten zusammenfügen
    df_wetter = combine_df(df_history, df_forecast)
    # show(df_wetter, block=True)
    
    # 3. Auf Duplikate prüfen (und ggf. entfernen)
    if check_for_unique_index(df_wetter, "df_wetter"):
        df_wetter = ensure_unique_index(df_wetter)
    check_for_unique_index(df_wetter, "df_wetter")

    # 4. Wetter und Preise zusammen führen, Index --> tz-naiv
    df_for_model = concat_df(df_wetter, df_price)
    ### entfernen der letzten zeile wirklich notwenig? ...ggf bessere lösung?
    # df_for_model = df_for_model.iloc[:-1] # testweise auskommentiert

    ### ggf. einfach UTC lassen, der einheitlich wegen???
    df_for_model = to_utc_naive_index(df_for_model)    

    # 5. Index erneut prüfen 
    print("="*30)
    check_index_timezone(df_for_model, "df_for_model")                 
    check_for_unique_index(df_for_model, "df_for_model")
    check_hourly_index_gaps(df_for_model, "df_for_model")
    print("="*30)   

    # 6. Für ML-Modelle zu tz-naiv und als .pkl speichern
    pd.to_pickle(df_for_model, f"{data_dir}df_for_model.pkl")
    print("df_for_model.pkl gespeichert")
    #show(df_for_model, block=True)

if __name__ == "__main__":
    main()

