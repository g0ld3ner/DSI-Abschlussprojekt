from datetime import datetime, timedelta, timezone
import time
import pickle
import pandas as pd
import openmeteo_requests # wird nicht mehr gebraucht
import requests_cache
from retry_requests import retry
from weather_API.bundeslaender_gewichte import sun_weights, wind_weights
from locations import location
from collections import defaultdict
from typing import Callable, Optional
# from pandasgui import show

data_dir = "data/"

api_sleep_time = 25

cache_session = requests_cache.CachedSession(f"{data_dir}.history_cache", expire_after=21600) #6h
session = retry(cache_session, retries=5, backoff_factor=0.5)

def current_day(minus: int = 0) -> datetime:
    return (datetime.now(timezone.utc) - timedelta(days=minus))

def x_days_back(date: datetime, x: int = 400) -> datetime: # Tage standardmäßig (None) abgefragt werden
    return date - timedelta(days=x)

def date_to_string(date: datetime) -> str: #die API braucht ja einen String
    return date.strftime("%Y-%m-%d")

def get_weather_by_location(lat: float, lon: float, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    print(f"Wetterdaten für {lat}, {lon}")

    if end_date is None:
        ### Die archive-API hängt immer 2 Tage hinterher, daher müssen diese 2 Tage aus der Forecast API aufgefüllt werden (daher minus=2)
        end_date = current_day(minus=2) # ob man das später nicht eleganter lösen kann mit der minus 2 ???
        end_date_str = date_to_string(end_date)
    else:
        end_date_str = end_date
    if start_date is None:
        start_date = x_days_back(end_date)
        start_date_str = date_to_string(start_date)
    else:
        start_date_str = start_date
        
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": [
            "temperature_2m", "wind_speed_100m", "sunshine_duration", "global_tilted_irradiance"
        ],
        "tilt": 35
    }
    print("Sende API-Anfrage...")
    start_zeitmessung = time.time()
    response = session.get(url, params=params)
    dauer = time.time() - start_zeitmessung
    print(f"Antwort erhalten nach {dauer:.2f} Sekunden => (Cache: {response.from_cache})")

    if response.status_code != 200:
        raise RuntimeError(f"API-Fehler: {response.status_code} {response.text[:200]}")

    data = response.json()

    # Abweichung der Koordinaten prüfen (wie im Forecast)
    tol = 0.11
    lat_api = data.get("latitude", -999)
    lon_api = data.get("longitude", -999)
    print(f"API-Koordinate: {lat_api:.5f}, {lon_api:.5f} (Angefragt: {lat:.5f}, {lon:.5f})")
    if abs(lat_api - lat) > tol or abs(lon_api - lon) > tol: 
        print(f"⚠️ WARNUNG: API liefert abweichende Koordinate (> {tol}°)")

    # aus dem cache? -> no sleep
    if not response.from_cache:
        print(f"Neue API-Anfrage in {api_sleep_time} Sekunden...")
        time.sleep(api_sleep_time)
    else:
        print("Cache-Treffer – kein Sleep.")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["date"] = pd.to_datetime(df["time"])
    df.drop("time", axis=1, inplace=True)
    print("-"*30)
    return df

def get_dataframe_dict(loc:dict, start_date:str, end_date:str, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> dict[str, list[pd.DataFrame]]:
    """
    Erstellt ein Dictionary von DataFrames (key: Bundesland, value: DF mit den API-Daten je Location)
    """
    df_dict = defaultdict(list) #dict values per default auf list
    bl_counter = 0
    count_progress = 0
    total = sum(len(coords) for coords in loc.values())

    for bl, coord in loc.items():
        bl_counter += 1
        print("."*30)
        print(f"Daten für Bundesland {bl_counter} von 17")
        for lat, lon in coord:
            count_progress += 1
            if progress_cb:
                progress_cb(count_progress, total, f"History für {bl}: {lat:.2f},{lon:.2f}\n(ca. 25 Sekunden pro Standort)")
            #API aufruf:
            df = get_weather_by_location(lat, lon, start_date, end_date)
            df_dict[bl].append(df)
    return df_dict

def bl_means(bl_dict:dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """
    Mittelwert der DataFrames für jedes Bundesland nach Datum.
    --> Dict mit BL als Key und ein aggregiertes (mean) DF als Value
    """
    return {bl: pd.concat(dfs, axis=0).groupby('date').mean() for bl, dfs in bl_dict.items()}

def add_features(dfs:dict[str,pd.DataFrame], sun_weights:dict[str,dict[str,float]], wind_weights:dict[str,dict[str,float]]
                ) -> dict[str,pd.DataFrame]:
    """
    Gewichtungen für Sonnen- und Windwerte sowie daraus berechnete Features zu den DataFrames der Bundesländer hinzufügen.
    """
    new_dfs = {}
    for bundesland in dfs:
        df = dfs[bundesland].copy()
        sun_spalte = []
        wind_spalte = []
        for i in range(len(df)):
            jahr = str(df.index[i].year)
            sun_wert = sun_weights.get(jahr, {}).get(bundesland, 0)
            wind_wert = wind_weights.get(jahr, {}).get(bundesland, 0)
            sun_spalte.append(sun_wert)
            wind_spalte.append(wind_wert)
        df["sun_weight"] = sun_spalte
        df["wind_weight"] = wind_spalte
        df["sunhours * gewichtung(sun)"] = df["sunshine_duration"] * df["sun_weight"]
        df["GTI * gewichtung(sun)"] = df["global_tilted_irradiance"] * df["sun_weight"]
        df["windspeed * gewichtung(wind)"] = df["wind_speed_100m"] * df["wind_weight"]
        new_dfs[bundesland] = df
    # first_bl, first_df = next(iter(new_dfs.items()))
    # print(f"werte für {first_bl}")
    # show(first_df)
    return new_dfs

def combine_all_dfs(dfs:dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Spaltennamen aus dem ersten DataFrame im Dictionary
    first_df = next(iter(dfs.values()))
    columns = first_df.columns
    # Überprüfen, ob alle DataFrames dieselben Spalten(namen) haben
    for key, df in dfs.items():
        if not df.columns.equals(columns):
            raise ValueError(f"DataFrame '{key}' hat andere Spaltennamen als erwartet.")
    
    sum_cols = ["sunhours * gewichtung(sun)", "GTI * gewichtung(sun)", "windspeed * gewichtung(wind)", "sun_weight", "wind_weight"]

    concated_df = pd.concat(dfs, axis=1) #Multiindex = Dict-Key (level=0)
    grouped = concated_df.T.groupby(level=1) #Spaltennamen (level=1) 
    sum_df  = grouped.sum().T[sum_cols]   
    mean_df = grouped.mean().T.drop(columns=sum_cols)
    combined_df = pd.concat([mean_df, sum_df], axis=1)[columns]  
    # for col in columns:
    #     col_list = [df[col] for df in dfs.values()]
    #     concat_cols = pd.concat(col_list, axis=1)
    #     if col in sum_cols:
    #         combined_df[col] = concat_cols.sum(axis=1)
    #     else:
    #         combined_df[col] = concat_cols.mean(axis=1)
    #show(combined_df)
    return combined_df

def add_source(df:pd.DataFrame) -> pd.DataFrame:
    df["Quelle"] = "history"
    return df

def func_all_funcs_history(start_date:str=None, end_date:str=None, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
    print("Dict mit DF's aller Locations wird erstellt...")
    bl_dict = get_dataframe_dict(location, start_date, end_date, progress_cb=progress_cb)
    print("Starte Preprocessing:")
    bl_mean_dict = bl_means(bl_dict)
    print("Means über die BL berechnen -> fertig")
    df_features = add_features(bl_mean_dict, sun_weights, wind_weights)
    print("Features hinzufügen -> fertig")
    df_combined = combine_all_dfs(df_features)
    print("DF's zusammenfügen -> fertig")
    df_final = add_source(df_combined)
    print("Quelle hinzufügen -> fertig")
    return df_final

if __name__ == "__main__":
    df = func_all_funcs_history()
    # show(df, block=True)




