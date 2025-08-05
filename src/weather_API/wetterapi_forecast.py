from datetime import datetime, timedelta
import time
import pandas as pd
import openmeteo_requests # wird nicht mehr gebraucht
import requests_cache
import requests #nur zum testen
from retry_requests import retry
from locations import location
from collections import defaultdict
# from pandasgui import show #nur für debuggingzwecke
from weather_API.bundeslaender_gewichte import sun_weights, wind_weights

data_dir = "data/"

api_sleep_time = 3

cache_session = requests_cache.CachedSession(f"{data_dir}.forecast_cache", expire_after=3600) #1h
session = retry(cache_session, retries=5, backoff_factor=0.5)

def get_weather_forecast_by_location(lat: float, lon: float, past_days:int|None = 2, forecast_days:int|None = 7) -> pd.DataFrame:
    print(f"Start Wetterdaten für {lat}, {lon}")

    if past_days is None:
        past_days = 2
    if forecast_days is None:
        forecast_days = 7

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m", "wind_speed_80m", "wind_speed_120m",
            "sunshine_duration", "global_tilted_irradiance"
        ],
        "past_days": 2, #past_days, #2
        "forecast_days": 7, #forecast_days, #7
        "tilt": 35
    }

    start = time.time()
    response = session.get(url, params=params)
    dauer = time.time() - start
    print(f"Antwort erhalten nach {dauer:.2f} Sekunden ==> (Cache: {response.from_cache})")

    if response.status_code != 200:
        raise RuntimeError(f"API-Fehler: {response.status_code} {response.text[:200]}")

    data = response.json()

   # Echtheitsprüfung der Koordinaten
    tol = 0.11
    lat_api = data.get("latitude", -999)
    lon_api = data.get("longitude", -999)
    print(f"API-Koordinate: {lat_api:.5f}, {lon_api:.5f} (Angefragt: {lat:.5f}, {lon:.5f})")
    if abs(lat_api - lat) > tol or abs(lon_api - lon) > tol: # Warnung bei abweichung ~>10km
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

def get_dataframe_dict(loc:dict, past_days:int, forecast_days:int) -> dict[str, list[pd.DataFrame]]:
    """
    Erstellt ein Dictionary von DataFrames (key: Bundesland, value: DF mit den API-Daten je Location)
    """
    df_dict = defaultdict(list) #dict values per default auf list
    counter = 0
    for bl, coord in loc.items():
        counter += 1
        print("."*30)
        print(f"Daten für Bundesland {counter} von 17")
        for lat, lon in coord:
            df = get_weather_forecast_by_location(lat, lon, past_days, forecast_days)
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
        df["wind_speed_100m"] = df[["wind_speed_120m", "wind_speed_80m"]].mean(axis=1)
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
    df["Quelle"] = "forecast"
    return df

def func_all_funcs_forecast(past_days:int = None, forecast_days:int = None) -> pd.DataFrame:
    print("Dict mit DF's aller Locations wird erstellt...")
    bl_dict = get_dataframe_dict(location, past_days, forecast_days)
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
    df = func_all_funcs_forecast()
    # show(df, block=True)

