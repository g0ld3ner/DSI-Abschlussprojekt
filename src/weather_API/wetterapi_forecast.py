from datetime import datetime, timedelta
import time
import pandas as pd
import openmeteo_requests
import requests_cache
import requests #nur zum testen
from retry_requests import retry
from locations import location as loc
from pandasgui import show
from bundeslaender_gewichte import sun_weights, wind_weights

data_dir = "data/"

api_sleep_time = 3

cache_session = requests_cache.CachedSession(f"{data_dir}.forecast_cache", expire_after=3600)
session = retry(cache_session, retries=5, backoff_factor=0.5)

def get_weather_forecast_by_location(lat: float, lon: float) -> pd.DataFrame:
    print(f"Start Wetterdaten für {lat}, {lon}")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m", "wind_speed_80m", "wind_speed_120m",
            "sunshine_duration", "global_tilted_irradiance"
        ],
        "past_days": 2,
        "forecast_days": 8,
        "tilt": 35
    }

    start = time.time()
    response = session.get(url, params=params)
    dauer = time.time() - start
    print(f"==> Antwort erhalten nach {dauer:.2f} Sekunden (Cache: {response.from_cache})")

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

def get_dataframe_list(coordinates: list) -> list:
    dataframes = []
    counter = 0
    for lat, lon in coordinates:
        counter += 1
        print(f"Forecast")
        print(f"Location Nr. {counter}")
        df = get_weather_forecast_by_location(lat, lon)
        dataframes.append(df)
        # time.sleep(api_sleep_time)
    return dataframes


def bl_df_dict(loc: dict) -> dict:
    counter = 0
    bl_dict = {}
    for bl, coordinates in loc.items():
        counter += 1
        print("."*30)
        print(f"Forecast")
        print(f"Bundesland Nr. {counter}")
        c = get_dataframe_list(coordinates)
        bl_dict[bl] = c
    return bl_dict


def bl_means(bl_dict: dict) -> dict:
    bl_mean_dict = {}
    for bl, liste in bl_dict.items():
        df = pd.concat(liste)
        df_mean = df.groupby("date").mean()
        bl_mean_dict[bl] = df_mean
    return bl_mean_dict

def windspeed_mean_80m_120m(dataframes: dict) -> dict:
    neue_dataframes = {}
    for bundesland in dataframes:
        df = dataframes[bundesland].copy()
        df["wind_speed_100m"] = df[["wind_speed_120m", "wind_speed_80m"]].mean(axis=1)
        neue_dataframes[bundesland] = df
    return neue_dataframes

def drop_unbrauchbare_spalten(dataframes: dict) -> dict:
    neue_dataframes = {}
    for bundesland in dataframes:
        df = dataframes[bundesland].copy()
        df = df.drop(columns=["wind_speed_120m", "wind_speed_80m"])
        neue_dataframes[bundesland] = df
    return neue_dataframes

def add_gewichtung(dataframes: dict, sun_weights: dict, wind_weights: dict) -> dict:
    neue_dataframes = {}
    for bundesland in dataframes:
        df = dataframes[bundesland].copy()
        if "date" in df.columns:
            df.set_index("date", inplace=True)
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
        neue_dataframes[bundesland] = df
    return neue_dataframes

#def anpassung_eingangsdaten1(dataframes: dict) -> dict:
#    neue_dataframes = {}
#    for bundesland in dataframes:
#        df = dataframes[bundesland].copy()
#        df["temperature_2m"] = df["temperature_2m"] / 2
#        df["sunshine_duration"] = df["sunshine_duration"] /2
#        df["global_tilted_irradiance"] = df["global_tilted_irradiance"] / 2
#        df["wind_speed_100m"] = df["wind_speed_100m"] / 2
#        df["sun_weight"] = df["sun_weight"] / 2
#        df["wind_weight"] = df["wind_weight"] / 2
#        neue_dataframes[bundesland] = df
#    return neue_dataframes

def add_sun_wind_features(dataframes: dict) -> dict:
    neue_dataframes = {}
    for bundesland in dataframes:
        df = dataframes[bundesland].copy()
        df["sunhours * gewichtung(sun)"] = df["sunshine_duration"] * df["sun_weight"]
        df["GTI * gewichtung(sun)"] = df["global_tilted_irradiance"] * df["sun_weight"]
        df["windspeed * gewichtung(wind)"] = df["wind_speed_100m"] * df["wind_weight"]
        neue_dataframes[bundesland] = df
    return neue_dataframes



def combine_dfs_columnwise_sum(dataframes: dict) -> pd.DataFrame:
    """
    Nimmt ein Dictionary von DataFrames entgegen, die alle dieselben Spaltennamen besitzen.
    Für jede Spalte wird zeilenweise die Summe aus den entsprechenden Spalten aller DataFrames berechnet.

    Parameter:
        dataframes: dict, wobei die Keys z. B. Namen (wie Bundesländer o.ä.) sind und die zugehörigen DataFrames enthalten.

    Rückgabe:
        Ein DataFrame mit denselben Spaltennamen, in dem jede Spalte die zeilenweise Summe
        der entsprechenden Spalten aus den übergebenen DataFrames enthält.
    """
    if not dataframes:
        return pd.DataFrame()

    # Holen der Spaltennamen aus dem ersten DataFrame im Dictionary
    first_df = next(iter(dataframes.values()))
    columns = first_df.columns

    # Überprüfen, ob alle DataFrames dieselben Spalten haben
    for key, df in dataframes.items():
        if not df.columns.equals(columns):
            raise ValueError(f"DataFrame '{key}' hat andere Spaltennamen als erwartet.")

    result = pd.DataFrame()

    # Für jede Spalte aus den DataFrames:
    for col in columns:
        # Extrahiere die Spalte 'col' aus jedem DataFrame
        series_list = [df[col] for df in dataframes.values()]
        # Zusammenführen der Series über den Index (outer join, damit alle Zeitstempel erhalten bleiben)
        merged = pd.concat(series_list, axis=1)
        # Berechne die zeilenweise Summe (NaN werden dabei standardmäßig ignoriert)
        result[col] = merged.sum(axis=1)

    return result


def combine_dfs_columnwise_mean(dataframes: dict) -> pd.DataFrame:
    """
    Nimmt ein Dictionary von DataFrames entgegen, die alle dieselben Spaltennamen besitzen.
    Für jede Spalte wird zeilenweise der Mittelwert aus den entsprechenden Spalten aller DataFrames berechnet.

    Parameter:
        dataframes: dict, wobei die Keys z. B. Namen (wie Bundesländer o.ä.) sind und die zugehörigen DataFrames enthalten.

    Rückgabe:
        Ein DataFrame mit denselben Spaltennamen, in dem jede Spalte den zeilenweisen Mittelwert
        der entsprechenden Spalten aus den übergebenen DataFrames enthält.
    """
    if not dataframes:
        return pd.DataFrame()

    # Holen der Spaltennamen aus dem ersten DataFrame im Dictionary
    first_df = next(iter(dataframes.values()))
    columns = first_df.columns

    # Überprüfen, ob alle DataFrames dieselben Spalten haben
    for key, df in dataframes.items():
        if not df.columns.equals(columns):
            raise ValueError(f"DataFrame '{key}' hat andere Spaltennamen als erwartet.")

    result = pd.DataFrame()

    # Für jede Spalte aus den DataFrames:
    for col in columns:
        # Extrahiere die Spalte 'col' aus jedem DataFrame
        series_list = [df[col] for df in dataframes.values()]
        # Zusammenführen der Series über den Index (outer join, damit alle Zeitstempel erhalten bleiben)
        merged = pd.concat(series_list, axis=1)
        # Berechne den zeilenweisen Mittelwert (NaN werden dabei standardmäßig ignoriert)
        result[col] = merged.mean(axis=1)

    return result

def combine_df(dataframe_mean:pd.DataFrame, dataframe_sum:pd.DataFrame) -> pd.DataFrame:
    df_m = dataframe_mean.copy()
    df_s = dataframe_sum.copy()

    spalten_ueberschreiben = ["sunhours * gewichtung(sun)","GTI * gewichtung(sun)","windspeed * gewichtung(wind)"]

    for spalte in spalten_ueberschreiben:
        df_m[spalte] = df_s[spalte]

    return df_m


def combine_df_angepasst(df_sum_weighted: pd.DataFrame, df_mean_unverändert: pd.DataFrame) -> pd.DataFrame:
    """
    Kombiniert zwei DataFrames:
    - übernimmt alle Spalten aus df_mean_unverändert
    - ersetzt bestimmte Spalten durch Werte aus df_sum_weighted (z. B. gewichtete Features)

    Annahme: Beide DataFrames haben denselben Index (Datum/Zeit) und kompatible Spalten.

    """
    df_combined = df_mean_unverändert.copy()

    spalten_ueberschreiben = [
        "sunhours * gewichtung(sun)",
        "GTI * gewichtung(sun)",
        "windspeed * gewichtung(wind)"
    ]

    for spalte in spalten_ueberschreiben:
        if spalte in df_sum_weighted.columns:
            df_combined[spalte] = df_sum_weighted[spalte]
        else:
            print(f"⚠️ Achtung: '{spalte}' nicht in Summen-DataFrame vorhanden.")

    return df_combined

#def anpassung_eingangsdaten2(dataframe_combine:pd.DataFrame) -> pd.DataFrame:
#
#    df = dataframe_combine.copy()
#    df["sunhours * gewichtung(sun)"] = df["sunhours * gewichtung(sun)"] *10
#    df["GTI * gewichtung(sun)"] = df["GTI * gewichtung(sun)"] *10
#    df["windspeed * gewichtung(wind)"] = df["windspeed * gewichtung(wind)"] *10
#    return df

def func_all_funcs_forecast() -> pd.DataFrame:
    print("start bl_df_dict...")

    df1 = bl_df_dict(loc)
    print("Alle locs der BL fertig...")

    df2 = bl_means(df1)
    print("Alle BL means fertig...")

    df3 = windspeed_mean_80m_120m(df2)
    print("Gesamt DF wind mean Fertig...")

    df4 = drop_unbrauchbare_spalten(df3)
    print("Gesamt DF drop spalten Fertig...")

    df5 = add_gewichtung(df4, sun_weights, wind_weights)
    print("Alle BL means fertig...")

    df6 = add_sun_wind_features(df5)
    print("Alle BL means fertig...")

    df7 = combine_dfs_columnwise_sum(df6)
    print("Alle BL means fertig...")

    df8 = combine_dfs_columnwise_mean(df5)
    print("Gesamt DF Fertig...")

    df9 = combine_df_angepasst(df7,df8)

    return df9


if __name__ == "__main__":
    df = func_all_funcs_forecast()
    # show(df, block=True)

