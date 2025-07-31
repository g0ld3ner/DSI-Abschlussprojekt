from datetime import datetime, timedelta, timezone
import time
import pickle
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from bundeslaender_gewichte import sun_weights, wind_weights
from locations import location as loc
from pandasgui import show

data_dir = "data/"

api_sleep_time = 25

cache_session = requests_cache.CachedSession(f"{data_dir}.history_cache", expire_after=604800)
session = retry(cache_session, retries=5, backoff_factor=0.5)

def current_day(minus: int = 0) -> datetime:
    return (datetime.now(timezone.utc) - timedelta(days=minus))

def x_days_back(date: datetime, x: int = 400) -> datetime:
    return date - timedelta(days=x)

def date_to_string(date: datetime) -> str: #die API braucht ja einen String
    return date.strftime("%Y-%m-%d")

def get_weather_by_location(lat: float, lon: float, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    print(f"Start Wetterdaten für {lat}, {lon}")

    if end_date is None:
        ### Die archive-API hängt immer 2 Tage hinterher, daher müssen diese 2 Tage aus der Forecast API aufgefüllt werden (daher minus=2)
        end_date = current_day(minus=2) # ob man das später nicht eleganter lösen kann mit der minus 2 ???
        end_date_str = date_to_string(end_date) 
    if start_date is None:
        start_date = x_days_back(end_date)
        start_date_str = date_to_string(start_date)

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
    print(f"==> Antwort erhalten nach {dauer:.2f} Sekunden (Cache: {response.from_cache})")

    

    if response.status_code != 200:
        raise RuntimeError(f"API-Fehler: {response.status_code} {response.text[:200]}")

    data = response.json()

    # Echtheitsprüfung der Koordinaten (wie im Forecast)
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

def get_dataframe_list(coordinates: list, start_day: str = None, end_day: str = None) -> list:
    dataframes = []
    counter = 0
    for lat, lon in coordinates:
        counter += 1
        print(f"Historische daten")
        print(f"Location Nr. {counter}")
        df = get_weather_by_location(lat, lon, start_day, end_day)
        dataframes.append(df)
        # time.sleep(api_sleep_time) # wird jetzt in der get_weather... dynamisch geregelt
    return dataframes


def bl_df_dict(loc: dict) -> dict:
    counter = 0
    bl_dict = {}
    for bl, coordinates in loc.items():
        counter += 1
        print("."*30)
        print(f"Historische daten")
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


def add_gewichtung(dataframes: dict, sun_weights: dict, wind_weights: dict) -> dict:
    neue_dataframes = {}
    for bundesland in dataframes:
        df = dataframes[bundesland].copy()
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


def combine_df(dataframe_mean: pd.DataFrame, dataframe_sum: pd.DataFrame) -> pd.DataFrame:
    df_m = dataframe_mean.copy()
    df_s = dataframe_sum.copy()

    spalten_ueberschreiben = ["sunhours * gewichtung(sun)", "GTI * gewichtung(sun)", "windspeed * gewichtung(wind)"]

    for spalte in spalten_ueberschreiben:
        df_m[spalte] = df_s[spalte]

    return df_m


def func_all_funcs_history() -> pd.DataFrame:
    bl_dict = bl_df_dict(loc)
    print("start bl_df_dict...")
    bl_mean_dict = bl_means(bl_dict)
    print("means berechnen fertig...")
    df_gewichte = add_gewichtung(bl_mean_dict, sun_weights, wind_weights)
    print("gewichtung append fertig...")
    df_features = add_sun_wind_features(df_gewichte)
    print("features berechnen fertig...")
    df_de_gesamt_sum = combine_dfs_columnwise_sum(df_features)
    print("DF Summen fertig....")
    df_de_gesamt_mean = combine_dfs_columnwise_mean(df_features)
    print("DF means fertig....")
    df_gesamt_dl = combine_df(df_de_gesamt_mean, df_de_gesamt_sum)
    print("DF Gesamt summen & means fertig....")
    ##### IDEE ###### das DF als pkl speichern, wenn es die pkl gibt, letztes datum nehmen,
    ########### den prozess ab dem datum bis heute laufen lassen, DF's zusammenführen
    return df_gesamt_dl


if __name__ == "__main__":
    df = func_all_funcs_history()
    # show(df, block=True)




