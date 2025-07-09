from datetime import datetime, timedelta
import time
import pickle
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from bundeslaender_gewichte import sun_weights, wind_weights
from locations import location as loc
from pandasgui import show

api_sleep_time = 120


def current_day(minus: int = 0) -> str:
    return (datetime.today() - timedelta(days=minus)).strftime("%Y-%m-%d")

def x_days_back(date: str, x:int = 800) -> str:
    given_date = datetime.strptime(date, "%Y-%m-%d").date()
    return (given_date - timedelta(days=x)).strftime("%Y-%m-%d")

def get_weather_by_location(lat: float, lon: float, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    print(f"Start Wetterdaten für {lat}, {lon}")

    if end_date is None:
        end_date = current_day(minus=2)

    if start_date is None:
        start_date = x_days_back(end_date)

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "wind_speed_100m", "sunshine_duration", "global_tilted_irradiance"],
        #"timezone": "Europe/Berlin",
        "tilt": 35
    }
    print("Sende API-Anfrage...")
    responses = openmeteo.weather_api(url, params=params)
    print("Antwort erhalten")
    response = responses[0]
    print("Verarbeite Daten...")
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "sunshine_duration": hourly.Variables(2).ValuesAsNumpy(),
        "global_tilted_irradiance": hourly.Variables(3).ValuesAsNumpy(),
        "wind_speed_100m": hourly.Variables(1).ValuesAsNumpy()
    }
    print("DataFrame wird erstellt")
    return pd.DataFrame(data=hourly_data)


def get_dataframe_list(coordinates: list, start_day: str = None, end_day: str = None) -> list:
    dataframes = []
    counter = 0
    for lat, lon in coordinates:
        counter += 1
        print(f"Historische daten")
        print(f"Location Nr. {counter}")
        df = get_weather_by_location(lat, lon, start_day, end_day)
        dataframes.append(df)
        time.sleep(api_sleep_time)
    return dataframes


def bl_df_dict(loc: dict) -> dict:
    counter = 0
    bl_dict = {}
    for bl, coordinates in loc.items():
        counter += 1
        print("------------------------")
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


# def combine_df_and_drop_unnoetige_columns(dataframe1, dataframe2) -> pd.DataFrame:

#     spalten_weg_1 = ["sunhours * gewichtung(sun)","GTI * gewichtung(sun)","windspeed * gewichtung(wind)"]

#     df1 = dataframe1.copy()
#     for spalte in spalten_weg_1:
#         if spalte in df1.columns:
#             df1.drop(columns=[spalte], inplace=True)

#     df2 = dataframe2.copy()
#     spalten_behalten_2 = []
#     for spalte in spalten_weg_1:
#         if spalte in df2.columns:
#             spalten_behalten_2.append(spalte)
#     df2 = df2[spalten_behalten_2]


#     df_kombiniert = pd.concat([df1, df2], axis=1)

#     return df_kombiniert

def rename_spalten(dataframes: dict) -> dict:
    neue_dataframes = {}
    for bundesland in dataframes:
        df = dataframes[bundesland].copy()
        df.rename(columns={
            'temperature_2m': f'temperature_2m_{bundesland}',
            'wind_speed_100m': f'wind_speed_100m_{bundesland}',
            'sunshine_duration': f'sunshine_duration_{bundesland}',
            'global_tilted_irradiance': f'global_tilted_irradiance_{bundesland}',
            'sun_weight': f'sun_weight_{bundesland}',
            'wind_weight': f'wind_weight_{bundesland}',
            'sunhours * gewichtung(sun)': f'sunhours * gewichtung(sun)_{bundesland}',
            'GTI * gewichtung(sun)': f'GTI * gewichtung(sun)_{bundesland}',
            'windspeed * gewichtung(wind)': f'windspeed * gewichtung(wind)_{bundesland}'
        }, inplace=True)
        neue_dataframes[bundesland] = df
    return neue_dataframes


def alle_df_bl_zusammen(dataframes: dict) -> pd.DataFrame:
    df_gesamt = None
    for bl, df in dataframes.items():
        if df_gesamt is None:
            df_gesamt = df.copy()
        else:
            df_gesamt = pd.merge(df_gesamt, df, left_index=True, right_index=True, how="left")
    return df_gesamt


def func_all_funcs() -> pd.DataFrame:
    print("start bl_df_dict...")
    bl_dict = bl_df_dict(loc)
    print("means berechnen fertig...")
    bl_mean_dict = bl_means(bl_dict)
    print("gewichtung append fertig...")
    df_gewichte = add_gewichtung(bl_mean_dict, sun_weights, wind_weights)
    print("features berechnen fertig...")
    df_features = add_sun_wind_features(df_gewichte)
    print("rename spalten fertig...")
    df_bl_names = rename_spalten(df_features)
    print("alles zusammen fertig...")
    df_gesamt = alle_df_bl_zusammen(df_bl_names)
    print("Fertig!")
    return df_gesamt


def func_all_funcs_de() -> pd.DataFrame:
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
    return df_gesamt_dl


if __name__ == "__main__":
    print("main läuft")
    df_gesamt = func_all_funcs_de()
    print("-" * 50)
    print("df_gesamt")
    print(df_gesamt)
    print("-" * 50)
    wt = "df_wetter_test.pkl"
    with open(wt, "wb") as file:
        pickle.dump(df_gesamt, file)
    show(df_gesamt, block=True)

    # df_all_bl = func_all_funcs()

    # all_bl = "df_all_bl.pkl"
    # with open(all_bl, "wb") as file:
    #     pickle.dump(df_all_bl, file)

    # show(df_all_bl)


