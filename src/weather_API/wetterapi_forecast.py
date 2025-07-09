from datetime import datetime, timedelta
import time
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from locations import location as loc
from pandasgui import show
from bundeslaender_gewichte import sun_weights, wind_weights

api_sleep_time = 10


def get_weather_forecast_by_location(lat: float, lon: float) -> pd.DataFrame:
    print(f"Start Wetterdaten für {lat}, {lon}")

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)


    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": ["temperature_2m", "wind_speed_80m", "wind_speed_120m", "sunshine_duration",
                   "global_tilted_irradiance"],
        "past_days": 2,
        "forecast_days": 7,
        "tilt": 35
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_80m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_120m = hourly.Variables(2).ValuesAsNumpy()
    hourly_sunshine_duration = hourly.Variables(3).ValuesAsNumpy()
    hourly_global_tilted_irradiance = hourly.Variables(4).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["wind_speed_80m"] = hourly_wind_speed_80m
    hourly_data["wind_speed_120m"] = hourly_wind_speed_120m
    hourly_data["sunshine_duration"] = hourly_sunshine_duration
    hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe


def get_dataframe_list(coordinates: list) -> list:
    dataframes = []
    counter = 0
    for lat, lon in coordinates:
        counter += 1
        print(f"Forecast")
        print(f"Location Nr. {counter}")
        df = get_weather_forecast_by_location(lat, lon)
        dataframes.append(df)
        time.sleep(api_sleep_time)
    return dataframes


def bl_df_dict(loc: dict) -> dict:
    counter = 0
    bl_dict = {}
    for bl, coordinates in loc.items():
        counter += 1
        print("------------------------")
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





"""
if __name__ == "__main__":
    print("main läuft")
    df_gesamt = func_all_funcs_forecast()
    show(df_gesamt, block=True)
"""

#test df1
#def gui_inspect_bl_df_dict(bl_dict: dict):
#    all_dfs = {}
#    for bundesland, df_list in bl_dict.items():
#        for i, df in enumerate(df_list):
#            key = f"{bundesland} - Standort {i+1}"
#            all_dfs[key] = df
#    show(**all_dfs)
#if __name__ == "__main__":
#    df1 = bl_df_dict(loc)                                          #korrekt
#    gui_inspect_bl_df_dict(df1)

#test df2, df3, df4, df5, df6
#def gui_inspect_bl_means_dict(bl_means_dict: dict):
#    all_dfs = {}
#    for bundesland, df in bl_means_dict.items():
#        key = f"{bundesland} - Mittelwert"
#        all_dfs[key] = df
#    show(**all_dfs)

#if __name__ == "__main__":
#    df1 = bl_df_dict(loc)                                           #korrekt
#    df2 = bl_means(df1)                                             #korrekt
#    df3 = windspeed_mean_80m_120m(df2)                              #korrekt
#    df4 = drop_unbrauchbare_spalten(df3)                            #korrekt
#    df5 = add_gewichtung(df4, sun_weights, wind_weights)            #korrekt
#    df6 = add_sun_wind_features(df5)                                #korrekt
#
#
#    gui_inspect_bl_means_dict(df6)

#Test df7

"""
def pruefe_combine_dfs_columnwise_sum(df_dict: dict, df_summed: pd.DataFrame):
    test_col = "sunhours * gewichtung(sun)"

    # Nimm beliebigen Zeitstempel zum Testen
    zeitstempel = df_summed.index[0]

    # Hole den erwarteten Summenwert durch manuelles Addieren aus den einzelnen BL-DataFrames
    erwartete_summe = 0
    for bl, df in df_dict.items():
        if zeitstempel in df.index:
            wert = df.loc[zeitstempel, test_col]
            erwartete_summe += wert

    berechnete_summe = df_summed.loc[zeitstempel, test_col]
    print(f"[TEST] Zeit: {zeitstempel}")
    print(f"Erwartet:  {erwartete_summe}")
    print(f"Tatsächlich: {berechnete_summe}")

    if abs(erwartete_summe - berechnete_summe) < 1e-6:
        print("✅ Test erfolgreich: Summenwert stimmt.")
    else:
        print("❌ Test fehlgeschlagen: Summenwert stimmt nicht.")

    # Optional: Weitere Plausibilitätstests
    assert not df_summed.isnull().values.any(), "❌ Es gibt NaN-Werte im summierten DataFrame!"
    print("✅ Keine NaNs im kombinierten DataFrame.")

if __name__ == "__main__":
    df1 = bl_df_dict(loc)
    df2 = bl_means(df1)
    df3 = windspeed_mean_80m_120m(df2)
    df4 = drop_unbrauchbare_spalten(df3)
    df5 = add_gewichtung(df4, sun_weights, wind_weights)
    df6 = add_sun_wind_features(df5)
    df7 = combine_dfs_columnwise_sum(df6)
    # Testfunktion aufrufen:
    pruefe_combine_dfs_columnwise_sum(df6, df7)                           #korrekt
    # DataFrame anzeigen
    show(df7, block=True)
"""

"""
def pruefe_combine_dfs_columnwise_mean(df_dict: dict, df_mean: pd.DataFrame):
    test_col = "sunhours * gewichtung(sun)"

    # Beispiel-Zeitstempel
    zeitstempel = df_mean.index[0]

    # Erwarteter Mittelwert (manuell berechnet)
    werte = []
    for bl, df in df_dict.items():
        if zeitstempel in df.index:
            werte.append(df.loc[zeitstempel, test_col])

    if not werte:
        print(f"⚠️ Kein Wert vorhanden für Zeitstempel {zeitstempel} in den Einzeldaten.")
        return

    erwarteter_mittelwert = sum(werte) / len(werte)
    berechneter_mittelwert = df_mean.loc[zeitstempel, test_col]

    print(f"[TEST] Zeit: {zeitstempel}")
    print(f"Erwartet:  {erwarteter_mittelwert}")
    print(f"Tatsächlich: {berechneter_mittelwert}")

    if abs(erwarteter_mittelwert - berechneter_mittelwert) < 1e-6:
        print("✅ Test erfolgreich: Mittelwert stimmt.")
    else:
        print("❌ Test fehlgeschlagen: Mittelwert stimmt nicht.")

    # Zusätzliche Plausibilitätsprüfung
    assert not df_mean.isnull().values.any(), "❌ Es gibt NaN-Werte im Mittelwert-DataFrame!"
    print("✅ Keine NaNs im kombinierten Mittelwert-DataFrame.")

if __name__ == "__main__":
    df1 = bl_df_dict(loc)
    df2 = bl_means(df1)
    df3 = windspeed_mean_80m_120m(df2)
    df4 = drop_unbrauchbare_spalten(df3)
    df5 = add_gewichtung(df4, sun_weights, wind_weights)
    df6 = add_sun_wind_features(df5)
    df8 = combine_dfs_columnwise_mean(df6)
    # Testfunktion aufrufen
    pruefe_combine_dfs_columnwise_mean(df6, df8)

    # DataFrame anzeigen
    show(df8, block=True)
"""

"""
def pruefe_combine_df(df_sum: pd.DataFrame, df_mean: pd.DataFrame, df_combined: pd.DataFrame):
    spalten_ueberschreiben = [
        "sunhours * gewichtung(sun)",
        "GTI * gewichtung(sun)",
        "windspeed * gewichtung(wind)"
    ]

    for spalte in spalten_ueberschreiben:
        unterschied = (df_combined[spalte] - df_sum[spalte]).abs().max()
        if unterschied < 1e-6:
            print(f"✅ Spalte '{spalte}' korrekt überschrieben mit Summenwerten.")
        else:
            print(f"❌ Spalte '{spalte}' wurde NICHT korrekt überschrieben!")
            print(f"Maximaler Unterschied: {unterschied}")

    # Prüfung auf Gleichheit anderer Spalten mit dem Mittelwert-DF
    for spalte in df_mean.columns:
        if spalte not in spalten_ueberschreiben:
            unterschied = (df_combined[spalte] - df_mean[spalte]).abs().max()
            if unterschied < 1e-6:
                print(f"✅ Spalte '{spalte}' korrekt vom Mittelwert übernommen.")
            else:
                print(f"❌ Spalte '{spalte}' unterscheidet sich vom Mittelwert!")
                print(f"Maximaler Unterschied: {unterschied}")


if __name__ == "__main__":
    df1 = bl_df_dict(loc)
    df2 = bl_means(df1)
    df3 = windspeed_mean_80m_120m(df2)
    df4 = drop_unbrauchbare_spalten(df3)
    df5 = add_gewichtung(df4, sun_weights, wind_weights)
    df6 = add_sun_wind_features(df5)
    df7 = combine_dfs_columnwise_sum(df6)
    df8 = combine_dfs_columnwise_mean(df5)
    df9 = combine_df_angepasst(df7,df8)

    # Testfunktion
    pruefe_combine_df(df7, df8, df9)

    # Ergebnis anzeigen
    show(df9, block=True)

"""