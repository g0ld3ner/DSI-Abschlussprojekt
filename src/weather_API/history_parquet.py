import pandas as pd
from pathlib import Path
import requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone
import time
from pandasgui import show

from locations import location

data_dir = "data/"
data_db = "wetter_history.parquet"
path = f"{data_dir}{data_db}"

api_sleep_time = 25

# API Session global cachen   #### am ende wieder entfernen!!!!!!!
cache_session = requests_cache.CachedSession(f"{data_dir}.parquet_cache", expire_after=3600)
session = retry(cache_session, retries=5, backoff_factor=0.2)

def load_parquet(path):
    if Path(path).exists():
        return pd.read_parquet(path)
    else:
        print("keine Datei gefunden! --> leeres DF erzeugt")
        return pd.DataFrame() #leeres DF als fallback

def save_parquet(df, path):
    df.to_parquet(path)
    print(f"Wetterdatenbank gespeichert: {path}")


def get_weather_from_api(lat:float, lon:float, start_date:datetime|str, end_date:datetime|str, bundesland:str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,  # Format "YYYY-MM-DD"
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "wind_speed_100m", "sunshine_duration", "global_tilted_irradiance"
        ],
        "tilt": 35
    }

    # Anfrage senden
    print("."*30)
    print(f"Sende API-Anfrage:\n - {lat:.2f}\n - {lon:.2f}\n - {start_date}\n - {end_date}\n - {bundesland}")
    start_api = time.time()
    response = session.get(url, params=params)
    dauer = time.time() - start_api
    print(f"Antwort erhalten nach {dauer:.2f} Sekunden (Cache: {response.from_cache})")
    print(f"Aus dem Cache?: {response.from_cache}")

    if response.status_code != 200:
        raise RuntimeError(f"API-Fehler: {response.status_code} {response.text[:200]}")
    
    data = response.json()

    # Metadaten auslesen
    lat_api = data.get("latitude", None)
    lon_api = data.get("longitude", None)
    elevation = data.get("elevation", None)
    generationtime_ms = data.get("generationtime_ms", None)
    api_call_date = pd.Timestamp.now(tz="UTC")

    # Wetterdaten in DataFrame
    df = pd.DataFrame(data["hourly"])

    # Zeitstempel als Datetime index setzen
    df["date"] = pd.to_datetime(df["time"], utc=True)
    df.drop("time", axis=1, inplace=True)
    df = df.set_index("date")

    # zusätzliche Spalten
    df["bundesland"] = bundesland
    df["latitude"] = lat
    df["longitude"] = lon
    df["lat_api"] = lat_api
    df["lon_api"] = lon_api
    df["elevation"] = elevation
    df["generationtime_ms"] = generationtime_ms
    df["api_call_date"] = api_call_date

    # aus dem cache? -> no sleep
    if not response.from_cache:
        print(f"Neue API-Anfrage in {api_sleep_time} Sekunden...")
        time.sleep(api_sleep_time)
    else:
        print("Cache-Treffer – kein Sleep.")

    print("."*30)
    return df




def update_rolling_window(path, location, days=400, api_sleep_time=25):
    """
    Stellt sicher, dass das Parquet für alle Locations mindestens die letzten `days` Tage (bis heute-2) abdeckt.
    Holt nur, was fehlt. Lässt alte Daten komplett stehen!
    """
    end = datetime.now(timezone.utc) - timedelta(days=2) ### Die fehlenden 2 Tage kommen aus der forecast-API
    start = end - timedelta(days=days)
    print(f"Rolling Window Update: {start.date()} bis {end.date()}")

    df = load_parquet(path)

    bl_counter = 0
    loc_counter = 0

    for bundesland, coords_list in location.items():
        bl_counter += 1
        print("="*30)
        print(f"Bundesland {bl_counter}")
        for lat, lon in coords_list:
            loc_counter += 1
            print("-"*30)
            print(f"Location {loc_counter}")
            # Maske für "Primärschlüssel" (außer Date-Index)
            mask = (
                (df["bundesland"] == bundesland) &
                (df["latitude"] == lat) &
                (df["longitude"] == lon)
            ) if not df.empty else pd.Series(dtype=bool)
            # DF mit allen einträgen für den Standort
            df_loc = df.loc[mask] if not df.empty else pd.DataFrame()

            #letzer Eintrag für den Standort
            last = df_loc.index.max() if not df_loc.empty else None

            if last is None or last < start:
                missing_start = start
            else:
                missing_start = last 

            if missing_start <= end: # Prüfen ob überhaupt NAchgeladen werden muss oder wir auf Stand sind
                start_str = missing_start.strftime("%Y-%m-%d")
                end_str = end.strftime("%Y-%m-%d")
                print(f"Nachladen für {bundesland} ({lat:.2f}, {lon:.2f}): {start_str} bis {end_str}")
                # API request mit...
                max_retries = 3 # ...versuchen
                for attempt in range(1, max_retries + 1):
                    try:
                        df_new = get_weather_from_api(lat, lon, start_str, end_str, bundesland)
                        break
                    except Exception as e:
                        print(f"API-Fehler für {bundesland} ({lat:.2f}, {lon:.2f}), Versuch {attempt}/{max_retries}: {e}")
                        if attempt == max_retries:
                            print(f"Abbruch für {bundesland} ({lat:.2f}, {lon:.2f}) nach {max_retries} Fehlversuchen.")
                            df_new = None
    
                # Neue Einträge an das DF anhängen          
                if df_new is not None:
                    df = append_data(df, df_new)
            else:
                print(f"Schon aktuell für {bundesland} ({lat:.2f}, {lon:.2f})")
    
    # in die DB speichern
    save_parquet(df, path)
    print(f"Update abgeschlossen. {len(df)} Zeilen gespeichert.")

    return df


def append_data(df_old, df_new):
    """
    Hängt neue Wetterdaten an bestehendes DataFrame an und entfernt alle Zeilen,
    die im Primärschlüssel (date, bundesland, latitude, longitude) doppelt sind.
    """
    if df_old is None or df_old.empty:
        return df_new
    # DF Zeilenweise anhängen
    df = pd.concat([df_old, df_new])
    # Index "date" zur Spalte "date" machen
    df = df.reset_index()  
    # Entfernt alle echten Dubletten anhand der "Primärschlüssel-Kombination"
    before = len(df)
    df = df.drop_duplicates(subset=["date", "bundesland", "latitude", "longitude"], keep="last")
    after = len(df)
    print(f"append_data: {before - after} Duplikate entfernt")
    # Zeitstempel wieder als Index "date" setzen
    df = df.set_index("date")
    return df


#########################################################


###### noch komplett überarbeiten
# def fill_all_gaps(path, locations_dict, api_sleep_time=25):
#     """
#     Füllt alle Lücken (tageweise) für jede Location im kompletten Parquet.
#     Ziel: Vollständige Schattendatenbank, alle Tage lückenlos.
#     """
#     df = load_parquet(path)

#     for bundesland, coords_list in locations_dict.items():
#         for lat, lon in coords_list:
#             # Filter für Standort
#             mask = (
#                 (df["bundesland"] == bundesland) &
#                 (df["latitude"] == lat) &
#                 (df["longitude"] == lon)
#             ) if not df.empty else pd.Series(dtype=bool)
#             df_loc = df.loc[mask] if not df.empty else pd.DataFrame()

#             # Wenn es keinen Eintrag gibt dann wird der Standort übersprungen
#             if df_loc.empty:
#                 print(f"Kein Eintrag für {bundesland} ({lat:.2f}, {lon:.2f}) – wird übersprungen!")
#                 continue  # Hier ggf. default-Start/Ende setzen!

#             # Alle Tage (UTC!) von erstem bis letztem Eintrag
#             all_days = pd.date_range(
#                 start=df_loc.index.min(),
#                 end=df_loc.index.max(),
#                 freq="D"
#             )

#             # Tage, für die mindestens 24 Einträge (Stunden) existieren
#             df_loc["date_only"] = df_loc.index.normalize()
#             full_days = (
#                 df_loc.groupby("date_only").size()
#                 .pipe(lambda s: s[s == 24])
#                 .index
#             )
#             missing_days = sorted(set(all_days.date) - set(full_days))

#             if not missing_days:
#                 print(f"Keine Lücken für {bundesland} ({lat:.2f}, {lon:.2f})!")
#                 continue

#             print(f"{len(missing_days)} fehlende Tage für {bundesland} ({lat:.2f}, {lon:.2f}): {missing_days}")

#             for day in missing_days:
#                 day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
#                 print(f"Nachladen Tag: {day_str} für {bundesland} ({lat:.2f}, {lon:.2f})")
#                 try:
#                     df_new = get_weather_from_api(lat, lon, day_str, day_str, bundesland)
#                     df = append_data(df, df_new)
#                 except Exception as e:
#                     print(f"API-Fehler am {day_str} für {bundesland} ({lat:.2f}, {lon:.2f}): {e}")

#     save_parquet(df, path)
#     print(f"Lückenprüfung abgeschlossen. {len(df)} Zeilen gespeichert.")

#     return df



if __name__ == "__main__":
    update_rolling_window(path, location)
    df_test = pd.read_parquet(path=path)
    show(df_test)



