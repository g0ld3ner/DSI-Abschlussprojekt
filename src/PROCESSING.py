import pandas as pd
import datetime
from WINSORIZING import weekend_and_winsorize
from WETTER_SIMULATION import simulate_weather_data

import BLOCKWISE
import SINGLE_STEP

import BLOCKWISEwetter

subpath = "data/"  # alle Ergebnis .pkl müssen in den unterordner "data"

def current_day(minus: int = 0) -> str:
    return (datetime.date.today() - datetime.timedelta(days=minus)).strftime("%Y-%m-%d %H:%M:%S")

######################### alles für die wettersimulation
def prepare_for_weathermodel(start=None, sun=None, wind=None) -> pd.DataFrame:
    if start == None:
        start = current_day(7)
    if sun == None:
        sun = 0.6
    if wind == None:
        wind = 0.5

    print(start)
    df = pd.read_pickle(subpath + "df_for_model.pkl")
    df = weekend_and_winsorize(df=df)
    df = simulate_weather_data(df=df,sun_scale=sun,wind_factor=wind)
    # nur für Daten ab 'start' die simulierten Windwerte übernehmen:
    mask = df.index >= start
    df.loc[mask, "windspeed * gewichtung(wind)"] = df.loc[mask, "wind_sim"]
    # und analog die simulierten GTI-Werte übernehmen:
    df.loc[mask, "GTI * gewichtung(sun)"] = df.loc[mask, "gti_sim"]
    pd.to_pickle(df,subpath + "ready_for_WETTER.pkl")
    return df

def weather_blockwise(start_date: str) -> tuple[dict, dict]:
    df_result_block, metrics_block = BLOCKWISEwetter.train_and_forecast_blockwise(BLOCKWISEwetter.CONFIG, forecast_start_date=start_date)
    print("SIMULATION DURCHGEFÜHRT")
    df_result_block.to_pickle(subpath + "df_blockwise_result_WETTER.pkl")
    print("PICKEL GESPEICHERT")
    df_loaded_block, load_metrics_block = BLOCKWISEwetter.load_and_forecast_blockwise(BLOCKWISEwetter.CONFIG, forecast_start_date=start_date)
    df_loaded_block.to_pickle(subpath + "df_blockwise_loaded_WETTER.pkl")
    # Metriken beider als Tuple[dict, dict]
    return load_metrics_block #,metrics_block

def processing_weather(start:int=7, sun = None, wind = None ):  #per Stremlit DAtum als auswahl übergeben wind und sonne per regler 0 - 1.1
    start_date = current_day(start)
    df = prepare_for_weathermodel(start=start_date,sun=sun,wind=wind)
    print("PREPARED")
    # BLOCKWISE und SINGLE_STEP nutzen die "ready_for_MODEL.pkl"
    metric_tuple_block = weather_blockwise(start_date=start_date)
    print("METRIK")
    # metric_tuple_single = train_single_step(start_date=start_date)
    return metric_tuple_block#, metric_tuple_single
#########################


####ich muss dass preparierete nutzen
def prepare_for_model() -> pd.DataFrame:
    df = pd.read_pickle(subpath + "df_for_model.pkl")
    df = weekend_and_winsorize(df=df)
    df = simulate_weather_data(df=df)
    pd.to_pickle(df,subpath + "ready_for_MODEL.pkl")
    return df


def train_blockwise(start_date: str) -> tuple[dict, dict]:
    df_result_block, metrics_block = BLOCKWISE.train_and_forecast_blockwise(BLOCKWISE.CONFIG, forecast_start_date=start_date)
    df_result_block.to_pickle(subpath + "df_blockwise_result.pkl")
    df_loaded_block, load_metrics_block = BLOCKWISE.load_and_forecast_blockwise(BLOCKWISE.CONFIG, forecast_start_date=start_date)
    df_loaded_block.to_pickle(subpath + "df_blockwise_loaded.pkl")
    # Metriken beider als Tuple[dict, dict]
    return metrics_block, load_metrics_block

def train_single_step(start_date: str) -> tuple[dict, dict]:
    df_result_single, metrics_single = SINGLE_STEP.train_and_forecast_single_step(SINGLE_STEP.CONFIG, forecast_start_date=start_date)
    df_result_single.to_pickle(subpath + "df_single_step_result.pkl")
    df_loaded_single, load_metrics_single = SINGLE_STEP.load_and_forecast_single_step(SINGLE_STEP.CONFIG, forecast_start_date=start_date)
    df_loaded_single.to_pickle(subpath + "df_single_step_loaded.pkl")
    # Metriken beider als Tuple[dict, dict]
    return metrics_single, load_metrics_single


def processing_all(start:int=9):
    """
    Führt den gesamten Verarbeitungsprozess durch, um die Daten für das Modell vorzubereiten und zu trainieren.

    Diese Funktion erstellt die Datei "ready_for_MODEL.pkl" aus der Datei "df_for_model.pkl", 
    welche aus dem Skript "wetterapi_gesamt.py" generiert wird. Anschließend werden zwei 
    Trainingsmethoden (BLOCKWISE und SINGLE_STEP) auf den vorbereiteten Daten ausgeführt.

    Args:
        start (int, optional): Die Anzahl der Tage, die vom aktuellen Datum subtrahiert werden, 
        um das Startdatum zu bestimmen. Standardwert ist 9.

    Returns:
        tuple: Ein Tupel bestehend aus den Metriken der beiden Trainingsmethoden:
            - metric_tuple_block: Metriken des blockweisen Trainings.
            - metric_tuple_single: Metriken des schrittweisen Trainings.
    """
    # erzeugt die "ready_for_MODEL.pkl" aus der "df_for_model.pkl", welche aus der "wetterapi_gesamt.py" kommt
    df = prepare_for_model()
    start_date = current_day(start)
    # BLOCKWISE und SINGLE_STEP nutzen die "ready_for_MODEL.pkl"
    metric_tuple_block = train_blockwise(start_date=start_date)
    metric_tuple_single = train_single_step(start_date=start_date)
    return metric_tuple_block, metric_tuple_single




if __name__ == '__main__':
########
    test_df = prepare_for_model()
    print(test_df.info())
    print("="*50)
########
    # start_date = "2025-04-01 00:00:00"
    start_date = current_day(8)
    print(start_date)
    print("="*50)
########
    df_result_block, metrics_block = BLOCKWISE.train_and_forecast_blockwise(BLOCKWISE.CONFIG, forecast_start_date=start_date)
    df_result_block.to_pickle(subpath + "df_blockwise_result.pkl")
    
    df_loaded_block, load_metrics_block = BLOCKWISE.load_and_forecast_blockwise(BLOCKWISE.CONFIG, forecast_start_date=start_date)
    df_loaded_block.to_pickle(subpath + "df_blockwise_loaded.pkl")
    
    print("[Blockwise] Fertig!")
    print("-"*50)
    print(df_result_block.info())
    print("-"*50)
    print(df_loaded_block.info())
    print("="*50)
########
    df_result_single, metrics_single = SINGLE_STEP.train_and_forecast_single_step(SINGLE_STEP.CONFIG, forecast_start_date=start_date)
    df_result_single.to_pickle(subpath + "df_single_step_result.pkl")
    
    df_loaded_single, load_metrics_single = SINGLE_STEP.load_and_forecast_single_step(SINGLE_STEP.CONFIG, forecast_start_date=start_date)
    df_loaded_single.to_pickle(subpath + "df_single_step_loaded.pkl")
    
    print("[Single-Step] Fertig!")
    print("-"*50)
    print(df_result_single.info())
    print("-"*50)
    print(df_loaded_single.info())
    print("="*50)
########