import pandas as pd
import numpy as np

def simulate_weather_data(df, sun_scale=0.6, wind_factor=0.5):
    """
    Simuliert Wetterdaten basierend auf dem übergebenen DataFrame.
    
    Erwartet:
      - df: DataFrame mit einem tz‑naiven DatetimeIndex und den Spalten:
            "GTI * gewichtung(sun)", "sunhours * gewichtung(sun)",
            "windspeed * gewichtung(wind)"
      - sun_scale: Skalierungsfaktor für die simulierten Sonnendaten (z. B. 1.0 entspricht 100%).
      - wind_factor: Skalierungsfaktor für die simulierten Winddaten (0 schaltet den Wind ab).
      
    Liefert:
      - Das DataFrame mit den zusätzlichen Spalten:
            "gti_peak_smooth", "gti_sim", "sunhours_sim", "wind_sim".
      Der ursprüngliche Index wird beibehalten.
    """
    df = df.copy()
    # Überprüfe, ob alle erforderlichen Spalten vorhanden sind.
    required_cols = ["GTI * gewichtung(sun)", "sunhours * gewichtung(sun)", "windspeed * gewichtung(wind)"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Erforderliche Spalte '{col}' fehlt im DataFrame.")
    
    # Es wird vorausgesetzt, dass df.index ein tz‑naiver DatetimeIndex ist.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Der DataFrame muss einen DatetimeIndex besitzen.")
    
    # Speichere den Originalindex, damit er später wiederhergestellt werden kann.
    df = df.copy()
    df["date"] = df.index
    
    # Erzeuge Hilfsspalten auf Monatsbasis (statt Woche) und Stunde.
    df["month"] = df.index.month
    df["hour"] = df.index.hour

    # Definiere eine Funktion zur Ermittlung des lokalen Maximums.
    # Werte unter 0.1 werden als Nacht interpretiert und liefern 0.
    def local_max_or_zero(series):
        nonzero = series[series > 0.1]
        return nonzero.max() if not nonzero.empty else 0.0

    # Gruppiere nach (Monat, Stunde) und bestimme den lokalen Maximalwert.
    peak_sun = df.groupby(["month", "hour"])["GTI * gewichtung(sun)"] \
                 .apply(local_max_or_zero) \
                 .reset_index(name="gti_peak")
    # Sortierung zur Übersicht: Erstelle einen 1D-Index aus Monat und Stunde.
    peak_sun["sort_index"] = (peak_sun["month"] - 1) * 24 + peak_sun["hour"]
    peak_sun.sort_values("sort_index", inplace=True)
    # Ohne Smoothing wird der ermittelte Wert direkt genutzt.
    peak_sun["gti_peak_smooth"] = peak_sun["gti_peak"]
    peak_sun.loc[peak_sun["gti_peak"] == 0, "gti_peak_smooth"] = 0.0

    # Merge: Führe die berechneten Werte anhand von month und hour in df ein.
    df = df.merge(peak_sun[["month", "hour", "gti_peak_smooth"]],
                  on=["month", "hour"], how="left")
    
    # Setze den ursprünglichen Index wieder ein.
    df.set_index("date", inplace=True)

    # Simulation der GTI-Daten: 
    # Die simulierten GTI-Werte ergeben sich aus der Oberkurve multipliziert mit der Sonnenskala.
    df["gti_sim"] = df["gti_peak_smooth"] * sun_scale
    df["sunhours_sim"] = df["sunhours * gewichtung(sun)"] * sun_scale

    ### Wind "zufälliger" simulieren (noch testen)
    # #############
    # # --- Wind-Simulation: stündliche Quantile + stochastische Schwankung ---
    # wq = df.groupby("hour")["windspeed * gewichtung(wind)"].quantile([0.0001, 0.999]).unstack()
    # if wq.shape[1] != 2:
    #     df["wind_sim"] = np.nan
    #     raise ValueError("Windquantile konnten nicht berechnet werden.")

    # wq.columns = ["wind_min", "wind_max"]
    # df = df.join(wq, on="hour")

    # if wind_factor == 0:
    #     # kompletter „Wind aus“
    #     df["wind_sim"] = 0.0
    # else:
    #     # Zielniveau innerhalb der stündlichen Bandbreite
    #     mu = df["wind_min"] + wind_factor * (df["wind_max"] - df["wind_min"])

    #     # Rauschparameter
    #     noise_scale = 0.18  # 0.10–0.25 typ. sinnvoll
    #     phi = 0.7           # AR(1): 0 = weißes Rauschen, 0.9 = sehr glatt
    #     smooth_window = 3   # optionales Nachglätten (Stunden); 1/0 = aus

    #     # Bandbreite pro Zeitschritt (numerisch stabil halten)
    #     band = (df["wind_max"] - df["wind_min"]).clip(lower=1e-6)
    #     sigma_t = noise_scale * band * np.sqrt(wind_factor)

    #     # Zeitlich korreliertes Rauschen (AR(1)) als Series über dem Index
    #     rng = np.random.default_rng(42)  # reproduzierbar
    #     eta = rng.normal(loc=0.0, scale=1.0, size=len(df))
    #     eps = pd.Series(0.0, index=df.index)
    #     for t in range(1, len(df)):
    #         eps.iloc[t] = phi * eps.iloc[t-1] + np.sqrt(1 - phi**2) * eta[t]

    #     # Stochastische Simulation um das Zielniveau
    #     wind_raw = mu + eps * sigma_t

    #     # Auf empirische Grenzen je Stunde beschneiden
    #     wind_clipped = wind_raw.clip(lower=df["wind_min"], upper=df["wind_max"])

    #     # Sanft glätten (optional)
    #     df["wind_sim"] = (
    #         wind_clipped.rolling(window=smooth_window, min_periods=1).mean()
    #         if smooth_window and smooth_window > 1 else wind_clipped
    #     )
    # ####################

    # Wind-Simulation: Pro Stunde werden Extremquantile (0.01%-Quantil und 99.9%-Quantil) ermittelt.
    wq = df.groupby("hour")["windspeed * gewichtung(wind)"].quantile([0.0001, 0.999]).unstack()
    if wq.shape[1] == 2:
        wq.columns = ["wind_min", "wind_max"]
        df = df.join(wq, on="hour")
        # Wenn wind_factor 0, dann Wind abschalten:
        if wind_factor == 0:
            df["wind_sim"] = 0
        else:
            df["wind_sim"] = df["wind_min"] + wind_factor * (df["wind_max"] - df["wind_min"])
    else:
        df["wind_sim"] = np.nan
        raise ValueError("Windquantile konnten nicht berechnet werden.")
    
    
    return df

# Beispiel zur Verwendung – wenn das Skript direkt ausgeführt wird:
if __name__ == '__main__':
    from pandasgui import show
    df_sample = pd.read_pickle("df_for_model.pkl")
    #print(df_sample.info())
    # Rufe die Simulation auf; hier können die Parameter angepasst werden.
    simulated_df = simulate_weather_data(df_sample, sun_scale=1.0, wind_factor=0.5)

    # Gib das Head des Ergebnis-DataFrames aus
    print("-"*50)
    print(simulated_df.head())
    print("-"*50)
    print(simulated_df.info())
    print("-"*50)
    print(simulated_df.describe())
    print("-"*50)
    show(simulated_df)
