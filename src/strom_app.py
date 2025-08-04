# Starten der App mit: "streamlit run src/strom_app.py"

import datetime
import os

import streamlit as st
import pydeck as pdk
import matplotlib.colors as mc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import root_mean_squared_error, mean_absolute_error,  mean_squared_error,  r2_score

### Eigene Module
import locations
import PROCESSING

### Streamlit config
st.set_page_config(
    page_title="Strompreis-Prognose",
    layout="wide",
    initial_sidebar_state="expanded"
)

### Funktionen
def lade_und_normalisiere_mehrere_daten(dateien: list, spalten: list) -> list:
    """
    Lädt mehrere Pickle-Dateien als DataFrames, sorgt für konsistenten Zeitindex,
    entfernt Duplikate und gibt alle DataFrames in einer Liste zurück.
    """
    dfs = []
    for datei, spalte in zip(dateien, spalten):
        df = pd.read_pickle(datei)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Europe/Berlin", nonexistent="shift_forward", ambiguous="NaT")
        else:
            df.index = df.index.tz_convert("Europe/Berlin")
        # Duplikate im Index entfernen
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep='first')]
        df = df[[spalte]]
        df = df.rename(columns={spalte: datei})
        dfs.append(df)
    return dfs

# -----------------
# Sektionen (Nav-Bar)
# -----------------

### 1. Stromdaten
def sektion_stromdaten():
    """Zeigt verschiedene Stromdaten-Visualisierungen basierend auf Dateien im docs-Ordner."""
    st.subheader("Kennzahlen aus 2025")
    # 1. Diagramm: Nettostromerzeugung Q2 2025
    st.markdown(
        "Das folgende Balkendiagramm zeigt die öffentliche Nettostromerzeugung in Deutschland im zweiten Quartal 2025 "
        "gemessen in Gigawattstunden (GWh). Mit einem Blick lässt sich erkennen, welche Technologien "
        "den größten Beitrag zur Stromproduktion geleistet haben."
    )
    try:
        strom_file = "docs/stromdaten_nettoerzeugung_q2_2025.xlsx"
        df_full = pd.read_excel(strom_file, sheet_name=0, header=None)
        columns = df_full.iloc[0, 1:]
        values = df_full.iloc[2, 1:]
        df_plot = pd.DataFrame({
            "Energiequelle": columns.values,
            "Energie (GWh)": values.values
        })
        fig = px.bar(
            df_plot,
            x="Energiequelle",
            y="Energie (GWh)",
            text="Energie (GWh)",
            color="Energiequelle",
            title="🔌 Öffentliche Nettostromerzeugung – Q2 2025",
            labels={"Energiequelle": "Stromquelle"}
        )
        fig.update_traces(
            texttemplate="%{text:.1f}",
            hovertemplate="<b>%{x}</b><br>Energie: %{y:.1f} GWh",
            marker_line_width=1,
            marker_line_color="white"
        )
        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            yaxis_title="Energie (GWh)",
            title_x=0,
            bargap=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.error("❌ Datei 'stromdaten_nettoerzeugung_q2_2025.xlsx' nicht gefunden.")
    except Exception as e:
        st.error(f"⚠️ Fehler beim Verarbeiten der Stromdatei: {e}")

    # 2. Diagramm: Jährlicher EE-Anteil ab 2015
    st.markdown("Darstellung des Anteils Erneuerbarer Energien an der öffentlicher Nettostromerzeugung und Last von 2015 bis 2025.")
    try:
        file_ee_year = "docs/ee_anteil_jaehrlich_2015_2025.xlsx"
        df_year = pd.read_excel(file_ee_year, sheet_name=0, header=None)
        df_year = df_year.iloc[2:, :3]
        df_year.columns = ["Jahr", "Anteil Last", "Anteil Erzeugung"]
        df_year = df_year.dropna()
        df_year["Jahr"] = pd.to_numeric(df_year["Jahr"], errors="coerce").astype(int)
        df_year["Anteil Last"] = pd.to_numeric(df_year["Anteil Last"], errors="coerce")
        df_year["Anteil Erzeugung"] = pd.to_numeric(df_year["Anteil Erzeugung"], errors="coerce")
        df_year = df_year[df_year["Jahr"] >= 2015]
        df_long = df_year.melt(id_vars="Jahr", var_name="Kategorie", value_name="Anteil (%)")
        fig_year = px.bar(
            df_long,
            x="Jahr",
            y="Anteil (%)",
            color="Kategorie",
            barmode="group",
            text="Anteil (%)",
            title="EE-Anteil an öffentlicher Stromerzeugung und Last (2015–2025)"
        )
        fig_year.update_traces(
            texttemplate="%{text:.1f} %",
            hoverinfo="skip",
            marker_line_width=1,
            marker_line_color="white"
        )
        fig_year.update_layout(
            title_x=0
        )
        st.plotly_chart(fig_year, use_container_width=True)
    except FileNotFoundError:
        st.error("❌ Datei 'ee_anteil_jaehrlich_2015_2025.xlsx' nicht gefunden.")
    except Exception as e:
        st.error(f"⚠️ Fehler beim Verarbeiten der Datei für Jahresvergleich: {e}")

    # 3. Diagramm: Installierte Netto-Leistung (SVG)
    st.markdown("Entwicklung der installierten Kraftwerksleistung pro Energieträger (2002–2024).")
    try:
        st.image("docs/installierte_leistung_stromerzeugung.svg")
    except FileNotFoundError:
        st.error("❌ SVG-Datei 'installierte_leistung_stromerzeugung.svg' nicht gefunden.")
    except Exception as e:
        st.error(f"⚠️ Fehler beim Laden der SVG-Datei: {e}")

    

### 2. Orte
def sektion_wetter_locations():
    """Zeigt eine interaktive Karte aller Wetterstandorte in Deutschland."""
    st.subheader("Standorte in Deutschland")
    st.markdown(
        "Die Karte zeigt die Standorte, für die Wetterdaten ermittelt werden.  \n"
        "Über alle Standorte eines Bundeslandes hinweg werden die Durschnittswerte "
        "ermittelt und anhand der anteiligen Erzeugung je Bundesland entsprechende Gewichtungen für die Modell-Prognosen vorgenommen.  \n"
        "  \n"
        "Die Punkte sind nach Bundesländern gruppiert und farblich gekennzeichnet."
    )

    try:
        # 1) Daten in ein DataFrame packen
        rows = []
        for state, coords in locations.location.items():
            color_str = locations.state_colors.get(state, "blue")
            # Farbstring (Name oder Hex) in RGBA umwandeln
            rgb = mc.to_rgb(color_str)  
            rgba = [int(c * 255) for c in rgb] + [160]  # A=160 für leichte Transparenz
            for lat, lon in coords:
                rows.append({
                    "lat": lat,
                    "lon": lon,
                    "state": state,
                    "color": rgba
                })
        df_coords = pd.DataFrame(rows)

        # 2) ScatterplotLayer anlegen
        layer = pdk.Layer(
            "ScatterplotLayer",
            df_coords,
            pickable=True,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=8000,
            auto_highlight=True
        )

        # 3) Kamera-Initialisierung
        view_state = pdk.ViewState(
            latitude=51.1657,
            longitude=10.4515,
            zoom=5,
            bearing=0,
            pitch=0
        )

        # 4) Deck zusammenbauen und rendern
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{state}"}
        )
        st.pydeck_chart(deck)

    except Exception as e:
        st.error(f"⚠️ Fehler beim Erstellen der Wetter-Standorte-Karte: {e}")

    #########

    st.markdown("Anteile an der Erzeugung nach Bundesländern (Wind und Solar).  \n"
                "Diese sind die Grundlage für die Gewichtung der Bundesländer.")
    try:
        st.image("docs/wind_solar_leistung_2025.svg")
    except FileNotFoundError:
        st.error("❌ SVG-Datei 'wind_solar_leistung_2025.svg' nicht gefunden.")
    except Exception as e:
        st.error(f"⚠️ Fehler beim Laden der SVG-Datei: {e}")

      

### 3. Echtzeit Wetter
def sektion_live_wetter():
    st.subheader("Wetterdaten (Ø aller Bundesländer)")
    st.text("Hier werden Daten der vergangenen Woche, sowie die Vorhersagen der nächsten Woche angezeigt")

    try:
        df_wetter = pd.read_pickle("data/df_for_model.pkl")
    except FileNotFoundError:
        st.error("❌ Datei 'df_for_model.pkl' nicht gefunden!")
        return

    # Zeitzone korrekt setzen (falls noch tz-naiv)
    if df_wetter.index.tz is None:
        df_wetter.index = (
            df_wetter.index
                     .tz_localize("UTC")            # falls ursprünglich UTC
                     .tz_convert("Europe/Berlin")
        )

    # 1) Transition-Punkt
    transition = df_wetter.loc[df_wetter["Quelle"] == "forecast"].index.min()

    # 2) Fenster ±7 Tage
    window_start = transition - pd.Timedelta(days=7)
    window_end   = transition + pd.Timedelta(days=7)

    # 3) Filtern
    df_window = df_wetter.loc[window_start:window_end]


    # Index umbenennen und resetten 
    df_window.index.name = "Zeit"
    df_plot = df_window.reset_index()
    
    # Temperaturdiagramm 
    # st.markdown("### 🌡️ Temperaturverlauf")
    fig1 = px.line(
        df_plot, x="Zeit", y="temperature_2m", markers=False,
        labels={"temperature_2m": "Temperatur [°C]", "Zeit": "Zeit"},
        title="🌡️ Temperaturverlauf",
        color_discrete_sequence=["red"],
    )
    fig1.add_shape(
    type="line",
    x0=transition, x1=transition,
    y0=0,  y1=1,
    xref="x", yref="paper",
    line=dict(color="grey", dash="dash")
)
    st.plotly_chart(fig1, use_container_width=True)


    # ☀️ Sonnenscheindauer (in Minuten)
    df_plot["sunshine_duration_min"] = df_plot["sunshine_duration"] / 60
    fig2 = px.line(
        df_plot, x="Zeit", y="sunshine_duration_min", markers=False,
        labels={"sunshine_duration_min": "Sonnenscheindauer [min/h]", "Zeit": "Zeit"},
        title="☀️ Sonnenscheindauer",
        color_discrete_sequence=["gold"]
    )
    fig2.add_shape(
        type="line",
        x0=transition, x1=transition, #cut
        y0=0,  y1=1,
        xref="x", yref="paper",
        line=dict(color="grey", dash="dash")
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 💨 Windgeschwindigkeit
    fig3 = px.line(
        df_plot, x="Zeit", y="wind_speed_100m", markers=False,
        labels={"wind_speed_100m": "Wind [km/h]", "Zeit": "Zeit"},
        title="💨 Windgeschwindigkeit",
        color_discrete_sequence=["blue"]
    )
    fig3.add_shape(
        type="line",
        x0=transition, x1=transition, #cut
        y0=0,  y1=1,
        xref="x", yref="paper",
        line=dict(color="grey", dash="dash")
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Globalstrahlung
    fig4 = px.line(
        df_plot,
        x="Zeit",
        y="global_tilted_irradiance",
        markers=False,
        labels={
            "global_tilted_irradiance": "Einstrahlung [W/m²]",
            "Zeit": "Zeit"
        },
        title="☀️ Globalstrahlung auf die 35° geneigte Ebene gegen Süden",
        color_discrete_sequence=["orange"]
    )
    fig4.add_shape(
        type="line",
        x0=transition, x1=transition, #cut
        y0=0,  y1=1,
        xref="x", yref="paper",
        line=dict(color="grey", dash="dash")
    )
    st.plotly_chart(fig4, use_container_width=True)


### 4. Preisvorhersage
def sektion_price_vs_market():
    st.subheader("Strompreis-Prognose")
    st.markdown(
        "Vergleich von Strompreisprognosen aus verschiedenen KI-Modellen mit den tatsächlichen Marktpreisen "
        "Es werden die letzten 7 Tage, für die historische Wetterdaten vorliegen, prognostiziert. Diese liegen API-Bedingt immer 2 Tage in der Vergangenheit."
    )

    heute = datetime.date.today()

    ### später wieder implementieren um beliebige Prognosezeiträume zu erfassen
    # # Prognosezeitraum in der Sidebar
    # st.sidebar.markdown("### 📆 Prognosezeitraum")
    # start_datum = st.sidebar.date_input("Startdatum", heute - datetime.timedelta(days=8), key="KIvsMP")
    start_datum = heute - datetime.timedelta(days=8)

    # Dateipfade + Spaltennamen
    dateien = [
        "data/df_blockwise_result.pkl", 
        "data/df_single_step_result.pkl", 
        #"data/df_xg_boost_history.pkl",
        "data/df_for_model.pkl" 
    ]
    spalten = [
        "prediction",  # MLP
        "prediction",  # LSTM
        # "price",       # XG-Boost
        "price"        # echte Preise
    ]
    model_namen = [
        "MLP Blockwise 24h",
        "LSTM Single Step 168h",
        # "XG-Boost",
        "Marktpreise"
    ]

    # Einlesen & Normierung der Zeitreihe
    dfs = lade_und_normalisiere_mehrere_daten(dateien, spalten)
    df_combined = pd.concat(dfs, axis=1)
    df_combined.columns = model_namen

    # Filter auf 7-Tage-Zeitraum
    tz = df_combined.index.tz
    start = pd.Timestamp(start_datum).tz_localize(tz)
    startzeit = df_combined[df_combined.index.date == start.date()].index.min()
    if pd.isna(startzeit):
        st.warning("⚠️ Für dieses Startdatum liegen keine Daten vor.")
        return

    df_combined = df_combined[df_combined.index >= startzeit].iloc[:168]

    # Plot
    fig = px.line(
        df_combined,
        color_discrete_sequence=["lightgreen", "lightblue", "red"],
        labels={"value": "Preis [€/MWh]", "index": "Zeit", "variable": "Modell"},
        title="Model vs. Realität",
        markers=False
    )
    fig.update_xaxes(title_text="Datum")
    fig.update_layout(title_x=0, dragmode="pan")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


    
    # def smape(y_true, y_pred): # Bei MAPE kann, wenn der Preis real 0 ist, eine Div/0 entstehen :-/ 
    #     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    #     mask = denominator != 0
    #     return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def rel_rmse(y_true, y_pred):
        rmse = root_mean_squared_error(y_true, y_pred)
        std_dev = np.std(y_true, ddof=1)
        return rmse / std_dev if std_dev != 0 else np.nan
    
    with st.expander("Fehlerkennzahlen & Marktvolatilität", expanded=True):
        # Nur die Modell-Namen, nicht die Marktpreise
        modelle = ["MLP Blockwise 24h", "LSTM Single Step 168h"]

        daten = []
        y_true_full = df_combined["Marktpreise"]

        # Marktpreis-Volatilität berechnen
        std_true = np.std(y_true_full.dropna(), ddof=1)

        for modell in modelle:
            y_pred_full = df_combined[modell]
            mask = y_true_full.notna() & y_pred_full.notna()
            y_true = y_true_full[mask]
            y_pred = y_pred_full[mask]
            resid = y_true - y_pred

            daten.append({
                "Modell":    modell,
                "RMSE":      f"{root_mean_squared_error(y_true, y_pred):.1f}",
                "MAE":       f"{mean_absolute_error(y_true, y_pred):.1f}",
                "StdErr":    f"{np.std(resid, ddof=1):.1f}",
                "R²":        f"{r2_score(y_true, y_pred):.3f}",
                # "MAPE (%)":  f"{(np.mean(abs(resid/y_true))*100):.1f}"
                # "SMAPE (%)": f"{smape(y_true, y_pred):.1f}" # MAPE durch SMAPE ersetzt
                "RMSE / StdDev Markt":  f"{rel_rmse(y_true,y_pred):.2f}" # vllt ne Erlärung zu schreiben?
            })


        df_stats = pd.DataFrame(daten).set_index("Modell")
        

        # 3) Layout: Tabelle & StdDev
        col1, col2 = st.columns([5,1])
        with col1:
            st.table(df_stats)
        with col2:
            st.metric("StdDev Marktpreis", f"{std_true:.1f} €/MWh")

    #ausklappbare Tabelle
    with st.expander("Vergleichstabelle anzeigen", expanded=False):
        df_table = df_combined.copy().reset_index().rename(columns={"index": "Zeit"})
        for sp in df_table.columns[1:]:
            df_table[sp] = df_table[sp].map(lambda x: f"{x:,.2f}".replace(".", ",") + " €")
        st.table(df_table)


### 5. Wettersimulation
def sektion_wetter_simulation():
    """
    Simuliert den Einfluss der Sonnenskala und Windfaktor auf Wetterdaten und zeigt diese als Zeitreihen-Diagramme.
    Ermöglicht Anpassungen per Sidebar und zeigt sowohl den Gesamtzeitraum als auch einen 7-Tage-Ausschnitt.
    """
    st.subheader("☁️ Wetterdaten-Simulation")
    try:
        df = pd.read_pickle("data/df_for_model.pkl")
    except FileNotFoundError:
        st.info("'df_for_model.pkl' nicht gefunden.")
        return

    @st.cache_data
    def load_data(file):
        return file

    if df is not None:
        df = load_data(df)

        # Prüfen, ob DatetimeIndex vorhanden ist
        if not isinstance(df.index, pd.DatetimeIndex):
            st.error("Index ist kein DatetimeIndex. Bitte konvertiere deine Daten.")
            return

        # Prüfung der erforderlichen Spalten
        gti_col = "GTI * gewichtung(sun)"
        sun_col = "sunhours * gewichtung(sun)"
        wind_col = "windspeed * gewichtung(wind)"
        for col in [gti_col, sun_col, wind_col]:
            if col not in df.columns:
                st.error(f"Spalte '{col}' fehlt im DataFrame.")
                return

        st.sidebar.header("Simulationsparameter")
        sun_scale = st.sidebar.slider("Sonnenskala (0-125%)", 0.0, 1.25, 1.0, 0.05)
        wind_factor = st.sidebar.slider("Windfaktor", 0.0, 1.25, 0.5, 0.05)
        week_offset = st.sidebar.number_input("Wöchentlicher Offset (0 = letzte 7 Tage, 1 = vorletzte 7 Tage, ...)",
                                              min_value=0, value=0, step=1)
        group_option = st.sidebar.selectbox("Gruppierung für Sonnendaten", options=["Monatlich", "Wöchentlich"])

        df["datetime"] = df.index
        df["hour"] = df.index.hour

        # Gruppierung
        if group_option == "Wöchentlich":
            df["week_of_year"] = df.index.isocalendar().week
            grouping_cols = ["week_of_year", "hour"]
            sort_index_expr = lambda row: (row["week_of_year"] - 1) * 24 + row["hour"]
        else:  # Monatlich
            df["month"] = df.index.month
            grouping_cols = ["month", "hour"]
            sort_index_expr = lambda row: (row["month"] - 1) * 24 + row["hour"]

        def local_max_or_zero(series):
            nonzero = series[series > 0.1]
            return nonzero.max() if not nonzero.empty else 0.0

        peak_sun = df.groupby(grouping_cols)[gti_col] \
            .apply(local_max_or_zero) \
            .reset_index(name="gti_peak")
        peak_sun["sort_index"] = peak_sun.apply(sort_index_expr, axis=1)
        peak_sun.sort_values("sort_index", inplace=True)
        peak_sun["gti_peak_smooth"] = peak_sun["gti_peak"]
        peak_sun.loc[peak_sun["gti_peak"] == 0, "gti_peak_smooth"] = 0.0

        df = df.merge(peak_sun[grouping_cols + ["gti_peak_smooth"]], on=grouping_cols, how="left")
        df.set_index("datetime", inplace=True)

        # Simulation: GTI & Sonne
        df["gti_sim"] = df["gti_peak_smooth"] * sun_scale
        df["sunhours_sim"] = df[sun_col] * sun_scale

        # Wind-Simulation
        wq = df.groupby("hour")[wind_col].quantile([0.0001, 0.999]).unstack()
        if wq.shape[1] == 2:
            wq.columns = ["wind_min", "wind_max"]
            df = df.join(wq, on="hour")
            if wind_factor == 0:
                df["wind_sim"] = 0
            else:
                df["wind_sim"] = df["wind_min"] + wind_factor * (df["wind_max"] - df["wind_min"])
        else:
            st.warning("Konnten keine Windquantile berechnen => wind_sim = NaN")
            df["wind_sim"] = np.nan

        st.header("Ergebnis: GTI & Wind-Simulation (gesamter Datensatz)")
        fig1, ax1 = plt.subplots(figsize=(12, 3))
        ax1.plot(df.index, df[gti_col], label="Original GTI", alpha=0.4)
        ax1.plot(df.index, df["gti_sim"], label="Simulierte GTI-Oberkurve", alpha=0.8)
        ax1.set_title("GTI (kompletter Datensatz)")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(df.index, df[wind_col], label="Original Wind", alpha=0.4)
        ax2.plot(df.index, df["wind_sim"], label="Simulierter Wind", alpha=0.8)
        ax2.set_title("Wind (kompletter Datensatz)")
        ax2.legend()
        st.pyplot(fig2)

        st.header(f"Ausgewählter 7-Tage-Zeitraum (Offset: {week_offset} Woche(n))")
        end_date_custom = df.index.max() - pd.Timedelta(weeks=week_offset)
        start_date_custom = end_date_custom - pd.Timedelta(days=7)
        custom_period = df[(df.index >= start_date_custom) & (df.index <= end_date_custom)]

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(custom_period.index, custom_period[gti_col], label="Original GTI", alpha=0.4)
        ax3.plot(custom_period.index, custom_period["gti_sim"], label="Simulierte GTI-Oberkurve", alpha=0.8)
        ax3.set_title(
            f"GTI – Zeitraum: {start_date_custom.strftime('%Y-%m-%d')} bis {end_date_custom.strftime('%Y-%m-%d')}")
        ax3.legend()
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.plot(custom_period.index, custom_period[wind_col], label="Original Wind", alpha=0.4)
        ax4.plot(custom_period.index, custom_period["wind_sim"], label="Simulierter Wind", alpha=0.8)
        ax4.set_title(
            f"Wind – Zeitraum: {start_date_custom.strftime('%Y-%m-%d')} bis {end_date_custom.strftime('%Y-%m-%d')}")
        ax4.legend()
        st.pyplot(fig4)

    else:
        st.info("'df_for_model.pkl' nicht gefunden.")



### 6. Simuliertes Wetter: Preis vs Real
def sektion_price_weather_simulation():
    """
    Vergleicht Strompreise aus Wettersimulationen und tatsächlichen Preisen.
    Lädt vorhandene Ergebnisdateien dynamisch, visualisiert alle vorhandenen Reihen
    und berechnet Fehlerkennzahlen, sofern möglich.
    """
    st.subheader("Strompreis nach Wettersimulation")

    sun_scale = st.sidebar.slider("Sonnenskala (0-110%)", 0.0, 1.1, 1.0, 0.05)
    wind_factor = st.sidebar.slider("Windfaktor", 0.0, 1.1, 0.5, 0.05)

    heute = datetime.date.today()

    ### für frei wählbaren Zeitraum
    # st.sidebar.markdown("### 📆 Prognosezeitraum")    
    # start_datum_wetter = st.sidebar.date_input(
    #     "Startdatum",
    #     value=heute - datetime.timedelta(days=7),
    #     max_value=heute,
    #     key="start_datum_123"
    # )
    # start_offset = (heute - start_datum_wetter).days
    
    ### Fixer Zeitraum
    start_datum_wetter = heute - datetime.timedelta(days=8)
    start_offset = (heute - start_datum_wetter).days

    if st.sidebar.button("Simulation starten"):
        PROCESSING.processing_weather(start=start_offset, sun=sun_scale, wind=wind_factor)

    # Dateipfade und Spaltennamen
    file_info = [
        ("data/df_blockwise_result.pkl", "prediction", "MLP Blockwise 24h"),
        ("data/df_blockwise_result_WETTER.pkl", "prediction", "MLP Wetter Simulation"),
        ("data/df_for_model.pkl", "price", "Marktpreise"),
    ]

    dfs = []
    used_model_namen = []
    used_spalten = []

    # Check existence, load only existing files
    for pfad, spalte, name in file_info:
        if os.path.exists(pfad):
            try:
                df = pd.read_pickle(pfad)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("Europe/Berlin", nonexistent="shift_forward", ambiguous="NaT")
                else:
                    df.index = df.index.tz_convert("Europe/Berlin")
                # Duplikate raus
                if not df.index.is_unique:
                    df = df[~df.index.duplicated(keep='first')]
                df = df[[spalte]].rename(columns={spalte: name})
                dfs.append(df)
                used_model_namen.append(name)
                used_spalten.append(spalte)
            except Exception as e:
                st.warning(f"Fehler beim Laden von {pfad}: {e}")
        else:
            st.info(f"Datei nicht gefunden und wird übersprungen: {pfad}")

    if len(dfs) < 2:
        st.warning("Es wurden weniger als zwei Zeitreihen geladen. Eine sinnvolle Visualisierung ist nicht möglich.")
        return

    try:
        df_combined = pd.concat(dfs, axis=1)
        df_combined.columns = used_model_namen

        tz = df_combined.index.tz
        start = pd.Timestamp(start_datum_wetter).tz_localize(tz)

        # Finde erste existierende Zeit an diesem Tag
        startzeit = df_combined[df_combined.index.date == start.date()].index.min()

        if pd.isna(startzeit):
            st.warning("⚠️ Für dieses Startdatum liegen keine Daten vor.")
        else:
            df_combined = df_combined[df_combined.index >= startzeit].iloc[:168]

            # Metriken berechnen (nur, falls mindestens zwei Kurven da sind)
            if "Marktpreise" in df_combined.columns and any(c for c in df_combined.columns if "MLP" in c):
                for model in [c for c in df_combined.columns if "MLP" in c]:
                    pred = df_combined[model]
                    real = df_combined["Marktpreise"]
                    mask = pred.notna() & real.notna()
                    if mask.sum() > 0:
                        rmse = root_mean_squared_error(real[mask], pred[mask])
                        mae  = mean_absolute_error(real[mask], pred[mask])
                        st.sidebar.markdown(f"### Fehlerkennzahlen {model}")
                        st.sidebar.metric(f"RMSE (€/MWh) {model}", f"{rmse:,.2f}")
                        st.sidebar.metric(f"MAE (€/MWh) {model}", f"{mae:,.2f}")

            # 📈 Diagramm anzeigen
            fig = px.line(
                df_combined,
                labels={"value": "Preis [€/MWh]", "index": "Zeit"},
                color_discrete_sequence=["lightgreen", "yellow", "red"],
                title="Wetterprognose vs. Wettersimulation vs. Reale Marktpreise",
                markers=False
            )
            fig.update_layout(title_x=0)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Fehler beim Laden oder Verarbeiten der Daten: {e}")



# -----------------



def main():
    """
    Einstiegspunkt der App und Navigationsleiste
    """
    st.sidebar.subheader("Prädiktive Analyse von Strompreisen basierend auf simulierten und berichtsbasierten Wetterdaten")
    st.sidebar.info("Aktuell muss die Datenextraktion über die Open-Meteo-API sowie die Modellberechnung noch manuell angestoßen werden.  \n" \
        " -> src/weather_API/wetterapi_main.py ausführen um die aktuellen Daten aus der API zu ziehen (~ 30 Minuten)  \n" \
        " -> src/PROCESSING.py ausführen um die Prognosen zu erstellen (je nach Hardware einige Minuten)")
    nav = [
        "Marktkennzahlen",
        "Wetter Locations",
        "Wetter-Daten",
        "Preisprognose",
        "Wetterdaten-Simulation",
        "Strompreis Wettersimulation"
    ]
    seite = st.sidebar.radio("Navigation", nav)

    if seite == nav[0]:
        sektion_stromdaten()
    elif seite == nav[1]:
        sektion_wetter_locations()
    elif seite == nav[2]:
        sektion_live_wetter()
    elif seite == nav[3]:
        sektion_price_vs_market()
    elif seite == nav[4]:
        sektion_wetter_simulation()
    elif seite == nav[5]:
        sektion_price_weather_simulation()
    
    else:
        st.error(f"Unbekannte Auswahl: {seite}")

if __name__ == "__main__":
    main()
    pass
