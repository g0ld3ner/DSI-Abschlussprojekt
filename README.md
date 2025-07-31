# DSI Abschlussprojekt
### Entwicklung einer interaktiven Streamlit-Applikation zur prädiktiven Analyse von Strompreisen basierend auf simulierten und berichtsbasierten Wetterdaten

## Projektbeschreibung
Dieses Projekt entstand im Rahmen meiner Weiterbildung beim DSI.
Eine interaktive Anwendung zur Analyse und Visualisierung von Strompreisen in Abhängigkeit von Wetterdaten.
Die Anwendung ermöglicht das Laden, Simulieren und Aufbereiten historischer Wetter- und Preisdaten sowie die Prognose zukünftiger Marktpreise (+7 Tage).

Kern des Projekts ist die Strompreisprognose verschiedener Machine-Learning-Modelle im Vergleich zu den tatsächlichen Marktpreisen.
Des Weiteren lassen sich Sonnenstrahlung (GTI) und Windgeschwindigkeit simulieren, um „Was-wäre-wenn“-Szenarien für Strompreise zu untersuchen.

Die Anwendung richtet sich an:
- Data Scientists
- Energie- und Wetteranalyst:innen
- Unternehmen/Behörden, die Preismodelle und Wettersimulationen nachvollziehen möchten

## Features
- Laden und Aufbereiten historischer Wetterdaten (API)
- Simulation von Wetterdaten-Szenarien
- Visualisierung von Strompreisen (Markt vs. ML-Prognose vs. Simulation) 

**Die App ist derzeit nur zu Demonstrationszwecken geeignet.**
- API-Abfragen sowie die Preisprognose müssen derzeit noch manuell angestoßen werden – bitte der Erklärung in der App folgen.

## Installation
1. Repository klonen:
   ```bash
   git clone https://github.com/g0ld3ner/DSI-Abschlussprojekt.git
   cd DSI-Abschlussprojekt
   ```
2. Virtuelle Umgebung erstellen und Abhängigkeiten installieren:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   (alternativ via Conda, siehe install_environment.txt)

## Docker Installation (optional)
1. Docker bauen:
   ```bash
   docker compose up -d --build
   ```
2. Die App ist anschließend über http://localhost:8501 erreichbar.



## Usage
1. Starte die Streamlit-App:
   ```bash
   streamlit run src/strom_app.py
   ```
2. Folge den Anweisungen in der Weboberfläche.

## Projektstruktur
```
projekt-root/
├── data/                  # Alle .pkl-Datendateien und Modelloutputs
│    └── ...pkl
├── docs/                  # Excel- und SVG-Dateien zur Visualisierung
│    ├── anteil_erneuerbare_2025.xlsx
│    ├── stromdaten_nettoerzeugung_q2_2025.xlsx
│    ├── installierte_leistung_stromerzeugung.svg
│    └── ...
├── src/                   # Quellcode, Streamlit-App, Hilfsmodule
│    ├── strom_app.py          # Haupt-App
│    ├── PROCESSING.py
│    ├── BLOCKWISE.py
│    ├── WINSORIZING.py
│    ├── ...
│    └── weather_API/          # Wetter-APIs, Standortdaten etc.
│         ├── wetterapi_gesamt.py
│         ├── wetterapi_forecast.py
│         ├── wettermodul_history.py
│         ├── bundeslaender_gewichte.py
│         ├── locations.py
│         └── ...
├── scripts/               # Hilfsskripte
│    └── export_requirements.py
├── requirements.txt       # Paketabhängigkeiten
├── environment.yml        # Für die Installation mit Conda
├── README.md              # (Diese Datei)
└── .gitignore             # 
```

## Kontakt & Lizenz
Dieses Projekt entstand im Rahmen des Data Science Abschlussprojekts (DSI).
Nutzung und Weitergabe bitte nur nach Rücksprache.
Bei Fragen oder Interesse an Erweiterungen:

https://github.com/g0ld3ner


## Changelog:
- 2025-06: Refactoring, Dokumentation, Fehlerhandling verbessert
- 2025-04: Erster Prototyp für die Abschlusspräsentation 

--------------------------------

## Ausblick:
- Automatisierung der Requests und ML-Berechnungen zur Bereitstellung kontinuierlicher Live-Prognosen.

