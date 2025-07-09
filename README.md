# DSI Abschlussprojekt
### Entwicklung einer interaktiven Streamlit-Applikation zur prädiktiven Analyse von Strompreisen basierend auf simulierten und berichtsbasierten Wetterdaten

## Projektbeschreibung
Dieses Projekt ist im Rahmen meiner Weiterbildung beim DSI entstanden.
Eine interaktive Anwendung zur Analyse und Visualisierung von Strompreisen in Abhänigkeit der Wetterdaten. Es können historische Wetter- und Preisdaten laden, simulieren und aufbereiten sowie Vorhersagen anzeigen.

Kern des Projekts ist  die Strompreisprognose verschiedener Machine-Learning-Modelle im Vergleich zu den tatsächlichen Marktpreisen.
Desweiteren können Intesität der Sonnenstrahlung (GTI) sowie Windgeschwindigkeiten simuliert werden um "Was wäre wenn"-Prognosen für den Strompreis zu ermitteln.

Die Anwendung richtet sich an:
- Data Scientists
- Energie- und Wetteranalyst:innen
- Unternehmen/Behörden, die Preismodelle und Wettersimulationen nachvollziehen wollen

## Features
- Laden und Aufbereiten historischer Wetterdaten (API)
- Simulation von Wetterdaten-Szenarien
- Visualisierung von Strompreisen und Wetterparametern

## Installation
1. Repository klonen:
   ```bash
   git clone <REPO-URL>
   cd <REPO-ORDNER>
   ```
2. Erstelle eine virtuelle Umgebung und installiere Abhängigkeiten:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

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
├── docs/                  # Excel- und SVG-Dateien für Visualisierung/Stromdaten
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
├── README.md              # (Diese Datei)
└── .gitignore             # 
```

## Contributing
1. Fork des Repos anlegen
2. Branch erstellen (`git checkout -b feature/name`)
3. Änderungen commiten
4. Pull Request erstellen

## Kontakt & Lizenz
Dieses Projekt entstand im Rahmen des Data Science Abschlussprojekts (DSI).
Nutzung und Weitergabe bitte nach Rücksprache.
Bei Fragen oder Interesse an Erweiterungen:

Deine E-Mail oder GitHub-Adresse hier eintragen



Changelog (Auszug)
2025-06: Refactoring, Doku, Fehlerhandling verbessert
2025-04: Erster Prototyp für die Abschlusspräsentation 

--------------------------------



