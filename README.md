# DSI Abschlussprojekt
### Entwicklung einer interaktiven Streamlit-Applikation zur prädiktiven Analyse von Strompreisen basierend auf simulierten und berichtsbasierten Wetterdaten

## Projektbeschreibung
Dieses Projekt entstand im Rahmen meiner Weiterbildung beim DSI.

**Die Implementierungszeit für die erste funktionsfähige Version betrug insgesamt nur rund zwei Wochen. Das Projekt ist daher in erster Linie als Tech-Demo bzw. Machbarkeitsprüfung zu verstehen und spiegelt nicht meinen heutigen Anspruch an Code-Qualität und Architektur wider.**

Eine interaktive Anwendung zur Analyse und Visualisierung von Strompreisen in Abhängigkeit von Wetterdaten.
Die Anwendung ermöglicht das Laden, Simulieren und Aufbereiten historischer Wetter- und Preisdaten sowie die Prognose zukünftiger Marktpreise (+7 Tage).
Kern des Projekts ist die Strompreisprognose verschiedener Machine-Learning-Modelle im Vergleich zu den tatsächlichen Marktpreisen.
Dazu wurden 60+ Koordinaten verteilt über ganz Deutschland gewählt, um eine möglichst homogene Wetterprognose für das ganze Land zu erhalten. Die Verteilung der Erzeuger ist dafür aber sehr heterogen (viele Windkraftwerke im Norden, viele Solarkraftwerke im Süden). Daher fließen die Daten der Koordinaten gewichtet nach "Anteil an der Erzeugung je Bundesland" in die Modellberechnung ein. 
Des Weiteren lassen sich Sonnenstrahlung (GTI) und Windgeschwindigkeit simulieren, um „Was-wäre-wenn“-Szenarien für Strompreise zu untersuchen.

**Weitere Details in der App. Link zur Demo: https://strom-app.mayone.de/** <-- Aktuell Offline, ich kümmere mich Zeitnah um ein neues Hosting

## Features
- Kurze Übersicht zu den Kennzahlen aus 2025
- Karte mit Koordinaten der Messpunkte für die Wetterdaten
- Diagramme zu aktuellen Wetterdaten (-7 Tage = History, +7 Tage = Forecast)
- Preisprognose der vergangenen 7 Tage gegenüber dem realen Strompreis
- Simulation Sonnenstrahlung (GTI) und Windgeschwindigkeit als Oberkurve
- Visualisierung von Strompreisen (Markt vs. ML-Prognose vs. Simulation -> "was wäre wenn")
- API-Abfragen für die Wetterdaten sowie die Berechnung der Preisprognose werden täglich zwischen 3 und 5 Uhr morgens ausgeführt
- API-Abfrage sowie Prognose können manuell angestoßen werden
   - Wetterdaten neu abfragen: ~30 Minuten (Limit der API)
   - Model-Prognose: je nach Hardware einige Minuten

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



## Usage (ohne Docker)
1. Starte die Streamlit-App:
   ```bash
   streamlit run src/strom_app.py
   ```
2. Folge den Anweisungen in der Weboberfläche.


## Kontakt & Lizenz
Dieses Projekt entstand im Rahmen des Data Science Abschlussprojekts (DSI).
Nutzung und Weitergabe bitte nur nach Rücksprache.
Bei Fragen oder Interesse:

https://github.com/g0ld3ner


## Changelog:
- 2025-08: Ausführlichere Projektbeschreibung
- 2025-07: Täglicher Cronjob, Visualisierung verbessert
- 2025-06: kleines Refactoring, Dokumentation und Fehlerhandling verbessert
- 2025-04: Erster Prototyp für die Abschlusspräsentation 

--------------------------------

## Ausblick:
- Umfassendes Refactoring!
- Tägliche Prognosen/Wetterdaten/Marktpreise in eine Datenbank speichern und abrufbar machen
- nicht nur Marktpreis vs Prognose prüfen, sondern auch Wetter-Forecast vs reales Wetter prüfen
- Ensemble beider Modelle?
- Testing einführen


