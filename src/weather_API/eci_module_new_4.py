import requests
from pprint import pprint
import pandas as pd
import numpy as np
import datetime

import countrys
import weekdays
#import holidays

import pickle

from pandasgui import show   # pip install pandasgui (und ggf.: pip install bokeh==2.4.3)


def current_day(minus: int = 0) -> str:
    return (datetime.date.today() - datetime.timedelta(days=minus)).strftime("%Y-%m-%d")

def x_days_back(date: str, x: int) -> str:
    given_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    return (given_date - datetime.timedelta(days=x)).strftime("%Y-%m-%d")

def merge_dfs_by_index(*dfs, join='outer', format_func=None):
    """
    Führt beliebig viele DataFrames anhand ihres Indexes zusammen und formatiert jeden Index vor dem Zusammenführen.

    Parameter:
    - *dfs: Beliebig viele DataFrame-Objekte, die zusammengeführt werden sollen.
    - join: Art des Joins ('outer' für Vereinigung aller Indizes, 'inner' für Schnittmenge). Standard ist 'outer'.
    - format_func: Eine Funktion, die auf jedes Element des Indexes angewendet wird, um es zu formatieren.
                   Falls None, wird der Index unverändert gelassen.

    Rückgabe:
    - Ein zusammengeführter DataFrame.
    """
    # Fallback-Funktion setzen, falls keine übergeben wurde
    if format_func is None:
        format_func = format_date

    formatted_dfs = []
    for df in dfs:
        df = df.copy()  # Kopie erstellen, um Originaldaten nicht zu verändern
        if format_func is not None:
            df.index = df.index.map(format_func)
        formatted_dfs.append(df)
    return pd.concat(formatted_dfs, axis=1, join=join)

def format_date(x) -> str:
    """
    Wandelt das übergebene Datum in das Format 'YYYY-MM-DD HH:MM:SS' um.

    Parameter:
    - x: Ein Datum, das von pd.to_datetime interpretiert werden kann.

    Rückgabe:
    - Ein String im Format 'YYYY-MM-DD HH:MM:SS'.
    """
    return pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')


class EnergyChartsInfo: ##### es fehlen noch die docstrings....

    base_url = "https://api.energy-charts.info"
    public_power = "/public_power" #Erzeugung (und Load?) 
    price = "/price" #Börsenstrompreis


    def __init__(self, start_day:str = None, end_day:str = None, country:str = "de"):
        self.country = country
        self.start_day = start_day
        self.end_day = end_day

        if self.end_day is None:
            self.end_day = current_day()
        if self.start_day is None:
            self.start_day = x_days_back(self.end_day, 800)

    def change_country(self, c:str = "de") -> str:
        country_list = list(countrys.country_codes_dict.keys())
        if c in country_list:
            self.country = c
            return f"Land wurde auf {countrys.country_codes_dict[c]} geändert"
        else:
            return "Über das Land sind keine daten verfügbar!"
        
    def change_start_end(self,start_date:str = None, end_date:str = None):
        if start_date is None:
            start_date = self.start_day
        if end_date is None:
            end_date = self.end_day

        self.start_day = start_date
        self.end_day = end_date
    
    def build_url(self, api:str, boerse:str = "DE-LU") ->str:

        #URL bauen:
        if api == self.price:
            api = f"{api}?bzn={boerse}"
        elif api == self.public_power:
            api = f"{api}?country={self.country}"
        else:
            api = f"{api}?" ##### Das muss man testen....

        return f"{self.base_url}{api}&start={self.start_day}&end={self.end_day}"
    
    def add_realtime_index(self, df:pd.DataFrame) -> pd.DataFrame:
        if "unix_seconds" not in df.columns:
            raise ValueError("DataFrame does not contain a 'unix_seconds' column.")
        else:
            df["date"] = pd.to_datetime(df["unix_seconds"], unit='s')
            df.set_index("date", inplace=True)
            return df

    def remove_unix_seconds(self, df:pd.DataFrame) -> pd.DataFrame:
        df.drop("unix_seconds", axis=1, inplace=True)
        return df

    def get_public_power_series(self) -> pd.DataFrame:
        #URL bauen, request senden:
        url = self.build_url(self.public_power)
        response = requests.get(url)
        #response prüfen
        if response.status_code != 200:
            print(f"API request nicht erfolgreich, Statuscode: {response.status_code}")
            return None
        #Dataframe erzeugen und in die passende Form bringen
        df = pd.DataFrame(response.json()["production_types"])
        print("DF -------------------------------")
        print(df) ########################
        headers = []
        data = []
        #elemente des DF auslesen und in eine Liste speichern
        for e in range(len(df)):
            headers.append(df.iloc[e, 0])
            data.append(df.iloc[e, 1])
        #Liste in ein array umwandeln und transponieren
        data = np.array(data)
        data = data.transpose()
        # np.array --> DF
        df2 = pd.DataFrame(data,columns=headers)
        print("DF2 -------------------------------")
        print(df2) ##################
        #Dataframe um eine Spalte mit Zeitstempeln erweitern
        df2['unix_seconds'] =response.json()['unix_seconds']
        #Um eine Spalte mit lesbarer Zeit erweitern
        df3 = self.add_realtime_index(df2)
        print("DF3 -------------------------------")
        print(df3) ####################
        df4 = self.remove_unix_seconds(df3)
        print("DF4 -------------------------------")
        print(df4) ####################
        df5 = self.to_frequency(df4, "h")
        return df5
    
    def get_price_series(self) -> pd.DataFrame:
        #URL bauen, request senden:
        url = self.build_url(self.price)
        response = requests.get(url)
        #response prüfen
        if response.status_code != 200:
            print(f"API request nicht erfolgreich, Statuscode: {response.status_code}")
            return pd.DataFrame()
        price_list = response.json()["price"]
        timestamp_list = response.json()["unix_seconds"]
        df = pd.DataFrame({"unix_seconds": timestamp_list, "price": price_list})
        df2 = self.add_realtime_index(df)
        df3 = self.remove_unix_seconds(df2)
        return df3
    
    ### Geht nicht wenn nicht sowohl werte für vor und nach dem stichtag sind, daher erstmal fallback auf alles nach dem Stichtag (s.o.)
    # def get_price_series(self) -> pd.DataFrame:
    #     #URL bauen, request senden:
    #     boerse_bis_2018 = "DE-AT-LU"   #### beim Stichtag trennen bis "2018-09-30 21:00:00"
    #     boerse_ab_2018 = "DE-LU"   #### beim Stichtag trennen ab "2018-09-30 22:00:00"
    #     url_bis = self.build_url(self.price, boerse_bis_2018)
    #     url_ab = self.build_url(self.price, boerse_ab_2018)
    #     print("URL -------------------------------")
    #     print(url_bis) #################################
    #     print(url_ab) #################################
    #     response_bis = requests.get(url_bis)
    #     response_ab = requests.get(url_ab)
    #     #response prüfen
    #     if response_ab.status_code != 200:
    #         print(f"API request (ab) nicht erfolgreich, Statuscode: {response_bis.status_code}")
    #         # return None
    #     if response_bis.status_code != 200:
    #         print(f"API request (bis) nicht erfolgreich, Statuscode: {response_ab.status_code}")
    #         # return None
    #     price_list_bis = response_bis.json()["price"]
    #     price_list_ab = response_ab.json()["price"]
    #     timestamp_list_bis = response_bis.json()["unix_seconds"]
    #     timestamp_list_ab = response_ab.json()["unix_seconds"]
    #     df_bis = pd.DataFrame({"unix_seconds": timestamp_list_bis, "price": price_list_bis})
    #     df_ab = pd.DataFrame({"unix_seconds": timestamp_list_ab, "price": price_list_ab})
        
    #     df_bis = self.add_realtime_index(df_bis)
    #     df_bis = self.remove_unix_seconds(df_bis)
    #     df_ab = self.add_realtime_index(df_ab)
    #     df_ab = self.remove_unix_seconds(df_ab)
    #     df_bis_ab = df_ab.copy() 
    #     series_bis_ab = df_ab['price'].fillna(df_bis['price'])
    #     df_bis_ab["price"] = series_bis_ab 
    #     return df_bis_ab

    ######## weiter testen, sehr langsam....
    def to_frequency(self, df:pd.DataFrame, freq:str = "h") -> pd.DataFrame:
        df = df.copy()
        # Datum und Uhrzeit zu datetime, auf Stunde/Tag/etc. "abrunden" und entsprechend gruppieren (h, D, W, MS/ME, YS/YE)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        #df_freq = df.groupby(pd.Grouper(freq=freq)).mean() # Alternativ: .agg(["mean", "median", "count"])
        df_freq = df.resample(freq).mean()  ### noch testen ob das besser ist
        return df_freq
    
    ######## testen
    def get_weekday(self, df:pd.DataFrame, weekday:int=1)-> pd.DataFrame:
        # Wochentag 1 (Montag) bis 7 (Sonntag)
        df["weekday"] = pd.to_datetime(df.index).day_name()
        if weekday < 1 or weekday > 7:
            print("Den Wochentag gibt es nicht (1-7)")
            return df
        df_weekday = df[df["weekday"] == weekdays.weekdays[weekday]]
        return df_weekday
    
    

    ####### testen und warshceinlich überarbeiten
    #def append_holidays(self, data:pd.DataFrame)-> pd.DataFrame:
     #   # noch eine Spalte für Feiertage (pip install holidays)
      #  c = self.country.upper()
       # data = data.copy() #damit der Originale DF nicht überschrieben wird...
        ##index in Datetime umwandeln (warum ist der nicht noch Datetime?)
        #if not isinstance(data.index, pd.DatetimeIndex):
        #    data.index = pd.to_datetime(data.index)
        # liste der im DF-Index vorhandenen Jahre erstellen
        #years = data.index.year.unique().tolist()
        # holiday dict-like Objekt für die Jahre erstellen
        #hd = holidays.country_holidays(country=c,years=years)
        # Spalte im DF hinzufügen
        #data['holiday'] = data.index.map(lambda x: hd.get(x)) #beim index ist es dann map und nicht apply ...
        #return data'


    def build_full_df(self) -> pd.DataFrame:
        pb_df = self.get_public_power_series()
        p_df = self.get_price_series()
        p_pb_df = merge_dfs_by_index(pb_df, p_df)
        return p_pb_df


if __name__ == "__main__": 

    test_eci = EnergyChartsInfo()   

    # df= test_eci.get_price_series()
    # show(df)

    # ###
    # public_power_df = test_eci.get_public_power_series()
    # print("PUBLIC POWER -------------------------------")
    # print(public_power_df)

    price_df = test_eci.get_price_series()
    print("PRICE -------------------------------")
    print(price_df)
    print("---------")
    show(price_df)
    # price_power_df = merge_dfs_by_index(public_power_df, price_df)
    
    # dateiname = f"ECI_von_{test_eci.start_day}_bis_{test_eci.end_day}_merge.pkl"
    # with open(dateiname, "wb") as file:
    #     pickle.dump(price_power_df, file)

    # print("PRICE POWER -------------------------------")
    # show(price_power_df)
    ###
    
    # show(public_power_df)
    # show(price_df)

    

    # hour_public_power_df = test_eci.to_frequency(public_power_df, "h")
    # show(hour_public_power_df)
    # day_public_power_df = test_eci.to_frequency(public_power_df, "D")
    # show(day_public_power_df)

    # weekday_public_power_df = test_eci.get_weekday(day_public_power_df)
    # show(weekday_public_power_df)

    # week_public_power_df = test_eci.to_frequency(public_power_df, "W")
    # show(week_public_power_df)
    # year_public_power_df = test_eci.to_frequency(public_power_df, "Y")
    # show(year_public_power_df)



    ########### TO-DO:
    ########### Preise für 2015 bis mitte 2018, entsprechend URL umbauen:
    ########### -->  'https://api.energy-charts.info/price?bzn=DE-AT-LU&start=2015-02-01&end=2015-03-15' 
    ########### --> 20.09.18  21 auf 22 Uhr = andere Börse
    ######### neue concat funktion testen
    ######## funktionen für dne umgang mit NaNs und Nullwerten
    ######## methoden testen
    ######## ggf currentday -1
    