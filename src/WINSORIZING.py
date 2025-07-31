import pandas as pd
# from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize

# Es waren immer mal absolute Extrem-Ausreißer in den Daten, das möchte ich hiermit etwas eindämmen.
def weekend_and_winsorize(df:pd.DataFrame, min_limit:float = 0.002, max_limit:float = 0.002 ) -> pd.DataFrame:
    df = df.copy()
    # Wochenende also bool (0, 1) hinzufügen
    df["weekday"] = pd.to_datetime(df.index).day_name()
    df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)
    # die geglätteten Preise als zusätzliche Spalte hinzufügen
    price_winsorized = winsorize(df['price'].dropna(), limits=[min_limit, max_limit])
    df.loc[df['price'].notnull(), 'price_winsorized'] = price_winsorized
    return df



