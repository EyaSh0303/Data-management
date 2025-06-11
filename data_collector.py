#%% 
import yfinance as yf
import pandas as pd
import sqlite3
from full_dict import full_dict, full_categories_dict
import time

start_date = "2010-01-01"
end_date = "2024-12-31"

def get_financial_data(ticker, start_date, end_date):
    """
    Télécharge les données pour un ticker donné via yfinance, calcule les rendements,
    traite les valeurs aberrantes (avec z-score) et remplace les valeurs manquantes par un forward fill.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[['Close', 'Volume']].copy()
        df['Category'] = full_categories_dict.get(ticker, 'Action')
        df['Returns'] = df['Close'].pct_change()
        z_score = (df['Returns'] - df['Returns'].mean()) / df['Returns'].std()
        df.loc[z_score.abs() > 3, 'Returns'] = None
        df['Returns'] = df['Returns'].fillna(method='ffill')
        df.index.name = 'Date'
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Erreur pour {ticker} : {e}")
        return None



keys = list(full_dict.keys())

# --- Étape 1 : Insertion dans market_data.db ---
# Ouvrir la connexion vers market_data.db avec timeout
conn_market = sqlite3.connect("market_data.db", timeout=30)
cursor_market = conn_market.cursor()

for i, symbol in enumerate(keys):
    print(f"[{i+1}/{len(keys)}] Importation de {symbol}")
    data = get_financial_data(symbol, start_date, end_date)
    if data is None or data.empty:
        continue

    table_name = symbol.replace("-", "_")
    

    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_str'] = data['Date'].dt.strftime('%Y-%m-%d')

    # Création de la table pour ce ticker dans market_data.db
    create_query = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        Date TEXT PRIMARY KEY,
        Close REAL,
        Volume INTEGER,
        Category TEXT,
        Returns REAL
    );
    """
    cursor_market.execute(create_query)
    records = data[['Date_str', 'Close', 'Volume', 'Category', 'Returns']].values.tolist()
    insert_query = f"""
    INSERT OR REPLACE INTO `{table_name}` (Date, Close, Volume, Category, Returns)
    VALUES (?, ?, ?, ?, ?);
    """
    cursor_market.executemany(insert_query, records)
    conn_market.commit()

conn_market.commit()
conn_market.close()

# %%
