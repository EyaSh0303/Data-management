#%%
import sqlite3
from faker import Faker
import os 

fake = Faker()
db_file = "fund_database.db"

def create_tables():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    create_tables_queries = [
        """
        CREATE TABLE IF NOT EXISTS Clients (
            client_id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT,
            prenom TEXT,
            profil TEXT CHECK(profil IN ('Low risk', 'Low turnover', 'High yield equity only'))
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS Products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT,
            type TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS Portfolios (
            profil TEXT,
            produits TEXT,
            nb_clients INTEGER DEFAULT 0
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS Managers (
            manager_id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom_complet TEXT,
            portefeuille TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS Deals (
            deal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            profil INTEGER,
            actif_achete INTEGER,
            actif_vendu INTEGER,
            FOREIGN KEY (profil) REFERENCES Portfolios(profil),
            FOREIGN KEY (actif_achete) REFERENCES Products(product_id),
            FOREIGN KEY (actif_vendu) REFERENCES Products(product_id)
        );
        """
    ]
    for query in create_tables_queries:
        cursor.execute(query)
    conn.commit()
    conn.close()


def generate_clients(n_low_risk, n_low_turnover, n_high_yield):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    clients = {
        'Low risk': n_low_risk, 
        'Low turnover': n_low_turnover,
        'High yield equity only': n_high_yield
    }
    
    data_to_insert = []
    for profil, nombre in clients.items():
        for _ in range(nombre):
            nom = fake.last_name()
            prenom = fake.first_name()
            data_to_insert.append((nom, prenom, profil))

    cursor.executemany("INSERT INTO Clients (nom, prenom, profil) VALUES (?, ?, ?)", data_to_insert)
    conn.commit()
    conn.close()

def generate_managers():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    managers = [
    ('Dhelia Cherifi', 'Low risk'),
    ('Adam Bousselmame', 'Low turnover'),
    ('Eya Shili', 'High yield equity only')
    ]
    cursor.executemany("INSERT INTO Managers (nom_complet,portefeuille) VALUES (?,?)", managers)
    conn.commit()
    conn.close()

def generate_portfolios(n_low_risk, n_low_turnover, n_high_yield):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    portfolios = [
        ('Low Volatility', n_low_risk),
        ('Low Turnover', n_low_turnover),
        ('Equity Only', n_high_yield)
    ]
    cursor.executemany("INSERT INTO Portfolios (profil, nb_clients) VALUES (?, ?)", portfolios)
    conn.commit()
    conn.close()


# %%
