#%%
import sqlite3
import json
import logging
from datetime import datetime
from strategies import low_volatility_strategy, equity_only_strategy, low_turnover_strategy
from full_dict import full_dict, full_categories_dict

# Connexion à la base de données
def get_db_connection():
    return sqlite3.connect("fund_database.db", timeout=30)

def is_first_monday(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # weekday() retourne 0 pour lundi ; on considère que le premier lundi a un jour entre 1 et 7.
    return dt.weekday() == 0 and dt.day <= 7

def insert_deals(cursor, deal_date, profil, trade_instructions):
    """
    Insère dans la table Deals une ligne pour chaque opération d'achat ou de vente.
    Traite différemment selon que l'instruction est :
      - un dictionnaire unique avec les clés "Ticker" et "Adjustment" (cas low turnover),
      - un dictionnaire regroupant plusieurs tickers (cas low volatility),
      - ou une liste de tickers (cas equity only).
    """
    # Traitement des achats
    if "Buy" in trade_instructions and trade_instructions["Buy"]:
        buy_inst = trade_instructions["Buy"]
        # Cas low turnover : dict unique avec "Ticker" et "Adjustment"
        if isinstance(buy_inst, dict) and {"Ticker", "Adjustment"} <= set(buy_inst.keys()):
            ticker = buy_inst["Ticker"]
            if ticker is not None:
                cursor.execute("""
                    INSERT INTO Deals (date, profil, actif_achete, actif_vendu)
                    VALUES (?, ?, ?, ?)
                """, (deal_date, profil, ticker, None))
        # Cas low volatility : dict regroupant plusieurs tickers
        elif isinstance(buy_inst, dict):
            for ticker, _ in buy_inst.items():
                cursor.execute("""
                    INSERT INTO Deals (date, profil, actif_achete, actif_vendu)
                    VALUES (?, ?, ?, ?)
                """, (deal_date, profil, ticker, None))
        # Cas equity only : liste de tickers
        elif isinstance(buy_inst, list):
            for ticker in buy_inst:
                cursor.execute("""
                    INSERT INTO Deals (date, profil, actif_achete, actif_vendu)
                    VALUES (?, ?, ?, ?)
                """, (deal_date, profil, ticker, None))
                
    # Traitement des ventes
    if "Sell" in trade_instructions and trade_instructions["Sell"]:
        sell_inst = trade_instructions["Sell"]
        if isinstance(sell_inst, dict) and {"Ticker", "Adjustment"} <= set(sell_inst.keys()):
            ticker = sell_inst["Ticker"]
            if ticker is not None:
                cursor.execute("""
                    INSERT INTO Deals (date, profil, actif_achete, actif_vendu)
                    VALUES (?, ?, ?, ?)
                """, (deal_date, profil, None, ticker))
        elif isinstance(sell_inst, dict):
            for ticker, _ in sell_inst.items():
                cursor.execute("""
                    INSERT INTO Deals (date, profil, actif_achete, actif_vendu)
                    VALUES (?, ?, ?, ?)
                """, (deal_date, profil, None, ticker))
        elif isinstance(sell_inst, list):
            for ticker in sell_inst:
                cursor.execute("""
                    INSERT INTO Deals (date, profil, actif_achete, actif_vendu)
                    VALUES (?, ?, ?, ?)
                """, (deal_date, profil, None, ticker))

def update_products(cursor, tickers):
    """
    Pour chaque ticker présent dans 'tickers', récupère le nom du produit via full_dict et
    le type du produit via full_categories_dict. On vérifie que le type figure bien dans
    l'ensemble des catégories de full_categories_dict. Sinon, on le fixe par défaut à "Alternative".
    Puis, le produit est inséré dans la table Products s'il n'existe pas déjà.
    """
    possible_categories = set(full_categories_dict.values())
    for ticker in tickers:
        product_name = full_dict.get(ticker, ticker)
        product_type = full_categories_dict.get(ticker)
        # Vérifier que le type est défini et qu'il figure dans l'ensemble des catégories connues
        if product_type is None or product_type not in possible_categories:
            product_type = "Alternative"
        cursor.execute("SELECT product_id FROM Products WHERE nom = ?", (product_name,))
        res = cursor.fetchone()
        if res is None:
            cursor.execute("INSERT INTO Products (nom, type) VALUES (?, ?)", (product_name, product_type))

def update_profiles_table(start_date, end_date, nbtrees=100):
    """
    Pour chaque profil présent dans la table Portfolios, exécute la stratégie correspondante en utilisant
    end_date pour le calcul de la stratégie et start_date pour l'insertion dans la table Deals
    (afin d'éviter le look-forward bias).
    
    Met à jour la colonne 'produits' de Portfolios avec le JSON des pondérations,
    insère les deals dans la table Deals et met à jour la table Products avec l'ensemble des tickers extraits.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Récupérer la liste distincte des profils dans Portfolios
    cursor.execute("SELECT DISTINCT profil FROM Portfolios")
    profils = [row[0].strip() for row in cursor.fetchall()]

    # Liste des tickers issus de full_dict et accumulés depuis les stratégies
    portfolio_assets = [key.replace("-", "_") for key in full_dict.keys()]
    all_tickers = set()

    for profil in profils:
        if profil == "Low Volatility":
            strategy_func = lambda: low_volatility_strategy(end_date, portfolio_assets, k_clusters=4)
        elif profil == "Low Turnover":
            # Vérifier si la date (start_date) est le premier lundi du mois.
            if is_first_monday(start_date):
                strategy_func = lambda: low_turnover_strategy(
                    end_date,
                    portfolio_assets,
                    universe_assets=portfolio_assets,
                    short_window=20,
                    long_window=60,
                )
            else:
                logging.info(f"Low Turnover: Stratégie non lancée car {end_date} n'est pas le premier lundi du mois.")
                continue
        elif profil == "Equity Only":
            strategy_func = lambda: equity_only_strategy(
                end_date,
                portfolio_assets,
                universe_assets=portfolio_assets,
                feature_windows={"1m": 21, "3m": 63, "6m": 126, "12m": 252},
                target_window=5,
                nbtrees=nbtrees,
            )
        else:
            logging.warning(f"Profil inconnu : {profil}")
            continue

        result = strategy_func()
        logging.info(f"Résultat de la stratégie pour {profil} : {result}")

        if isinstance(result, tuple):
            portfolio_weights, trade_instructions = result
        else:
            portfolio_weights = result
            trade_instructions = {}

        if portfolio_weights is None:
            logging.warning(f"Stratégie pour {profil} a échoué.")
            continue

        # Accumuler les tickers présents dans la composition du portefeuille
        all_tickers.update(portfolio_weights.keys())

        produits_json = json.dumps(portfolio_weights)
        cursor.execute(
            """
            UPDATE Portfolios SET produits = ? WHERE profil = ?
            """,
            (produits_json, profil),
        )

        # Insérer les deals en utilisant start_date pour la colonne date
        if trade_instructions:
            insert_deals(cursor, start_date, profil, trade_instructions)

    # Mise à jour de la table Products avec l'ensemble des tickers récupérés
    update_products(cursor, all_tickers)

    conn.commit()
    conn.close()


# %%
