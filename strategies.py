#%%
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import cvxpy as cp
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import logging
from full_dict import full_dict, full_categories_dict
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Connexion à la base de données
def get_db_connection():
    return sqlite3.connect("fund_database.db", timeout=30)

# --- Fonctions de chargement et calcul des indicateurs ---

def load_market_data(ticker, start_date, end_date):
    conn = sqlite3.connect("market_data.db")
    query = f"""SELECT Date, Close, Volume, Category, Returns 
                FROM "{ticker}" 
                WHERE Date BETWEEN ? AND ? 
                ORDER BY Date;"""
    try:
        df = pd.read_sql(query, conn, params=(start_date, end_date))
    finally:
        conn.close()
    if df.empty:
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    today_dt = datetime.today()
    if end_dt.date() >= today_dt.date():
        df = df[df.index < today_dt.replace(hour=0, minute=0, second=0, microsecond=0)]
    return df

def annualized_return(series):
    daily_returns = series.pct_change().dropna()
    cum_return = (1 + daily_returns).prod() - 1
    nb_days = len(daily_returns)
    return (1 + cum_return)**(252/nb_days) - 1 if nb_days > 0 else 0

def annualized_volatility(series):
    daily_returns = series.pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)

def sharpe_ratio(series, rf=0.0):
    r_a = annualized_return(series)
    vol_a = annualized_volatility(series)
    return (r_a - rf) / vol_a if vol_a != 0 else 0

# --- Stratégies ---

def low_volatility_strategy(end_date, portfolio_assets, universe_assets=None, k_clusters=4, plot_clusters=False):
    """
    Applique la stratégie low volatility et renvoie un tuple :
      (portfolio_weights, trade_instructions)
    On peut choisir si oui ou non on affiche les graphiques avec les clusters déterminés (plot_cluster = False par défaut)
    """
    start_date = "2010-01-01"
    
    # code pour vérifier qu'il n'y a pas de look forward bias : 
    # print(f"J'entraîne un cluster de {start_date} à {end_date}")

    if universe_assets is None or len(universe_assets) == 0:
        universe_assets = portfolio_assets

    data_dict = {}
    features = []
    tickers_valides = []
    for asset in universe_assets:
        df = load_market_data(asset, start_date, end_date)
        if df is not None and not df.empty and len(df) > 30:
            ann_ret = annualized_return(df['Close'])
            ann_vol = annualized_volatility(df['Close'])
            sr = sharpe_ratio(df['Close'])
            features.append([ann_ret, ann_vol, sr])
            tickers_valides.append(asset)
            data_dict[asset] = df['Close']

    if not tickers_valides:
        logging.warning("Low volatility: Aucun actif valide pour le clustering.")
        return None, None

    features_arr = np.array(features)
    valid_indices = ~np.isnan(features_arr).any(axis=1)
    features_arr = features_arr[valid_indices]
    tickers_valides = [tickers_valides[i] for i in range(len(tickers_valides)) if valid_indices[i]]
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    labels = kmeans.fit_predict(features_arr)

    if plot_clusters:
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(features_arr[:,1], features_arr[:,2], c=labels, cmap='viridis', alpha=0.7)
        plt.xlabel("Volatilité annualisée")
        plt.ylabel("Ratio de Sharpe")
        plt.title("Clusters (Volatilité vs Sharpe)")
        plt.colorbar(scatter, label="Cluster")
        for i, ticker in enumerate(tickers_valides):
            plt.annotate(ticker, (features_arr[i,1], features_arr[i,2]), fontsize=8, alpha=0.7)
        plt.show()

    cluster_info = {}
    for c in range(k_clusters):
        indices = np.where(labels == c)[0]
        if len(indices) == 0:
            cluster_info[c] = (0, 999, 0, 0)
            continue
        sub = features_arr[indices]
        mean_ret = np.mean(sub[:, 0])
        mean_vol = np.mean(sub[:, 1])
        mean_sr  = np.mean(sub[:, 2])
        cluster_info[c] = (mean_ret, mean_vol, mean_sr, len(indices))

    sorted_clusters = sorted(cluster_info.items(), key=lambda kv: kv[1][1])
    chosen_cluster = None
    for cid, (_, _, m_sr, count) in sorted_clusters:
        if m_sr > 0 and count > 10:
            chosen_cluster = cid
            break

    if chosen_cluster is None:
        logging.info("Low volatility: Aucun cluster ne satisfait le critère.")
        return None, None

    subset_indices = np.where(labels == chosen_cluster)[0]
    subset_tickers = [tickers_valides[i] for i in subset_indices]
    logging.info(f"Low volatility: Tickers dans le cluster choisi: {subset_tickers}")

    if len(subset_tickers) < 2:
        logging.info("Low volatility: Cluster trop petit.")
        return None, None

    logging.info("Low volatility: Optimisation Markowitz.")
    price_data = {tck: data_dict[tck] for tck in subset_tickers}
    prices_df = pd.DataFrame(price_data)
    returns_df = prices_df.pct_change().dropna()
    if returns_df.empty:
        logging.warning("Low volatility: returns_df vide.")
        return None, None

    mu = returns_df.mean()
    cov = returns_df.cov().values
    n = len(subset_tickers)
    w = cp.Variable(n)
    portfolio_variance = cp.quad_form(w, cp.psd_wrap(cov))
    constraints = [cp.sum(w) == 1]
    expected_return = mu.values @ w
    constraints.append(expected_return >= mu.mean())
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    try:
        problem.solve(solver=cp.SCS, verbose=False)
    except cp.error.SolverError as e:
        logging.error(f"Low volatility: Solver error: {e}")
        return None, None

    optimal_weights = w.value
    if optimal_weights is None:
        logging.warning("Low volatility: Aucune solution Markowitz.")
        return None, None

    portfolio_weights = {ticker: optimal_weights[i] for i, ticker in enumerate(subset_tickers)}
    trade_instructions = {
         "Buy": {ticker: weight for ticker, weight in portfolio_weights.items() if weight > 0},
         "Sell": {ticker: weight for ticker, weight in portfolio_weights.items() if weight < 0}
    }

    return portfolio_weights, trade_instructions

def compute_momentum(df, short_window=20, long_window=60):
    sma_short = df['Close'].rolling(window=short_window, min_periods=short_window).mean()
    sma_long  = df['Close'].rolling(window=long_window, min_periods=long_window).mean()
    if pd.isnull(sma_short.iloc[-1]) or pd.isnull(sma_long.iloc[-1]) or sma_long.iloc[-1] == 0:
        return None
    return (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]

def low_turnover_strategy(end_date, portfolio_assets, universe_assets=None, short_window=20, long_window=60):
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=150)
    start_date = start_dt.strftime("%Y-%m-%d")
    if universe_assets is None or len(universe_assets) == 0:
        universe_assets = portfolio_assets
    momentum_signals = {}
    categories = {}

    # code pour vérifier qu'il n'y a pas de look forward bias : 
    # print(f"Je calcule le momentum de {start_date} à {end_date}")

    for asset in universe_assets:
        df = load_market_data(asset, start_date, end_date)
        if df is not None and len(df) >= long_window:
            momentum = compute_momentum(df, short_window, long_window)
            if momentum is not None:
                momentum_signals[asset] = momentum
                if 'Category' in df.columns:
                    categories[asset] = df['Category'].iloc[0]
                else:
                    categories[asset] = full_categories_dict.get(asset, "Equity")
    
    if not momentum_signals:
        logging.warning("Low Turnover: Pas d'actifs valides.")
        return None, None

    equities = {ticker: momentum for ticker, momentum in momentum_signals.items() if categories.get(ticker, "") == "Equity"}
    bonds    = {ticker: momentum for ticker, momentum in momentum_signals.items() if categories.get(ticker, "") == "Bond"}
    
    target_allocation = {"Equity": 0.2, "Bond": 0.8}
    init_equity = {ticker: target_allocation["Equity"]/len(equities) for ticker in equities} if equities else {}
    init_bond   = {ticker: target_allocation["Bond"]/len(bonds) for ticker in bonds} if bonds else {}
    
    best_equity = max(equities, key=equities.get) if equities else None
    worst_equity = min(equities, key=equities.get) if equities else None
    best_bond = max(bonds, key=bonds.get) if bonds else None
    worst_bond = min(bonds, key=bonds.get) if bonds else None

    candidate_buy = best_equity if best_equity and (not best_bond or equities[best_equity] >= bonds[best_bond]) else best_bond
    candidate_sell = worst_equity if worst_equity and (not worst_bond or equities[worst_equity] <= bonds[worst_bond]) else worst_bond

    new_equity = init_equity.copy()
    new_bond = init_bond.copy()
    trade_instructions = {}

    if candidate_buy is not None:
        if candidate_buy in equities:
            adjustment = equities[candidate_buy] if equities[candidate_buy] > 0 else 0
            new_equity[candidate_buy] = init_equity[candidate_buy] * (1 + adjustment)
            trade_instructions["Buy"] = {"Ticker": candidate_buy, "Adjustment": new_equity[candidate_buy] - init_equity[candidate_buy]}
            total_eq = sum(new_equity.values())
            for ticker in new_equity:
                new_equity[ticker] = new_equity[ticker] * target_allocation["Equity"] / total_eq
        elif candidate_buy in bonds:
            adjustment = bonds[candidate_buy] if bonds[candidate_buy] > 0 else 0
            new_bond[candidate_buy] = init_bond[candidate_buy] * (1 + adjustment)
            trade_instructions["Buy"] = {"Ticker": candidate_buy, "Adjustment": new_bond[candidate_buy] - init_bond[candidate_buy]}
            total_bond = sum(new_bond.values())
            for ticker in new_bond:
                new_bond[ticker] = new_bond[ticker] * target_allocation["Bond"] / total_bond

    if candidate_sell is not None:
        if candidate_sell in equities:
            adjustment = abs(equities[candidate_sell]) if equities[candidate_sell] < 0 else 0
            new_equity[candidate_sell] = init_equity[candidate_sell] * (1 - adjustment)
            trade_instructions["Sell"] = {"Ticker": candidate_sell, "Adjustment": init_equity[candidate_sell] - new_equity[candidate_sell]}
            total_eq = sum(new_equity.values())
            for ticker in new_equity:
                new_equity[ticker] = new_equity[ticker] * target_allocation["Equity"] / total_eq
        elif candidate_sell in bonds:
            adjustment = abs(bonds[candidate_sell]) if bonds[candidate_sell] < 0 else 0
            new_bond[candidate_sell] = init_bond[candidate_sell] * (1 - adjustment)
            trade_instructions["Sell"] = {"Ticker": candidate_sell, "Adjustment": init_bond[candidate_sell] - new_bond[candidate_sell]}
            total_bond = sum(new_bond.values())
            for ticker in new_bond:
                new_bond[ticker] = new_bond[ticker] * target_allocation["Bond"] / total_bond

    portfolio_weights = {}
    portfolio_weights.update(new_equity)
    portfolio_weights.update(new_bond)
    return portfolio_weights, trade_instructions

# --- Stratégie Equity Only ---
trained_rf_model = None  # Variable globale pour conserver le modèle en session

# Chemin du modèle pré-entraîné
MODEL_PATH = "trained_rf_model.pkl"

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

def save_trained_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def equity_only_strategy(end_date, portfolio_assets, universe_assets=None,
                         feature_windows={"1m":21, "3m":63, "6m":126, "12m":252},
                         target_window=5, nbtrees=100):
    global trained_rf_model
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=3000)
    start_date = start_dt.strftime("%Y-%m-%d")

    # Charger le modèle pré-entraîné s'il existe
    if trained_rf_model is None:
        trained_rf_model = load_trained_model()
    
    if universe_assets is None:
        universe_assets = portfolio_assets

    # On ne garde que les actions dont la catégorie est "Equity"
    equities = [ticker for ticker in universe_assets if full_categories_dict.get(ticker, "Equity") == "Equity"]
    
    if not equities:
        logging.warning("Equity Only: Aucune action disponible.")
        return None, None
    
    training_data = []
    current_features = {}
    for ticker in tqdm(equities, desc="Equity Only: Traitement"):
        if trained_rf_model is None:
            samples = generate_training_samples(ticker, start_date, end_date, feature_windows, target_window)
            if samples:
                training_data.extend(samples)
                
        df = load_market_data(ticker, start_date, end_date)
        if df is None or len(df) < max(feature_windows.values()):
            continue
        current_sharpe = sharpe_ratio(df['Close'])
        current_vol = annualized_volatility(df['Close'])
        current_perfs = []
        valid = True
        for key, win in feature_windows.items():
            if len(df) < win:
                valid = False
                break
            perf_val = (df['Close'].iloc[-1] / df['Close'].iloc[-win]) - 1
            current_perfs.append(perf_val)
        if not valid:
            continue
        current_features[ticker] = [current_sharpe, current_vol] + current_perfs
    
    if trained_rf_model is None:
        print(f"J'entraîne une forêt aléatoire du {start_date} jusqu'à {end_date}")  # éviter le look forward bias
        if not training_data:
            logging.warning("Equity Only: Pas assez de données pour entraîner le modèle.")
            return None, None
        X_train = np.array([features for features, target in training_data])
        y_train = np.array([target for features, target in training_data])
        mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[mask]
        y_train = y_train[mask]
        rf = RandomForestRegressor(n_estimators=nbtrees, random_state=42)
        rf.fit(X_train, y_train)
        trained_rf_model = rf
        save_trained_model(rf)
    else:
        rf = trained_rf_model

    predictions = {}
    for ticker, features in tqdm(current_features.items(), desc="Equity Only: Prédictions"):
        features_arr = np.array(features)
        if not np.isnan(features_arr).any():
            pred = rf.predict([features_arr])[0]
            predictions[ticker] = pred
    
    # code pour vérifier qu'il n'y a pas de look forward bias : 
    # print(f"J'entraîne un forêt aléatoire de {start_date} à {end_date}")

    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    buy_candidates = [ticker for ticker, _ in sorted_predictions[:5]]
    sell_candidates = [ticker for ticker, _ in sorted_predictions[-5:]]
    
    n_equities = len(current_features)
    init_weight = 1 / n_equities
    portfolio_weights = {ticker: init_weight for ticker in current_features}
    alpha = 2
    for ticker in buy_candidates:
        portfolio_weights[ticker] = init_weight * (1 + alpha)
    for ticker in sell_candidates:
        portfolio_weights[ticker] = init_weight * (1 - alpha)
    total = sum(portfolio_weights.values())
    for ticker in portfolio_weights:
        portfolio_weights[ticker] /= total
    trade_instructions = {"Buy": buy_candidates, "Sell": sell_candidates}
    return portfolio_weights, trade_instructions




def generate_training_samples(ticker, start_date, end_date, feature_windows, target_window=21):
    """
    Génère des échantillons d'entraînement pour un actif donné à partir des données historiques de marché.

    Paramètres :
        ticker (str) : Le symbole boursier de l'actif.
        start_date (str) : La date de début pour les données historiques au format "YYYY-MM-DD".
        end_date (str) : La date de fin pour les données historiques au format "YYYY-MM-DD".
        feature_windows (dict) : Un dictionnaire où les clés sont des noms descriptifs (par exemple, "1m", "3m") 
                                 et les valeurs sont le nombre de jours pour chaque fenêtre de caractéristiques.
        target_window (int) : Le nombre de jours pour calculer le rendement cible (par défaut 21).

    Retourne :
        list : Une liste de tuples où chaque tuple contient :
               - features (list) : Une liste de caractéristiques calculées (par exemple, ratio de Sharpe, volatilité, métriques de performance).
               - target (float) : Le rendement cible pour la fenêtre cible spécifiée.
    """
    df = load_market_data(ticker, start_date, end_date)
    if df is None or len(df) < max(feature_windows.values()) + target_window:
        return []
    samples = []
    max_window = max(feature_windows.values())
    for i in range(max_window, len(df) - target_window):
        window_data = df['Close'].iloc[i-max_window:i]
        feat_sharpe = sharpe_ratio(window_data)
        feat_vol = annualized_volatility(window_data)
        feat_perfs = []
        valid = True
        for key, win in feature_windows.items():
            if i < win:
                valid = False
                break
            perf_val = (df['Close'].iloc[i] / df['Close'].iloc[i-win]) - 1
            feat_perfs.append(perf_val)
        if not valid:
            continue
        features = [feat_sharpe, feat_vol] + feat_perfs
        target = (df['Close'].iloc[i+target_window] / df['Close'].iloc[i]) - 1
        samples.append((features, target))
    return samples


# %%
