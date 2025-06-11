# %%
import pandas as pd 
import numpy as np 
import sqlite3
import json
from datetime import datetime, timedelta

# Connexions aux bases
conn_fund = sqlite3.connect("fund_database.db")
conn_market = sqlite3.connect("market_data.db")
cursor = conn_market.cursor()

def compute_beta(portfolio_returns, market_returns):
    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else 1

def taux_sans_risque(date1, max_attempts=4):
    current_date = date1
    attempts = 0
    while attempts < max_attempts:
        cursor.execute("SELECT Close FROM '^TNX' WHERE date = ?", (current_date,))
        result = cursor.fetchone()
        if result is not None and result[0] is not None:
            return result[0]
        dt = datetime.strptime(current_date, "%Y-%m-%d")
        dt_prev = dt - timedelta(days=1)
        current_date = dt_prev.strftime("%Y-%m-%d")
        attempts += 1
    return 0

def calcul_portfolio_volatilite(returns_series, composition):
    common_dates = None
    for ticker, series in returns_series.items():
        if common_dates is None:
            common_dates = series.index
        else:
            common_dates = common_dates.intersection(series.index)
    if common_dates is None or common_dates.empty or len(common_dates) < 2:
        return 0
    aligned_returns = {ticker: series.loc[common_dates] for ticker, series in returns_series.items()}
    df_aligned = pd.DataFrame(aligned_returns)
    weights = np.array([composition[t] for t in df_aligned.columns], dtype=float)
    if weights.sum() == 0:
        return 0
    elif abs(weights.sum() - 1.0) > 1e-6:
        weights = weights / weights.sum()
    cov_matrix = df_aligned.cov()
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    daily_volatility = np.sqrt(portfolio_variance)
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility


def ratio_sharpe(rendement, volatilite, date1):
    rf_val = taux_sans_risque(date1)
    rf = rf_val / 100 if rf_val is not None else 0
    if volatilite == 0:
        return 0
    return (rendement - rf) / volatilite


def ratio_sortino(rendements_indices, rendement_portefeuille, date1):
    target_return = taux_sans_risque(date1)/100
    returns = np.array(list(rendements_indices.values()))
    downside_returns = returns[returns < target_return]    
    if len(downside_returns) == 0:
        return np.nan
    downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
    if downside_deviation == 0:
        return np.nan
    ratio = (rendement_portefeuille - target_return) / downside_deviation
    return ratio


def rendement_marche(date1):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0] != "^TNX"]
    rendements = []
    for table in tables:
        cursor.execute(f"SELECT * FROM '{table}' WHERE Date = ?", (date1,))
        row = cursor.fetchone()
        if row:
            rendement = row[-1]
            if rendement is not None:
                rendements.append(rendement)
    if rendements:
        return sum(rendements) / len(rendements)
    else:
        return 0

def alpha_de_jensen(rendement_portefeuille, composition, date1):
    window_days = 30
    start_date = (datetime.strptime(date1, "%Y-%m-%d") - timedelta(days=window_days)).strftime("%Y-%m-%d")
    portfolio_returns_series = {}
    for ticker in composition:
        try:
            df_asset = pd.read_sql_query(
                f"SELECT Date, Close FROM '{ticker}' WHERE Date BETWEEN ? AND ? ORDER BY Date ASC",
                conn_market, params=(start_date, date1)
            )
            if df_asset.empty or len(df_asset) < 2:
                continue
            df_asset['Date'] = pd.to_datetime(df_asset['Date'])
            df_asset.sort_values('Date', inplace=True)
            df_asset.set_index('Date', inplace=True)
            daily_returns = df_asset['Close'].pct_change().dropna()
            portfolio_returns_series[ticker] = daily_returns
        except Exception:
            continue
    if not portfolio_returns_series:
        computed_beta = 1
    else:
        common_dates = None
        for series in portfolio_returns_series.values():
            if common_dates is None:
                common_dates = series.index
            else:
                common_dates = common_dates.intersection(series.index)
        if common_dates is None or common_dates.empty or len(common_dates) < 2:
            computed_beta = 1
        else:
            df_aligned = pd.DataFrame({t: s.loc[common_dates] for t, s in portfolio_returns_series.items()})
            weights = np.array([composition[t] for t in df_aligned.columns], dtype=float)
            if weights.sum() == 0:
                computed_beta = 1
            else:
                weights /= weights.sum()
                portfolio_returns = (df_aligned * weights).sum(axis=1)
                try:
                    benchmark_df = pd.read_sql_query(
                        f"SELECT Date, Close FROM 'SPY' WHERE Date BETWEEN ? AND ? ORDER BY Date ASC",
                        conn_market, params=(start_date, date1)
                    )
                    if benchmark_df.empty or len(benchmark_df) < 2:
                        computed_beta = 1
                    else:
                        benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
                        benchmark_df.sort_values('Date', inplace=True)
                        benchmark_df.set_index('Date', inplace=True)
                        benchmark_returns = benchmark_df['Close'].pct_change().dropna()
                        common_benchmark_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                        if len(common_benchmark_dates) == 0:
                            computed_beta = 1
                        else:
                            pr_array = portfolio_returns.loc[common_benchmark_dates].values
                            bm_array = benchmark_returns.loc[common_benchmark_dates].values
                            computed_beta = compute_beta(pr_array, bm_array)
                except Exception:
                    computed_beta = 1
    rm = rendement_marche(date1)
    rf_val = taux_sans_risque(date1)
    rf = rf_val / 100 if rf_val is not None else 0.0
    return rendement_portefeuille - (rf + computed_beta * (rm - rf))

def VaR(rendements_indices, composition, alpha=0.05):
    rendements = np.array([rendements_indices[i] * composition[i] for i in composition])
    return np.percentile(rendements, 100 * alpha)

def calculer_performance_semaine(date1, date2):
    performances_list = []
    df_portfolios = pd.read_sql_query("SELECT rowid AS portfolio_id, * FROM Portfolios", conn_fund)
    for _, row in df_portfolios.iterrows():
        profil = row["profil"]
        produits_json = row["produits"]
        portfolio_id = row["portfolio_id"]
        strategie = profil
        if not produits_json:
            continue
        try:
            composition = json.loads(produits_json)
        except Exception:
            continue
        rendements_indices = {}
        returns_series = {}
        prix_debut, prix_fin = {}, {}
        for indice, poids in composition.items():
            try:
                df_indice = pd.read_sql_query(f"""
                    SELECT * FROM '{indice}' 
                    WHERE Date BETWEEN '{date1}' AND '{date2}'
                    ORDER BY Date ASC
                """, conn_market)
                if len(df_indice) >= 2:
                    prix_debut[indice] = df_indice["Close"].iloc[0]
                    prix_fin[indice] = df_indice["Close"].iloc[-1]
                    rendement = (prix_fin[indice] - prix_debut[indice]) / prix_debut[indice]
                    rendements_indices[indice] = rendement
                    daily_returns = df_indice["Close"].pct_change().dropna()
                    returns_series[indice] = daily_returns
                else:
                    rendements_indices[indice] = 0
            except Exception:
                rendements_indices[indice] = 0
        rendement_portefeuille = sum(rendements_indices[indice] * poids for indice, poids in composition.items())
        volatilite = calcul_portfolio_volatilite(returns_series, composition)
        sharpe = ratio_sharpe(rendement_portefeuille, volatilite, date1)
        sortino = ratio_sortino(rendements_indices, rendement_portefeuille, date1)
        alpha = alpha_de_jensen(rendement_portefeuille, composition, date1)
        var = VaR(rendements_indices, composition)
        performances_list.append({
            "portfolio_id": portfolio_id,
            "profil": profil,
            "strategie": strategie,
            "date_debut": date1,
            "date_fin": date2,
            "rendement": rendement_portefeuille,
            "volatilite": volatilite,
            "sharpe": sharpe,
            "sortino": sortino,
            "alpha": alpha,
            "VaR": var,
        })
    return pd.DataFrame(performances_list)


def compute_aggregated_performance(df_weekly):
    agg_return = (1 + df_weekly['rendement']).prod() - 1
    agg_volatility = df_weekly['volatilite'].mean()
    agg_sharpe = df_weekly['sharpe'].mean()
    agg_sortino = df_weekly['sortino'].mean()
    agg_alpha   = df_weekly['alpha'].mean()
    agg_VaR = np.percentile(df_weekly['rendement'], 5)
    agg_perf = {
        'aggregated_return': agg_return,
        'aggregated_volatility': agg_volatility,
        'aggregated_sharpe': agg_sharpe,
        'aggregated_sortino': agg_sortino,
        'aggregated_alpha': agg_alpha,
        'aggregated_VaR': agg_VaR
    }
    return agg_perf
def compute_aggregated_performance_by_portfolio(df_weekly):
    """
    Calcule, pour chaque profil (portefeuille), les indicateurs agrégés :
      - aggregated_return (rendement cumulé)
      - aggregated_volatility (moyenne des volatilités hebdomadaires)
      - aggregated_sharpe (moyenne du sharpe)
      - aggregated_sortino (moyenne du sortino)
      - aggregated_alpha (moyenne de l'alpha)
      - aggregated_VaR (5e percentile des rendements)
    Retourne un DataFrame avec une ligne par profil.
    """
    # On réinitialise l'index pour que 'profil' soit une colonne
    df_reset = df_weekly.reset_index()
    def aggregator(subdf):
        subdf_numeric = subdf.select_dtypes(include=[np.number])
        aggregated_return = (1 + subdf_numeric['rendement']).prod() - 1
        aggregated_volatility = subdf_numeric['volatilite'].mean()
        aggregated_sharpe = subdf_numeric['sharpe'].mean()
        aggregated_sortino = subdf_numeric['sortino'].mean()
        aggregated_alpha = subdf_numeric['alpha'].mean()
        aggregated_VaR = np.percentile(subdf_numeric['rendement'], 5)
        # On récupère le profil à partir du premier élément de la colonne 'profil'
        profil = subdf['profil'].iloc[0] if 'profil' in subdf.columns else ""
        return pd.Series({
            'profil': profil,
            'aggregated_return': aggregated_return,
            'aggregated_volatility': aggregated_volatility,
            'aggregated_sharpe': aggregated_sharpe,
            'aggregated_sortino': aggregated_sortino,
            'aggregated_alpha': aggregated_alpha,
            'aggregated_VaR': aggregated_VaR
        })
    
    df_agg = df_reset.groupby('profil').apply(aggregator).reset_index(drop=True)
    return df_agg




def add_charts_by_profile_to_excel(writer, df_total):
    """
    Pour chaque profil présent dans df_total, cette fonction crée des graphiques illustrant l'évolution hebdomadaire
    des indicateurs clés (rendement, volatilité, alpha, Sharpe, VaR). Pour l'axe des abscisses, elle utilise la colonne
    'date_fin' convertie en format 'dd/mm/yyyy'. Chaque groupe de graphiques (pour un même profil) est inséré verticalement
    dans une nouvelle feuille "Charts".
    """
    # Utiliser le workbook de writer
    wb = writer.book

    # Réinitialiser l'index pour avoir accès à toutes les colonnes
    df_reset = df_total.reset_index()
    # Convertir 'date_fin' en datetime et trier
    df_reset['date_fin'] = pd.to_datetime(df_reset['date_fin'], errors='coerce')
    df_reset = df_reset.sort_values('date_fin')
    
    # Créer la feuille "Charts"
    charts_ws = wb.add_worksheet("Charts")
    
    # Obtenir la liste unique des profils
    profiles = df_reset['profil'].unique()
    
    # Paramètres pour le placement des graphiques
    start_row = 0
    chart_height = 15  # hauteur approximative en nombre de lignes
    vertical_spacing = 2  # espacement entre graphiques
    
    for profile in profiles:
        # Filtrer les données pour le profil courant
        df_profile = df_reset[df_reset['profil'] == profile].copy()
        # Créer une colonne formatée pour la date
        df_profile['date_fin_str'] = df_profile['date_fin'].dt.strftime('%d/%m/%Y')
        
        # Créer une feuille temporaire cachée pour ce profil (pour servir de source aux graphiques)
        temp_sheet_name = f"Data_Graph_{profile}"
        wb.add_worksheet(temp_sheet_name)
        df_profile.to_excel(writer, sheet_name=temp_sheet_name, index=False)
        
        # Récupérer les indices des colonnes dans df_profile
        date_idx = df_profile.columns.get_loc('date_fin_str')
        # Les indicateurs à tracer
        indicators = ['rendement', 'volatilite', 'alpha', 'sharpe', 'VaR']
        
        # Nombre de lignes dans le DataFrame (hors entête)
        num_rows = df_profile.shape[0]
        
        # Pour chaque indicateur, créer un graphique
        for indicator in indicators:
            try:
                col_idx = df_profile.columns.get_loc(indicator)
            except Exception:
                continue  # si la colonne n'existe pas, on passe
            
            chart = wb.add_chart({'type': 'line'})
            chart.add_series({
                'name':       indicator,
                'categories': [temp_sheet_name, 1, date_idx, num_rows, date_idx],
                'values':     [temp_sheet_name, 1, col_idx, num_rows, col_idx],
                'line':       {'color': ('blue' if indicator=='rendement' else
                                          'red' if indicator=='volatilite' else
                                          'green' if indicator=='alpha' else
                                          'orange' if indicator=='sharpe' else
                                          'purple')}
            })
            chart.set_title({'name': f"{indicator.capitalize()} - {profile}"})
            chart.set_x_axis({
                'name': 'Date Fin',
                'date_axis': True,
                'num_format': 'dd/mm/yyyy'
            })
            chart.set_y_axis({'name': indicator.capitalize()})
            # Insérer le graphique dans la feuille "Charts" à la position calculée
            cell = f"B{start_row + 2}"
            charts_ws.insert_chart(cell, chart, {'x_scale': 1.5, 'y_scale': 1.5})
            start_row += chart_height + vertical_spacing
        
        start_row += vertical_spacing * 3

# %%
