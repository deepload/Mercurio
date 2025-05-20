#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto API Helper
----------------
Ce module contient des fonctions utilitaires pour les appels directs à l'API crypto d'Alpaca.
Il prend en charge les différentes structures de réponse, y compris les réponses paginées.
"""

import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import yfinance as yf

# Configuration du logging
logger = logging.getLogger(__name__)

def fetch_crypto_data_direct(
    symbol: str,
    api_key: str,
    api_secret: str,
    timeframe: str = "1Min",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Effectue un appel direct à l'API crypto d'Alpaca et gère correctement toutes les structures de réponse.
    Args:
        symbol: Symbole crypto au format BTC/USD
        api_key: Clé API Alpaca
        api_secret: Secret API Alpaca
        timeframe: Intervalle de temps (1Min, 5Min, 15Min, 1H, 1D)
        start_date: Date de début (format ISO)
        end_date: Date de fin (format ISO)
        limit: Nombre maximum de barres à récupérer
    Returns:
        DataFrame pandas avec les données historiques, colonnes: timestamp, open, high, low, close, volume
    """
    endpoint = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "limit": limit
    }
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    logger.info(f"Appel API crypto pour {symbol} avec URL: {endpoint}")
    all_bars = []
    next_token = None
    try:
        while True:
            if next_token:
                params["page_token"] = next_token
            response = requests.get(endpoint, params=params, headers=headers)
            if response.status_code != 200:
                logger.error(f"Erreur API {response.status_code}: {response.text}")
                break
            data = response.json()
            logger.debug(f"Clés disponibles dans la réponse: {list(data.keys())}")
            bars_data = []
            if 'bars' in data and isinstance(data['bars'], list):
                bars_data = data['bars']
                logger.info(f"Structure de liste directe détectée pour {symbol}")
            elif 'bars' in data and isinstance(data['bars'], dict):
                if symbol in data['bars']:
                    bars_data = data['bars'][symbol]
                    logger.info(f"Structure de dictionnaire standard détectée pour {symbol}")
                elif data['bars']:
                    available_symbols = list(data['bars'].keys())
                    logger.info(f"Symboles disponibles dans la réponse: {available_symbols}")
                    for available_symbol in available_symbols:
                        if symbol.replace('/', '') in available_symbol or \
                           available_symbol in symbol or \
                           symbol in available_symbol:
                            logger.info(f"Correspondance partielle trouvée: {available_symbol} pour {symbol}")
                            bars_data = data['bars'][available_symbol]
                            break
            if bars_data:
                all_bars.extend(bars_data)
                logger.info(f"Récupéré {len(bars_data)} barres pour {symbol}")
            if 'next_page_token' in data and data['next_page_token']:
                next_token = data['next_page_token']
                logger.info(f"Page suivante disponible avec jeton: {next_token[:10]}...")
            else:
                break
        if all_bars:
            logger.info(f"Données crypto reçues avec succès pour {symbol} ({len(all_bars)} barres)")
            df = pd.DataFrame(all_bars)
            if 'timestamp' not in df.columns and 't' in df.columns:
                df['timestamp'] = df['t']
            if 'timestamp' in df.columns:
                if isinstance(df['timestamp'].iloc[0], (int, float)):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            column_mapping = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    logger.warning(f"Colonne {col} manquante dans les données")
            return df
        else:
            logger.warning(f"Aucune barre récupérée pour {symbol}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
        return pd.DataFrame()


def fetch_crypto_data_with_fallback(
    symbol: str,
    api_key: str,
    api_secret: str,
    timeframe: str = "1Min",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Tente de récupérer les données OHLCV crypto depuis Alpaca, puis bascule sur Yahoo Finance si Alpaca échoue.
    Affiche le résultat dans la console.
    """
    print(f"[INFO] Trying to fetch {symbol} from Alpaca...")
    df = fetch_crypto_data_direct(
        symbol=symbol,
        api_key=api_key,
        api_secret=api_secret,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    if not df.empty:
        print(f"[SUCCESS] Data for {symbol} fetched from Alpaca:")
        print(df.head())
        return df
    else:
        print(f"[WARNING] Alpaca failed or returned no data for {symbol}. Trying Yahoo Finance...")
        # Convert symbol to Yahoo format (e.g., BTC-USD)
        yahoo_symbol = symbol.replace('/', '-')
        try:
            yf_ticker = yf.Ticker(yahoo_symbol)
            # Yahoo supports interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            yf_timeframe = timeframe.lower()
            if yf_timeframe == '1min':
                yf_timeframe = '1m'
            elif yf_timeframe == '5min':
                yf_timeframe = '5m'
            elif yf_timeframe == '15min':
                yf_timeframe = '15m'
            elif yf_timeframe == '1h':
                yf_timeframe = '1h'
            elif yf_timeframe == '1d':
                yf_timeframe = '1d'
            else:
                yf_timeframe = '1d'  # default fallback

            df_yf = yf_ticker.history(
                interval=yf_timeframe,
                start=start_date,
                end=end_date
            )
            if not df_yf.empty:
                df_yf = df_yf.reset_index()
                df_yf.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Datetime': 'timestamp',
                    'Date': 'timestamp',
                }, inplace=True)
                print(f"[SUCCESS] Data for {symbol} fetched from Yahoo Finance:")
                print(df_yf.head())
                return df_yf
            else:
                print(f"[ERROR] Yahoo Finance returned no data for {symbol}.")
                return pd.DataFrame()
        except Exception as ex:
            print(f"[ERROR] Exception fetching {symbol} from Yahoo Finance: {ex}")
            return pd.DataFrame()

    """
    Effectue un appel direct à l'API crypto d'Alpaca et gère correctement toutes les structures de réponse.
    
    Args:
        symbol: Symbole crypto au format BTC/USD
        api_key: Clé API Alpaca
        api_secret: Secret API Alpaca
        timeframe: Intervalle de temps (1Min, 5Min, 15Min, 1H, 1D)
        start_date: Date de début (format ISO)
        end_date: Date de fin (format ISO)
        limit: Nombre maximum de barres à récupérer
        
    Returns:
        DataFrame pandas avec les données historiques, colonnes: timestamp, open, high, low, close, volume
    """
    # Endpoint API
    endpoint = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
    
    # Paramètres de base
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "limit": limit
    }
    
    # Ajout des paramètres optionnels
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date
    
    # En-têtes d'authentification
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    
    logger.info(f"Appel API crypto pour {symbol} avec URL: {endpoint}")
    
    # Stockage pour toutes les barres récupérées
    all_bars = []
    next_token = None
    
    try:
        # Possibilité de plusieurs pages
        while True:
            # Ajouter le jeton de pagination si disponible
            if next_token:
                params["page_token"] = next_token
            
            # Faire l'appel API
            response = requests.get(endpoint, params=params, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Erreur API {response.status_code}: {response.text}")
                break
            
            data = response.json()
            
            # Logger les clés pour le débogage
            logger.debug(f"Clés disponibles dans la réponse: {list(data.keys())}")
            
            # Traiter les différentes structures de réponse possibles
            bars_data = []
            
            # Cas 1: Structure avec 'bars' comme liste directe
            if 'bars' in data and isinstance(data['bars'], list):
                bars_data = data['bars']
                logger.info(f"Structure de liste directe détectée pour {symbol}")
            
            # Cas 2: Structure avec 'bars' comme dictionnaire indexé par symbole
            elif 'bars' in data and isinstance(data['bars'], dict):
                if symbol in data['bars']:
                    bars_data = data['bars'][symbol]
                    logger.info(f"Structure de dictionnaire standard détectée pour {symbol}")
                elif data['bars']:  # Le dictionnaire contient d'autres données
                    # Vérifier si une version modifiée du symbole est présente
                    available_symbols = list(data['bars'].keys())
                    logger.info(f"Symboles disponibles dans la réponse: {available_symbols}")
                    
                    # Essayer de trouver des correspondances partielles
                    for available_symbol in available_symbols:
                        if symbol.replace('/', '') in available_symbol or \
                           available_symbol in symbol or \
                           symbol in available_symbol:
                            logger.info(f"Correspondance partielle trouvée: {available_symbol} pour {symbol}")
                            bars_data = data['bars'][available_symbol]
                            break
            
            # Si on a trouvé des barres, les ajouter au résultat
            if bars_data:
                all_bars.extend(bars_data)
                logger.info(f"Récupéré {len(bars_data)} barres pour {symbol}")
            
            # Vérifier s'il y a plus de pages
            if 'next_page_token' in data and data['next_page_token']:
                next_token = data['next_page_token']
                logger.info(f"Page suivante disponible avec jeton: {next_token[:10]}...")
            else:
                # Plus de pages, sortir de la boucle
                break
        
        # Créer un DataFrame à partir des barres
        if all_bars:
            logger.info(f"Données crypto reçues avec succès pour {symbol} ({len(all_bars)} barres)")
            
            # Convertir en DataFrame
            df = pd.DataFrame(all_bars)
            
            # Standardiser les noms de colonnes si nécessaire
            if 'timestamp' not in df.columns and 't' in df.columns:
                df['timestamp'] = df['t']
            
            # Convertir les timestamps si nécessaire
            if 'timestamp' in df.columns:
                if isinstance(df['timestamp'].iloc[0], (int, float)):
                    # Convertir en datetime si c'est un entier (timestamp unix)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    # Sinon, essayer de parser comme string ISO
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Standardiser d'autres noms de colonnes si nécessaire
            column_mapping = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # S'assurer que toutes les colonnes essentielles sont présentes
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    logger.warning(f"Colonne {col} manquante dans les données")
            
            return df
        else:
            logger.warning(f"Aucune barre récupérée pour {symbol}")
            return pd.DataFrame()  # DataFrame vide
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
        return pd.DataFrame()  # DataFrame vide
