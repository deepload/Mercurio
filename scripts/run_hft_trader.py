#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mercurio AI - Script de Trading Haute Fréquence (HFT)
-----------------------------------------------------

Ce script implémente un trader haute fréquence utilisant les WebSockets d'Alpaca
pour obtenir des mises à jour de prix et d'ordres en temps réel. Il est conçu
pour fonctionner avec une latence minimale et traiter plusieurs symboles simultanément.

Caractéristiques:
- Utilise les WebSockets pour les données en temps réel (au lieu des requêtes REST)
- Traitement asynchrone des données et ordres
- Optimisé pour les stratégies à haute fréquence
- Support des données L2 (carnet d'ordres) avec l'API Alpaca Premium
"""

import os
import sys
import json
import time
import signal
import asyncio
import logging
import argparse
import threading
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import deque, defaultdict
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# API Alpaca
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream

# Import des modules Mercurio
# Import des services Mercurio
from app.services.market_data import MarketDataService
try:
    from app.services.trading import TradingService
except ImportError:
    TradingService = None
from app.strategies.base import BaseStrategy

# Fonction pour détecter le niveau d'accès Alpaca
def detect_alpaca_level(api_key=None, api_secret=None, base_url=None, data_url=None):
    """
    Détecte le niveau d'abonnement Alpaca disponible en testant les fonctionnalités
    
    Args:
        api_key: Clé API Alpaca
        api_secret: Secret API Alpaca
        base_url: URL de base pour l'API Alpaca
        data_url: URL des données pour l'API Alpaca
        
    Returns:
        int: Niveau d'abonnement (3 = premium, 2 = standard+, 1 = standard, 0 = non détecté)
    """
    if not api_key or not api_secret:
        # Récupérer les clés d'API depuis les variables d'environnement
        load_dotenv()
        # Déterminer le mode (paper ou live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        if alpaca_mode == "live":
            api_key = os.getenv("ALPACA_LIVE_KEY")
            api_secret = os.getenv("ALPACA_LIVE_SECRET")
            base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        else:  # mode paper par défaut
            api_key = os.getenv("ALPACA_PAPER_KEY")
            api_secret = os.getenv("ALPACA_PAPER_SECRET")
            base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
        
        data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
    
    # Initialiser le client API
    try:
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
            data_url=data_url
        )
        
        logger.info("Test du niveau d'abonnement Alpaca...")
        
        # Test niveau 3 (premium) - Accès aux données en temps réel
        try:
            # Tester une fonctionnalité spécifique au niveau 3: données en temps réel plus précises
            end = datetime.now()
            start = end - timedelta(minutes=15)
            symbol = "BTC/USD"  # Une paire crypto populaire
            bars = api.get_crypto_bars(symbol, '1Min', start=start.isoformat(), end=end.isoformat())
            if len(bars) > 0:
                logger.info("✅ Niveau 3 (Premium) détecté - Accès complet aux données temps réel")
                return 3
        except Exception as e:
            logger.debug(f"Test niveau 3 échoué: {str(e)}")
        
        # Test niveau 2 - Données historiques étendues
        try:
            # Tester des données historiques (disponibles dans les niveaux 2 et 3)
            end = datetime.now()
            start = end - timedelta(days=30)  # 30 jours de données
            bars = api.get_crypto_bars('BTC/USD', '1Day', start=start.isoformat(), end=end.isoformat())
            if len(bars) > 20:  # Si on a plus de 20 jours, c'est probablement niveau 2+
                logger.info("✅ Niveau 2 détecté - Accès aux données historiques étendues")
                return 2
        except Exception as e:
            logger.debug(f"Test niveau 2 échoué: {str(e)}")
        
        # Test niveau 1 - Fonctionnalités de base
        try:
            account = api.get_account()
            if account and account.status == 'ACTIVE':
                logger.info("✅ Niveau 1 détecté - Accès aux fonctionnalités de base")
                return 1
        except Exception as e:
            logger.debug(f"Test niveau 1 échoué: {str(e)}")
        
        # Si tous les tests échouent
        logger.warning("❌ Aucun niveau d'API détecté, vérifiez vos clés API")
        return 0
    except Exception as e:
        logger.error(f"Erreur lors de la détection du niveau d'API: {e}")
        return 0
try:
    from app.strategies.moving_average_ml import MovingAverageMLStrategy
    
    # Implémentation concrète de MovingAverageMLStrategy pour le HFT
    class HFTMovingAverageMLStrategy(MovingAverageMLStrategy):
        def __init__(self, **kwargs):
            # Paramètres optimisés pour HFT
            kwargs.setdefault('short_window_min', 3)
            kwargs.setdefault('short_window_max', 20)
            kwargs.setdefault('long_window_min', 10)
            kwargs.setdefault('long_window_max', 50)
            super().__init__(**kwargs)
        
        def backtest(self, data):
            # Implémentation simple du backtest
            signals = self.generate_signals(data)
            return signals
        
        def load_data(self, *args, **kwargs):
            # Le chargement de données est géré par le HFTrader
            return None
        
        def predict(self, data):
            # Génération de signaux basés sur les indicateurs techniques
            signals = self.generate_signals(data)
            
            # Ajouter une confiance basée sur la force du signal
            if signals.empty:
                return None
                
            last_signal = signals.iloc[-1]
            signal_dict = {
                'action': 'buy' if last_signal['signal'] > 0 else 'sell' if last_signal['signal'] < 0 else 'hold',
                'confidence': abs(last_signal['signal_strength']) if 'signal_strength' in last_signal else 0.6,
                'reason': 'MovingAverageML signal' 
            }
            
            return signal_dict
        
        def preprocess_data(self, data):
            # Simple prétraitement, assurez-vous d'avoir les colonnes requises
            if isinstance(data, pd.DataFrame):
                return data
            return pd.DataFrame(data)
    
    # Remplacer la classe abstraite par notre implémentation concrète
    MovingAverageMLStrategy = HFTMovingAverageMLStrategy
    
except ImportError:
    MovingAverageMLStrategy = None

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"hft_trader_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hft_trader")

# Global constants
MARKET_DATA_INTERVAL = 1  # secondes
BACKTEST_INTERVAL = 0.1  # secondes

# Charger la liste personnalisée de cryptos depuis le fichier .env
# Assurez-vous que la variable PERSONALIZED_CRYPTO_LIST est définie dans le fichier .env
default_crypto_list = "BTC/USD,ETH/USD,SOL/USD"
custom_crypto_list_str = os.getenv("PERSONALIZED_CRYPTO_LIST", default_crypto_list)
PERSONALIZED_CRYPTO_LIST = [s.strip() for s in custom_crypto_list_str.split(',')]

# Version sans slash pour l'API HFT
PERSONALIZED_CRYPTO_LIST_NO_SLASH = [
    symbol.replace("/", "") for symbol in PERSONALIZED_CRYPTO_LIST
]

# Nombre de barres historiques à garder en mémoire
MAX_HISTORICAL_BARS = 100  
MAX_TRADE_FREQUENCY = 5  # Secondes minimales entre deux trades pour un même symbole

# Seuils pour éviter les dust positions
MIN_POSITION_VALUE_USD = 10.0  # Valeur minimale en USD pour une position
MIN_POSITION_SIZE = {  # Tailles minimales spécifiques par crypto
    "BTCUSD": 0.0001,  # ~5-6 USD minimum
    "ETHUSD": 0.005,   # ~10-15 USD minimum
    "DEFAULT": 0.01    # Valeur par défaut pour autres cryptos
}
DEFAULT_SYMBOLS = {
    "STOCK": ["AAPL", "AMZN", "MSFT", "GOOGL", "TSLA"], 
    "CRYPTO": ["BTCUSD", "ETHUSD", "SOLUSD"]
}

# Variables globales
running = True  # Contrôle la boucle principale

# Signal handler pour arrêt propre
def signal_handler(sig, frame):
    global running
    logger.info('Signal d\'arrêt reçu. Arrêt propre en cours...')
    running = False

# Enregistrer le handler
signal.signal(signal.SIGINT, signal_handler)

# Énumérations
class AssetType(Enum):
    STOCK = auto()
    CRYPTO = auto()

class StrategyType(Enum):
    MOVING_AVERAGE = "moving_average"
    MOVING_AVERAGE_ML = "moving_average_ml"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ORDERBOOK_IMBALANCE = "orderbook_imbalance"

class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class HFTrader:
    """Trader haute fréquence utilisant les WebSockets d'Alpaca"""
    
    def __init__(
        self,
        symbols: List[str] = None,
        api_key: str = None, 
        api_secret: str = None,
        base_url: str = None,
        data_url: str = None,
        strategy_type: StrategyType = StrategyType.MOVING_AVERAGE,
        asset_type: AssetType = AssetType.STOCK,
        position_size_pct: float = 0.02,   # 2% du portefeuille par position par défaut
        stop_loss_pct: float = 0.01,       # Stop loss à 1% par défaut
        take_profit_pct: float = 0.05,     # Take profit à 5% par défaut
        api_level: int = 0,                # Niveau d'API Alpaca (0=auto-detect, 1-3=niveau spécifique)
        max_positions: int = 5,            # Max 5 positions simultanées
        is_paper: bool = True,             # Paper trading par défaut
        use_custom_symbols: bool = False   # Utiliser les symboles personnalisés
    ):
        """Initialiser le trader HF"""
        # Utiliser les symboles personnalisés si demandé
        if use_custom_symbols:
            logger.info("Utilisation de la liste de symboles personnalisée")
            if asset_type == AssetType.CRYPTO:
                # Utiliser tous les symboles fournis - sans limitation
                if symbols and len(symbols) > 0:
                    # Si des symboles ont été spécifiés (par --symbols ou --custom-symbols-file), utiliser ceux-ci
                    logger.info(f"Utilisation de {len(symbols)} symboles crypto personnalisés")
                else:
                    # Sinon, utiliser la liste par défaut complète
                    symbols = PERSONALIZED_CRYPTO_LIST_NO_SLASH
                    logger.info(f"Utilisation de la liste complète de {len(symbols)} symboles crypto par défaut")
            else:
                # Pour les actions, utiliser tous les symboles spécifiés ou une liste par défaut
                if not symbols or len(symbols) == 0:
                    symbols = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"]
                logger.info(f"Utilisation de {len(symbols)} symboles d'actions")

        elif asset_type == AssetType.CRYPTO:
            symbols = DEFAULT_SYMBOLS["CRYPTO"]
        else:
            symbols = DEFAULT_SYMBOLS["STOCK"]
        
        # Normaliser les symboles crypto (enlever les / si présents)
        if asset_type == AssetType.CRYPTO:
            self.symbols = [symbol.replace('/', '') for symbol in symbols]
        else:
            self.symbols = symbols
        self.strategy_type = strategy_type
        self.asset_type = asset_type
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.api_level = api_level
        self.is_paper = is_paper
        
        # Paramètres supplémentaires pour les stratégies
        self.fast_ma_period = 5
        self.slow_ma_period = 15
        self.momentum_lookback = 10
        self.mean_reversion_zscore = 1.5
        self.market_check_interval = 1  # En secondes
        self.is_backtest = False
        
        # Chargement des variables d'environnement
        load_dotenv()
        
        # Déterminer le mode Alpaca (paper ou live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        
        # Configuration selon le mode
        if alpaca_mode == "live":
            self.api_key = os.getenv("ALPACA_LIVE_KEY")
            self.api_secret = os.getenv("ALPACA_LIVE_SECRET")
            self.base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
            self.is_paper = False
        else:  # mode paper par défaut
            self.api_key = os.getenv("ALPACA_PAPER_KEY")
            self.api_secret = os.getenv("ALPACA_PAPER_SECRET")
            self.base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
            self.is_paper = True
            
        # Pour les crypto, configurer l'asset class dans l'environnement
        if self.asset_type == AssetType.CRYPTO:
            logger.info("Configuration pour le trading de crypto")
            os.environ["ALPACA_ASSET_CLASS"] = "crypto"   # Initialiser l'API Alpaca
        # Initialisation de l'API REST Alpaca sans le paramètre data_url
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url
        )
        
        # Initialiser le stream (WebSockets)
        if self.asset_type == AssetType.CRYPTO:
            # Configuration spécifique pour les cryptomonnaies
            self.stream = Stream(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                data_stream_url="wss://stream.data.alpaca.markets/v1beta2/crypto",
                raw_data=True
            )
        else:
            # Configuration pour les actions
            self.stream = Stream(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                data_feed='iex' if self.api_level >= 2 else 'sip',  # Utiliser IEX pour niveau 2+
                raw_data=True
            )
        
        # Structures de données pour stocker l'historique
        self.price_data = {symbol: deque(maxlen=MAX_HISTORICAL_BARS) for symbol in self.symbols}
        self.order_book = {symbol: {'bids': {}, 'asks': {}} for symbol in self.symbols}
        self.last_tick = {symbol: None for symbol in self.symbols}
        self.last_trade_time = {symbol: datetime.now() - timedelta(minutes=10) for symbol in self.symbols}
        
        # Initialisation des API et des services
        # Ajout de l'API REST d'Alpaca plus bas
        self._initialize_services()
        
        # Structures de données pour le trading
        self.positions = {}  # Symbole -> informations sur la position
        self.pending_orders = set()  # Set des IDs d'ordres en attente
        
        # Initialiser la stratégie
        self.strategies = {}
        self.initialize_strategy()
    
    def _initialize_services(self):
        """Initialiser les services mercurio et API Alpaca avec détection automatique du niveau d'API"""
        try:
            # Initialiser les services marchande et API Alpaca avec détection automatique du niveau d'API
            if not self.api_key or not self.api_secret:
                # Charger les variables d'environnement
                load_dotenv()
                
                alpaca_level = int(os.getenv("ALPACA_SUBSCRIPTION_LEVEL", "1"))
                self.api_level = self.api_level or alpaca_level
                
                # Déterminer le mode (paper ou live)
                alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
                
                if alpaca_mode == "live":
                    self.api_key = os.getenv("ALPACA_LIVE_KEY")
                    self.api_secret = os.getenv("ALPACA_LIVE_SECRET")
                    self.base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
                else:  # mode paper par défaut
                    self.api_key = os.getenv("ALPACA_PAPER_KEY")
                    self.api_secret = os.getenv("ALPACA_PAPER_SECRET")
                    self.base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
                
                self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
                
                logger.info(f"Niveau d'API Alpaca configuré manuellement: {self.api_level}")
            else:
                # Détecter automatiquement le niveau d'abonnement Alpaca
                if self.api_level is None:
                    self.api_level = detect_alpaca_level(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        base_url=self.base_url,
                        data_url=self.data_url
                    )
                    logger.info(f"Niveau d'API Alpaca détecté: {self.api_level}")
            
            # Configuration directe de l'API Alpaca selon la version supportée
            try:
                # Première méthode: API sans data_url (compatible avec versions plus récentes)
                self.api = tradeapi.REST(
                    key_id=self.api_key,
                    secret_key=self.api_secret,
                    base_url=self.base_url
                )
                
                # Configuration manuelle de data_url si possible
                if hasattr(self.api, 'data_url'):
                    self.api.data_url = self.data_url
                    logger.info(f"Configuration manuelle de data_url: {self.data_url}")
            except Exception as e:
                # Fallback: Méthode plus ancienne qui peut supporter data_url directement
                logger.warning(f"Erreur lors de l'initialisation standard: {e}, tentative de méthode alternative")
                try:
                    self.api = tradeapi.REST(
                        key_id=self.api_key,
                        secret_key=self.api_secret,
                        base_url=self.base_url,
                        api_version='v2'
                    )
                    logger.info("API Alpaca initialisée avec succès (méthode alternative)")
                except Exception as e2:
                    logger.error(f"Erreur critique lors de l'initialisation de l'API Alpaca: {e2}")
                    raise
            
            # Définir les variables d'environnement pour les services Mercurio
            os.environ["ALPACA_KEY_ID"] = self.api_key
            os.environ["ALPACA_SECRET_KEY"] = self.api_secret
            os.environ["ALPACA_BASE_URL"] = self.base_url
            os.environ["ALPACA_DATA_URL"] = self.data_url
            os.environ["ALPACA_SUBSCRIPTION_LEVEL"] = str(self.api_level)
            os.environ["ALPACA_IS_PAPER"] = str(self.is_paper).lower()
            
            # Initialiser les services Mercurio
            # Vérifier si MarketDataService existe et est correctement importé
            if MarketDataService is not None:
                self.market_data_service = MarketDataService()
                logger.info("Service de données de marché initialisé avec succès")
            else:
                logger.warning("MarketDataService n'est pas disponible, utilisation de l'API Alpaca directement")
                self.market_data_service = None
            
            # Configuration spéciale pour le trading de crypto si nécessaire
            if self.asset_type == AssetType.CRYPTO:
                logger.info("Configuration du service de données pour les cryptomonnaies")
                
                if self.api_level >= 3:
                    logger.info("Niveau Premium (3+) détecté. Utilisation des fonctionnalités crypto avancées")
                    # Formats des symboles pour API crypto - standardiser en format avec slash
                    self.symbols = [self._ensure_symbol_format(s) for s in self.symbols]
                    logger.warning("Niveau Basic (1) détecté. Accès limité aux données crypto.")
                    # Ajuster les paramètres pour éviter les limites de taux
                    self.market_check_interval = max(self.market_check_interval, 5)  # Minimum 5 secondes
            
            # Créer le service de trading si disponible
            if TradingService is not None:
                self.trading_service = TradingService(is_paper=self.is_paper)
                logger.info(f"Service de trading initialisé en mode {'paper' if self.is_paper else 'live'}")
            else:
                self.trading_service = None
                logger.info("Service de trading non disponible")
            
            logger.info("Services Mercurio initialisés avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des services Mercurio: {e}")
            self.market_data_service = None
            self.trading_service = None
            return False
    
    def initialize_strategy(self) -> bool:
        """Initialiser la stratégie de trading"""
        try:
            if self.strategy_type == StrategyType.MOVING_AVERAGE:
                # Paramètres optimisés pour HFT
                from app.strategies.moving_average import MovingAverageStrategy
                self.strategies[self.strategy_type] = MovingAverageStrategy(
                    market_data_service=self.market_data_service,
                    trading_service=self.trading_service,
                    short_window=5,  # Très court pour HFT
                    long_window=15   # Court pour HFT
                )
            elif self.strategy_type == StrategyType.MOVING_AVERAGE_ML:
                if MovingAverageMLStrategy:
                    self.strategies[self.strategy_type] = MovingAverageMLStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service,
                        short_window_min=2,   # Ultra court pour HFT
                        short_window_max=10,
                        long_window_min=8, 
                        long_window_max=30,
                        optimize_interval=10   # Réoptimiser fréquemment
                    )
                else:
                    logger.error("MovingAverageMLStrategy non disponible")
                    return False
            elif self.strategy_type == StrategyType.MEAN_REVERSION:
                # La stratégie mean reversion sera implémentée directement dans ce script
                self.strategies[self.strategy_type] = self.mean_reversion_strategy
            elif self.strategy_type == StrategyType.MOMENTUM:
                # La stratégie momentum sera implémentée directement dans ce script
                self.strategies[self.strategy_type] = self.momentum_strategy
            elif self.strategy_type == StrategyType.ORDERBOOK_IMBALANCE:
                # La stratégie d'imbalance sera implémentée directement dans ce script
                self.strategies[self.strategy_type] = self.orderbook_imbalance_strategy
            
            logger.info(f"Stratégie {self.strategy_type.value} initialisée avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la stratégie {self.strategy_type.value}: {e}")
            return False
    
    async def start(self):
        """Démarrer le trader haute fréquence"""
        global running
        running = True
        
        # Gérer les signaux d'arrêt dans le thread principal
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialiser les structures nécessaires
        self.order_book = {symbol: {'bids': {}, 'asks': {}} for symbol in self.symbols}
        self.last_tick = {symbol: {'price': None, 'timestamp': None} for symbol in self.symbols}
        self.positions = {}
        self.price_data = {symbol: [] for symbol in self.symbols}
        self.active_orders = {}  # Pour stocker les ordres actifs
        self.pending_orders = set()  # Pour stocker les IDs d'ordres en attente
        self.last_trade_time = {symbol: datetime.now() - timedelta(hours=1) for symbol in self.symbols}
        
        # Vérifier le mode websocket
        self.use_websockets = hasattr(self, 'no_stream') and not self.no_stream
        
        # Rafraîchir les positions actuelles
        try:
            await self.refresh_positions()
        except Exception as e:
            logger.warning(f"Impossible de récupérer les positions initiales: {e}")
        
        logger.info(f"Démarrage du trader haute fréquence avec {len(self.symbols)} symboles")
        logger.info(f"Mode {'Paper' if self.is_paper else 'Live'}, Niveau API: {self.api_level}")
        
        # Précharger des données historiques
        try:
            await self.load_historical_data()
        except Exception as e:
            logger.warning(f"Problème lors du chargement des données historiques: {e}")
        
        # Choisir entre mode websocket et mode polling
        if self.use_websockets:
            try:
                # Initialiser le client WebSocket
                self.stream = tradeapi.Stream(key_id=self.api_keys['key'],
                                     secret_key=self.api_keys['secret'],
                                     base_url=self.api_url,
                                     data_feed='iex' if self.api_level >= 2 else 'sip')
                
                # Configurer les abonnements WebSocket
                for symbol in self.symbols:
                    if self.asset_type == AssetType.STOCK:
                        # Pour stocks - entourer chaque appel dans un try-except
                        try:
                            self.stream.subscribe_trades(self.handle_trade, symbol)
                        except Exception as e:
                            logger.warning(f"Erreur lors de l'abonnement aux trades pour {symbol}: {e}")
                        
                        try:
                            self.stream.subscribe_quotes(self.handle_quote, symbol)
                        except Exception as e:
                            logger.warning(f"Erreur lors de l'abonnement aux quotes pour {symbol}: {e}")
                            
                        try:
                            self.stream.subscribe_bars(self.handle_minute_bar, symbol)
                        except Exception as e:
                            logger.warning(f"Erreur lors de l'abonnement aux flux pour {symbol}: {e}")
                    else:
                        # Pour crypto - entourer chaque appel dans un try-except
                        try:
                            # Vérifier si les méthodes existent
                            if hasattr(self.stream, 'subscribe_crypto_trades'):
                                self.stream.subscribe_crypto_trades(self.handle_crypto_trade, symbol)
                        except Exception as e:
                            logger.warning(f"Erreur lors de l'abonnement aux trades crypto pour {symbol}: {e}")
                            
                        try:
                            if hasattr(self.stream, 'subscribe_crypto_quotes'):
                                self.stream.subscribe_crypto_quotes(self.handle_crypto_quote, symbol)
                        except Exception as e:
                            logger.warning(f"Erreur lors de l'abonnement aux quotes crypto pour {symbol}: {e}")
                            
                        try:
                            # Barres crypto si disponibles
                            if hasattr(self.stream, 'subscribe_crypto_bars'):
                                self.stream.subscribe_crypto_bars(self.handle_crypto_minute_bar, symbol)
                        except Exception as e:
                            logger.warning(f"Erreur lors de l'abonnement aux barres crypto pour {symbol}: {e}")
                
                # Enregistrer les mises à jour des ordres
                self.stream.subscribe_trade_updates(self.handle_trade_updates)
                
                # Démarrer la boucle principale d'analyse AVANT de démarrer le stream
                asyncio.create_task(self._run_main_loop())
                
                # Démarrer le stream dans un thread séparé
                self.stream_thread = threading.Thread(target=self._run_stream, daemon=True)
                self.stream_thread.start()
                logger.info("Stream WebSocket démarré")
                
                # Attendre que la boucle principale se termine
                while running:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Erreur lors du démarrage du stream WebSocket: {e}, passage en mode polling")
                self.use_websockets = False
                self._setup_polling_fallback()
        else:
            # Mode sans WebSocket, utilisation du polling
            logger.info("Mode sans WebSocket activé, utilisation du polling")
            self._setup_polling_fallback()
            
            # Lancer la boucle principale en mode asynchrone
            await self._run_main_loop()
    
    def _run_stream(self):
        """Exécute le stream WebSocket sans gestion de signal (pour être compatible avec les threads non-principaux)"""
        try:
            # Exécuter le stream (bloquant)
            self.stream.run()
        except Exception as e:
            # Gérer les erreurs de WebSocket
            if "HTTP 404" in str(e) or "WebSocket connection" in str(e):
                logger.error(f"Erreur WebSocket: {e}")
                if hasattr(self, 'use_websockets') and self.use_websockets:
                    self.use_websockets = False
                    # Passer en mode polling sans utiliser le signal
                    self._setup_polling_fallback()
            else:
                logger.error(f"Erreur dans le stream: {e}")
                logger.info("Activation du mode de secours par polling")
                self._setup_polling_fallback()

    
    async def _run_main_loop(self):
        """Boucle principale async pour le traitement des données"""
        # Compteur pour l'affichage périodique du solde
        balance_check_counter = 0
        
        # Boucle principale
        while running:
            try:
                # Vérifier les positions et gérer les stop loss / take profit
                await self.manage_positions()
                
                # Analyser les symboles pour de nouveaux signaux
                logger.debug(f"Début de l'analyse des symboles. Nombre de symboles: {len(self.symbols)}")
                await self.analyze_symbols()
                logger.debug("Fin de l'analyse des symboles.")
                
                # Afficher périodiquement le solde disponible (toutes les 10 itérations)
                balance_check_counter += 1
                if balance_check_counter >= 10:  # Environ toutes les 10 secondes avec l'intervalle par défaut
                    try:
                        account_info = self.api.get_account()
                        buying_power = float(account_info.buying_power)
                        cash = float(account_info.cash)
                        equity = float(account_info.equity)
                        
                        logger.info("===== INFORMATION DU COMPTE ALPACA =====")
                        logger.info(f"Solde disponible: ${buying_power:.2f}")
                        logger.info(f"Liquidités: ${cash:.2f}")
                        logger.info(f"Valeur totale: ${equity:.2f}")
                        logger.info("=======================================\n")
                    except Exception as e:
                        logger.error(f"Erreur lors de la récupération du solde: {e}")
                    
                    # Réinitialiser le compteur
                    balance_check_counter = 0
                
                # Courte pause pour éviter d'utiliser trop de CPU
                await asyncio.sleep(MARKET_DATA_INTERVAL)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle principale: {e}")
                await asyncio.sleep(1)  # Pause en cas d'erreur
        
    
    async def load_historical_data(self):
        """Précharger des données historiques pour initialiser les stratégies"""
        logger.info("Chargement des données historiques...")
        
        now = datetime.now()
        start = now - timedelta(minutes=MAX_HISTORICAL_BARS)  # Dernier jour
        
        # Déterminer le timeframe en fonction du niveau API
        timeframe = "1Min"  # Par défaut pour la haute fréquence
        
        # Pour les API premium (niveau 3), on peut utiliser une résolution plus fine
        if self.api_level >= 3 and self.asset_type == AssetType.CRYPTO:
            # Optimiser pour le trading haute fréquence
            timeframe = "1Min"  # Ou même secondes si disponible
            logger.info(f"Utilisation du timeframe {timeframe} pour le niveau API premium {self.api_level}")
        
        for symbol in self.symbols:
            try:
                # Utiliser le MarketDataService de Mercurio qui gère correctement les API Alpaca
                # Ce service dispose notamment d'une méthode spéciale v1beta3 pour les crypto
                if hasattr(self, 'market_data_service'):
                    try:
                        # Format de date attendu par get_historical_data
                        if self.asset_type == AssetType.STOCK:
                            # Pour les actions, utiliser le timeframe 1 minute
                            # Vérifier si get_historical_data est une coroutine ou une méthode normale
                            if asyncio.iscoroutinefunction(self.market_data_service.get_historical_data):
                                # Si c'est une coroutine, l'appeler directement avec await
                                df = await self.market_data_service.get_historical_data(symbol, start, now, "1Min")
                            else:
                                df = await asyncio.to_thread(self.market_data_service.get_historical_data, 
                                                          symbol, start, now, "1Min")
                        else:  # Pour les crypto
                            # Format correct du symbole pour l'API crypto
                            crypto_symbol = symbol
                            if "/" not in crypto_symbol and "-" in crypto_symbol:
                                # Convertir format avec tiret en format avec slash (ex: BTC-USD -> BTC/USD)
                                crypto_symbol = crypto_symbol.replace("-", "/")
                                logger.info(f"Conversion du format du symbole pour API crypto: {symbol} -> {crypto_symbol}")
                            
                            # Optimisation spécifique pour le niveau d'API détecté
                            logger.info(f"Utilisation de l'API crypto pour {crypto_symbol} (niveau {self.api_level})")
                            
                            if asyncio.iscoroutinefunction(self.market_data_service.get_historical_data):
                                df = await self.market_data_service.get_historical_data(crypto_symbol, start, now, timeframe)
                            else:
                                df = await asyncio.to_thread(self.market_data_service.get_historical_data, 
                                                          crypto_symbol, start, now, timeframe)
                        
                        # Vérifier si des données ont été retournées
                        if len(df) > 0:
                            # Convertir le dataframe pour le format attendu
                            # Assurez-vous que timestamp est présent et est l'index
                            if df.index.name == 'timestamp' or 'timestamp' in df.index.names:
                                # Reset l'index pour avoir timestamp comme colonne
                                df = df.reset_index()
                            
                            # Convertir en dict pour compatibilité avec notre structure
                            for idx, row in df.iterrows():
                                bar_dict = {
                                    'timestamp': pd.Timestamp(row['timestamp']).to_pydatetime() if 'timestamp' in row else now,
                                    'open': row['open'] if 'open' in row else row.get('o', 0),
                                    'high': row['high'] if 'high' in row else row.get('h', 0),
                                    'low': row['low'] if 'low' in row else row.get('l', 0),
                                    'close': row['close'] if 'close' in row else row.get('c', 0),
                                    'volume': row['volume'] if 'volume' in row else row.get('v', 0)
                                }
                                self.price_data[symbol].append(bar_dict)
                            
                            logger.info(f"Chargé {len(df)} barres historiques pour {symbol} via MarketDataService")
                        else:
                            # Si aucune donnée n'est disponible et qu'on est en mode premium,
                            # essayer une approche alternative pour les cryptos
                            if self.asset_type == AssetType.CRYPTO and self.api_level >= 3:
                                logger.warning(f"Données historiques non disponibles pour {symbol} malgré l'accès premium")
                                
                                # Pour les niveaux premium, essayer d'appeler directement l'API Alpaca crypto
                                try:
                                    logger.info(f"Tentative d'appel direct à l'API crypto pour {symbol}")
                                    
                                    # Formater le symbole correctement pour l'API crypto
                                    crypto_symbol = self._ensure_symbol_format(symbol)
                                    logger.info(f"Utilisation du symbole formaté: {crypto_symbol} pour l'appel direct API")
                                    
                                    # Utiliser notre méthode d'appel direct à l'API crypto
                                    crypto_data = self._direct_crypto_api_call(symbol, timeframe, limit=MAX_HISTORICAL_BARS)
                                    
                                    if crypto_data and 'bars' in crypto_data:
                                        bars_data = crypto_data['bars']
                                        # Vérifier si les données sont présentes sous le symbole formaté
                                        available_symbols = list(bars_data.keys())
                                        logger.info(f"Symboles disponibles dans la réponse: {available_symbols}")
                                        
                                        if crypto_symbol in bars_data and len(bars_data[crypto_symbol]) > 0:
                                            # Traiter les données récupérées directement
                                            for bar in bars_data[crypto_symbol]:
                                                bar_dict = {
                                                    'timestamp': pd.Timestamp(bar['t']).to_pydatetime(),
                                                    'open': float(bar['o']),
                                                    'high': float(bar['h']),
                                                    'low': float(bar['l']),
                                                    'close': float(bar['c']),
                                                    'volume': float(bar['v'])
                                                }
                                                self.price_data[symbol].append(bar_dict)
                                            
                                            logger.info(f"Récupéré {len(bars_data[crypto_symbol])} barres via API crypto directe pour {symbol}")
                                            continue  # Continuer à la prochaine itération de boucle
                                        else:
                                            # Essayer de trouver une correspondance dans les symboles disponibles
                                            found_match = False
                                            for avail_symbol in available_symbols:
                                                # Comparer sans slashes, tirets, etc.
                                                if (symbol.replace('/', '').replace('-', '') in avail_symbol.replace('/', '').replace('-', '') or
                                                    avail_symbol.replace('/', '').replace('-', '') in symbol.replace('/', '').replace('-', '')):
                                                    logger.info(f"Symbole correspondant trouvé: {avail_symbol}")
                                                    
                                                    # Traiter les données pour ce symbole
                                                    for bar in bars_data[avail_symbol]:
                                                        bar_dict = {
                                                            'timestamp': pd.Timestamp(bar['t']).to_pydatetime(),
                                                            'open': float(bar['o']),
                                                            'high': float(bar['h']),
                                                            'low': float(bar['l']),
                                                            'close': float(bar['c']),
                                                            'volume': float(bar['v'])
                                                        }
                                                        self.price_data[symbol].append(bar_dict)
                                                    
                                                    logger.info(f"Récupéré {len(bars_data[avail_symbol])} barres via symbole correspondant pour {symbol}")
                                                    found_match = True
                                                    break
                                            
                                            if found_match:
                                                continue  # Continuer à la prochaine itération de boucle
                                except Exception as e:
                                    logger.error(f"Échec de l'appel direct à l'API crypto: {str(e)[:100]}")
                            
                            # Si tout échoue, essayer de récupérer au moins le dernier prix
                            logger.warning(f"Données historiques non disponibles pour {symbol}, tentative de récupération du dernier prix")
                            try:
                                # Vérifier si la méthode est une coroutine
                                if asyncio.iscoroutinefunction(self.market_data_service.get_latest_price):
                                    price = await self.market_data_service.get_latest_price(symbol)
                                else:
                                    price = await asyncio.to_thread(self.market_data_service.get_latest_price, symbol)
                                    
                                if price:
                                    # Créer une entrée avec le dernier prix connu
                                    bar_dict = {
                                        'timestamp': now,
                                        'open': price,
                                        'high': price,
                                        'low': price,
                                        'close': price,
                                        'volume': 0
                                    }
                                    self.price_data[symbol].append(bar_dict)
                                    logger.info(f"Utilisé le dernier prix disponible pour {symbol}: {price}")
                                else:
                                    logger.warning(f"Aucune donnée historique disponible pour {symbol}")
                            except Exception as e:
                                logger.warning(f"Aucune donnée historique disponible pour {symbol}: {e}")
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'utilisation du MarketDataService pour {symbol}: {e}")
                else:
                    # Fallback à l'ancien code si market_data_service n'est pas disponible
                    logger.warning(f"MarketDataService non disponible, utilisation de l'API directe pour {symbol}")
                    try:
                        if self.asset_type == AssetType.STOCK:
                            # Récupérer les données historiques pour les actions
                            bars = self.api.get_bars(
                                symbol, 
                                tradeapi.TimeFrame.Minute, 
                                start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                limit=MAX_HISTORICAL_BARS
                            ).df
                            
                            if len(bars) > 0:
                                # Convertir en dict pour compatibilité avec notre structure
                                bars = bars.reset_index()
                                for _, bar in bars.iterrows():
                                    bar_dict = {
                                        'timestamp': pd.Timestamp(bar['timestamp']).to_pydatetime(),
                                        'open': bar['open'],
                                        'high': bar['high'],
                                        'low': bar['low'],
                                        'close': bar['close'],
                                        'volume': bar['volume']
                                    }
                                    self.price_data[symbol].append(bar_dict)
                                logger.info(f"Chargé {len(bars)} barres historiques pour {symbol}")
                            else:
                                logger.warning(f"Aucune donnée historique disponible pour {symbol}")
                        else:  # Récupérer les données historiques pour les cryptos
                            # Essayer d'obtenir au moins le dernier prix avec les méthodes disponibles
                            try:
                                # Pour les cryptos, essayer différentes méthodes API
                                if self.asset_type == AssetType.CRYPTO:
                                    # Essayer avec l'API crypto spécifique si disponible
                                    if hasattr(self.api, 'get_crypto_bars'):
                                        bars = self.api.get_crypto_bars(
                                            symbol, 
                                            tradeapi.TimeFrame.Minute, 
                                            start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                            now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                            limit=MAX_HISTORICAL_BARS
                                        ).df
                                        
                                        if len(bars) > 0:
                                            # Process the bars
                                            bars = bars.reset_index()
                                            for _, bar in bars.iterrows():
                                                bar_dict = {
                                                    'timestamp': pd.Timestamp(bar['timestamp']).to_pydatetime(),
                                                    'open': bar['open'],
                                                    'high': bar['high'],
                                                    'low': bar['low'],
                                                    'close': bar['close'],
                                                    'volume': bar['volume']
                                                }
                                                self.price_data[symbol].append(bar_dict)
                                            logger.info(f"Chargé {len(bars)} barres historiques crypto pour {symbol}")
                                            continue
                                        
                                # Essayer d'obtenir la dernière barre (fallback)
                                latest_bar = self.api.get_latest_bar(symbol)
                                bar_dict = {
                                    'timestamp': latest_bar.t,
                                    'open': latest_bar.o,
                                    'high': latest_bar.h,
                                    'low': latest_bar.l,
                                    'close': latest_bar.c,
                                    'volume': latest_bar.v
                                }
                                self.price_data[symbol].append(bar_dict)
                                logger.info(f"Utilisé la dernière barre disponible pour {symbol}")
                            except Exception as bar_error:
                                logger.debug(f"Échec récupération barre pour {symbol}: {bar_error}")
                                try:
                                    # Essayer d'obtenir la dernière transaction
                                    if self.asset_type == AssetType.CRYPTO and hasattr(self.api, 'get_latest_crypto_trade'):
                                        latest_trade = self.api.get_latest_crypto_trade(symbol) 
                                    else:
                                        latest_trade = self.api.get_latest_trade(symbol)
                                        
                                    bar_dict = {
                                        'timestamp': latest_trade.t,
                                        'open': latest_trade.p,
                                        'high': latest_trade.p,
                                        'low': latest_trade.p,
                                        'close': latest_trade.p,
                                        'volume': latest_trade.s
                                    }
                                    self.price_data[symbol].append(bar_dict)
                                    logger.info(f"Utilisé la dernière transaction disponible pour {symbol}")
                                except Exception:
                                    # En dernier recours, utiliser des prix par défaut
                                    default_prices = {
                                        "BTCUSD": 55000.0, "ETHUSD": 2500.0, "DOGEUSD": 0.15,
                                        "SOLUSD": 120.0, "AVAXUSD": 30.0, "LINKUSD": 15.0,
                                        "LTCUSD": 80.0, "XRPUSD": 0.5, "BATUSD": 0.2,
                                        "PEPEUSD": 0.000005, "SHIBUSD": 0.000009, "TRUMPUSD": 11.0,
                                        "AAVEUSD": 80.0, "XTZUSD": 0.8, "CRVUSD": 0.6,
                                        "UNIUSD": 7.0, "MKRUSD": 1200.0, "YFIUSD": 7500.0,
                                        "BCHUSD": 250.0, "SUSHIUSD": 0.7, "USDCUSD": 1.0, "USDTUSD": 1.0
                                    }
                                    price = default_prices.get(symbol, 10.0)  # 10.0 comme prix par défaut générique
                                    bar_dict = {
                                        'timestamp': now,
                                        'open': price,
                                        'high': price,
                                        'low': price,
                                        'close': price,
                                        'volume': 0
                                    }
                                    self.price_data[symbol].append(bar_dict)
                                    logger.warning(f"Utilisation d'un prix par défaut pour {symbol}: {price}")
                    except Exception as e:
                        logger.warning(f"Erreur lors du chargement des données directes pour {symbol}: {e}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données historiques pour {symbol}: {e}")

    
    # ----- Handlers WebSocket -----
    
    async def handle_trade(self, trade):
        """Gestionnaire pour les trades (actions)"""
        symbol = trade.symbol
        self.last_tick[symbol] = {
            'timestamp': trade.timestamp,
            'price': trade.price,
            'size': trade.size
        }
    
    async def handle_quote(self, quote):
        """Gestionnaire pour les quotes (actions)"""
        symbol = quote.symbol
        # Mettre à jour l'order book simplifié
        self.order_book[symbol]['bids'] = {quote.bid_price: quote.bid_size}
        self.order_book[symbol]['asks'] = {quote.ask_price: quote.ask_size}
    
    async def handle_minute_bar(self, bar):
        """Gestionnaire pour les barres minute (actions)"""
        symbol = bar.symbol
        bar_dict = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        
        self.price_data[symbol].append(bar_dict)
        
        # Déclencher l'analyse uniquement sur la réception d'une nouvelle barre
        await self.analyze_symbol(symbol)
    
    async def handle_crypto_trade(self, trade):
        """Gestionnaire pour les trades crypto"""
        symbol = trade.symbol
        self.last_tick[symbol] = {
            'timestamp': trade.timestamp,
            'price': trade.price,
            'size': trade.size
        }
    
    async def handle_crypto_quote(self, quote):
        """Gestionnaire pour les quotes crypto"""
        symbol = quote.symbol
        # Mettre à jour l'order book simplifié
        self.order_book[symbol]['bids'] = {quote.bid_price: quote.bid_size}
        self.order_book[symbol]['asks'] = {quote.ask_price: quote.ask_size}
    
    async def handle_crypto_minute_bar(self, bar):
        """Gestionnaire pour les barres minute crypto"""
        symbol = bar.symbol
        bar_dict = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        
        self.price_data[symbol].append(bar_dict)
        
        # Déclencher l'analyse uniquement sur la réception d'une nouvelle barre
        await self.analyze_symbol(symbol)
    
    async def handle_trade_updates(self, update):
        """Gérer les mises à jour des ordres"""
        try:
            # Déterminer le format de l'update (objet ou dictionnaire)
            if isinstance(update, dict):
                # Format dict
                logger.debug(f"Update d'ordre en format dict: {update}")
                order_id = update.get('id', 'inconnu')
                order_status = update.get('status', 'inconnu')
                symbol = update.get('symbol', 'inconnu')
                event_type = update.get('event', order_status)
                
                # Récupérer plus d'infos si disponibles
                qty = update.get('qty', 'N/A')
                filled_qty = update.get('filled_qty', 'N/A')
                filled_avg_price = update.get('filled_avg_price', 'N/A')
                limit_price = update.get('limit_price', None)
                reject_reason = update.get('reject_reason', 'Raison inconnue')
                
            else:
                # Format objet (avec attributs)
                logger.debug(f"Update d'ordre en format objet: {update}")
                
                # Vérifier si les attributs de base existent
                if not hasattr(update, 'event'):
                    logger.warning(f"Format d'update d'ordre non reconnu: {update}")
                    return
                
                event_type = update.event
                
                # Vérifier si nous avons un attribut 'order' ou si l'objet lui-même est l'ordre
                if hasattr(update, 'order'):
                    order = update.order
                else:
                    order = update
                
                # Extraire les infos avec securité
                order_id = getattr(order, 'id', 'inconnu')
                symbol = getattr(order, 'symbol', 'inconnu')
                order_status = getattr(order, 'status', event_type)
                
                # Récupérer plus d'infos si disponibles
                qty = getattr(order, 'qty', 'N/A')
                filled_qty = getattr(order, 'filled_qty', 'N/A')
                filled_avg_price = getattr(order, 'filled_avg_price', 'N/A')
                limit_price = getattr(order, 'limit_price', None)
                reject_reason = getattr(order, 'reject_reason', 'Raison inconnue')
            
            # Enregistrer les informations pertinentes selon l'événement
            if event_type == 'fill' or order_status == 'filled':
                logger.info(f"Ordre rempli: {order_id} pour {symbol}, {qty} @ {filled_avg_price}")
            elif event_type == 'partial_fill':
                logger.info(f"Ordre partiellement rempli: {order_id} pour {symbol}, {filled_qty}/{qty} @ {filled_avg_price}")
            elif event_type == 'rejected' or order_status == 'rejected':
                logger.warning(f"Ordre rejeté: {order_id} pour {symbol}, raison: {reject_reason}")
            elif event_type == 'canceled' or order_status == 'canceled':
                logger.info(f"Ordre annulé: {order_id} pour {symbol}")
            elif event_type == 'new' or order_status == 'new':
                price_info = f"@ {limit_price}" if limit_price else "au marché"
                logger.info(f"Nouvel ordre: {order_id} pour {symbol}, {qty} {price_info}")
            else:
                logger.info(f"Mise à jour d'ordre: {order_id} pour {symbol}, événement: {event_type}, statut: {order_status}")
            
            # Mettre à jour notre suivi des ordres si nécessaire
            if order_status in ['filled', 'canceled', 'rejected', 'expired']:
                # Gérer les ordres actifs
                if hasattr(self, 'active_orders') and order_id in self.active_orders:
                    del self.active_orders[order_id]
                    logger.debug(f"Ordre {order_id} retiré des ordres actifs")
                
                # Gérer les ordres en attente
                if hasattr(self, 'pending_orders') and order_id in self.pending_orders:
                    self.pending_orders.remove(order_id)
                    logger.debug(f"Ordre {order_id} retiré des ordres en attente")
                
                # Mettre à jour les positions si rempli
                if order_status == 'filled':
                    await self.refresh_positions()
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la mise à jour d'ordre: {e}")
            logger.debug(f"Données de l'ordre: {update}")
            # Ne pas laisser une exception arrêter la boucle principale
    
    # ----- Méthodes d'analyse et de signal -----
    
    async def analyze_symbols(self):
        """Analyser tous les symboles pour des opportunités de trading"""
        for symbol in self.symbols:
            logger.debug(f"Analyse du symbole: {symbol}")
            await self.analyze_symbol(symbol)
    
    async def analyze_symbol(self, symbol):
        """Analyser un symbole spécifique pour signal de trading"""
        # Vérifier si on a suffisamment de données
        if len(self.price_data[symbol]) < 20:  # Minimum requis pour la plupart des stratégies
            logger.debug(f"Pas assez de données pour {symbol} (n={len(self.price_data[symbol])})")
            return
        
        # Vérifier si on peut trader ce symbole (fréquence de trade limitée)
        since_last_trade = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
        if since_last_trade < MAX_TRADE_FREQUENCY:
            logger.debug(f"Fréquence de trade limitée pour {symbol}: {since_last_trade:.2f}s < {MAX_TRADE_FREQUENCY}s")
            return
        
        # Convertir les données pour l'analyse
        df = self._prepare_dataframe(symbol)
        
        # Générer un signal de trading selon la stratégie sélectionnée
        signal = None
        
        if self.strategy_type == StrategyType.MOVING_AVERAGE:
            if self.strategies[self.strategy_type]:
                strategy_result = self.strategies[self.strategy_type].get_signal(symbol, df)
                logger.debug(f"Signal MOVING_AVERAGE pour {symbol}: {strategy_result}")
                signal = {
                    "action": strategy_result["action"],
                    "confidence": strategy_result["confidence"],
                    "reason": f"MovingAverage: {strategy_result.get('params', {})}"
                }
        
        elif self.strategy_type == StrategyType.MOVING_AVERAGE_ML:
            if self.strategies[self.strategy_type]:
                strategy_result = self.strategies[self.strategy_type].get_signal(symbol, df)
                logger.debug(f"Signal MOVING_AVERAGE_ML pour {symbol}: {strategy_result}")
                signal = {
                    "action": strategy_result["action"],
                    "confidence": strategy_result["confidence"],
                    "reason": f"MovingAverageML: {strategy_result.get('params', {})}"
                }
        
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            signal = await self.mean_reversion_strategy(symbol, df)
            logger.debug(f"Signal MEAN_REVERSION pour {symbol}: {signal}")
        
        elif self.strategy_type == StrategyType.MOMENTUM:
            signal = await self.momentum_strategy(symbol, df)
            logger.debug(f"Signal MOMENTUM pour {symbol}: {signal}")
        
        elif self.strategy_type == StrategyType.ORDERBOOK_IMBALANCE:
            signal = await self.orderbook_imbalance_strategy(symbol)
            logger.debug(f"Signal ORDERBOOK_IMBALANCE pour {symbol}: {signal}")
        
        # Exécuter le signal si valide
        if signal and signal.get("action") is not None:
            logger.info(f"Signal valide pour {symbol}: action={signal.get('action')}, confiance={signal.get('confidence')}, raison={signal.get('reason')}")
            await self.execute_signal(symbol, signal)
        else:
            logger.debug(f"Aucun signal valide pour {symbol} ou action=None. Signal: {signal}")
    
    def _prepare_dataframe(self, symbol) -> pd.DataFrame:
        """Préparer un DataFrame pandas à partir des données historiques"""
        # Convertir nos données stockées en DataFrame
        data = list(self.price_data[symbol])
        
        if not data:
            return pd.DataFrame()
        
        # Créer un DataFrame avec les données OHLCV
        df = pd.DataFrame(data)
        
        # S'assurer que les colonnes existent
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                return pd.DataFrame()
        
        # Trier par timestamp
        df = df.sort_values('timestamp')
        
        # Ajouter le dernier tick si disponible
        if self.last_tick[symbol] and 'price' in self.last_tick[symbol]:
            # Mettre à jour le dernier prix de clôture avec le dernier tick
            if len(df) > 0:
                df.iloc[-1, df.columns.get_loc('close')] = self.last_tick[symbol]['price']
        
        return df
    
    # ----- Stratégies de trading intégrées -----
    
    async def mean_reversion_strategy(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Stratégie de Mean Reversion optimisée pour HFT"""
        if len(data) < 30:
            return {"action": None, "confidence": 0}
            
        # Calculer les bandes de Bollinger avec fenêtres courtes pour HFT
        window = 15  # Fenêtre courte pour réactivité
        data['sma'] = data['close'].rolling(window=window).mean()
        data['std'] = data['close'].rolling(window=window).std()
        data['upper_band'] = data['sma'] + (data['std'] * 2)
        data['lower_band'] = data['sma'] - (data['std'] * 2)
        data['z_score'] = (data['close'] - data['sma']) / data['std']
        
        # Obtenir les derniers indicateurs
        last_close = data['close'].iloc[-1]
        last_upper = data['upper_band'].iloc[-1]
        last_lower = data['lower_band'].iloc[-1]
        last_z_score = data['z_score'].iloc[-1]
        
        # Générer un signal basé sur la position par rapport aux bandes
        from app.db.models import TradeAction
        
        action = TradeAction.HOLD
        confidence = 0.5
        
        # Signal de sur-achat (vente)
        if last_z_score > 2.0:
            action = TradeAction.SELL
            confidence = min(0.5 + abs(last_z_score - 2.0) / 2.0, 0.95)
            
        # Signal de sur-vente (achat)
        elif last_z_score < -2.0:
            action = TradeAction.BUY
            confidence = min(0.5 + abs(last_z_score + 2.0) / 2.0, 0.95)
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": f"Mean Reversion: z-score={last_z_score:.2f}, upp={last_upper:.2f}, low={last_lower:.2f}"
        }
    
    async def momentum_strategy(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Stratégie de Momentum optimisée pour HFT"""
        if len(data) < 30:
            return {"action": None, "confidence": 0}
            
        # Calculer les indicateurs de momentum
        data['rsi'] = self._calculate_rsi(data['close'], 8)  # RSI rapide
        data['price_change'] = data['close'].pct_change(3)  # Changement sur 3 périodes
        data['volume_change'] = data['volume'].pct_change(3)  # Changement de volume
        
        # Moyennes mobiles courtes
        data['ema5'] = data['close'].ewm(span=5).mean()  # EMA très rapide
        data['ema10'] = data['close'].ewm(span=10).mean()  # EMA rapide
        
        # Obtenir les derniers indicateurs
        last_rsi = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50
        last_price_change = data['price_change'].iloc[-1] if not pd.isna(data['price_change'].iloc[-1]) else 0
        last_vol_change = data['volume_change'].iloc[-1] if not pd.isna(data['volume_change'].iloc[-1]) else 0
        last_ema_diff = (data['ema5'].iloc[-1] / data['ema10'].iloc[-1] - 1) * 100 if not pd.isna(data['ema5'].iloc[-1]) else 0
        
        # Générer un signal basé sur la combinaison d'indicateurs
        from app.db.models import TradeAction
        
        action = TradeAction.HOLD
        confidence = 0.5
        reason = ""
        
        # Signal d'achat fort
        if (last_rsi > 50 and last_rsi < 70 and  # RSI en tendance haussière mais pas sur-acheté
                last_price_change > 0.001 and    # Prix en hausse
                last_vol_change > 0.2 and        # Volume en hausse
                last_ema_diff > 0.1):           # EMA5 au-dessus de EMA10
                
            action = TradeAction.BUY
            confidence = 0.7 + (last_price_change * 100)  # Confiance proportionnelle au momentum
            confidence = min(confidence, 0.95)  # Plafonner la confiance à 95%
            reason = f"Momentum: RSI={last_rsi:.2f}, PriceChange={last_price_change:.4f}, EMA_Diff={last_ema_diff:.2f}%"
            
        # Signal de vente fort
        elif (last_rsi < 50 and last_rsi > 30 and  # RSI en tendance baissière mais pas sur-vendu
                last_price_change < -0.001 and     # Prix en baisse
                last_ema_diff < -0.1):            # EMA5 sous EMA10
                
            action = TradeAction.SELL
            confidence = 0.7 + abs(last_price_change * 100)  # Confiance proportionnelle au momentum
            confidence = min(confidence, 0.95)  # Plafonner la confiance à 95%
            reason = f"Momentum: RSI={last_rsi:.2f}, PriceChange={last_price_change:.4f}, EMA_Diff={last_ema_diff:.2f}%"
            
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason or f"Momentum: No signal, RSI={last_rsi:.2f}"
        }
    
    async def orderbook_imbalance_strategy(self, symbol: str) -> Dict[str, Any]:
        """Stratégie basée sur le déséquilibre du carnet d'ordres (nécessite API niveau 3)"""
        # Vérifier si on a accès aux données du carnet d'ordres
        if self.api_level < 3 or not self.order_book[symbol]:
            return {"action": None, "confidence": 0}
            
        bids = self.order_book[symbol]['bids']
        asks = self.order_book[symbol]['asks']
        
        if not bids or not asks:
            return {"action": None, "confidence": 0}
            
        # Calculer le volume total d'achat et de vente
        bid_volume = sum(bids.values())
        ask_volume = sum(asks.values())
        
        # Calculer l'imbalance ratio
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return {"action": None, "confidence": 0}
            
        imbalance_ratio = (bid_volume - ask_volume) / total_volume  # Entre -1 et 1
        
        # Générer un signal basé sur l'imbalance
        from app.db.models import TradeAction
        
        action = TradeAction.HOLD
        confidence = 0.5
        
        # Fort déséquilibre côté acheteurs (signal d'achat)
        if imbalance_ratio > 0.2:
            action = TradeAction.BUY
            confidence = 0.5 + abs(imbalance_ratio) / 2  # Confiance proportionnelle à l'imbalance
            
        # Fort déséquilibre côté vendeurs (signal de vente)
        elif imbalance_ratio < -0.2:
            action = TradeAction.SELL
            confidence = 0.5 + abs(imbalance_ratio) / 2  # Confiance proportionnelle à l'imbalance
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": f"Order Book Imbalance: {imbalance_ratio:.4f} (bid_vol={bid_volume}, ask_vol={ask_volume})"
        }
    
    # ----- Méthodes utilitaires -----
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculer le RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ----- Gestion des positions et exécution des ordres -----
    
    async def refresh_positions(self):
        """Mettre à jour la liste des positions actuelles"""
        try:
            positions = self.api.list_positions()
            self.positions = {p.symbol: {
                'qty': float(p.qty),
                'entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)  # P&L en pourcentage
            } for p in positions}
            
            logger.info(f"Positions mises à jour: {len(self.positions)} positions actives")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des positions: {e}")
    
    async def manage_positions(self):
        """Gérer les positions existantes (stop loss, take profit)"""
        if not self.positions:
            await self.refresh_positions()
            
        for symbol, position in list(self.positions.items()):
            try:
                # Vérifier si le symbole est encore dans notre liste de symboles à trader
                if symbol not in self.symbols:
                    continue
                    
                # Récupérer les données de position
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', 0)
                qty = position.get('qty', 0)
                
                # Vérifier si les données sont valides pour éviter division par zéro
                if not entry_price or not current_price or abs(qty) < 1e-8:
                    logger.debug(f"Position ignorée pour {symbol}: prix ou quantité trop faible/nulle (entry: {entry_price}, current: {current_price}, qty: {qty})")
                    continue
                
                # Calculer le pourcentage de profit/perte
                if qty > 0:  # Position longue
                    # Protection contre division par zéro
                    if entry_price <= 0:
                        logger.warning(f"Prix d'entrée invalide pour {symbol}: {entry_price}")
                        continue
                        
                    pct_change = (current_price / entry_price) - 1
                    
                    # Take profit
                    if pct_change >= self.take_profit_pct:
                        logger.info(f"Take Profit déclenché pour {symbol}: +{pct_change*100:.2f}%")
                        await self.execute_order(symbol, "sell", qty, "Take Profit")
                        
                    # Stop loss
                    elif pct_change <= -self.stop_loss_pct:
                        logger.info(f"Stop Loss déclenché pour {symbol}: {pct_change*100:.2f}%")
                        await self.execute_order(symbol, "sell", qty, "Stop Loss")
                        
                elif qty < 0:  # Position courte (short)
                    # Protection contre division par zéro
                    if entry_price <= 0:
                        logger.warning(f"Prix d'entrée invalide pour {symbol}: {entry_price}")
                        continue
                        
                    pct_change = 1 - (current_price / entry_price)
                    
                    # Take profit
                    if pct_change >= self.take_profit_pct:
                        logger.info(f"Take Profit déclenché pour {symbol} (short): +{pct_change*100:.2f}%")
                        await self.execute_order(symbol, "buy", abs(qty), "Take Profit (short)")
                        
                    # Stop loss
                    elif pct_change <= -self.stop_loss_pct:
                        logger.info(f"Stop Loss déclenché pour {symbol} (short): {pct_change*100:.2f}%")
                        await self.execute_order(symbol, "buy", abs(qty), "Stop Loss (short)")
                        
            except Exception as e:
                logger.error(f"Erreur lors de la gestion de la position {symbol}: {e}")
                logger.debug(f"Détails de la position qui a causé l'erreur: {position}")
                # Continue avec les autres positions même si une erreur se produit
    
    async def execute_signal(self, symbol: str, signal: Dict[str, Any]):
        """Exécuter un signal de trading"""
        # Utiliser notre propre énumération TradeAction
        
        # Vérifier la confiance minimale requise
        if signal.get("confidence", 0) < 0.6:
            logger.info(f"Confiance insuffisante pour {symbol}: {signal.get('confidence', 0):.2f} < 0.6")
            return
            
        # Déterminer le type d'action
        action = signal.get("action")
        logger.debug(f"Type d'action pour {symbol}: {action}")
        
        if action == TradeAction.BUY:
            # Vérifier si on a déjà une position sur ce symbole
            if symbol in self.positions and self.positions[symbol]['qty'] > 0:
                logger.info(f"Position déjà ouverte sur {symbol}, pas d'achat supplémentaire")
                return
                
            # Vérifier si on n'a pas atteint le nombre max de positions
            if len(self.positions) >= self.max_positions:
                logger.info(f"Nombre maximum de positions atteint: {len(self.positions)}/{self.max_positions}")
                return
                
            # Calculer la quantité à acheter
            quantity = await self._calculate_position_size(symbol)
            logger.debug(f"Quantité calculée à acheter pour {symbol}: {quantity}")
            if quantity <= 0:
                logger.info(f"Quantité calculée nulle ou négative pour {symbol}: {quantity}")
                return
                
            # Exécuter l'achat
            logger.info(f"Envoi d'un ordre d'achat pour {symbol}, quantité={quantity}")
            await self.execute_order(symbol, "buy", quantity, signal.get("reason", "Signal d'achat"))
            
        elif action == TradeAction.SELL:
            # Vérifier si on a une position longue sur ce symbole
            if symbol in self.positions and self.positions[symbol]['qty'] > 0:
                # Vendre toute la position
                quantity = self.positions[symbol]['qty']
                logger.info(f"Envoi d'un ordre de vente pour {symbol}, quantité={quantity}")
                await self.execute_order(symbol, "sell", quantity, signal.get("reason", "Signal de vente"))
                
            # Ou vérifier si on veut ouvrir une position short (si permis par le compte)
            elif self._is_shorting_enabled() and signal.get("confidence", 0) > 0.8:
                quantity = await self._calculate_position_size(symbol)
                logger.debug(f"Quantité calculée pour short sur {symbol}: {quantity}")
                if quantity <= 0:
                    logger.info(f"Quantité calculée nulle ou négative pour short sur {symbol}: {quantity}")
                    return
                    
                logger.info(f"Envoi d'un ordre short pour {symbol}, quantité={quantity}")
                await self.execute_order(symbol, "sell", quantity, signal.get("reason", "Signal de vente à découvert"))
        
        # Mettre à jour le timestamp du dernier trade pour limiter la fréquence
        self.last_trade_time[symbol] = datetime.now()
        logger.debug(f"Timestamp du dernier trade mis à jour pour {symbol}")
    
    async def _check_balance(self, symbol: str, side: str, quantity: float) -> bool:
        """Vérifier si le solde est suffisant pour l'ordre"""
        try:
            if side.lower() == "buy":
                # Vérifier le cash disponible pour un achat
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                
                # Estimer le coût de l'ordre
                price = None
                try:
                    if self.asset_type == AssetType.STOCK:
                        # Pour les actions, utiliser get_latest_quote
                        latest_quote = self.api.get_latest_quote(symbol)
                        price = float(latest_quote.ap)  # ask price
                    else:
                        # Pour les cryptos, utiliser plusieurs méthodes alternatives
                        try:
                            # Méthode 1: Essayer d'obtenir la dernière transaction
                            trade = self.api.get_latest_trade(symbol)
                            price = float(trade.p)
                        except Exception:
                            try:
                                # Méthode 2: Utiliser la dernière barre de prix
                                bar = self.api.get_latest_bar(symbol)
                                price = float(bar.c)
                            except Exception:
                                # Méthode 3: Utiliser le service de données de marché Mercurio
                                try:
                                    if hasattr(self, 'market_data_service'):
                                        price = float(self.market_data_service.get_latest_price(symbol))
                                    else:
                                        # Utiliser le prix stocké en cache si disponible
                                        if symbol in self.last_tick and self.last_tick[symbol]['initialized']:
                                            price = self.last_tick[symbol]['price']
                                        else:
                                            # Utiliser un prix par défaut en dernier recours
                                            default_prices = {
                                                "BTCUSD": 55000.0, "ETHUSD": 2500.0, "DOGEUSD": 0.15,
                                                "SOLUSD": 120.0, "AVAXUSD": 30.0, "LINKUSD": 15.0,
                                                "LTCUSD": 80.0, "XRPUSD": 0.5, "BATUSD": 0.2
                                            }
                                            price = default_prices.get(symbol, 10.0)  # 10.0 comme prix par défaut générique
                                            logger.warning(f"Utilisation d'un prix par défaut pour {symbol}: {price}")
                                except Exception as e:
                                    logger.error(f"Toutes les méthodes de récupération de prix ont échoué pour {symbol}: {e}")
                                    raise
                    
                    # Si un prix a été trouvé, calculer le coût estimé
                    if price:
                        estimated_cost = price * quantity
                            
                        if estimated_cost > buying_power:
                            logger.warning(f"Solde insuffisant pour acheter {quantity} {symbol}: ${estimated_cost:.2f} requis, ${buying_power:.2f} disponible")
                            return False
                except Exception as e:
                    logger.warning(f"Erreur lors de la récupération du prix pour {symbol}: {e}")
                    # Si on ne peut pas estimer le coût, on suppose que c'est OK
                    return True
            else:  # sell
                # Vérifier les positions détenues pour une vente
                try:
                    position = self.api.get_position(symbol)
                    available_qty = float(position.qty)
                    
                    # Vérifier si la position est une "dust position" (trop petite pour être traitée)
                    current_price = float(position.current_price)
                    position_value = abs(available_qty * current_price)
                    
                    if self.asset_type == AssetType.CRYPTO:
                        # Seuil minimal spécifique au symbole ou valeur par défaut
                        min_size = MIN_POSITION_SIZE.get(symbol, MIN_POSITION_SIZE["DEFAULT"])
                        
                        if abs(available_qty) < min_size:
                            logger.warning(f"Position trop petite pour {symbol}: {available_qty} < {min_size} (minimum requis)")
                            return False
                            
                        if position_value < MIN_POSITION_VALUE_USD:
                            logger.warning(f"Valeur de position trop faible pour {symbol}: ${position_value:.2f} < ${MIN_POSITION_VALUE_USD} (minimum requis)")
                            return False
                    else:
                        # Pour les actions, on garde le seuil minimal très bas
                        if available_qty <= 0.000001:
                            logger.warning(f"Position trop petite ou nulle pour {symbol}: {available_qty}, impossible de vendre")
                            return False
                    
                    if available_qty < quantity:
                        # Différence trop faible, arrondir
                        if abs(available_qty - quantity) < 0.000001:
                            logger.info(f"Ajustement automatique de la quantité pour {symbol}: {quantity} -> {available_qty}")
                            return True  # Nous utiliserons la quantité disponible dans execute_order
                        else:
                            logger.warning(f"Solde insuffisant pour {symbol}: {quantity} demandé, {available_qty} disponible")
                            return False
                except Exception as e:
                    # Si l'exception est due à l'absence de position, on ne peut pas vendre
                    if "position does not exist" in str(e).lower():
                        logger.warning(f"Aucune position pour {symbol}, impossible de vendre")
                        return False
                    logger.warning(f"Erreur lors de la vérification de la position pour {symbol}: {e}")
                    # Par prudence, on suppose que c'est NON
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du solde pour {symbol}: {e}")
            return False
    
    async def execute_order(self, symbol, side, quantity, reason=""):
        """Exécuter un ordre d'achat ou de vente"""
        try:
            # Vérifier si les ordres sont désactivés (mode backtest)
            if hasattr(self, 'no_orders') and self.no_orders:
                logger.info(f"[BACKTEST MODE] Ordre simulé: {side} {quantity} {symbol} ({reason})")
                return None
                
            # Pour les crypto, vérifier et ajuster la quantité pour éviter les dust positions
            if self.asset_type == AssetType.CRYPTO:
                # S'assurer que la quantité est supérieure au seuil minimal
                min_size = MIN_POSITION_SIZE.get(symbol, MIN_POSITION_SIZE["DEFAULT"])
                
                # Si la quantité est inférieure au minimum, l'augmenter
                if side.lower() == "buy" and quantity < min_size:
                    logger.info(f"Ajustement de la quantité pour {symbol}: {quantity} -> {min_size} (minimum requis)")
                    quantity = min_size
                
                # Vérifier si la valeur estimative est suffisante
                try:
                    # Estimer la valeur de l'ordre
                    price = None
                    if hasattr(self, 'last_tick') and symbol in self.last_tick and self.last_tick[symbol]:
                        price = self.last_tick[symbol].get('price')
                    elif hasattr(self, 'price_data') and symbol in self.price_data and self.price_data[symbol]:
                        last_bar = self.price_data[symbol][-1]
                        price = last_bar.get('close')
                    
                    if price and price > 0:
                        estimated_value = price * quantity
                        if estimated_value < MIN_POSITION_VALUE_USD:
                            # Recalculer la quantité minimale pour atteindre MIN_POSITION_VALUE_USD
                            adjusted_quantity = MIN_POSITION_VALUE_USD / price
                            logger.info(f"Valeur trop faible pour {symbol} (${estimated_value:.2f}). Ajustement de la quantité: {quantity} -> {adjusted_quantity:.8f}")
                            quantity = adjusted_quantity
                except Exception as e:
                    logger.warning(f"Impossible d'estimer la valeur de l'ordre pour {symbol}: {e}")
            
            # Vérifier le solde avant de passer l'ordre
            if not await self._check_balance(symbol, side, quantity):
                logger.warning(f"Annulation de l'ordre pour {symbol} en raison de solde insuffisant")
                return None
                
            # Si c'est une vente, vérifier la quantité disponible réelle
            if side.lower() == "sell":
                try:
                    position = self.api.get_position(symbol)
                    available_qty = float(position.qty)
                    
                    # Utilisez toujours la quantité disponible exacte pour les ventes
                    # et ne pas dépendre d'une comparaison de valeurs flottantes
                    if available_qty <= 0.000001:  # Trop petit pour être vendu
                        logger.warning(f"Position trop petite pour {symbol}: {available_qty}, impossible de vendre")
                        return None
                        
                    # Pour les ventes, toujours utiliser la quantité disponible exacte
                    # plutôt que de comparer des nombres flottants qui peuvent avoir des erreurs de précision
                    if quantity != available_qty:
                        logger.info(f"Ajustement de la vente pour {symbol}: {quantity} -> {available_qty} (diff: {abs(quantity - available_qty):.12f})")
                        quantity = available_qty
                        
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération de la position pour {symbol}: {e}")
                    return None
                    
            # Arrondir la quantité selon les règles du marché
            original_qty = quantity
            quantity = self._round_quantity(quantity, symbol)
                
            if original_qty != quantity:
                logger.info(f"Quantité arrondie pour {symbol}: {original_qty} -> {quantity}")
                
            # Vérifier que la quantité est positive
            if quantity <= 0:
                logger.warning(f"Annulation de l'ordre pour {symbol}: quantité nulle ou négative ({quantity})")
                return None
                
            # Exécuter l'ordre via le service Mercurio si disponible
            if self.trading_service:
                try:
                    # Vérifier si la méthode place_order existe
                    if hasattr(self.trading_service, 'place_order'):
                        order = await self.trading_service.place_order(
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            order_type="market",
                            time_in_force="gtc"
                        )
                    # Sinon, essayer submit_order comme alternative
                    elif hasattr(self.trading_service, 'submit_order'):
                        order = await self.trading_service.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side=side,
                            type="market",
                            time_in_force="gtc"
                        )
                    else:
                        raise AttributeError("Méthode de placement d'ordre non disponible")
                        
                    order_id = order.id
                except Exception as e:
                    logger.error(f"Erreur lors de l'utilisation du TradingService: {e}, utilisation de l'API Alpaca directe")
                    # Fallback vers API Alpaca directe
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side=side,
                        type="market",
                        time_in_force="gtc"
                    )
                    order_id = order.id
            # Ou utiliser directement l'API Alpaca
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type="market",
                    time_in_force="gtc"
                )
                order_id = order.id
                
            # Enregistrer l'ordre en attente
            self.pending_orders.add(order_id)
            
            logger.info(f"Ordre soumis: {side} {quantity} {symbol} - Raison: {reason}")
            return order_id
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'ordre {side} {quantity} {symbol}: {e}")
            return None
    
    async def _calculate_position_size(self, symbol: str) -> float:
        """Calculer la taille de position optimale"""
        try:
            # Récupérer les informations de compte
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            equity = float(account.equity)
            
            # Utiliser l'avoir plutôt que le pouvoir d'achat pour les calculs
            available_capital = min(buying_power, equity)
            
            # Ajuster la taille de position selon la volatilité
            volatility_factor = 1.0  # Par défaut
            
            # Calculer la volatilité si on a suffisamment de données
            if len(self.price_data[symbol]) > 20:
                df = self._prepare_dataframe(symbol)
                if not df.empty:
                    # Ajuster selon la volatilité sur les 20 dernières périodes
                    volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
                    if not pd.isna(volatility):
                        # Réduire la taille de position si la volatilité est élevée
                        volatility_factor = max(0.3, 1.0 - volatility * 10)  # Min 30% de la taille normale
            
            # Obtenir le prix actuel
            current_price = 0
            if self.last_tick[symbol] and 'price' in self.last_tick[symbol]:
                current_price = self.last_tick[symbol]['price']
            else:
                # Utiliser le dernier prix du DataFrame si disponible
                df = self._prepare_dataframe(symbol)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
            
            if current_price <= 0:
                logger.error(f"Prix invalide pour {symbol}")
                return 0
                
            # Calculer la position en $ en fonction du capital disponible, du pourcentage alloué
            # par position et du facteur de volatilité
            position_value = available_capital * self.position_size_pct * volatility_factor
            
            # Quantité = valeur de la position / prix actuel
            quantity = position_value / current_price
            
            logger.info(f"Taille de position pour {symbol}: {quantity:.4f} units (${position_value:.2f}, vol={volatility_factor:.2f})")
            return quantity
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la taille de position pour {symbol}: {e}")
            return 0
    
    def _signal_handler(self, sig, frame):
        """Gestionnaire de signal pour arrêter proprement le trader"""
        global running
        logger.info("Signal d'arrêt reçu. Arrêt propre en cours...")
        running = False
        
        # Arrêter proprement tous les threads
        if hasattr(self, 'stream') and self.stream:
            try:
                if hasattr(self.stream, 'stop') and callable(self.stream.stop):
                    self.stream.stop()
            except Exception as e:
                logger.warning(f"Erreur lors de l'arrêt du stream: {e}")
                
        # Gérer le nettoyage de manière synchrone pour éviter les problèmes d'await
        try:
            # Générer un rapport plutôt que cleanup() qui est asynchrone
            self._generate_final_report()
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage final: {e}")
            
        logger.info("Trader HF arrêté.")
        
    def _generate_final_report(self):
        """Générer un rapport de performance final de manière synchrone"""
        try:
            # Récupérer les positions actuelles en mode synchrone
            positions = {}
            try:
                account = self.api.get_account()
                positions_api = self.api.list_positions()
                
                for pos in positions_api:
                    symbol = pos.symbol
                    positions[symbol] = {
                        'quantity': float(pos.qty),
                        'entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc),
                        'market_value': float(pos.market_value)
                    }
                    
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération des positions finales: {e}")
            
            # Générer le rapport
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "open_positions": len(positions),
                "positions": positions
            }
            
            # Écrire le rapport dans un fichier
            report_file = f"hft_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Rapport de performance généré: {report_file}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport final: {e}")
        
    def _direct_crypto_api_call(self, symbol, timeframe="1Min", limit=100):
        """Effectue un appel direct à l'API Alpaca crypto pour récupérer des données historiques"""
        try:
            # Construction de l'URL pour l'API crypto (endpoint v1beta3)
            base_url = "https://data.alpaca.markets"
            api_version = "v1beta3"
            
            # Pour les barres historiques
            if timeframe:
                endpoint = f"{base_url}/{api_version}/crypto/us/bars"
                
                # Pour l'API v1beta3, les symboles doivent être au format BTC/USD
                formatted_symbol = self._ensure_symbol_format(symbol)
                if "/" not in formatted_symbol and formatted_symbol.endswith("USD"):
                    # Conversion forcée au format correct
                    base = formatted_symbol[:-3]
                    formatted_symbol = f"{base}/USD"
                
                # Préparer les paramètres - utiliser UTC pour la compatibilité
                from datetime import timezone
                now = datetime.now(timezone.utc)
                start_time = (now - timedelta(minutes=limit)).isoformat()
                end_time = now.isoformat()
                
                params = {
                    "symbols": formatted_symbol,
                    "timeframe": timeframe,
                    "start": start_time,
                    "end": end_time,
                    "limit": limit
                }
                
                logger.info(f"Appel API crypto pour {formatted_symbol} avec URL: {endpoint}?symbols={formatted_symbol}")
            else:
                # Pour les derniers prix (latest)
                endpoint = f"{base_url}/{api_version}/crypto/us/latest/bars"
                params = {"symbols": symbol}
                logger.info(f"Appel API crypto pour dernier prix: {endpoint}?symbols={symbol}")
            
            # Configurer les entêtes avec l'authentification
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            # Effectuer la requête
            import requests
            response = requests.get(endpoint, params=params, headers=headers)
            
            # Vérifier si la réponse est valide
            if response.status_code == 200:
                data = response.json()
                
                # Vérifier la structure des données
                if 'bars' in data and formatted_symbol in data['bars']:
                    logger.info(f"Données crypto reçues avec succès pour {symbol} ({len(data['bars'][formatted_symbol])} barres)")
                    return data
                else:
                    logger.error(f"Structure de données inattendue: {data.keys() if isinstance(data, dict) else 'non dictionnaire'}")
                    return None
            else:
                # Log l'erreur
                logger.error(f"Erreur API Alpaca: {response.status_code} / {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Échec de l'appel direct à l'API crypto: {str(e)}")
            return None
    
    def _ensure_symbol_format(self, symbol):
        """Standardise le format des symboles crypto (BTC/USD ou BTCUSD selon le contexte)"""
        if self.asset_type != AssetType.CRYPTO:
            return symbol  # Ne rien faire pour les symboles non-crypto
            
        # Pour l'API crypto d'Alpaca niveau 3 (premium), on utilise le format avec slash (BTC/USD)
        if self.api_level >= 3:
            if "/" not in symbol:
                # Conversion BTCUSD -> BTC/USD
                if symbol.endswith("USD"):
                    base = symbol[:-3]
                    quote = "USD"
                    formatted_symbol = f"{base}/{quote}"
                    logger.debug(f"Conversion du format de symbole: {symbol} -> {formatted_symbol}")
                    return formatted_symbol
                else:
                    # Autre format inconnu, laisser tel quel
                    return symbol
            else:
                # Déjà au format avec slash
                return symbol
        else:
            # Pour les autres niveaux d'API, utiliser le format sans slash (BTCUSD)
            if "/" in symbol:
                # Conversion BTC/USD -> BTCUSD
                formatted_symbol = symbol.replace("/", "")
                logger.debug(f"Conversion du format de symbole: {symbol} -> {formatted_symbol}")
                return formatted_symbol
            else:
                # Déjà au format sans slash
                return symbol

    def _round_quantity(self, quantity, symbol):
        """Arrondir la quantité selon les règles du marché"""
        # Protection contre les valeurs négatives ou nulles
        if quantity <= 0:
            return 0.0
            
        # Valeurs par défaut pour les cryptos usuelles
        precision = 8  # Précision par défaut pour les crypto
        min_size = MIN_POSITION_SIZE.get(symbol.replace("/", ""), 0.0001)  # Minimum par défaut       
        # Crypto: arrondir avec des règles spécifiques
        if self.asset_type == AssetType.CRYPTO:
            # Vérifier si la correction de précision est activée
            if hasattr(self, 'crypto_precision_fix') and self.crypto_precision_fix:
                # Pour les crypto-monnaies de faible valeur (comme SHIB, DOGE), utiliser plus de précision
                if "SHIB" in symbol or "PEPE" in symbol:
                    # Ces tokens ont besoin de beaucoup de décimales à cause de leur faible valeur
                    precision = 12
                    # Arrondir vers le bas pour garantir que nous ne dépassons jamais le solde disponible
                    rounded = math.floor(quantity * 10**precision) / 10**precision
                    # Si la valeur est trop petite, renvoyer 0
                    if rounded < 1e-10:
                        return 0.0
                    return rounded
                elif "DOGE" in symbol:
                    # DOGE a besoin d'une précision légèrement différente
                    precision = 8
                    # Arrondir vers le bas pour éviter les erreurs d'insuffisance de solde
                    return math.floor(quantity * 10**precision) / 10**precision
                # Pour BTC, ajuster la précision à cause de sa valeur élevée
                elif "BTC" in symbol:
                    precision = 8
                # Pour ETH, SOL, AVAX et autres crypto majeures
                elif any(token in symbol for token in ["ETH", "SOL", "AVAX", "LINK", "XRP", "DOT", "LTC"]):
                    precision = 8
                else:
                    # Pour les autres cryptos, utiliser une précision par défaut
                    precision = 8
                
                # Arrondir TOUJOURS vers le bas pour éviter les problèmes de solde insuffisant
                rounded = math.floor(quantity * 10**precision) / 10**precision
                
                # Vérifier si la quantité est extrêmement petite
                if rounded < 1e-8:
                    return 0.0  # Retourner zéro pour les quantités trop petites
                    
                logger.debug(f"Arrondi effectué pour {symbol}: {quantity} -> {rounded} (précision: {precision})")
                return rounded
            else:
                # Comportement par défaut - arrondir vers le bas avec 8 décimales
                precision = 8
                return math.floor(quantity * 10**precision) / 10**precision
        else:  # Actions: arrondir selon les règles NYSE/NASDAQ
            return math.floor(quantity)  # Arrondir vers le bas pour les actions aussi
    
    def _is_shorting_enabled(self) -> bool:
        """Vérifier si le compte permet le trading à découvert"""
        try:
            account = self.api.get_account()
            return account.shorting_enabled
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du shorting: {e}")
            return False
    
    def _setup_polling_fallback(self):
        """Configurer le mode de repli par polling lorsque les WebSockets échouent"""
        logger.info("Configuration du mode polling")
        self.use_websockets = False
        self.polling_interval = max(1, int(self.market_check_interval) // 2) if hasattr(self, 'market_check_interval') else 1
        
        # Initialiser les données de prix si nécessaire
        if not hasattr(self, 'price_data'):
            self.price_data = {symbol: [] for symbol in self.symbols}
            
        # Pré-initialiser les derniers ticks pour chaque symbole
        self.last_tick = {}
        for symbol in self.symbols:
            self.last_tick[symbol] = {
                'price': 0.0,  # Valeur par défaut
                'timestamp': datetime.now(),
                'initialized': False  # Indicateur de premier chargement
            }
            
        # Charger les données initiales pour chaque symbole de manière sécurisée
        try:
            self._load_initial_ticker_data()
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des données initiales: {e}")
            
        # Démarrer la boucle de polling dans un thread séparé
        def polling_loop():
            # Créer un event loop dédié pour ce thread
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                while running:
                    try:
                        for symbol in self.symbols:
                            try:
                                if not self.last_tick[symbol].get('initialized', False):
                                    continue  # Attendre l'initialisation
                                    
                                # Récupérer les derniers prix
                                if self.asset_type == AssetType.STOCK:
                                    try:
                                        latest_bar = self.api.get_latest_bar(symbol)
                                        price = float(latest_bar.c)
                                        timestamp = latest_bar.t
                                    except Exception as e:
                                        logger.debug(f"Erreur lors de la récupération du prix pour {symbol}: {e}")
                                        continue
                                else:  # Crypto
                                    try:
                                        # Essayer d'abord avec get_latest_trade qui est plus commun
                                        trade = self.api.get_latest_trade(symbol)
                                        price = float(trade.p)
                                        timestamp = trade.t
                                    except Exception as e1:
                                        try:
                                            # Alternative: essayer get_latest_crypto_bar ou get_last_crypto_bar 
                                            # (selon la version de l'API)
                                            if hasattr(self.api, 'get_latest_crypto_bar'):
                                                bar = self.api.get_latest_crypto_bar(symbol)
                                                price = float(bar.c)
                                                timestamp = bar.t
                                            elif hasattr(self.api, 'get_latest_bar'):
                                                # Utiliser get_latest_bar qui fonctionne aussi avec les crypto
                                                bar = self.api.get_latest_bar(symbol)
                                                price = float(bar.c)
                                                timestamp = bar.t
                                            else:
                                                # Dernier recours: obtenir la dernière barre via get_barset
                                                barset = self.api.get_barset(symbol, "1Min", limit=1)
                                                if symbol in barset and len(barset[symbol]) > 0:
                                                    bar = barset[symbol][0]
                                                    price = float(bar.c)
                                                    timestamp = bar.t
                                                else:
                                                    logger.warning(f"Impossible de récupérer les données pour {symbol}")
                                                    continue
                                        except Exception as e2:
                                            logger.debug(f"Erreur lors de la récupération du prix crypto pour {symbol}: {e1} / {e2}")
                                            continue
                                
                                # Mettre à jour les données
                                self.last_tick[symbol] = {
                                    'price': price,
                                    'timestamp': timestamp,
                                    'initialized': True
                                }
                                
                                # Simuler un événement de bar minute pour déclencher l'analyse
                                bar_dict = {
                                    'timestamp': pd.Timestamp.now(),
                                    'open': price,
                                    'high': price,
                                    'low': price,
                                    'close': price,
                                    'volume': 0
                                }
                                self.price_data[symbol].append(bar_dict)
                                
                                # Lancer l'analyse dans l'event loop courant
                                try:
                                    loop.run_until_complete(self.analyze_symbol(symbol))
                                except Exception as e:
                                    logger.warning(f"Erreur lors de l'analyse pour {symbol}: {e}")
                                
                            except Exception as e:
                                logger.warning(f"Erreur de polling pour {symbol}: {e}")
                        
                        # Mettre à jour les positions dans l'event loop courant
                        try:
                            loop.run_until_complete(self.refresh_positions())
                        except Exception as e:
                            logger.warning(f"Erreur lors de la mise à jour des positions: {e}")
                            
                    except Exception as e:
                        logger.error(f"Erreur générale dans la boucle de polling: {e}")
                    
                    time.sleep(self.polling_interval)
            except Exception as e:
                logger.error(f"Erreur critique dans le thread de polling: {e}")
                
        # Démarrer le thread de polling
        self.polling_thread = threading.Thread(target=polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info(f"Mode polling démarré avec intervalle de {self.polling_interval}s")
        
    def _load_initial_ticker_data(self):
        """Charge les données initiales pour chaque symbole"""
        for symbol in self.symbols:
            try:
                # D'abord, vérifier si nous avons déjà des données historiques chargées
                if symbol in self.price_data and self.price_data[symbol] and len(self.price_data[symbol]) > 0:
                    # Utiliser la dernière barre des données historiques
                    last_bar = self.price_data[symbol][-1]
                    price = float(last_bar['close'])
                    timestamp = last_bar['timestamp']
                    logger.info(f"Utilisation des données historiques pour le prix initial de {symbol}")
                    self.last_tick[symbol] = {
                        'price': price,
                        'timestamp': timestamp,
                        'initialized': True
                    }
                    logger.info(f"Prix initial pour {symbol}: {price}")
                    continue
                
                # Si nous n'avons pas de données historiques, essayer l'API Alpaca
                if self.asset_type == AssetType.STOCK:
                    bar = self.api.get_latest_bar(symbol)
                    self.last_tick[symbol] = {
                        'price': float(bar.c),
                        'timestamp': bar.t,
                        'initialized': True
                    }
                else:  # Crypto
                    # Essayer d'abord le MarketDataService pour la cohérence
                    try:
                        # Vérifier si le market_data_service est correctement initialisé
                        if not self.market_data_service:
                            raise Exception("MarketDataService non initialisé")
                        
                        # Déterminer le bon format de symbole selon le niveau d'API
                        api_symbol = self._ensure_symbol_format(symbol)
                        
                        # Méthode 1: Essayer d'obtenir une citation en direct sans utiliser le MarketDataService asynchrone
                        # Cette approche fonctionne pour les versions récentes de l'API Alpaca
                        quote_price = None
                        # Pour l'API niveau 3, on peut essayer d'utiliser les méthodes crypto spécifiques
                        try:
                            # Essayer les différentes méthodes disponibles dans l'API Alpaca pour les crypto
                            if self.api_level >= 3:
                                # Format sans slash pour les méthodes natives de l'API
                                no_slash_symbol = api_symbol.replace('/', '')
                                
                                if hasattr(self.api, 'get_crypto_latest_trade'):
                                    # Méthode get_crypto_latest_trade (plus récente)
                                    trade = self.api.get_crypto_latest_trade(no_slash_symbol)
                                    quote_price = float(trade.p)
                                    timestamp = trade.t
                                    logger.info(f"Prix initial pour {symbol} obtenu via get_crypto_latest_trade: {quote_price}")
                                elif hasattr(self.api, 'get_crypto_latest_quote'):
                                    # Méthode get_crypto_latest_quote (alternative)
                                    quote = self.api.get_crypto_latest_quote(no_slash_symbol)
                                    quote_price = float(quote.ap)  # ask price
                                    timestamp = quote.t
                                    logger.info(f"Prix initial pour {symbol} obtenu via get_crypto_latest_quote: {quote_price}")
                                elif hasattr(self.api, 'get_latest_crypto_quote'):
                                    # Méthode get_latest_crypto_quote (ancienne version)
                                    quote = self.api.get_latest_crypto_quote(no_slash_symbol)
                                    quote_price = float(quote.ap)
                                    timestamp = quote.t
                                    logger.info(f"Prix initial pour {symbol} obtenu via get_latest_crypto_quote: {quote_price}")
                            
                            if quote_price is not None:
                                price = quote_price
                            else:
                                raise Exception("Aucune méthode de citation crypto disponible")
                                
                        except Exception as quote_error:
                            logger.warning(f"Erreur lors de la récupération de citation crypto: {quote_error}")
                            raise  # Propager l'erreur pour essayer la méthode suivante
                    except Exception as e:
                        try:
                            # Utiliser l'API crypto spécifique pour obtenir les prix
                            # Note: Alpaca a une API spécifique pour les crypto qui utilise /crypto/ au lieu de /stocks/
                            crypto_bar = None
                            
                            # Essayer les méthodes spécifiques aux crypto d'abord
                            if hasattr(self.api, 'get_crypto_latest_bar'):
                                crypto_bar = self.api.get_crypto_latest_bar(symbol)
                            elif hasattr(self.api, 'get_latest_crypto_bar'):
                                crypto_bar = self.api.get_latest_crypto_bar(symbol)
                            
                            if crypto_bar:
                                price = float(crypto_bar.c)
                                timestamp = crypto_bar.t
                                logger.info(f"Prix initial pour {symbol} obtenu via API Crypto Alpaca: {price}")
                            else:
                                # Fallback sur la méthode générique mais avec le bon endpoint
                                # Construire l'URL correcte pour l'API crypto d'Alpaca
                                data_url = "https://data.alpaca.markets"
                            
                            # Pour l'API crypto Alpaca, utiliser le bon endpoint selon le format du symbole
                            # 1. Préparer le symbole selon son format
                            if "/" in symbol:
                                # Format avec slash (BTC/USD) - bon pour v1beta3
                                symbol_with_slash = symbol
                                symbol_no_slash = symbol.replace("/", "")
                            else:
                                # Format sans slash (BTCUSD) - convertir pour v1beta3
                                symbol_no_slash = symbol
                                # Supposons que le format est [BASE]USD
                                if symbol.endswith("USD"):
                                    symbol_base = symbol[:-3]  # supprimer "USD"
                                    symbol_with_slash = f"{symbol_base}/USD"
                                else:
                                    # Gardons-le tel quel si format inconnu
                                    symbol_with_slash = symbol
                            
                            # 2. Construire l'URL selon le niveau d'API
                            if self.api_level >= 3:
                                # Niveau premium: API v1beta3 avec format symbol/quote
                                request_url = f"{data_url}/v1beta3/crypto/us/latest/bars?symbols={symbol_with_slash}"
                                logger.info(f"Utilisation de l'API premium v1beta3 pour {symbol_with_slash}")
                            else:
                                # Niveaux standard: API v2 avec format sans slash
                                request_url = f"{data_url}/v2/crypto/{symbol_no_slash}/bars/latest"
                                logger.info(f"Utilisation de l'API standard v2 pour {symbol_no_slash}")
                            
                            logger.info(f"Utilisation de l'URL: {request_url} pour obtenir le prix de {symbol}")
                            
                            # Effectuer la requête avec l'authentification Alpaca
                            headers = {
                                'APCA-API-KEY-ID': self.api_key,
                                'APCA-API-SECRET-KEY': self.api_secret
                            }
                            
                            import requests
                            response = requests.get(request_url, headers=headers)
                            response.raise_for_status()  # Raise exception si erreur HTTP
                            json_data = response.json()
                            
                            # Traiter la réponse en fonction du format attendu selon l'API utilisée
                            if self.api_level >= 3 and "/v1beta3/" in request_url:
                                # Format de réponse pour v1beta3: {"bars": {"BTC/USD": [{...}, ...]}}
                                if 'bars' in json_data and symbol_with_slash in json_data['bars']:
                                    bars_data = json_data['bars'][symbol_with_slash]
                                    if bars_data and len(bars_data) > 0:
                                        latest_bar = bars_data[-1]  # Prendre la barre la plus récente
                                        price = float(latest_bar['c'])
                                        timestamp = datetime.strptime(latest_bar['t'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                                        logger.info(f"Prix initial pour {symbol} (v1beta3): {price}")
                                    else:
                                        raise Exception(f"Aucune donnée de barre pour {symbol_with_slash}")
                                else:
                                    raise Exception(f"Format de réponse inattendu: {json_data}")
                            else:
                                # Format de réponse pour v2/crypto/{symbol}/bars/latest: 
                                # Soit {'bar': {...}} ou {'bars': [...]}
                                if 'bar' in json_data:
                                    # Format single bar
                                    bar_data = json_data['bar']
                                    price = float(bar_data['c'])
                                    timestamp = datetime.strptime(bar_data['t'], '%Y-%m-%dT%H:%M:%SZ')
                                elif 'bars' in json_data and len(json_data['bars']) > 0:
                                    # Format multiple bars array
                                    latest_bar = json_data['bars'][-1]
                                    price = float(latest_bar['c'])
                                    timestamp = datetime.strptime(latest_bar['t'], '%Y-%m-%dT%H:%M:%SZ')
                                else:
                                    raise Exception(f"Format de réponse inattendu: {json_data}")
                                
                                logger.info(f"Prix initial pour {symbol} obtenu via API REST Crypto: {price}")
                        except Exception as e1:
                            try:
                                # Si les appels spécifiques aux crypto échouent, essayer les appels génériques
                                # Cet appel utilise probablement l'endpoint /stocks/ qui peut échouer pour les crypto
                                trade = self.api.get_latest_trade(symbol)
                                price = float(trade.p)
                                timestamp = trade.t
                                logger.warning(f"Utilisation de l'API actions pour la crypto {symbol}: prix={price}")
                            except Exception as e2:
                                logger.warning(f"Impossible d'obtenir le dernier prix pour {symbol}: {e1} / {e2}")
                                # Utiliser un prix par défaut pour les cryptos courantes
                                default_prices = {
                                    'BTCUSD': 55000.0,
                                    'ETHUSD': 2500.0,
                                    'SOLUSD': 120.0,
                                    'AVAXUSD': 30.0,
                                    'LTCUSD': 80.0,
                                    'LINKUSD': 15.0,
                                    'AAVEUSD': 80.0,
                                    'UNIUSD': 7.0,
                                    'DOTUSD': 10.0,
                                    'BCHUSDT': 10.0,
                                    'ETHUSDT': 10.0,
                                    'BTCUSDT': 10.0,
                                    'LINKUSDT': 10.0,
                                    'LTCUSDT': 10.0,
                                    'AAVEUSDT': 10.0,
                                    'UNIUSDT': 10.0,
                                    'DOGEUSDT': 10.0,
                                    'SUSHIUSDT': 10.0,
                                    'YFIUSDT': 10.0,
                                    'DOGEUSD': 0.15,
                                    'USDCUSD': 1.0,
                                    'USDTUSD': 1.0,
                                    'GRTUSD': 10.0,
                                    'CRVUSD': 0.6,
                                    'BATUSD': 0.2,
                                    'XRPUSD': 0.5,
                                    'XTZUSD': 0.8,
                                    'SHIBUSD': 9e-6,
                                    'PEPEUSD': 5e-6,
                                    'BCHUSD': 250.0,
                                    'MKRUSD': 1200.0,
                                    'YFIUSD': 7500.0,
                                    'SUSHIUSD': 0.7,
                                    'TRUMPUSD': 11.0,
                                }
                                logger.warning(f"Utilisation d'un prix par défaut pour {symbol}")
                                price = default_prices.get(symbol, 10.0)  # Valeur par défaut si le symbole n'est pas dans la liste
                                timestamp = datetime.now()
                    
                    self.last_tick[symbol] = {
                        'price': price,
                        'timestamp': timestamp,
                        'initialized': True
                    }
                logger.info(f"Prix initial pour {symbol}: {self.last_tick[symbol]['price']}")
            except Exception as e:
                logger.warning(f"Impossible de charger les données initiales pour {symbol}: {e}")
                # Laisser initialized=False pour que le symbole soit ignoré jusqu'à ce qu'on puisse obtenir des données
        
    async def cleanup(self):
        """Nettoyer les ressources avant de terminer"""
        try:
            # Fermer proprement le stream
            self.stream.stop()
            
            # Générer un rapport final
            await self.generate_report()
            
            logger.info("Nettoyage terminé")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
    
    async def generate_report(self):
        """Générer un rapport de performance"""
        try:
            # Mettre à jour les positions une dernière fois
            await self.refresh_positions()
            
            # Récupérer les ordres exécutés
            orders = self.api.list_orders(status="closed", limit=100)
            filled_orders = [o for o in orders if o.status == "filled"]
            
            # Calculer les performances
            total_trades = len(filled_orders)
            profitable_trades = sum(1 for p in self.positions.values() if p['unrealized_plpc'] > 0)
            
            # Générer le rapport
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_duration": "N/A",  # À calculer plus tard
                "total_trades": total_trades,
                "open_positions": len(self.positions),
                "profitable_positions": profitable_trades,
                "total_pnl": sum(p['unrealized_pl'] for p in self.positions.values()),
                "positions": self.positions
            }
            
            # Écrire le rapport dans un fichier
            report_file = f"hft_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Rapport de performance généré: {report_file}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")

# ----- Fonction principale -----
def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Trader haute fréquence pour crypto et actions")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                      help="Liste des symboles à trader (ex: AAPL MSFT GOOGL ou BTC/USD ETH/USD)")
    parser.add_argument("--strategy", type=str, 
                      choices=[s.value for s in StrategyType], 
                      default=StrategyType.MOVING_AVERAGE.value,
                      help="Stratégie de trading à utiliser")
    parser.add_argument("--asset-type", type=str, choices=["stock", "crypto"], default="stock",
                      help="Type d'actif à trader")
    parser.add_argument("--position-size", type=float, default=0.01,
                      help="Taille de position en pourcentage du portefeuille (default: 0.01 = 1%)")
    parser.add_argument("--stop-loss", type=float, default=0.002,
                      help="Stop loss en pourcentage (default: 0.002 = 0.2%)")
    parser.add_argument("--take-profit", type=float, default=0.005,
                      help="Take profit en pourcentage (default: 0.005 = 0.5%)")
    parser.add_argument("--api-level", type=int, choices=[1, 2, 3], default=3,
                      help="Niveau d'API Alpaca à utiliser (1=basique, 2=standard+, 3=premium)")
    parser.add_argument("--max-positions", type=int, default=5,
                      help="Nombre maximum de positions simultanées")
    
    # Arguments standards communs avec les autres scripts
    parser.add_argument("--use-custom-symbols", action="store_true", 
                      help="Utiliser la liste personnalisée de symboles au lieu du filtre automatique")
    parser.add_argument("--custom-symbols-file", type=str, default=None,
                      help="Chemin vers un fichier contenant la liste des symboles personnalisés (un symbole par ligne)")
    parser.add_argument("--crypto-precision-fix", action="store_true",
                      help="Activer la correction de précision pour les crypto-monnaies à faible valeur")
    parser.add_argument("--market-check-interval", type=int, default=1,
                      help="Intervalle en secondes entre les vérifications du marché (default: 1s)")
    parser.add_argument("--fast-ma", type=int, default=5,
                      help="Période de la moyenne mobile rapide (default: 5)")
    parser.add_argument("--slow-ma", type=int, default=15,
                      help="Période de la moyenne mobile lente (default: 15)")
    parser.add_argument("--momentum-lookback", type=int, default=10,
                      help="Période de lookback pour la stratégie momentum (default: 10)")
    parser.add_argument("--mean-reversion-zscore", type=float, default=1.5,
                      help="Seuil de Z-score pour la stratégie mean reversion (default: 1.5)")
    parser.add_argument("--config", type=str, default=None,
                      help="Chemin vers un fichier de configuration JSON")
    
    # Mode paper/live
    parser.add_argument("--paper", action="store_true", default=True,
                      help="Utiliser le paper trading (défaut: True)")
    parser.add_argument("--live", action="store_true", default=False,
                      help="Utiliser le live trading (désactivé par défaut)")
    
    # Options d'exécution
    parser.add_argument("--verbose", action="store_true",
                      help="Afficher des messages de débogage détaillés")
    parser.add_argument("--duration", type=str, choices=["1h", "4h", "8h", "night", "continuous"], 
                      default="continuous",
                      help="Durée maximale de la session (1h, 4h, 8h, night=9h ou continuous)")
    parser.add_argument("--debug", action="store_true",
                      help="Activer le mode debug (plus de logs)")
    parser.add_argument("--log-file", type=str, default=None,
                      help="Chemin vers un fichier de log spécifique")
    parser.add_argument("--no-stream", action="store_true", 
                      help="Désactiver le streaming (utilise des requêtes régulières à la place des WebSockets)")
    parser.add_argument("--backtest-mode", action="store_true",
                      help="Exécuter en mode backtest sans passer d'ordres réels")
    
    args = parser.parse_args()
    
    # Déterminer le mode paper/live
    is_paper = not args.live  # Par défaut paper trading à moins que --live soit spécifié
    
    # Déterminer le type d'actif
    asset_type = AssetType.STOCK if args.asset_type.lower() == "stock" else AssetType.CRYPTO
    
    # Convertir la stratégie
    strategy_type = next((s for s in StrategyType if s.value == args.strategy), StrategyType.MOVING_AVERAGE)
    
    # Gérer les symboles personnalisés 
    symbols = args.symbols
    
    # Charger les symboles depuis un fichier si spécifié
    if args.custom_symbols_file and args.use_custom_symbols:
        import csv
        try:
            symbols_from_file = []
            if args.custom_symbols_file.lower().endswith('.csv'):
                with open(args.custom_symbols_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    # Try to skip header if present
                    first_row = next(reader, None)
                    if first_row and ("symbol" in first_row[0].lower() or "ticker" in first_row[0].lower()):
                        pass  # header detected, skip
                    else:
                        if first_row:
                            symbols_from_file.append(first_row[0].strip())
                    for row in reader:
                        if row and row[0].strip():
                            symbols_from_file.append(row[0].strip())
            else:
                with open(args.custom_symbols_file, 'r') as f:
                    symbols_from_file = [line.strip() for line in f if line.strip()]

            if symbols_from_file:
                processed_symbols = []
                for symbol in symbols_from_file:
                    if "/" not in symbol and symbol.endswith("USD"):
                        symbol_base = symbol[:-3]
                        processed_symbol = f"{symbol_base}/USD"
                        logger.info(f"Conversion du symbole: {symbol} -> {processed_symbol}")
                        processed_symbols.append(processed_symbol)
                    else:
                        processed_symbols.append(symbol)
                symbols = processed_symbols
                logger.info(f"Symboles chargés depuis {args.custom_symbols_file}: {len(symbols)} symboles")
            else:
                logger.warning(f"Aucun symbole trouvé dans {args.custom_symbols_file}, utilisation des symboles en ligne de commande")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des symboles depuis {args.custom_symbols_file}: {e}")
            logger.info("Utilisation des symboles spécifiés en ligne de commande")
    
    if not args.use_custom_symbols:
        # Si --use-custom-symbols n'est pas spécifié, utiliser les symboles par défaut
        symbols = None
        
    # Lire la configuration depuis un fichier JSON si spécifié
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration chargée depuis {args.config}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
    
    # Configurer le niveau de logging
    if args.debug or args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Mode débogage activé")
        
    # Créer et démarrer le trader
    trader = HFTrader(
        symbols=symbols,
        strategy_type=strategy_type,
        asset_type=asset_type,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        api_level=args.api_level,
        max_positions=args.max_positions,
        is_paper=is_paper,
        use_custom_symbols=args.use_custom_symbols
    )
    
    # Configurer l'option de correction de précision pour les crypto-monnaies
    if args.crypto_precision_fix:
        trader.crypto_precision_fix = True
    
    # Configuration supplémentaire du trader basée sur les nouveaux arguments
    if args.fast_ma:
        trader.fast_ma_period = args.fast_ma
    if args.slow_ma:
        trader.slow_ma_period = args.slow_ma
    if args.momentum_lookback:
        trader.momentum_lookback = args.momentum_lookback
    if args.mean_reversion_zscore:
        trader.mean_reversion_zscore = args.mean_reversion_zscore
    if args.market_check_interval:
        trader.market_check_interval = args.market_check_interval
    
    # Mode backtest
    if args.backtest_mode:
        trader.is_backtest = True
    
    # Afficher le résumé de la configuration
    logger.info("=" * 60)
    logger.info(f"TRADER HAUTE FREQUENCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"Mode: {'PAPER' if is_paper else 'LIVE'}")
    logger.info(f"Type d'actif: {asset_type.name}")
    logger.info(f"Symboles: {trader.symbols}")
    logger.info(f"Stratégie: {strategy_type.value}")
    logger.info(f"Niveau API Alpaca: {args.api_level}")
    logger.info(f"Taille de position: {args.position_size*100}%")
    logger.info(f"Stop Loss: {args.stop_loss*100}%, Take Profit: {args.take_profit*100}%")
    logger.info(f"Maximum positions simultanées: {args.max_positions}")
    logger.info("=" * 60)
    logger.info("Appuyez sur Ctrl+C pour arrêter le trader proprement.")
    
    # Démarrer la boucle asynchrone
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        logger.info("Arrêt manuel du trader.")
    except Exception as e:
        logger.error(f"Erreur générale: {e}")
    finally:
        logger.info("Trader HF arrêté.")

if __name__ == "__main__":
    main()
