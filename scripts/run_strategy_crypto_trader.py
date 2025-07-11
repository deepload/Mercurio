#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour lancer le trader de cryptos avec une stratégie spécifique
Ce script va lancer le trader de cryptos Alpaca avec une stratégie explicite et une 
durée de session paramétrable, parfait pour les sessions de nuit ou de jour.
"""

import sys
import os
import argparse
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import signal
import atexit

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer le trader de crypto
from alpaca_crypto_trader import AlpacaCryptoTrader, SessionDuration

# Importer les stratégies avancées
from app.strategies.lstm_predictor import LSTMPredictorStrategy
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.llm_strategy_v2 import LLMStrategyV2
from app.strategies.llm_strategy_v3 import LLMStrategyV3
from app.strategies.sentiment.news_api_agent import NewsAPIAgent

# Importer les fournisseurs de données
from app.services.providers.binance import BinanceProvider
from app.services.providers.factory import MarketDataProviderFactory

# Suppression des messages d'erreur asyncio "Event loop is closed"
import asyncio
import warnings
import sys

# Solution radicale pour supprimer les erreurs "Event loop is closed"
# en remplaçant le destructeur de _ProactorBasePipeTransport
if sys.platform == 'win32':
    try:
        # Patch pour Windows uniquement
        import asyncio.proactor_events
        
        # Sauvegarde de la méthode originale
        original_del = asyncio.proactor_events._ProactorBasePipeTransport.__del__
        
        # Remplacement par une version silencieuse qui ne génère pas d'erreur
        def silent_del(self):
            try:
                original_del(self)
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise
        
        # Application du patch
        asyncio.proactor_events._ProactorBasePipeTransport.__del__ = silent_del
    except (ImportError, AttributeError):
        pass  # Si le module n'est pas trouvé ou si la classe n'existe pas

# Désactiver les avertissements liés à asyncio
logging.getLogger('asyncio').setLevel(logging.ERROR)

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
            base_url=base_url
            # Removed data_url as it's not supported in newer versions
        )
        
        logger.info("Test du niveau d'abonnement Alpaca...")
        #return 3
        # Test niveau 3 (premium) - Accès aux données en temps réel
        try:
            # Tester une fonctionnalité spécifique au niveau 3: données en temps réel plus précises
            end = datetime.now()
            start = end - timedelta(minutes=15)
            symbol = "BTC/USD"  # Une paire crypto populaire
            bars = api.get_crypto_bars(symbol, '1Min', start.isoformat(), end.isoformat())
            if len(bars) > 0 and hasattr(bars[0], 'trade_count'):
                logger.info("[OK] Niveau 3 (Premium) détecté - Accès complet aux données temps réel")
                return 3
        except Exception as e:
            logger.debug(f"Test niveau 3 échoué: {str(e)}")
        
        # Test niveau 2 - Données historiques étendues
        try:
            # Tester des données historiques (disponibles dans les niveaux 2 et 3)
            end = datetime.now()
            start = end - timedelta(days=30)  # 30 jours de données
            bars = api.get_crypto_bars('BTC/USD', '1Day', start.isoformat(), end.isoformat())
            if len(bars) > 20:  # Si on a plus de 20 jours, c'est probablement niveau 2+
                logger.info("✅ Niveau 2 détecté - Accès aux données historiques étendues")
                return 2
        except Exception as e:
            logger.debug(f"Test niveau 2 échoué: {str(e)}")
        
        # Test niveau 1 - Fonctionnalités de base
        try:
            # Tester les fonctionnalités de base (disponibles dans tous les niveaux)
            account = api.get_account()
            logger.info("✅ Niveau 1 détecté - Accès aux fonctionnalités de base")
            return 1
        except Exception as e:
            logger.debug(f"Test niveau 1 échoué: {str(e)}")
        
        logger.warning("❌ Aucun niveau d'abonnement détecté - Vérifiez vos identifiants API")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Alpaca: {str(e)}")
        return 0

# Configuration du logger avec encodage UTF-8
try:
    # Configurer l'encodage de la console sur Windows
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    
    # Configuration du logger avec encodage UTF-8
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", encoding='utf-8')
except Exception:
    # Fallback si l'encodage échoue
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger("strategy_crypto_trader")

# Importation de l'utilitaire d'arrêt propre
try:
    from scripts.graceful_exit import is_running, register_thread, register_cleanup, register_liquidation_handler
    logger.info("Utilitaire d'arrêt propre chargé avec succès")
    USE_GRACEFUL_EXIT = True
except ImportError:
    # Fonction de repli si le module n'est pas disponible
    logger.warning("Utilitaire d'arrêt propre non disponible, utilisation du mécanisme standard")
    USE_GRACEFUL_EXIT = False
    # Variables globales pour la gestion des signaux
    running = True
    
    def is_running():
        global running
        return running
        
    def register_thread(thread):
        pass
        
    def register_cleanup(callback):
        pass
        
    def register_liquidation_handler(callback):
        pass
        
    # Gestionnaire de signal traditionnel
    def signal_handler(sig, frame):
        global running
        logger.info("Signal d'arrêt reçu, arrêt en cours...")
        running = False

# Liste personnalisée de cryptos à trader
def load_crypto_symbols_from_file(file_path):
    """Charge les symboles de crypto depuis un fichier externe"""
    try:
        symbols = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
        logger.info(f"Chargé {len(symbols)} symboles depuis {file_path}")
        return symbols
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
        return []

def load_crypto_symbols_from_env():
    """Charge les symboles de crypto depuis la variable d'environnement"""
    try:
        env_symbols = os.getenv("PERSONALIZED_CRYPTO_LIST", "")
        if env_symbols:
            symbols = [s.strip() for s in env_symbols.split(',')]
            logger.info(f"Chargé {len(symbols)} symboles depuis .env")
            return symbols
        else:
            logger.warning("Aucun symbole trouvé dans .env")
            return []
    except Exception as e:
        logger.error(f"Erreur lors du chargement des symboles depuis .env: {e}")
        return []

# Liste par défaut
DEFAULT_CRYPTO_LIST = [
    "BCH/USDT",
    "MKR/USD",
    "LTC/USD",
    "BAT/USD",
    "ETH/USD",
    "SOL/USD",
    "BTC/USD",
    "AVAX/USD", 
    "BCH/BTC", 
    "DOT/USDC", 
    "USDG/USD", 
    "BCH/USD", 
    "UNI/BTC",  
    "YFI/USD", 
    "GRT/USD",  
    "LINK/USD",   
    "SUSHI/USD",  
    "UNI/USD", 
    "CRV/USD", 
    "XRP/USD",  
    "AAVE/USD",  
    "TRUMP/USD", 
    "XTZ/USD", 
    "DOT/USD",
    "SHIB/USD",
    "PEPE/USD",
    "DOGE/USD",
]
# DEFAULT_CRYPTO_LIST = [
#     "AAVE/USD", "AAVE/USDT", "AVAX/USD", "BAT/USD", "BCH/USD", 
#     "BCH/USDT", "BTC/USD", "BTC/USDT", "CRV/USD", "DOGE/USD", 
#     "DOGE/USDT", "DOT/USD", "ETH/USD", "ETH/USDT", "GRT/USD", 
#     "LINK/USD", "LINK/USDT", "LTC/USD", "LTC/USDT", "MKR/USD", 
#     "PEPE/USD", "SHIB/USD", "SOL/USD", "SUSHI/USD", "SUSHI/USDT", 
#     "TRUMP/USD", "UNI/USD", "UNI/USDT", "USDC/USD", "USDT/USD", 
#     "XRP/USD", "XTZ/USD", "YFI/USD", "YFI/USDT"
# ]


# Sera initialisé durant l'exécution
PERSONALIZED_CRYPTO_LIST = []

# Implémentation des stratégies de trading
class BaseStrategy:
    """Classe de base pour toutes les stratégies"""
    def __init__(self, **kwargs):
        self.position_size = kwargs.get("position_size", 0.02)
        self.stop_loss = kwargs.get("stop_loss", 0.03)
        self.take_profit = kwargs.get("take_profit", 0.06)
        self.name = "BaseStrategy"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyser les données et générer un signal"""
        raise NotImplementedError("Les stratégies dérivées doivent implémenter cette méthode")

class MomentumStrategy(BaseStrategy):
    """Stratégie basée sur le momentum des prix"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = kwargs.get("lookback_period", 20)
        self.name = "Momentum"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un signal basé sur le momentum"""
        if len(data) < self.lookback_period + 10:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes"}
        
        # Calculer le momentum (changement de prix sur la période)
        data['returns'] = data['close'].pct_change(self.lookback_period)
        data['momentum'] = data['returns'].rolling(window=10).mean()
        
        current_momentum = data['momentum'].iloc[-1]
        momentum_signal = "neutral"
        signal_strength = abs(current_momentum) * 10  # Normaliser entre 0-1
        
        if current_momentum > 0.02:  # Momentum positif significatif
            momentum_signal = "buy"
        elif current_momentum < -0.02:  # Momentum négatif significatif
            momentum_signal = "sell"
            
        return {
            "signal": momentum_signal,
            "strength": min(signal_strength, 1.0),
            "reason": f"Momentum {current_momentum:.4f}"
        }

class MeanReversionStrategy(BaseStrategy):
    """Stratégie basée sur le retour à la moyenne"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = kwargs.get("lookback_period", 20)
        self.name = "Mean Reversion"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un signal basé sur le retour à la moyenne"""
        if len(data) < self.lookback_period + 10:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes"}
        
        # Calculer la moyenne mobile et les bandes de Bollinger
        data['sma'] = data['close'].rolling(window=self.lookback_period).mean()
        data['std'] = data['close'].rolling(window=self.lookback_period).std()
        data['upper_band'] = data['sma'] + (data['std'] * 2)
        data['lower_band'] = data['sma'] - (data['std'] * 2)
        
        current_price = data['close'].iloc[-1]
        upper_band = data['upper_band'].iloc[-1]
        lower_band = data['lower_band'].iloc[-1]
        sma = data['sma'].iloc[-1]
        
        # Distance normalisée par rapport aux bandes
        if current_price > upper_band:
            # Sur-acheté - signal de vente
            distance = (current_price - upper_band) / (upper_band - sma)
            signal = "sell"
            reason = f"Prix {current_price:.4f} au-dessus de la bande supérieure {upper_band:.4f}"
        elif current_price < lower_band:
            # Sur-vendu - signal d'achat
            distance = (lower_band - current_price) / (sma - lower_band)
            signal = "buy"
            reason = f"Prix {current_price:.4f} en-dessous de la bande inférieure {lower_band:.4f}"
        else:
            # Entre les bandes - neutre
            if current_price > sma:
                distance = (current_price - sma) / (upper_band - sma)
                signal = "neutral_bearish"  # Tendance baissière potentielle
            else:
                distance = (sma - current_price) / (sma - lower_band)
                signal = "neutral_bullish"  # Tendance haussière potentielle
            reason = f"Prix {current_price:.4f} entre les bandes (SMA: {sma:.4f})"
        
        return {
            "signal": signal,
            "strength": min(distance * 1.5, 1.0),  # Normaliser entre 0-1
            "reason": reason
        }

class BreakoutStrategy(BaseStrategy):
    """Stratégie basée sur les ruptures de niveaux"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = kwargs.get("lookback_period", 20)
        self.name = "Breakout"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un signal basé sur les ruptures de niveaux"""
        if len(data) < self.lookback_period + 5:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes"}
        
        # Calculer les niveaux de support et résistance
        lookback_data = data.iloc[-self.lookback_period-5:-5]
        resistance = lookback_data['high'].max()
        support = lookback_data['low'].min()
        
        # Vérifier le volume
        avg_volume = lookback_data['volume'].mean()
        current_volume = data['volume'].iloc[-1]
        volume_factor = min(current_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
        
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        price_range = resistance - support
        
        if price_range <= 0:
            return {"signal": "neutral", "strength": 0, "reason": "Plage de prix trop faible"}
        
        # Normaliser la distance par rapport à la plage
        if current_price > resistance and previous_price <= resistance:
            # Breakout haussier
            distance = (current_price - resistance) / price_range
            signal = "buy"
            reason = f"Breakout haussier: {current_price:.4f} > {resistance:.4f} (résistance)"
        elif current_price < support and previous_price >= support:
            # Breakout baissier
            distance = (support - current_price) / price_range
            signal = "sell"
            reason = f"Breakout baissier: {current_price:.4f} < {support:.4f} (support)"
        else:
            # Pas de breakout
            signal = "neutral"
            if current_price > (resistance + support) / 2:
                reason = f"Prix {current_price:.4f} proche de la résistance {resistance:.4f}"
                distance = (current_price - ((resistance + support) / 2)) / (resistance - ((resistance + support) / 2))
            else:
                reason = f"Prix {current_price:.4f} proche du support {support:.4f}"
                distance = (((resistance + support) / 2) - current_price) / (((resistance + support) / 2) - support)
        
        signal_strength = min(distance * volume_factor, 1.0)
        
        return {
            "signal": signal,
            "strength": signal_strength,
            "reason": reason
        }

class StatisticalArbitrageStrategy(BaseStrategy):
    """Stratégie d'arbitrage statistique pour paires de cryptos"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volatility_lookback = kwargs.get("volatility_lookback", 10)
        self.name = "Statistical Arbitrage"
    
    def analyze(self, data: pd.DataFrame, pair_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Génère un signal basé sur l'arbitrage statistique"""
        if pair_data is None or len(data) < self.volatility_lookback + 5 or len(pair_data) < self.volatility_lookback + 5:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes pour l'arbitrage"}
        
        # Normaliser les prix
        data_normalized = data['close'] / data['close'].iloc[0]
        pair_normalized = pair_data['close'] / pair_data['close'].iloc[0]
        
        # Calculer le ratio et la moyenne mobile
        ratio = data_normalized / pair_normalized
        ratio_mean = ratio.rolling(window=self.volatility_lookback).mean()
        ratio_std = ratio.rolling(window=self.volatility_lookback).std()
        
        if ratio_std.iloc[-1] == 0:
            return {"signal": "neutral", "strength": 0, "reason": "Volatilité du ratio trop faible"}
        
        # Calculer le z-score du ratio actuel
        current_ratio = ratio.iloc[-1]
        mean_ratio = ratio_mean.iloc[-1]
        std_ratio = ratio_std.iloc[-1]
        z_score = (current_ratio - mean_ratio) / std_ratio
        
        # Générer un signal basé sur le z-score
        signal = "neutral"
        if z_score > 2.0:  # Ratio anormalement élevé - la paire 1 est surperformante
            signal = "sell"  # Vendre la première paire
            reason = f"Ratio anormalement élevé: z-score = {z_score:.4f}"
        elif z_score < -2.0:  # Ratio anormalement bas - la paire 1 est sous-performante
            signal = "buy"  # Acheter la première paire
            reason = f"Ratio anormalement bas: z-score = {z_score:.4f}"
        else:
            reason = f"Ratio normal: z-score = {z_score:.4f}"
        
        signal_strength = min(abs(z_score) / 3.0, 1.0)  # Normaliser entre 0-1
        
        return {
            "signal": signal,
            "strength": signal_strength,
            "reason": reason
        }

# Énumération des stratégies disponibles
class StrategyType(str, Enum):
    MOVING_AVERAGE = "moving_average"  # Stratégie par défaut d'AlpacaCryptoTrader
    MOVING_AVERAGE_ML = "moving_average_ml"  # Version améliorée avec ML pour optimiser les paramètres
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    STATISTICAL_ARBITRAGE = "stat_arb"
    TRANSFORMER = "transformer"  # Stratégie basée sur un modèle Transformer de deep learning
    LSTM = "lstm"
    LLM = "llm"  # Stratégie basée sur un modèle LLM de deep learning
    LLM_V2 = "llm_v2"  # Stratégie avancée LLM combinant analyse technique et sentiment
    LLM_V3 = "llm_v3"  # Stratégie multi-agent LLM avec architecture Claude Trader, Analyste, Coordinateur

def get_strategy_class(strategy_type: str) -> Optional[type]:
    """Récupère la classe de stratégie en fonction du type spécifié"""
    # Import des stratégies avancées uniquement si nécessaires
    if strategy_type.lower() == StrategyType.TRANSFORMER.lower():
        try:
            from app.strategies.transformer_strategy import TransformerStrategy
            return TransformerStrategy
        except ImportError:
            logger.warning("Stratégie Transformer non disponible, utilisation de la stratégie par défaut")
            return None
    
    # Nouvelle stratégie LLM_V2
    if strategy_type.lower() == StrategyType.LLM_V2.lower():
        try:
            from app.strategies.llm_strategy_v2 import LLMStrategyV2
            return LLMStrategyV2
        except ImportError:
            logger.warning("Stratégie LLM_V2 non disponible, utilisation de la stratégie par défaut")
            return None
            
    # Nouvelle stratégie LLM_V3 (multi-agent)
    if strategy_type.lower() == StrategyType.LLM_V3.lower():
        try:
            from app.strategies.llm_strategy_v3 import LLMStrategyV3
            return LLMStrategyV3
        except ImportError:
            logger.warning("Stratégie LLM_V3 non disponible, utilisation de la stratégie par défaut")
            return None
    
    # Import de MovingAverageMLStrategy si nécessaire
    if strategy_type.lower() == StrategyType.MOVING_AVERAGE_ML.lower():
        try:
            from app.strategies.moving_average_ml import MovingAverageMLStrategy
            # Adapter la stratégie pour le trading crypto
            class CryptoMovingAverageMLStrategy(MovingAverageMLStrategy):
                """Version adaptée de MovingAverageMLStrategy pour les cryptomonnaies"""
                def __init__(self, **kwargs):
                    # Paramètres spécifiques adaptés aux crypto (plus courte période, plus de volatilité)
                    kwargs.setdefault('short_window_min', 3)  # Fenêtres plus courtes pour les crypto
                    kwargs.setdefault('short_window_max', 24)
                    kwargs.setdefault('long_window_min', 20)
                    kwargs.setdefault('long_window_max', 80)
                    kwargs.setdefault('optimize_interval', 15)  # Optimisation plus fréquente
                    kwargs.setdefault('symbol', "BTC/USD")    # Symbole par défaut pour crypto
                    super().__init__(**kwargs)
                    self.name = "Moving Average ML (Crypto)"
                    
                def train(self, symbol: str = None) -> bool:
                    """Méthode d'entraînement adaptée pour les cryptos (données 24/7)"""
                    # Adapter la fenêtre temporelle pour les crypto qui tradent 24/7
                    return super().train(symbol=symbol)
                    
                def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
                    """Analyse des données crypto pour générer des signaux"""
                    # Cette méthode est requise par l'interface BaseStrategy de crypto_trader
                    if data is None or len(data) < max(self.short_window, self.long_window) + 5:
                        return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes"}
                        
                    # Générer le signal avec la méthode get_signal de MovingAverageMLStrategy
                    ml_signal = self.get_signal(self.symbol, data)
                    
                    # Convertir le format du signal pour qu'il soit compatible avec crypto_trader
                    action_mapping = {
                        "BUY": "buy",
                        "SELL": "sell",
                        "HOLD": "neutral"
                    }
                    
                    return {
                        "signal": action_mapping.get(ml_signal["action"].name, "neutral"),
                        "strength": float(ml_signal["confidence"]),
                        "reason": f"ML Signal - Short MA: {ml_signal['params']['short_ma']:.2f}, Long MA: {ml_signal['params']['long_ma']:.2f}"
                    }
                
                def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
                    """Méthode de backtesting pour la stratégie"""
                    # Implémentation simple de backtesting
                    if data is None or len(data) < 100:
                        return {"profit": 0, "trades": 0, "win_rate": 0}
                        
                    # Préparer les données
                    df = data.copy()
                    df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
                    df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
                    df = df.dropna()
                    
                    # Simuler les signaux
                    df['signal'] = 0
                    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1  # Achat
                    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1  # Vente
                    
                    # Calculer les rendements
                    df['returns'] = df['close'].pct_change()
                    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
                    
                    # Métriques
                    cumulative_return = (1 + df['strategy_returns'].fillna(0)).cumprod().iloc[-1] - 1
                    trades = df['signal'].diff().abs().sum() // 2
                    win_rate = 0
                    if trades > 0:
                        wins = ((df['strategy_returns'] > 0).sum())
                        win_rate = wins / trades
                        
                    return {
                        "profit": cumulative_return * 100,  # en pourcentage
                        "trades": int(trades),
                        "win_rate": win_rate * 100  # en pourcentage
                    }
                
                def predict(self, data: pd.DataFrame) -> float:
                    """Prédiction de la direction du marché"""
                    # Utiliser notre modèle pour prédire la direction
                    if self.ml_model is None or data is None or len(data) < 30:
                        return 0.0
                        
                    try:
                        # Préparer les données
                        features_df = self._prepare_features(data)
                        
                        # Extraire les caractéristiques pour la prédiction
                        last_features = features_df.iloc[-1][[                        
                            'ma_diff', 'ma_diff_pct', 'volatility_ratio',
                            'trend_5d', 'trend_10d', 'trend_20d', 'volume_ratio',
                            'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5'
                        ]].values.reshape(1, -1)
                        
                        # Normaliser
                        last_features_scaled = self.scaler.transform(last_features)
                        
                        # Prédire la probabilité de hausse
                        probas = self.ml_model.predict_proba(last_features_scaled)[0]
                        prediction = (probas[1] - 0.5) * 2  # Normaliser entre -1 et 1
                        
                        return prediction
                    except Exception as e:
                        logger.warning(f"Erreur lors de la prédiction: {e}")
                        return 0.0
                
                def load_data(self, symbol: str, interval: str = '1day', limit: int = 200) -> pd.DataFrame:
                    """Charger les données pour un symbole"""
                    # Déléguer au service de données de marché si disponible
                    if self.market_data_service:
                        end = datetime.now()
                        start = end - timedelta(days=limit)  # Utiliser limit comme nombre de jours
                        
                        return self.market_data_service.get_historical_data(
                            symbol=symbol,
                            interval=interval,
                            start=start,
                            end=end
                        )
                    return None
                
                def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
                    """Prétraitement des données pour l'analyse"""
                    # S'assurer que les données existent
                    if data is None or len(data) < 30:
                        return None
                        
                    # Utiliser notre méthode existante
                    return self._prepare_features(data)
            
            return CryptoMovingAverageMLStrategy
        except ImportError:
            logger.warning("Stratégie MovingAverageML non disponible, utilisation de la stratégie par défaut")
            return None
        
    # Import direct de LLMStrategyV2 pour éviter une référence circulaire
    try:
        from app.strategies.llm_strategy_v2 import LLMStrategyV2
        llm_v2_class = LLMStrategyV2
    except ImportError:
        logger.warning("Stratégie LLM_V2 non disponible, utilisation de la stratégie par défaut")
        llm_v2_class = None
        
    # Import de LLMStrategyV3 pour le mapping
    try:
        from app.strategies.llm_strategy_v3 import LLMStrategyV3
        llm_v3_class = LLMStrategyV3
    except ImportError:
        logger.warning("Stratégie LLM_V3 non disponible, utilisation de la stratégie par défaut")
        llm_v3_class = None
        
    strategy_map = {
        StrategyType.MOMENTUM: MomentumStrategy,
        StrategyType.MEAN_REVERSION: MeanReversionStrategy,
        StrategyType.BREAKOUT: BreakoutStrategy,
        StrategyType.STATISTICAL_ARBITRAGE: StatisticalArbitrageStrategy,
        StrategyType.LSTM: LSTMPredictorStrategy, 
        StrategyType.LLM: LLMStrategy,
        StrategyType.LLM_V2: llm_v2_class,
        StrategyType.LLM_V3: llm_v3_class
    }
    return strategy_map.get(strategy_type.lower())

class MarketCondition(Enum):
    NORMAL = auto()
    VOLATILE = auto()
    INACTIVE = auto()
    DANGEROUS = auto()

def parse_arguments():
    """Analyse les arguments de ligne de commande
    
    Returns:
        argparse.Namespace: Les arguments analysés
    """
    parser = argparse.ArgumentParser(description="Lancer le trader crypto avec une stratégie spécifique")
    
    # Arguments pour la stratégie
    parser.add_argument("--strategy", type=str, choices=[s.value for s in StrategyType], 
                      default=StrategyType.MOVING_AVERAGE.value,
                      help="Stratégie de trading à utiliser")
    parser.add_argument("--duration", type=str, 
                      default="night",
                      help="Durée de la session (1h, 4h, 8h, day, night, weekend, continuous)")
    
    # Arguments pour les symboles
    parser.add_argument("--use-custom-symbols", action="store_true", 
                      help="Utiliser la liste personnalisée de symboles au lieu du filtre automatique")
    parser.add_argument("--symbols-file", type=str, 
                      help="Fichier contenant la liste des cryptomonnaies (une par ligne)")
    parser.add_argument("--use-env-symbols", action="store_true", 
                      help="Utiliser les symboles définis dans la variable d'environnement PERSONALIZED_CRYPTO_LIST")
    parser.add_argument("--symbols", type=str, 
                      help="Liste de symboles séparés par des virgules (ex: BTC/USD,ETH/USD)")
    
    # Arguments pour les stratégies spécifiques
    parser.add_argument("--momentum-lookback", type=int, default=20,
                      help="Période de lookback pour la stratégie momentum (default: 20)")
    parser.add_argument("--mean-reversion-lookback", type=int, default=20,
                      help="Période de lookback pour la stratégie mean reversion (default: 20)")
    parser.add_argument("--breakout-lookback", type=int, default=20,
                      help="Période de lookback pour la stratégie breakout (default: 20)")
    parser.add_argument("--volatility-lookback", type=int, default=10,
                      help="Période de lookback pour le calcul de la volatilité (default: 10)")
    
    # Arguments pour le trading
    parser.add_argument("--api-level", type=int, choices=[1, 2, 3], default=0,
                      help="Niveau d'API Alpaca à utiliser (1=basique, 2=standard+, 3=premium). Par défaut: auto-détection)")
    parser.add_argument("--position-size", type=float, default=0.02, 
                      help="Taille de position en pourcentage du portefeuille (default: 0.02 = 2%)")
    parser.add_argument("--stop-loss", type=float, default=0.03, 
                      help="Stop loss en pourcentage (default: 0.03 = 3%)")
    parser.add_argument("--take-profit", type=float, default=0.06, 
                      help="Take profit en pourcentage (default: 0.06 = 6%)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Afficher les informations détaillées")
    
    # Arguments spécifiques à la stratégie moving_average
    parser.add_argument("--fast-ma", type=int, default=5,
                      help="Période de la moyenne mobile rapide en minutes - uniquement pour la stratégie moving_average (default: 5)")
    parser.add_argument("--slow-ma", type=int, default=15,
                      help="Période de la moyenne mobile lente en minutes - uniquement pour la stratégie moving_average (default: 15)")
    
    # Arguments pour les stratégies avancées
    # Transformer
    parser.add_argument("--sequence-length", type=int, default=30, 
                      help="Longueur de séquence pour Transformer")
    parser.add_argument("--prediction-horizon", type=int, default=5, 
                      help="Horizon de prédiction pour Transformer/LSTM")
    parser.add_argument("--d-model", type=int, default=64, 
                      help="Dimension du modèle Transformer")
    parser.add_argument("--nhead", type=int, default=4, 
                      help="Nombre de têtes d'attention pour Transformer")
    parser.add_argument("--num-layers", type=int, default=2, 
                      help="Nombre de couches pour Transformer")
    parser.add_argument("--dropout", type=float, default=0.1, 
                      help="Taux de dropout pour Transformer")
    parser.add_argument("--signal-threshold", type=float, default=0.6, 
                      help="Seuil de signal pour Transformer")
    
    # LSTM
    parser.add_argument("--lstm-units", type=int, default=50, 
                      help="Nombre d'unités LSTM")
    parser.add_argument("--lstm-dropout", type=float, default=0.2, 
                      help="Taux de dropout pour LSTM")
    parser.add_argument("--lstm-epochs", type=int, default=50, 
                      help="Nombre d'époques pour l'entraînement LSTM")
    parser.add_argument("--lstm-batch-size", type=int, default=32, 
                      help="Taille du batch pour l'entraînement LSTM")
    
    # LLM
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo", 
                      help="Nom du modèle LLM à utiliser (pour LLM et LLM_V2)")
    parser.add_argument("--use-local-model", action="store_true", 
                      help="Utiliser un modèle LLM local")
    parser.add_argument("--local-model-path", type=str, 
                      help="Chemin vers le modèle LLM local")
    parser.add_argument("--api-key", type=str, 
                      help="Clé API pour le modèle LLM (si nécessaire)")
    parser.add_argument("--newsapi-key", type=str, 
                      help="Clé API pour NewsAPI.org (si non définie dans .env)")
    parser.add_argument("--sentiment-threshold", type=float, default=0.6, 
                      help="Seuil de sentiment pour la stratégie LLM")
    parser.add_argument("--sentiment-weight", type=float, default=0.5, 
                      help="Poids donné à l'analyse de sentiment vs technique (0-1)")
    parser.add_argument("--min-confidence", type=float, default=0.65, 
                      help="Seuil minimal de confiance pour les signaux de trading")
    parser.add_argument("--news-lookback", type=int, default=24, 
                      help="Période de recherche d'actualités (en heures)")
    
    # LLM_V3 (multi-agent)
    parser.add_argument("--trader-model-name", type=str, default="claude-3-7-sonnet-20250219", 
                      help="Nom du modèle Claude pour l'agent Trader (LLM_V3)")
    parser.add_argument("--analyst-model-name", type=str, default="claude-3-7-sonnet-20250219", 
                      help="Nom du modèle Claude pour l'agent Analyste (LLM_V3)")
    parser.add_argument("--coordinator-model-name", type=str, default="claude-3-7-sonnet-20250219", 
                      help="Nom du modèle Claude pour l'agent Coordinateur (LLM_V3)")
    
    # Options communes
    parser.add_argument("--use-gpu", action="store_true", 
                      help="Utiliser le GPU pour l'entraînement (si disponible)")
    parser.add_argument("--retrain", action="store_true", 
                      help="Réentraîner le modèle avant utilisation")
    
    # Option pour le fournisseur de données de marché
    parser.add_argument("--data-provider", type=str, choices=["alpaca", "yahoo", "binance"], default="binance",
                      help="Fournisseur de données de marché à utiliser (binance recommandé pour crypto)")
    
    return parser.parse_args()


def main():
    """Fonction principale du script"""
    # Analyser les arguments de ligne de commande
    args = parse_arguments()
    # Tous les arguments sont maintenant définis dans la fonction parse_arguments()
    
    # Tous les arguments sont maintenant définis dans la fonction parse_arguments()
    
    # Déterminer la durée de session
    duration_map = {
        "1h": SessionDuration.ONE_HOUR,
        "4h": SessionDuration.FOUR_HOURS,
        "8h": SessionDuration.EIGHT_HOURS,
        "night": SessionDuration.NIGHT_RUN
    }
    
    # Vérifier si c'est une durée personnalisée (format: Xh)
    custom_duration = None
    if args.duration not in duration_map:
        # Vérifier le format (nombre + 'h')
        if args.duration.endswith('h'):
            try:
                # Extraire le nombre d'heures et convertir en secondes
                hours = int(args.duration[:-1])
                custom_duration = hours * 3600
                print(f"Durée personnalisée: {hours} heures ({custom_duration} secondes)")
            except ValueError:
                print(f"Format de durée invalide: {args.duration}, utilisation de la durée par défaut 'night'")
                args.duration = "night"
    
    # Obtenir la durée de session
    if custom_duration:
        session_duration = SessionDuration.CUSTOM
    else:
        session_duration = duration_map.get(args.duration, SessionDuration.NIGHT_RUN)
    
    # Détecter ou utiliser le niveau d'API spécifié
    api_level = args.api_level
    if api_level == 0:  # Auto-détection
        api_level = detect_alpaca_level()
    
    print("=" * 60)
    print(f"DÉMARRAGE DU TRADER CRYPTO AVEC STRATÉGIE - {datetime.now()}")
    print("=" * 60)
    print(f"Stratégie sélectionnée: {args.strategy}")
    print(f"Ce trader va tourner en mode PAPER pendant environ {args.duration}")
    print(f"Position size: {args.position_size * 100}%")
    print(f"Stop-loss: {args.stop_loss * 100}%")
    print(f"Take-profit: {args.take_profit * 100}%")
    print(f"Niveau d'API Alpaca: {api_level if api_level > 0 else 'Non détecté - utilisation du niveau 1'}")
    print(f"Fournisseur de données: {args.data_provider}")
    
    if args.strategy == StrategyType.MOVING_AVERAGE:
        print(f"MA rapide: {args.fast_ma} minutes")
        print(f"MA lente: {args.slow_ma} minutes")
    elif args.strategy == StrategyType.MOMENTUM:
        print(f"Momentum lookback: {args.momentum_lookback} périodes")
    elif args.strategy == StrategyType.MEAN_REVERSION:
        print(f"Mean reversion lookback: {args.mean_reversion_lookback} périodes")
    elif args.strategy == StrategyType.BREAKOUT:
        print(f"Breakout lookback: {args.breakout_lookback} périodes")
    elif args.strategy == StrategyType.STATISTICAL_ARBITRAGE:
        print(f"Volatility lookback: {args.volatility_lookback} périodes")
    elif args.strategy == StrategyType.TRANSFORMER:
        print(f"Transformer configuration:")
        print(f"  - Sequence length: {args.sequence_length}")
        print(f"  - Prediction horizon: {args.prediction_horizon}")
        print(f"  - Model dimension: {args.d_model}")
        print(f"  - Attention heads: {args.nhead}")
        print(f"  - Layers: {args.num_layers}")
        print(f"  - Dropout: {args.dropout}")
        print(f"  - Signal threshold: {args.signal_threshold}")
        print(f"  - GPU: {'Activé' if args.use_gpu else 'Désactivé'}")
        print(f"  - Réentraînement: {'Oui' if args.retrain else 'Non'}")
    elif args.strategy == StrategyType.LSTM:
        print(f"LSTM configuration:")
        print(f"  - Sequence length: {args.sequence_length}")
        print(f"  - Prediction horizon: {args.prediction_horizon}")
        print(f"  - LSTM units: {args.lstm_units}")
        print(f"  - Dropout: {args.lstm_dropout}")
        print(f"  - Epochs: {args.lstm_epochs}")
        print(f"  - Batch size: {args.lstm_batch_size}")
        print(f"  - GPU: {'Activé' if args.use_gpu else 'Désactivé'}")
    elif args.strategy in [StrategyType.LLM.value, StrategyType.LLM_V2.value]:
        print(f"LLM configuration:")
        print(f"  - Model name: {args.model_name}")
        print(f"  - Use local model: {'Oui' if args.use_local_model else 'Non'}")
        if args.use_local_model and args.local_model_path:
            print(f"  - Local model path: {args.local_model_path}")
        print(f"  - Sentiment threshold: {args.sentiment_threshold}")
        print(f"  - Sentiment weight: {args.sentiment_weight}")
        print(f"  - News lookback hours: {args.news_lookback}")
        
        # Vérifier si la clé NewsAPI est disponible
        newsapi_key = args.newsapi_key or os.environ.get("NEWSAPI_KEY")
        if newsapi_key:
            print(f"  - NewsAPI: Configuré")
        else:
            print(f"  - NewsAPI: Non configuré (utilisation du mode fallback)")
            
        # Vérifier si la clé API Anthropic est disponible (pour Claude)
        if "claude" in args.model_name.lower():
            anthropic_api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                print(f"  - Anthropic API: Configuré pour {args.model_name}")
            else:
                print(f"  - Anthropic API: Non configuré (requis pour Claude)")
                logger.warning("La clé API Anthropic est requise pour utiliser Claude. Définissez ANTHROPIC_API_KEY dans .env ou utilisez --api-key.")
        elif "gpt" in args.model_name.lower() or "openai" in args.model_name.lower():
            openai_api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                print(f"  - OpenAI API: Configuré pour {args.model_name}")
            else:
                print(f"  - OpenAI API: Non configuré (requis pour GPT)")
                logger.warning("La clé API OpenAI est requise pour utiliser GPT. Définissez OPENAI_API_KEY dans .env ou utilisez --api-key.")
    
    print("=" * 60)
    
    # Si la stratégie est moving_average, utiliser AlpacaCryptoTrader directement
    if args.strategy == StrategyType.MOVING_AVERAGE:
        # Créer le trader avec la durée de session spécifiée
        trader = AlpacaCryptoTrader(session_duration=session_duration, data_provider=args.data_provider)
        
        # Clarifier l'architecture hybride (données vs trading)
        if args.data_provider != "alpaca":
            print(f"\n[ARCHITECTURE HYBRIDE]")
            print(f"- {args.data_provider.capitalize()} sera utilisé comme source de données de marché (prix, historiques)")
            print(f"- Alpaca sera utilisé uniquement pour l'exécution des trades et la gestion du compte")
            print(f"- Les messages 'API Alpaca' concernent le trading, pas les données de marché\n")
        
        # Configurer les paramètres
        trader.position_size_pct = args.position_size
        trader.stop_loss_pct = args.stop_loss
        trader.take_profit_pct = args.take_profit
        trader.fast_ma_period = args.fast_ma
        trader.slow_ma_period = args.slow_ma
        
        # Configurer le niveau d'API
        if api_level > 0:
            print(f"Configuration du niveau d'API Alpaca: {api_level}")
            trader.subscription_level = api_level
        
        # Utiliser la liste personnalisée de symboles
        trader.custom_symbols = PERSONALIZED_CRYPTO_LIST
        trader.use_custom_symbols = args.use_custom_symbols
        
        # Démarrer le trader avec la stratégie par défaut
        print(f"Démarrage du trader avec la stratégie de moyenne mobile")
        if custom_duration:
            trader.start(custom_duration)
        else:
            trader.start()
    else:
        # Pour les autres stratégies, utiliser une version simplifiée
        # qui fonctionne avec AlpacaCryptoTrader en adaptant les signaux
        strategy_class = get_strategy_class(args.strategy)
        if not strategy_class:
            print(f"Erreur: Stratégie {args.strategy} non disponible")
            return
        
        # Paramètres spécifiques à la stratégie
        strategy_params = {}
        strategy_type = args.strategy
        if strategy_type == StrategyType.MOMENTUM.value:
            strategy_params = {
                "lookback_period": args.momentum_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif strategy_type == StrategyType.MEAN_REVERSION.value:
            strategy_params = {
                "lookback_period": args.mean_reversion_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif strategy_type == StrategyType.BREAKOUT.value:
            strategy_params = {
                "lookback_period": args.breakout_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif strategy_type == StrategyType.STATISTICAL_ARBITRAGE.value:
            strategy_params = {
                "volatility_lookback": args.volatility_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif strategy_type == StrategyType.TRANSFORMER.value:
            strategy_params = {
                "sequence_length": args.sequence_length,
                "prediction_horizon": args.prediction_horizon,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "signal_threshold": args.signal_threshold,
                "use_gpu": args.use_gpu,
                "retrain": args.retrain,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif strategy_type == StrategyType.LSTM.value:
            strategy_params = {
                "sequence_length": args.sequence_length,
                "prediction_horizon": args.prediction_horizon,
                "lstm_units": args.lstm_units,
                "dropout_rate": args.lstm_dropout,
                "epochs": args.lstm_epochs,
                "batch_size": args.lstm_batch_size,
                "use_gpu": args.use_gpu,
                "retrain": args.retrain,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif strategy_type == StrategyType.LLM.value:
            strategy_params = {
                "model_name": args.model_name,
                "use_local_model": args.use_local_model,
                "local_model_path": args.local_model_path,
                "api_key": args.api_key,
                "sentiment_threshold": args.sentiment_threshold,
                "news_lookback_hours": args.news_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif strategy_type == StrategyType.LLM_V2.value:
            strategy_params = {
                "model_name": args.model_name,
                "use_local_model": args.use_local_model,
                "local_model_path": args.local_model_path,
                "api_key": args.api_key,
                "sentiment_threshold": args.sentiment_threshold,
                "sentiment_weight": args.sentiment_weight,
                "min_confidence": args.min_confidence,
                "news_lookback_hours": args.news_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit,
                "technical_weight": 1.0 - args.sentiment_weight  # Poids complémentaire pour l'analyse technique
            }
        
        # Créer l'instance de stratégie
        strategy_instance = strategy_class(**strategy_params)
        print(f"Stratégie {strategy_instance.name} initialisée avec succès")
        print("Utilisation du trader Alpaca de base avec adaptation des signaux")
        
        # Enregistrer la stratégie pour utilisation future
        global STRATEGY_INSTANCE
        STRATEGY_INSTANCE = strategy_instance
        
        # Créer le trader avec la durée de session spécifiée
        trader = AlpacaCryptoTrader(session_duration=session_duration, data_provider=args.data_provider)
        
        # Clarifier l'architecture hybride (données vs trading)
        if args.data_provider != "alpaca":
            print(f"\n[ARCHITECTURE HYBRIDE]")
            print(f"- {args.data_provider.capitalize()} sera utilisé comme source de données de marché (prix, historiques)")
            print(f"- Alpaca sera utilisé uniquement pour l'exécution des trades et la gestion du compte")
            print(f"- Les messages 'API Alpaca' concernent le trading, pas les données de marché\n")
        
        # Configurer le niveau d'API
        if api_level > 0:
            print(f"Configuration du niveau d'API Alpaca: {api_level}")
            trader.subscription_level = api_level
        
        # Configurer les paramètres
        trader.position_size_pct = args.position_size
        trader.stop_loss_pct = args.stop_loss
        trader.take_profit_pct = args.take_profit
        
        # La stratégie personnalisée sera utilisée dans un script séparé
        # qui sera exécuté ultérieurement avec les mêmes paramètres
        strategy_type = args.strategy
        strategy_file = f"custom_strategy_{strategy_type}_params.json"
        
        # Sauvegarder les paramètres dans un fichier pour utilisation future
        with open(strategy_file, "w") as f:
            json.dump({
                "strategy_type": strategy_type,
                "params": strategy_params,
                "symbols": PERSONALIZED_CRYPTO_LIST if args.use_custom_symbols else []
            }, f, indent=2)
        
        print(f"Configuration de stratégie enregistrée dans {strategy_file}")
        
        # Utiliser la liste personnalisée de symboles
        trader.custom_symbols = PERSONALIZED_CRYPTO_LIST
        trader.use_custom_symbols = args.use_custom_symbols
        
        # Démarrer le trader avec la stratégie par défaut adaptée
        print(f"Démarrage du trader avec adaptation pour la stratégie {strategy_type}")
        if custom_duration:
            trader.start(custom_duration)
        else:
            trader.start()
    
    print("=" * 60)
    print("SESSION DE TRADING TERMINÉE")
    print("=" * 60)
    print("Un rapport détaillé a été généré dans le dossier courant")
    print("=" * 60)

def initialize_trader(session_duration, data_provider, api_level=0, position_size=0.02, stop_loss=0.03, take_profit=0.06):
    """
    Initialise le trader avec les paramètres spécifiés et affiche les messages appropriés
    
    Args:
        session_duration: Durée de la session de trading
        data_provider: Fournisseur de données de marché à utiliser
        api_level: Niveau d'API Alpaca (0=auto-détection, 1=basique, 2=standard+, 3=premium)
        position_size: Taille de position en % du portefeuille
        stop_loss: Stop loss en % en dessous du prix d'entrée
        take_profit: Take profit en % au-dessus du prix d'entrée
    
    Returns:
        Instance du trader initialisé
    """
    # Initialiser le trader avec les paramètres spécifiés
    trader = AlpacaCryptoTrader(
        is_paper=True,  # Toujours utiliser le mode paper trading pour les tests
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        data_provider=data_provider,
        api_level=api_level
    )
    
    # Afficher un message explicatif sur l'architecture hybride si Binance est utilisé comme fournisseur de données
    if data_provider == "binance":
        print("\n" + "=" * 80)
        print("ARCHITECTURE HYBRIDE: Binance pour les données de marché, Alpaca pour l'exécution")
        print("Les données de marché (prix, historique) sont récupérées via Binance")
        print("L'exécution des ordres et la gestion du compte sont gérées via Alpaca")
        print("=" * 80 + "\n")
    
    # Stocker l'instance du trader dans la variable globale pour l'arrêt propre
    global trader_instance
    trader_instance = trader
    
    return trader


def main():
    """Fonction principale du script"""
    # Analyser les arguments de ligne de commande
    args = parse_arguments()
    
    # Afficher un message sur le fournisseur de données
    print(f"Fournisseur de données sélectionné: {args.data_provider}")
    if args.data_provider == "binance":
        print("Binance est le fournisseur recommandé pour les données crypto - accès sans API key requis")
    elif args.data_provider == "alpaca":
        print("Note: Alpaca peut renvoyer des erreurs 404 pour certains symboles crypto si vous n'avez pas d'abonnement")
    
    # Chargement des symboles personnalisés si demandé
    global PERSONALIZED_CRYPTO_LIST
    
    # 1. Priorité aux symboles passés en ligne de commande
    if args.symbols:
        symbol_list = [s.strip() for s in args.symbols.split(',')]
        PERSONALIZED_CRYPTO_LIST = symbol_list
        print(f"Utilisation des symboles spécifiés en ligne de commande: {', '.join(symbol_list)}")
    
    # 2. Ensuite, vérifier le fichier de symboles
    elif args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file, 'r') as f:
            symbol_list = [line.strip() for line in f.readlines() if line.strip()]
        PERSONALIZED_CRYPTO_LIST = symbol_list
        print(f"Chargement de {len(symbol_list)} symboles depuis {args.symbols_file}")
    
    # 3. Ensuite, vérifier la variable d'environnement
    elif args.use_env_symbols and os.environ.get('PERSONALIZED_CRYPTO_LIST'):
        symbol_list = os.environ.get('PERSONALIZED_CRYPTO_LIST').split(',')
        PERSONALIZED_CRYPTO_LIST = [s.strip() for s in symbol_list]
        print(f"Utilisation des symboles depuis la variable d'environnement: {', '.join(PERSONALIZED_CRYPTO_LIST)}")
    
    # 4. Si --use-custom-symbols est utilisé sans autre source de symboles, utiliser DEFAULT_CRYPTO_LIST
    elif args.use_custom_symbols and not PERSONALIZED_CRYPTO_LIST:
        PERSONALIZED_CRYPTO_LIST = DEFAULT_CRYPTO_LIST
        print(f"Utilisation de la liste par défaut avec {len(DEFAULT_CRYPTO_LIST)} cryptomonnaies")
    
    # Déterminer la durée de la session
    session_duration = args.duration
    
    # Configurer la stratégie en fonction des arguments
    strategy_type = args.strategy
    strategy_params = {}
    
    # Paramètres spécifiques à chaque stratégie
    if strategy_type == StrategyType.MOVING_AVERAGE.value:
        strategy_params = {
            "fast_ma": args.fast_ma,
            "slow_ma": args.slow_ma
        }
    elif strategy_type == StrategyType.TRANSFORMER.value:
        strategy_params = {
            "sequence_length": args.sequence_length,
            "prediction_horizon": args.prediction_horizon,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "signal_threshold": args.signal_threshold,
            "use_gpu": args.use_gpu,
            "retrain": args.retrain
        }
    elif strategy_type == StrategyType.LSTM.value:
        strategy_params = {
            "lstm_units": args.lstm_units,
            "lstm_dropout": args.lstm_dropout,
            "lstm_epochs": args.lstm_epochs,
            "lstm_batch_size": args.lstm_batch_size,
            "prediction_horizon": args.prediction_horizon,
            "use_gpu": args.use_gpu,
            "retrain": args.retrain
        }
    elif args.strategy in [StrategyType.LLM.value, StrategyType.LLM_V2.value]:
        # Récupérer la clé NewsAPI depuis les arguments ou l'environnement
        newsapi_key = args.newsapi_key or os.environ.get("NEWSAPI_KEY")
        
        strategy_params = {
            "model_name": args.model_name,
            "use_local_model": args.use_local_model,
            "local_model_path": args.local_model_path,
            "api_key": args.api_key,
            "sentiment_threshold": args.sentiment_threshold,
            "sentiment_weight": args.sentiment_weight,
            "min_confidence": args.min_confidence,
            "news_lookback_hours": args.news_lookback  # Assurez-vous que le nom correspond au paramètre dans LLMStrategyV2
        }
        
        # Ajouter la clé NewsAPI si disponible
        if newsapi_key:
            strategy_params["newsapi_key"] = newsapi_key
    
    elif args.strategy == StrategyType.LLM_V3.value:
        # Récupérer la clé NewsAPI depuis les arguments ou l'environnement
        newsapi_key = args.newsapi_key or os.environ.get("NEWSAPI_KEY")
        
        # Paramètres spécifiques à la stratégie multi-agent LLM_V3
        strategy_params = {
            "trader_model_name": args.trader_model_name,
            "analyst_model_name": args.analyst_model_name,
            "coordinator_model_name": args.coordinator_model_name,
            "api_key": args.api_key,
            "use_local_model": args.use_local_model,
            "local_model_path": args.local_model_path,
            "sentiment_weight": args.sentiment_weight,
            "min_confidence": args.min_confidence,
            "news_lookback_hours": args.news_lookback,
            "position_size": args.position_size,
            "stop_loss": args.stop_loss,
            "take_profit": args.take_profit,
            "data_provider": args.data_provider
        }
        
        # Ajouter la clé NewsAPI si disponible
        if newsapi_key:
            strategy_params["newsapi_key"] = newsapi_key
    
    # Initialiser le trader avec les paramètres spécifiés et la stratégie configurée
    trader = initialize_trader(
        api_level=args.api_level,
        data_provider=args.data_provider,
        position_size=args.position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        strategy_type=strategy_type,
        strategy_params=strategy_params
    )
    
    # Utiliser la liste personnalisée de symboles si spécifiée
    if args.use_custom_symbols:
        # Si PERSONALIZED_CRYPTO_LIST est vide, utiliser DEFAULT_CRYPTO_LIST
        if not PERSONALIZED_CRYPTO_LIST:
            PERSONALIZED_CRYPTO_LIST = DEFAULT_CRYPTO_LIST
            print(f"Utilisation de la liste par défaut avec {len(DEFAULT_CRYPTO_LIST)} cryptomonnaies")
        trader.custom_symbols = PERSONALIZED_CRYPTO_LIST
        trader.use_custom_symbols = True
    elif PERSONALIZED_CRYPTO_LIST:
        trader.custom_symbols = PERSONALIZED_CRYPTO_LIST
        trader.use_custom_symbols = True
    
    # Enregistrer la configuration de stratégie si nécessaire
    strategy_file = f"custom_strategy_{strategy_type}_params.json"
    with open(strategy_file, "w") as f:
        json.dump({
            "strategy_type": strategy_type,
            "params": strategy_params,
            "symbols": PERSONALIZED_CRYPTO_LIST if PERSONALIZED_CRYPTO_LIST else []
        }, f, indent=2)
    print(f"Configuration de stratégie enregistrée dans {strategy_file}")
    
    # Convertir la durée de session en secondes pour la méthode start()
    duration_seconds = None
    if isinstance(session_duration, int):
        duration_seconds = session_duration
    elif isinstance(session_duration, str):
        if session_duration.endswith('h'):
            try:
                hours = float(session_duration[:-1])
                duration_seconds = int(hours * 3600)  # Conversion en secondes
            except ValueError:
                print(f"Format de durée invalide: {session_duration}, utilisation de la durée par défaut")
        elif session_duration == "night":
            # Session de nuit (8 heures par défaut)
            duration_seconds = 8 * 3600
    
    # Démarrer le trader avec la stratégie configurée
    print(f"Démarrage du trader avec la stratégie {strategy_type} pour {duration_seconds//3600 if duration_seconds else 'une durée indéterminée'} heures")
    trader.start(duration_seconds)
    
    print("=" * 60)
    print("SESSION DE TRADING TERMINÉE")
    print("=" * 60)
    print("Un rapport détaillé a été généré dans le dossier courant")
    print("=" * 60)


def cleanup_resources():
    """Nettoie les ressources avant de quitter"""
    logger.info("Nettoyage des ressources avant de quitter...")
    # Fermer les connexions, sauvegarder les données, etc.
    # Cette fonction peut être étendue selon les besoins
    
    # Afficher un message de confirmation
    logger.info("Nettoyage terminé. Au revoir!")


def cleanup_handler(sig, frame):
    """Gestionnaire de signal pour un arrêt propre"""
    logger.info(f"Signal {sig} reçu. Arrêt propre...")
    cleanup_resources()
    sys.exit(0)


def run_crypto_trader():
    """Point d'entrée principal du script, gère les signaux et les exceptions"""
    # Enregistrer les gestionnaires de nettoyage pour une sortie propre
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruption clavier détectée. Arrêt propre...")
        cleanup_resources()
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        traceback.print_exc()
        cleanup_resources()
        sys.exit(1)
    finally:
        cleanup_resources()

# Variable globale pour stocker l'instance du trader
trader_instance = None


# Fonction pour générer un rapport final et nettoyer
def cleanup_resources():
    """Nettoyer les ressources et générer le rapport final avant de quitter"""
    global trader_instance
    
    logger.info("Nettoyage des ressources et finalisation du rapport...")
    
    # Arrêter proprement le trader s'il a été initialisé
    if trader_instance is not None:
        try:
            logger.info("Arrêt propre du trader...")
            trader_instance.stop()
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du trader: {e}")

    logger.info("Rapport généré et ressources nettoyées")


def initialize_trader(api_level=0, data_provider="binance", position_size=0.02, stop_loss=0.03, take_profit=0.06, strategy_type=None, strategy_params=None):
    """Initialise le trader avec les paramètres spécifiés
    
    Args:
        api_level (int): Niveau d'API Alpaca à utiliser
        data_provider (str): Fournisseur de données à utiliser
        position_size (float): Taille de position en pourcentage du portefeuille
        stop_loss (float): Stop loss en pourcentage
        take_profit (float): Take profit en pourcentage
        strategy_type (str): Type de stratégie à utiliser
        strategy_params (dict): Paramètres de la stratégie
        
    Returns:
        AlpacaCryptoTrader: Le trader initialisé
    """
    # Initialiser le trader
    trader = AlpacaCryptoTrader(data_provider=data_provider)
    
    # Afficher des informations sur le fournisseur de données
    print(f"Fournisseur de données sélectionné: {data_provider}")
    if data_provider == "binance":
        print("Binance est le fournisseur recommandé pour les données crypto - accès sans API key requis")
    
    # Afficher un message clair sur l'architecture hybride
    if data_provider != "alpaca":
        print(f"\n[ARCHITECTURE HYBRIDE]")
        print(f"- {data_provider.capitalize()} sera utilisé comme source de données de marché (prix, historiques)")
        print(f"- Alpaca sera utilisé uniquement pour l'exécution des trades et la gestion du compte")
        print(f"- Les messages 'API Alpaca' concernent le trading, pas les données de marché\n")
    
    # Configurer le niveau d'API
    if api_level > 0:
        print(f"Configuration du niveau d'API Alpaca: {api_level}")
        trader.subscription_level = api_level

    # Configurer les paramètres de trading
    trader.position_size_pct = position_size
    trader.stop_loss_pct = stop_loss
    trader.take_profit_pct = take_profit
    
    # Configurer la stratégie si spécifiée
    if strategy_type:
        # Stocker les informations de stratégie dans le trader
        trader.strategy_type = strategy_type
        trader.strategy_params = strategy_params or {}
        
        # Afficher les informations sur la stratégie sélectionnée
        print(f"\nStratégie configurée: {strategy_type}")
        if strategy_params:
            print(f"Paramètres de stratégie: {strategy_params}")

    return trader


# Point d'entrée principal du script

if __name__ == "__main__":
    run_crypto_trader()
