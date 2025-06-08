#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de trading crypto utilisant la stratégie LLM_V3 avec agents Claude
Ce script charge la stratégie LLM_V3, l'assigne au trader Alpaca et exécute la boucle de trading
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from enum import Enum, auto

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports des modules du projet
from app.strategies.llm_strategy_v3 import LLMStrategyV3
from alpaca_crypto_trader import AlpacaCryptoTrader, SessionDuration
from dotenv import load_dotenv

# Configuration du logger spécifique pour ce script
log_file = f"logs/llm_v3_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Créer le répertoire logs s'il n'existe pas
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_v3_trader")

# Liste personnalisée de cryptos pour le trading
PERSONALIZED_CRYPTO_LIST = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
    "MATIC/USD", "DOT/USD", "ADA/USD", "XRP/USD", "DOGE/USD"
]

def parse_arguments():
    """Analyser les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Trader crypto avec stratégie LLM_V3")
    
    # Paramètres de session
    parser.add_argument("--duration", type=str, default="1h", 
                        choices=["1h", "4h", "8h", "night", "custom"],
                        help="Durée de la session de trading")
    parser.add_argument("--custom-duration", type=int, default=3600,
                        help="Durée personnalisée en secondes (si --duration=custom)")
    
    # Paramètres de trading
    parser.add_argument("--position-size", type=float, default=0.02,
                        help="Taille de position en pourcentage du portefeuille")
    parser.add_argument("--stop-loss", type=float, default=0.03,
                        help="Pourcentage de stop loss")
    parser.add_argument("--take-profit", type=float, default=0.06,
                        help="Pourcentage de take profit")
    
    # Paramètres de données
    parser.add_argument("--data-provider", type=str, default="binance",
                        choices=["alpaca", "binance", "yahoo"],
                        help="Fournisseur de données de marché")
    parser.add_argument("--use-custom-symbols", action="store_true",
                        help="Utiliser la liste personnalisée de symboles")
    
    # Paramètres API
    parser.add_argument("--api-level", type=int, default=1,
                        help="Niveau d'API Alpaca (1=standard, 2=premium)")
    
    # Paramètres spécifiques à LLM_V3
    parser.add_argument("--model-name", type=str, default="claude-3-opus-20240229",
                        help="Nom du modèle LLM à utiliser")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Clé API pour le modèle LLM (si non définie dans .env)")
    parser.add_argument("--debug", action="store_true",
                        help="Activer le mode debug avec logs détaillés")
    
    return parser.parse_args()

def main():
    """Fonction principale du script"""
    # Analyser les arguments
    args = parse_arguments()
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Configurer le niveau de log
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Mode DEBUG activé")
    
    # Déterminer la durée de session
    if args.duration == "1h":
        session_duration = SessionDuration.ONE_HOUR
    elif args.duration == "4h":
        session_duration = SessionDuration.FOUR_HOURS
    elif args.duration == "8h":
        session_duration = SessionDuration.EIGHT_HOURS
    elif args.duration == "night":
        session_duration = SessionDuration.NIGHT_RUN
    else:  # custom
        session_duration = args.custom_duration
    
    # Charger les paramètres de la stratégie depuis le fichier JSON
    strategy_file = "custom_strategy_llm_v3_params.json"
    try:
        with open(strategy_file, "r") as f:
            strategy_config = json.load(f)
            strategy_params = strategy_config.get("params", {})
            custom_symbols = strategy_config.get("symbols", [])
            
            logger.info(f"Paramètres de stratégie chargés depuis {strategy_file}")
    except FileNotFoundError:
        logger.warning(f"Fichier de configuration {strategy_file} non trouvé, utilisation des paramètres par défaut")
        strategy_params = {
            "model_name": args.model_name,
            "api_key": args.api_key,
            "position_size": args.position_size,
            "stop_loss": args.stop_loss,
            "take_profit": args.take_profit
        }
        custom_symbols = []
    
    # Créer l'instance de stratégie LLM_V3
    try:
        # Vérifier que la classe LLMStrategyV3 est correctement importée
        logger.info(f"Vérification de la classe LLMStrategyV3: {LLMStrategyV3}")
        
        # Adapter les paramètres au format attendu par LLMStrategyV3
        adapted_params = {}
        
        # Paramètres des modèles
        model_name = strategy_params.get('model_name', 'claude-3-5-sonnet-20240620')
        adapted_params['trader_model_name'] = strategy_params.get('trader_model_name', model_name)
        adapted_params['analyst_model_name'] = strategy_params.get('analyst_model_name', model_name)
        adapted_params['coordinator_model_name'] = strategy_params.get('coordinator_model_name', model_name)
        
        # Autres paramètres
        adapted_params['api_key'] = strategy_params.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        adapted_params['use_local_model'] = strategy_params.get('use_local_model', False)
        adapted_params['local_model_path'] = strategy_params.get('local_model_path')
        adapted_params['sentiment_weight'] = float(strategy_params.get('sentiment_weight', 0.6))
        adapted_params['min_confidence'] = float(strategy_params.get('min_confidence', 0.65))
        adapted_params['news_lookback_hours'] = int(strategy_params.get('news_lookback_hours', 2))
        adapted_params['position_size'] = float(strategy_params.get('position_size', 0.02))
        adapted_params['stop_loss'] = float(strategy_params.get('stop_loss', 0.03))
        adapted_params['take_profit'] = float(strategy_params.get('take_profit', 0.06))
        adapted_params['data_provider'] = strategy_params.get('data_provider', 'binance')
        adapted_params['newsapi_key'] = strategy_params.get('newsapi_key')
        
        # Afficher les paramètres adaptés
        logger.info(f"Paramètres adaptés: {json.dumps(adapted_params, indent=2)}")
        
        # Vérifier les clés API
        if adapted_params['api_key']:
            logger.info("Clé API Anthropic trouvée")
        else:
            logger.warning("Aucune clé API Anthropic trouvée")
            
        if adapted_params['newsapi_key']:
            logger.info("Clé API NewsAPI trouvée")
        else:
            logger.warning("Aucune clé API NewsAPI trouvée")
            
        # Initialiser la stratégie avec les paramètres adaptés
        logger.info("Tentative d'initialisation de LLMStrategyV3...")
        strategy_instance = LLMStrategyV3(**adapted_params)
        logger.info(f"Stratégie {strategy_instance.name} initialisée avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la stratégie: {e}")
        import traceback
        logger.error(f"Traceback complet: {traceback.format_exc()}")
        # Écrire le traceback dans un fichier séparé pour une analyse plus facile
        with open("logs/error_trace.log", "w") as f:
            f.write(traceback.format_exc())
        return
    
    # Créer le trader avec la durée de session spécifiée
    trader = AlpacaCryptoTrader(session_duration=session_duration, data_provider=args.data_provider)
    
    # Assigner la stratégie au trader
    trader.set_strategy(strategy_instance)
    logger.info(f"Stratégie {strategy_instance.__class__.__name__} assignée au trader")
    
    # Configurer les paramètres du trader
    trader.position_size_pct = args.position_size
    trader.stop_loss_pct = args.stop_loss
    trader.take_profit_pct = args.take_profit
    
    # Configurer les symboles personnalisés si nécessaire
    if args.use_custom_symbols:
        trader.use_custom_symbols = True
        trader.custom_symbols = custom_symbols if custom_symbols else PERSONALIZED_CRYPTO_LIST
        logger.info(f"Utilisation de {len(trader.custom_symbols)} symboles personnalisés")
    
    # Configurer le niveau d'API
    if args.api_level > 0:
        logger.info(f"Configuration du niveau d'API Alpaca: {args.api_level}")
        trader.subscription_level = args.api_level
    
    # Démarrer le trader
    logger.info("Démarrage du trader LLM_V3...")
    try:
        trader.start()
    except KeyboardInterrupt:
        logger.info("Interruption manuelle détectée, arrêt propre...")
        trader.stop()
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du trader: {e}")
        trader.stop()
    
    logger.info("Session de trading terminée")

if __name__ == "__main__":
    main()
