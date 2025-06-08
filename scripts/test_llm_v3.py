#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour la stratégie LLM_V3
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Créer le dossier logs s'il n'existe pas
os.makedirs("logs", exist_ok=True)

def main():
    try:
        # Import de la classe LLMStrategyV3
        from app.strategies.llm_strategy_v3 import LLMStrategyV3
        logger.info("Import de LLMStrategyV3 réussi")
        
        # Paramètres minimaux pour l'initialisation
        params = {
            "trader_model_name": "claude-3-5-sonnet-20240620",
            "analyst_model_name": "claude-3-5-sonnet-20240620",
            "coordinator_model_name": "claude-3-5-sonnet-20240620",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "newsapi_key": os.getenv("NEWSAPI_KEY"),
            "sentiment_weight": 0.6,
            "min_confidence": 0.65,
            "news_lookback_hours": 2,
            "position_size": 0.02,
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "data_provider": "binance",
            "use_local_model": False
        }
        
        logger.info(f"Paramètres: {json.dumps(params, indent=2)}")
        
        # Initialisation de la stratégie
        logger.info("Tentative d'initialisation de LLMStrategyV3...")
        strategy = LLMStrategyV3(**params)
        logger.info(f"Stratégie {strategy.name} initialisée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        with open("logs/test_error.log", "w") as f:
            f.write(traceback.format_exc())
        logger.error(f"Traceback écrit dans logs/test_error.log")

if __name__ == "__main__":
    main()
