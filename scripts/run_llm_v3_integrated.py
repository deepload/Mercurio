#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script intégré pour la stratégie LLM_V3 combinant le trading actif et l'entraînement
Alterne automatiquement entre:
1. Trading pendant les heures de marché (utilisant run_strategy_crypto_trader.py)
2. Entraînement des modèles pendant les périodes d'inactivité (utilisant llm_v3_training.py)
"""

import os
import sys
import argparse
import logging
import json
import time
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def is_market_active() -> bool:
    """
    Détermine si le marché crypto est actif
    Pour les cryptos, le marché est toujours actif, mais nous pouvons définir
    des périodes de faible activité pour l'entraînement
    
    Returns:
        True si le marché est considéré comme actif, False sinon
    """
    # Pour les cryptos, nous considérons que le marché est moins actif entre 2h et 5h UTC
    # C'est une période où les volumes sont généralement plus faibles
    now = datetime.utcnow()
    hour = now.hour
    
    # Période de faible activité: entre 2h et 5h UTC
    if 2 <= hour < 5:
        return False
    
    return True

def run_trading(config_file: str, symbols: List[str], duration: str = "1h") -> None:
    """
    Exécute le trading avec la stratégie LLM_V3
    
    Args:
        config_file: Chemin vers le fichier de configuration
        symbols: Liste des symboles à trader
        duration: Durée du trading
    """
    logger.info(f"Démarrage du trading pour {len(symbols)} symboles")
    
    # Charger la configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            strategy_params = config.get("params", {})
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        strategy_params = {}
    
    # Construire la commande avec le chemin absolu vers le script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_strategy_crypto_trader.py")
    cmd = [
        sys.executable,
        script_path,
        "--strategy", "llm_v3",
        "--duration", duration
    ]
    
    # Ajouter les symboles (en format CSV pour run_strategy_crypto_trader.py)
    if symbols:
        cmd.extend(["--symbols", ",".join(symbols)])
        
    # Ajouter les paramètres spécifiques à LLM_V3 depuis le fichier de configuration
    if "trader_model_name" in strategy_params:
        cmd.extend(["--trader-model-name", strategy_params["trader_model_name"]])
    if "analyst_model_name" in strategy_params:
        cmd.extend(["--analyst-model-name", strategy_params["analyst_model_name"]])
    if "coordinator_model_name" in strategy_params:
        cmd.extend(["--coordinator-model-name", strategy_params["coordinator_model_name"]])
    if "news_lookback_hours" in strategy_params:
        cmd.extend(["--news-lookback", str(strategy_params["news_lookback_hours"])])
    if "min_confidence" in strategy_params:
        cmd.extend(["--min-confidence", str(strategy_params["min_confidence"])])
    
    # Exécuter la commande
    try:
        subprocess.run(cmd, check=True)
        logger.info("Session de trading terminée avec succès")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution du trading: {e}")

def run_training(config_file: str, symbols: List[str]) -> None:
    """
    Exécute l'entraînement de la stratégie LLM_V3
    
    Args:
        config_file: Chemin vers le fichier de configuration
        symbols: Liste des symboles à analyser
    """
    logger.info(f"Démarrage de l'entraînement pour {len(symbols)} symboles")
    
    # Construire la commande avec le chemin absolu vers le script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_v3_training.py")
    cmd = [
        sys.executable,
        script_path,
        "--config", config_file,
        "--export-stats"
    ]
    
    # Ajouter les symboles (en une seule fois avec nargs=+)
    if symbols:
        cmd.append("--symbols")
        cmd.extend(symbols)
    
    # Exécuter la commande
    try:
        subprocess.run(cmd, check=True)
        logger.info("Session d'entraînement terminée avec succès")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de l'entraînement: {e}")

def load_symbols(config_file: str) -> List[str]:
    """
    Charge la liste des symboles depuis le fichier de configuration
    
    Args:
        config_file: Chemin vers le fichier de configuration
        
    Returns:
        Liste des symboles
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Récupérer les symboles depuis la configuration
        symbols = config.get("symbols", [])
        
        if not symbols:
            logger.warning("Aucun symbole trouvé dans la configuration")
            return ["BTC/USDT", "ETH/USDT"]  # Symboles par défaut
            
        return symbols
    except Exception as e:
        logger.error(f"Erreur lors du chargement des symboles: {e}")
        return ["BTC/USDT", "ETH/USDT"]  # Symboles par défaut en cas d'erreur

def main():
    """
    Point d'entrée principal du script
    """
    parser = argparse.ArgumentParser(description="Trading et entraînement intégrés pour la stratégie LLM_V3")
    default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_strategy_llm_v3_params.json")
    parser.add_argument("--config", type=str, default=default_config_path, 
                        help="Chemin vers le fichier de configuration")
    parser.add_argument("--symbols", type=str, nargs="+", 
                        help="Liste des symboles à trader (par défaut: depuis la configuration)")
    parser.add_argument("--duration", type=str, default="continuous",
                        help="Durée d'exécution (continuous, 1h, 4h, etc.)")
    parser.add_argument("--trading-interval", type=str, default="1h",
                        help="Intervalle de trading pendant les heures actives")
    parser.add_argument("--training-interval", type=int, default=1,
                        help="Intervalle d'entraînement en heures pendant les périodes d'inactivité")
    parser.add_argument("--no-training", action="store_true",
                        help="Désactiver l'entraînement automatique")
    
    args = parser.parse_args()
    
    # Charger les symboles
    symbols = args.symbols if args.symbols else load_symbols(args.config)
    
    logger.info(f"Démarrage du système intégré LLM_V3 pour {len(symbols)} symboles")
    logger.info(f"Symboles: {', '.join(symbols)}")
    
    # Boucle principale
    start_time = datetime.now()
    
    try:
        while True:
            current_time = datetime.now()
            
            # Vérifier si la durée totale est écoulée (sauf si continue)
            if args.duration != "continuous":
                # Extraire la valeur numérique et l'unité
                import re
                match = re.match(r"(\d+)([hd])", args.duration)
                if match:
                    value, unit = match.groups()
                    value = int(value)
                    
                    # Calculer la durée en heures
                    if unit == "h":
                        duration_hours = value
                    elif unit == "d":
                        duration_hours = value * 24
                    else:
                        duration_hours = 1
                    
                    # Vérifier si la durée est écoulée
                    elapsed_hours = (current_time - start_time).total_seconds() / 3600
                    if elapsed_hours >= duration_hours:
                        logger.info(f"Durée d'exécution écoulée ({args.duration})")
                        break
            
            # Vérifier si le marché est actif
            if is_market_active():
                logger.info("Marché actif: exécution du trading")
                run_trading(args.config, symbols, args.trading_interval)
            elif not args.no_training:
                logger.info("Marché inactif: exécution de l'entraînement")
                run_training(args.config, symbols)
                
                # Attendre l'intervalle d'entraînement avant la prochaine vérification
                logger.info(f"Attente de {args.training_interval} heure(s) avant la prochaine vérification")
                time.sleep(args.training_interval * 3600)
            else:
                logger.info("Marché inactif et entraînement désactivé: attente")
                time.sleep(300)  # Attendre 5 minutes
    
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur: arrêt du système")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
        return 1
    
    logger.info("Système intégré LLM_V3 arrêté")
    return 0

if __name__ == "__main__":
    sys.exit(main())
