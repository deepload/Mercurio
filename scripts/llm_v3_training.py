#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement et d'optimisation pour la stratégie LLM_V3
Permet d'exécuter l'apprentissage en boucle fermée pendant les périodes d'inactivité du marché
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.strategies.llm_strategy_v3 import LLMStrategyV3

# Configuration du logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier JSON
    
    Args:
        config_file: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return {}

def save_training_report(report: Dict[str, Any], output_file: str) -> None:
    """
    Sauvegarde le rapport d'entraînement dans un fichier JSON
    
    Args:
        report: Rapport d'entraînement
        output_file: Chemin du fichier de sortie
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Rapport d'entraînement sauvegardé dans {output_file}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du rapport: {e}")

def main():
    """
    Point d'entrée principal du script
    """
    parser = argparse.ArgumentParser(description="Entraînement et optimisation de la stratégie LLM_V3")
    parser.add_argument("--config", type=str, default="../custom_strategy_llm_v3_params.json", 
                        help="Chemin vers le fichier de configuration")
    parser.add_argument("--symbols", type=str, nargs="+", 
                        help="Liste des symboles à analyser (par défaut: tous)")
    parser.add_argument("--output", type=str, default="../reports/llm_v3_training_report.json",
                        help="Chemin du fichier de sortie pour le rapport")
    parser.add_argument("--days", type=int, default=30,
                        help="Nombre de jours à analyser")
    parser.add_argument("--export-stats", action="store_true",
                        help="Exporter les statistiques de performance")
    parser.add_argument("--stats-output", type=str, default="../reports/llm_v3_performance_stats.json",
                        help="Chemin du fichier de sortie pour les statistiques")
    
    args = parser.parse_args()
    
    try:
        # Charger la configuration
        config = load_config(args.config)
        
        if not config:
            logger.error("Configuration invalide ou manquante")
            return 1
        
        # Initialiser la stratégie LLM_V3
        strategy = LLMStrategyV3(**config)
        
        # Exécuter l'entraînement et l'optimisation
        start_time = datetime.now()
        logger.info(f"Démarrage de l'entraînement à {start_time.isoformat()}")
        
        training_results = strategy.train_and_optimize(symbols=args.symbols)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Ajouter des métadonnées au rapport
        training_results["metadata"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "config_file": args.config,
            "symbols": args.symbols
        }
        
        # Sauvegarder le rapport
        save_training_report(training_results, args.output)
        
        # Exporter les statistiques si demandé
        if args.export_stats:
            stats = strategy.export_performance_stats(output_path=args.stats_output, days=args.days)
            logger.info(f"Statistiques exportées: {len(stats.get('symbols', {}))} symboles analysés")
        
        logger.info(f"Entraînement terminé en {duration:.2f} secondes")
        logger.info(f"Symboles traités: {training_results.get('symbols_processed', 0)}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
