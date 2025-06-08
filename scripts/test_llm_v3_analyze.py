"""
Script de test pour isoler et tester la méthode analyze de LLMStrategyV3
"""

import os
import sys
import logging
import pandas as pd
import asyncio
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Créer le dossier logs s'il n'existe pas
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
logger.info(f"Dossier de logs vérifié: {log_dir}")

# Vérifier que le fichier de log existe ou peut être créé
log_file = os.path.join(log_dir, 'claude_prompts.log')
try:
    with open(log_file, 'a') as f:
        pass
    logger.info(f"Fichier de log vérifié: {log_file}")
except Exception as e:
    logger.error(f"Erreur lors de la vérification du fichier de log: {e}")

# Importer la stratégie LLMStrategyV3
try:
    from app.strategies.llm_strategy_v3 import LLMStrategyV3
    logger.info("Import de LLMStrategyV3 réussi")
except Exception as e:
    logger.error(f"Erreur lors de l'import de LLMStrategyV3: {e}")
    sys.exit(1)

async def main():
    """Fonction principale pour tester la méthode analyze de LLMStrategyV3"""
    try:
        # Vérifier si le répertoire logs existe
        if not os.path.exists('logs'):
            os.makedirs('logs')
            logger.info("Répertoire logs créé")
        
        # Vérifier si le fichier de log existe
        log_file = 'logs/claude_prompts.log'
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("[{}] - {} - ===== INITIALISATION DU LOGGER CLAUDE PROMPTS =====\n".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ""
                ))
                f.write("Le logger est prêt à enregistrer les prompts et réponses Claude\n===========\n")
            logger.info(f"Fichier de log {log_file} créé")
        else:
            logger.info(f"Fichier de log {log_file} existe déjà")
            
        # Vérifier les clés API
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        newsapi_key = os.environ.get("NEWSAPI_KEY")
        
        if not anthropic_key:
            logger.error("ANTHROPIC_API_KEY non définie dans les variables d'environnement")
        else:
            logger.info("ANTHROPIC_API_KEY trouvée")
            
        if not newsapi_key:
            logger.error("NEWSAPI_KEY non définie dans les variables d'environnement")
        else:
            logger.info("NEWSAPI_KEY trouvée")
            
        # Paramètres de test
        params = {
            "trader_model_name": "claude-3-7-sonnet-20240620",
            "analyst_model_name": "claude-3-7-sonnet-20240620",
            "coordinator_model_name": "claude-3-7-sonnet-20240620",
            "api_key": anthropic_key,
            "newsapi_key": newsapi_key,
            "sentiment_weight": 0.6,
            "min_confidence": 0.65,
            "news_lookback_hours": 2,
            "position_size": 0.02,
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "data_provider": "binance",
            "use_local_model": False
        }
        
        logger.info(f"Paramètres: {params}")
        
        # Initialiser la stratégie
        logger.info("Tentative d'initialisation de LLMStrategyV3...")
        strategy = LLMStrategyV3(**params)
        logger.info("Stratégie LLMStrategyV3 initialisée avec succès")
        
        # Créer des données de test
        logger.info("Création de données de test...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        # Créer un DataFrame avec des données de test
        data = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(len(dates))],
            'high': [101 + i * 0.1 for i in range(len(dates))],
            'low': [99 + i * 0.1 for i in range(len(dates))],
            'close': [100.5 + i * 0.1 for i in range(len(dates))],
            'volume': [1000 + i * 10 for i in range(len(dates))]
        }, index=dates)
        
        # Ajouter la colonne symbol
        data['symbol'] = 'BTC/USD'
        
        logger.info(f"Données de test créées: {len(data)} lignes")
        
        # Appeler la méthode analyze
        logger.info("Appel de la méthode analyze...")
        
        # Vérifier que la méthode analyze est bien asynchrone
        import inspect
        if inspect.iscoroutinefunction(strategy.analyze):
            logger.info("La méthode analyze est bien asynchrone")
        else:
            logger.warning("La méthode analyze n'est PAS asynchrone!")
        
        # Vérifier le contenu du fichier de log avant l'appel
        log_file = 'logs/claude_prompts.log'
        with open(log_file, 'r') as f:
            log_content_before = f.read()
            logger.debug(f"Contenu du fichier de log avant l'appel:\n{log_content_before[-500:] if len(log_content_before) > 500 else log_content_before}")
        
        # Appeler la méthode analyze
        try:
            # La méthode analyze n'est pas asynchrone, donc pas besoin de await
            result = strategy.analyze(data, symbol='BTC/USD')
            logger.info("Appel à analyze terminé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à analyze: {e}")
            traceback.print_exc()
            result = None
        
        # Vérifier le contenu du fichier de log après l'appel
        try:
            with open(log_file, 'r') as f:
                log_content_after = f.read()
                new_content = log_content_after[len(log_content_before):]
                logger.debug(f"Nouveau contenu du fichier de log après l'appel:\n{new_content}")
                
                if len(new_content) > 0:
                    logger.info("De nouveaux logs ont été ajoutés pendant l'appel à analyze")
                else:
                    logger.warning("Aucun nouveau log n'a été ajouté pendant l'appel à analyze")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier de log: {e}")
        
        # Afficher le résultat
        if result:
            logger.info(f"Résultat de l'analyse: {result}")
        else:
            logger.warning("Pas de résultat d'analyse disponible")
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Charger les variables d'environnement depuis .env si disponible
    try:
        from dotenv import load_dotenv
        logger.info("Loading environment variables from .env file")
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env loading")
    
    # Exécuter la fonction principale de façon asynchrone
    asyncio.run(main())
