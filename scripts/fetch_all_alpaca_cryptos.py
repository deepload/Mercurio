#!/usr/bin/env python
"""
Script simplifié pour récupérer toutes les cryptomonnaies disponibles sur Alpaca.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Charger les variables d'environnement
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / '.env')

def get_all_alpaca_cryptos():
    """Récupère toutes les cryptomonnaies disponibles sur Alpaca"""
    try:
        # Récupérer les clés API
        # Support both legacy and new env variable names
        api_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET') or os.getenv('ALPACA_SECRET')

        if not api_key or not api_secret:
            logger.error("Clés API Alpaca non définies dans le fichier .env (ALPACA_API_KEY/ALPACA_KEY et ALPACA_API_SECRET/ALPACA_SECRET)")
            return []
        elif os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_API_SECRET'):
            logger.info("Using ALPACA_API_KEY and ALPACA_API_SECRET from .env")
        else:
            logger.info("Using ALPACA_KEY and ALPACA_SECRET from .env")
        
        # Créer une connexion à l'API Alpaca
        logger.info("Connexion à l'API Alpaca...")
        api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')
        
        # Récupérer tous les assets disponibles
        assets = api.list_assets()
        
        # Filtrer pour ne garder que les cryptomonnaies (compatibilité Alpaca SDK et _raw['class'])
        crypto_assets = [
            asset for asset in assets
            if (
                getattr(asset, 'asset_class', None) == 'crypto'
                or getattr(asset, 'class_', None) == 'crypto'
                or (hasattr(asset, '_raw') and asset._raw.get('class') == 'crypto')
            )
        ]

        if not crypto_assets and assets:
            logger.warning("Aucun asset avec 'asset_class', 'class_' ou _raw['class'] == 'crypto'. Voici les attributs du premier asset pour debug:")
            logger.warning(vars(assets[0]))

        # Récupérer les symboles
        crypto_symbols = []
        for asset in crypto_assets:
            # Format avec /
            symbol_with_slash = asset.symbol  # Ex: BTC/USD
            # Format avec -
            symbol_with_dash = asset.symbol.replace('/', '-')  # Ex: BTC-USD
            
            crypto_symbols.append({
                "symbol": asset.symbol,
                "symbol_dash": symbol_with_dash,
                "name": asset.name,
                "status": asset.status,
                "tradable": asset.tradable
            })
        
        return crypto_symbols
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des cryptomonnaies: {e}")
        return []

def save_crypto_lists(crypto_assets):
    """Sauvegarde les listes de cryptomonnaies dans différents formats"""
    try:
        # Créer un dossier data s'il n'existe pas
        data_dir = root_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # 1. Sauvegarder la liste complète avec détails
        logger.info("Sauvegarde de la liste détaillée...")
        with open(data_dir / "all_crypto_details.csv", "w") as f:
            f.write("symbol,symbol_dash,name,status,tradable\n")
            for asset in crypto_assets:
                f.write(f"{asset['symbol']},{asset['symbol_dash']},{asset['name']},{asset['status']},{asset['tradable']}\n")
        
        # 2. Sauvegarder la liste au format custom_crypto_symbols.txt (BTCUSD)
        logger.info("Sauvegarde au format sans séparateur...")
        with open(data_dir / "all_crypto_symbols_no_separator.txt", "w") as f:
            for asset in crypto_assets:
                if asset['tradable']:
                    symbol = asset['symbol'].replace('/', '')
                    f.write(f"{symbol}\n")
        
        # 3. Sauvegarder la liste au format custom_crypto_symbols_new.txt (BTC/USD)
        logger.info("Sauvegarde au format avec slash...")
        with open(data_dir / "all_crypto_symbols_slash.txt", "w") as f:
            for asset in crypto_assets:
                if asset['tradable']:
                    f.write(f"{asset['symbol']}\n")
        
        # 4. Sauvegarder la liste au format avec tiret (BTC-USD)
        logger.info("Sauvegarde au format avec tiret...")
        with open(data_dir / "all_crypto_symbols_dash.txt", "w") as f:
            for asset in crypto_assets:
                if asset['tradable']:
                    f.write(f"{asset['symbol_dash']}\n")
        
        # 5. Générer la ligne pour le fichier .env
        logger.info("Génération de la ligne pour .env...")
        tradable_symbols = [asset['symbol'] for asset in crypto_assets if asset['tradable']]
        env_line = f"PERSONALIZED_CRYPTO_LIST={','.join(tradable_symbols[:30])}"
        with open(data_dir / "crypto_env_line.txt", "w") as f:
            f.write(env_line)
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des listes: {e}")
        return False

if __name__ == "__main__":
    logger.info("Récupération des cryptomonnaies disponibles sur Alpaca...")
    crypto_assets = get_all_alpaca_cryptos()
    
    if crypto_assets:
        tradable_count = sum(1 for asset in crypto_assets if asset['tradable'])
        total_count = len(crypto_assets)
        
        logger.info(f"Total des cryptomonnaies trouvées: {total_count}")
        logger.info(f"Cryptomonnaies négociables: {tradable_count}")
        
        # Sauvegarder les listes
        if save_crypto_lists(crypto_assets):
            logger.info("Les listes ont été sauvegardées avec succès dans le dossier 'data'")
            
            # Afficher toutes les cryptos négociables
            logger.info("Liste complète des cryptomonnaies négociables:")
            for asset in crypto_assets:
                if asset['tradable']:
                    logger.info(f"  - {asset['symbol']} ({asset['name']})")
        else:
            logger.error("Échec de la sauvegarde des listes")
    else:
        logger.error("Aucune cryptomonnaie n'a pu être récupérée")
