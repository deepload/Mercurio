#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour le fournisseur Binance
Ce script teste l'intégration du fournisseur Binance dans Mercurio AI
"""

import sys
import os
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("binance_provider_test")

# Importer les services et fournisseurs
from app.services.market_data import MarketDataService
from app.services.providers.binance import BinanceProvider
from app.services.providers.factory import MarketDataProviderFactory

async def test_historical_data(provider, symbol):
    """Tester la récupération de données historiques"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    logger.info(f"Récupération des données historiques pour {symbol} du {start_date.date()} au {end_date.date()}")
    
    try:
        data = await provider.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1h"
        )
        
        if isinstance(data, pd.DataFrame) and not data.empty:
            logger.info(f"✓ Données historiques récupérées avec succès: {len(data)} entrées")
            logger.info(f"Premières entrées:\n{data.head(3)}")
            return True
        else:
            logger.error(f"✗ Échec: Données vides ou format incorrect")
            return False
    except Exception as e:
        logger.error(f"✗ Erreur lors de la récupération des données historiques: {e}")
        return False

async def test_latest_price(provider, symbol):
    """Tester la récupération du dernier prix"""
    logger.info(f"Récupération du dernier prix pour {symbol}")
    
    try:
        price = await provider.get_latest_price(symbol=symbol)
        if price and price > 0:
            logger.info(f"✓ Prix récupéré avec succès: {price}")
            return True
        else:
            logger.error(f"✗ Échec: Prix invalide ou nul")
            return False
    except Exception as e:
        logger.error(f"✗ Erreur lors de la récupération du dernier prix: {e}")
        return False

async def test_market_symbols(provider):
    """Tester la récupération des symboles disponibles"""
    logger.info("Récupération des symboles de marché disponibles")
    
    try:
        symbols = await provider.get_market_symbols()
        if symbols and len(symbols) > 0:
            logger.info(f"✓ Symboles récupérés avec succès: {len(symbols)} symboles")
            logger.info(f"Exemples: {', '.join(symbols[:5])}")
            return True
        else:
            logger.error(f"✗ Échec: Aucun symbole récupéré")
            return False
    except Exception as e:
        logger.error(f"✗ Erreur lors de la récupération des symboles: {e}")
        return False

async def test_symbol_conversion(provider):
    """Tester la conversion des symboles"""
    test_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", 
        "BTC/USDT", "ETH/USDT", "SOL/USDT",
        "BTC-USD", "ETH-USD", "SOL-USD"
    ]
    
    logger.info("Test de conversion des symboles")
    success = True
    
    for symbol in test_symbols:
        try:
            converted = provider._convert_symbol(symbol)
            logger.info(f"✓ Conversion de {symbol} -> {converted}")
        except Exception as e:
            logger.error(f"✗ Erreur lors de la conversion de {symbol}: {e}")
            success = False
    
    return success

async def main():
    """Fonction principale de test"""
    logger.info("=== TEST DU FOURNISSEUR BINANCE ===")
    
    # Test 1: Vérifier que le fournisseur est correctement enregistré dans la factory
    logger.info("\n=== TEST 1: Vérification de l'enregistrement dans la factory ===")
    factory = MarketDataProviderFactory()
    
    # Vérifier si Binance est disponible en essayant de l'obtenir directement
    try:
        binance_provider = factory.get_provider("binance")
        if binance_provider and binance_provider.name.lower() == "binance":
            logger.info("✓ Fournisseur Binance correctement enregistré dans la factory")
        else:
            logger.error("✗ Fournisseur Binance non disponible dans la factory")
            return False
    except Exception as e:
        logger.error(f"✗ Erreur lors de la récupération du fournisseur Binance: {e}")
        return False
    
    # Test 2: Vérifier que le fournisseur est sélectionné par défaut pour les symboles crypto
    logger.info("\n=== TEST 2: Vérification de la sélection par défaut pour les symboles crypto ===")
    crypto_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    for symbol in crypto_symbols:
        default_provider = factory.get_default_provider(symbol)
        if default_provider and default_provider.name.lower() == "binance":
            logger.info(f"✓ Binance sélectionné par défaut pour {symbol}")
        else:
            logger.warning(f"! Binance non sélectionné par défaut pour {symbol}, fournisseur actuel: {default_provider.name if default_provider else 'Aucun'}")
    
    # Test 3: Créer une instance directe du fournisseur Binance
    logger.info("\n=== TEST 3: Création d'une instance directe du fournisseur Binance ===")
    binance_provider = BinanceProvider()
    
    # Test 4: Tester la conversion des symboles
    logger.info("\n=== TEST 4: Test de conversion des symboles ===")
    await test_symbol_conversion(binance_provider)
    
    # Test 5: Tester la récupération des symboles de marché
    logger.info("\n=== TEST 5: Test de récupération des symboles de marché ===")
    await test_market_symbols(binance_provider)
    
    # Test 6: Tester la récupération de données historiques
    logger.info("\n=== TEST 6: Test de récupération de données historiques ===")
    test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    for symbol in test_symbols:
        await test_historical_data(binance_provider, symbol)
    
    # Test 7: Tester la récupération du dernier prix
    logger.info("\n=== TEST 7: Test de récupération du dernier prix ===")
    for symbol in test_symbols:
        await test_latest_price(binance_provider, symbol)
    
    # Test 8: Tester via le MarketDataService
    logger.info("\n=== TEST 8: Test via le MarketDataService ===")
    market_data_service = MarketDataService(provider_name="binance")
    
    if market_data_service.active_provider and market_data_service.active_provider.name.lower() == "binance":
        logger.info("✓ MarketDataService utilise correctement le fournisseur Binance")
        
        # Tester la récupération de données via le service
        symbol = "BTC/USD"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        try:
            data = await market_data_service.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe="1h"
            )
            
            if isinstance(data, pd.DataFrame) and not data.empty:
                logger.info(f"✓ MarketDataService: Données historiques récupérées avec succès via le service: {len(data)} entrées")
            else:
                logger.error(f"✗ MarketDataService: Échec de récupération des données historiques")
        except Exception as e:
            logger.error(f"✗ MarketDataService: Erreur lors de la récupération des données historiques: {e}")
    else:
        logger.error(f"✗ MarketDataService n'utilise pas le fournisseur Binance, fournisseur actuel: {market_data_service.active_provider.name if market_data_service.active_provider else 'Aucun'}")
    
    logger.info("\n=== TESTS TERMINÉS ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du test: {e}")
        import traceback
        logger.error(traceback.format_exc())
