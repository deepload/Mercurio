import sys
import os
from alpaca_crypto_trader import AlpacaCryptoTrader, SessionDuration

# Test simple pour vérifier l'intégration de Binance
if __name__ == "__main__":
    print("Test de l'intégration de Binance")
    
    # Créer le trader avec Binance comme fournisseur de données
    trader = AlpacaCryptoTrader(
        session_duration=SessionDuration.ONE_HOUR,
        data_provider="binance"
    )
    
    # Afficher les informations
    print(f"Fournisseur de données configuré: {trader.data_provider}")
    
    # Tester la récupération des données pour un symbole
    symbol = "BTC/USD"
    print(f"Récupération des données pour {symbol}...")
    
    # Initialiser le trader
    trader.initialize()
    
    # Traiter le symbole
    trader.process_symbol(symbol)
    
    print("Test terminé")
