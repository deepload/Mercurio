#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Crypto Day Trading Script
--------------------------------
Script autonome pour le daytrading de cryptomonnaies via Alpaca API,
optimisé pour l'abonnement de niveau 3 (AlgoTrader Plus).

Utilisation:
    python alpaca_crypto_trader.py --duration 1h --log-level INFO
"""

import os
import time
import signal
import logging
import argparse
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# API Alpaca
import alpaca_trade_api as tradeapi
import pandas as pd
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logger
# Configurer la journalisation pour enregistrer dans un fichier
log_file = f"crypto_trader_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alpaca_crypto_trader")

# Variables globales pour la gestion des signaux
running = True
session_end_time = None

# Enums pour la durée de session
class SessionDuration(int, Enum):
    ONE_HOUR = 3600
    FOUR_HOURS = 14400
    EIGHT_HOURS = 28800
    NIGHT_RUN = 32400  # 9 heures (pour couvrir toute la nuit)
    CUSTOM = 0

class AlpacaCryptoTrader:
    """
    Système de daytrading crypto utilisant directement l'API Alpaca
    
    Caractéristiques:
    - Utilise l'API Alpaca pour trader des cryptos en mode paper
    - Stratégie simple de croisement de moyennes mobiles
    - Plusieurs durées de session (1h, 4h, 8h, nuit)
    - Paramètres de trading configurables
    - Journalisation complète et rapport de performance
    """
    
    def __init__(self, session_duration: SessionDuration = SessionDuration.ONE_HOUR):
        """Initialiser le système de trading crypto"""
        self.session_duration = session_duration
        
        # Déterminer le mode Alpaca (paper ou live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        
        # Configuration selon le mode
        if alpaca_mode == "live":
            self.api_key = os.getenv("ALPACA_LIVE_KEY")
            self.api_secret = os.getenv("ALPACA_LIVE_SECRET")
            self.base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            logger.info("Configuré pour le trading LIVE (réel)")
        else:  # mode paper par défaut
            self.api_key = os.getenv("ALPACA_PAPER_KEY")
            self.api_secret = os.getenv("ALPACA_PAPER_SECRET")
            self.base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            logger.info("Configuré pour le trading PAPER (simulation)")
            
        # URL des données de marché
        self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        # Niveau d'abonnement Alpaca
        self.subscription_level = int(os.getenv("ALPACA_SUBSCRIPTION_LEVEL", "1"))
        logger.info(f"Utilisation du niveau d'abonnement Alpaca: {self.subscription_level}")
        
        # Client API Alpaca
        self.api = None
        
        # Paramètres de trading
        self.symbols = []  # Sera rempli avec les symboles crypto disponibles
        self.custom_symbols = []  # Liste personnalisée de symboles à utiliser
        self.use_custom_symbols = False  # Si True, utilise custom_symbols au lieu de la liste filtrée
        self.fast_ma_period = 5   # 5 minutes pour la moyenne mobile rapide
        self.slow_ma_period = 15  # 15 minutes pour la moyenne mobile lente
        self.position_size_pct = 0.02  # 2% du portefeuille par position
        self.stop_loss_pct = 0.03  # 3% de stop loss
        self.take_profit_pct = 0.06  # 6% de prise de profit
        
        # Paramètres pour le trailing stop-loss
        self.use_trailing_stop = True  # Activer le trailing stop-loss par défaut
        self.trailing_stop_pct = 0.02  # 2% de trailing stop-loss (distance en pourcentage)
        self.trailing_stop_activation_pct = 0.015  # Activer le trailing stop après 1.5% de gain
        
        # Suivi de l'état
        self.positions = {}
        self.highest_prices = {}  # Pour suivre le prix le plus élevé atteint par chaque position
        self.portfolio_value = 0.0
        self.initial_portfolio_value = 0.0
        self.session_start_time = None
        self.session_end_time = None
        self.trade_history = []  # Pour enregistrer l'historique des transactions
        self.running = True  # Variable pour contrôler l'exécution
        
        logger.info("AlpacaCryptoTrader initialisé")
        
    def initialize(self):
        """Initialiser les services et charger la configuration"""
        try:
            # Initialiser le client API Alpaca
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                api_version='v2'
            )
            logger.info("API Alpaca initialisée avec succès")
            
            # Réinitialiser le dictionnaire des prix les plus élevés
            self.highest_prices = {}
            
            # Vérifier que le client est correctement initialisé
            account = self.api.get_account()
            if account:
                self.portfolio_value = float(account.portfolio_value)
                self.initial_portfolio_value = self.portfolio_value
                logger.info(f"Compte Alpaca connecté: {account.id}")
                logger.info(f"Valeur initiale du portefeuille: ${self.portfolio_value:.2f}")
                logger.info(f"Mode trading: {account.status}")
                
                # Vérifier la disponibilité du trading crypto
                assets = self.api.list_assets(asset_class='crypto')
                
                if not self.use_custom_symbols:
                    # Filtrer pour ne garder que les paires USD pures (exclure explicitement USDT/USDC)
                    self.symbols = [asset.symbol for asset in assets if asset.tradable 
                                  and '/USD' in asset.symbol 
                                  and not '/USDT' in asset.symbol 
                                  and not '/USDC' in asset.symbol]
                    
                    if self.symbols:
                        logger.info(f"Trouvé {len(self.symbols)} symboles crypto disponibles (USD seulement)")
                        logger.info(f"Exemples: {', '.join(self.symbols[:5])}")
                    else:
                        logger.warning("Aucun symbole crypto disponible avec USD")
                else:
                    # Utiliser la liste personnalisée et vérifier que les symboles sont tradables
                    tradable_assets = [asset.symbol for asset in assets if asset.tradable]
                    self.symbols = [symbol for symbol in self.custom_symbols if symbol in tradable_assets]
                    logger.info(f"Utilisation d'une liste personnalisée de {len(self.symbols)} symboles crypto")
                    if self.symbols:
                        logger.info(f"Exemples: {', '.join(self.symbols[:5])}")
                    else:
                        logger.warning("Aucun symbole personnalisé n'est tradable")
                    
                # Vérifier le solde disponible en USD
                try:
                    account = self.api.get_account()
                    cash = float(account.cash)
                    logger.info(f"Solde USD disponible: ${cash:.2f}")
                except Exception as e:
                    logger.warning(f"Impossible de récupérer le solde USD: {e}")
                    pass
                
                return True
            else:
                logger.error("Impossible de récupérer les informations du compte")
                return False
                
        except Exception as e:
            logger.error(f"Erreur d'initialisation: {e}")
            raise
        
    def stop(self):
        """Arrêter proprement le trader"""
        self.running = False
        logger.info("Arrêt du trader demandé, finalisation des opérations en cours...")
        
        # Générer un rapport final si nécessaire
        try:
            self.generate_final_report()
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport final: {e}")
        
        logger.info("Trader arrêté avec succès")
        
    def generate_final_report(self):
        """Générer un rapport final sur les performances"""
        logger.info("=" * 60)
        logger.info("RAPPORT FINAL DE TRADING")
        logger.info("=" * 60)
        
        # Afficher les positions actuelles
        try:
            positions = self.api.list_positions()
            if positions:
                logger.info(f"Positions ouvertes: {len(positions)}")
                for pos in positions:
                    market_value = float(pos.market_value) if hasattr(pos, 'market_value') else 0
                    unrealized_pl = float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0
                    logger.info(f"  {pos.symbol}: {pos.qty} @ {pos.avg_entry_price} - PnL: ${unrealized_pl:.2f}")
            else:
                logger.info("Aucune position ouverte")
        except Exception as e:
            logger.warning(f"Impossible de récupérer les positions: {e}")
            
        # Récupérer la valeur du portefeuille
        try:
            account = self.api.get_account()
            logger.info(f"Valeur du portefeuille: ${float(account.portfolio_value):.2f}")
            logger.info(f"Espèces disponibles: ${float(account.cash):.2f}")
        except Exception as e:
            logger.warning(f"Impossible de récupérer les informations du compte: {e}")
        
        logger.info("=" * 60)
        logger.info("FIN DU RAPPORT")
        logger.info("=" * 60)
    
    def start(self, duration_seconds: Optional[int] = None):
        """Démarrer la session de trading crypto"""
        self.session_start_time = datetime.now()
        
        if duration_seconds is not None:
            self.session_end_time = self.session_start_time + timedelta(seconds=duration_seconds)
        else:
            self.session_end_time = self.session_start_time + timedelta(seconds=int(self.session_duration))
            
        logger.info(f"Démarrage de la session de trading crypto à {self.session_start_time}")
        logger.info(f"La session se terminera à {self.session_end_time}")
        
        # Initialiser le trader
        initialized = self.initialize()
        if not initialized:
            logger.error("Échec de l'initialisation, abandon")
            self.generate_performance_report()
            return
            
        # Démarrer la boucle de trading
        self.trading_loop()
        
        # Générer un rapport de performance à la fin
        self.generate_performance_report()
            
    def trading_loop(self):
        """Boucle principale de trading"""
        global running
        
        try:
            while running and datetime.now() < self.session_end_time:
                # Déterminer les symboles à trader pour cette itération
                # Limiter aux 10 premières cryptos pour éviter les limites de taux si pas de liste personnalisée
                trading_symbols = self.symbols[:10] if (len(self.symbols) > 10 and not self.use_custom_symbols) else self.symbols
                
                # Afficher le solde disponible à chaque itération
                try:
                    account_info = self.api.get_account()
                    buying_power = float(account_info.buying_power)
                    cash = float(account_info.cash)
                    equity = float(account_info.equity)
                    
                    logger.info("\n===== INFORMATION DU COMPTE ALPACA =====")
                    logger.info(f"Solde disponible: ${buying_power:.2f}")
                    logger.info(f"Liquidités: ${cash:.2f}")
                    logger.info(f"Valeur totale: ${equity:.2f}")
                    
                    # Afficher les positions ouvertes
                    try:
                        positions = self.api.list_positions()
                        if positions:
                            logger.info("\n----- POSITIONS OUVERTES -----")
                            for position in positions:
                                symbol = position.symbol
                                qty = float(position.qty)
                                current_price = float(position.current_price)
                                market_value = float(position.market_value)
                                entry_price = float(position.avg_entry_price)
                                profit_loss = float(position.unrealized_pl)
                                profit_loss_pct = float(position.unrealized_plpc) * 100
                                logger.info(f"{symbol}: {qty} @ ${entry_price:.2f} | Prix actuel: ${current_price:.2f} | Valeur: ${market_value:.2f} | P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                            logger.info("--------------------------")
                        else:
                            logger.info("Pas de positions ouvertes")
                    except Exception as e:
                        logger.error(f"Erreur lors de la récupération des positions: {e}")
                    
                    logger.info("=======================================\n")
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération du solde Alpaca: {e}")
                
                # Traiter chaque symbole
                for symbol in trading_symbols:
                    try:
                        self.process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"Erreur de traitement de {symbol}: {e}")
                
                # Mettre à jour l'état du portefeuille
                self.update_portfolio_state()
                
                # Attendre 60 secondes avant la prochaine itération
                time_remaining = int((self.session_end_time - datetime.now()).total_seconds() / 60)
                logger.info(f"Attente de 60 secondes avant le prochain cycle. Fin de session dans {time_remaining} minutes")
                time.sleep(60)
                
        except Exception as e:
            logger.error(f"Erreur dans la boucle de trading: {e}")
        finally:
            logger.info("Boucle de trading terminée")
    
            
    def process_symbol(self, symbol: str):
        """Traiter un symbole de trading"""
        logger.info(f"Traitement de {symbol}")
        
        # Obtenir les données historiques (intervalles de 5 minutes pour les dernières 24 heures)
        end = datetime.now()
        start = end - timedelta(days=1)
        
        try:
            # Formater les dates pour l'API
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            # Obtenir les barres de prix (corriger l'erreur expected list, str found)
            bars = self.api.get_crypto_bars(
                [symbol],  # Passer une liste au lieu d'une chaîne
                timeframe='5Min',
                start=start_str,
                end=end_str
            ).df
            
            if bars.empty:
                logger.warning(f"Pas de données historiques disponibles pour {symbol}")
                return
            
            # Si les données sont multi-index (symbole, timestamp), prendre juste le symbole concerné
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.loc[symbol]
                
            # Calculer les moyennes mobiles
            bars['fast_ma'] = bars['close'].rolling(window=self.fast_ma_period).mean()
            bars['slow_ma'] = bars['close'].rolling(window=self.slow_ma_period).mean()
            
            # Obtenir la position actuelle
            position = None
            try:
                position = self.api.get_position(symbol)
            except:
                pass  # Pas de position existante
            
            # Obtenir le prix actuel (compatible avec abonnement niveau 1)
            try:
                # Pour le niveau 1, on peut utiliser la dernière barre des dernières 5 minutes comme prix actuel
                if not bars.empty:
                    current_price = float(bars.iloc[-1]['close'])
                    logger.info(f"{symbol} prix actuel (dernière barre): ${current_price:.4f}")
                else:
                    logger.error(f"Pas de données disponibles pour obtenir le prix actuel de {symbol}")
                    return
            except Exception as e:
                logger.error(f"Impossible d'obtenir le prix actuel pour {symbol}: {e}")
                return
            
            # Logique de trading - Croisement de moyennes mobiles
            if len(bars) >= self.slow_ma_period:
                last_row = bars.iloc[-1]
                prev_row = bars.iloc[-2]
                
                # Vérifier le signal d'achat: MA rapide croise au-dessus de la MA lente
                buy_signal = (
                    prev_row['fast_ma'] <= prev_row['slow_ma'] and 
                    last_row['fast_ma'] > last_row['slow_ma']
                )
                
                # Vérifier le signal de vente: MA rapide croise en dessous de la MA lente
                sell_signal = (
                    prev_row['fast_ma'] >= prev_row['slow_ma'] and 
                    last_row['fast_ma'] < last_row['slow_ma']
                )
                
                # Exécuter les signaux
                if buy_signal and not position:
                    self.execute_buy(symbol, current_price)
                elif sell_signal and position:
                    self.execute_sell(symbol, current_price, position)
                
                # Vérifier le stop loss, take profit et trailing stop
                if position:
                    entry_price = float(position.avg_entry_price)
                    if entry_price > 0:
                        pnl_pct = (current_price - entry_price) / entry_price
                        
                        # Stop loss normal
                        if pnl_pct <= -self.stop_loss_pct:
                            logger.info(f"{symbol} a atteint le stop loss à {pnl_pct:.2%}")
                            self.execute_sell(symbol, current_price, position)
                        # Take profit normal
                        elif pnl_pct >= self.take_profit_pct:
                            logger.info(f"{symbol} a atteint le take profit à {pnl_pct:.2%}")
                            self.execute_sell(symbol, current_price, position)
                        # Gestion du trailing stop
                        elif self.use_trailing_stop:
                            # Mettre à jour le prix le plus élevé pour ce symbole si nécessaire
                            if symbol not in self.highest_prices:
                                self.highest_prices[symbol] = entry_price
                                 
                            # Mettre à jour le prix le plus élevé si le prix actuel est plus élevé
                            if current_price > self.highest_prices[symbol]:
                                self.highest_prices[symbol] = current_price
                                highest_pnl_pct = (self.highest_prices[symbol] - entry_price) / entry_price
                                logger.debug(f"{symbol} - Nouveau prix max: ${self.highest_prices[symbol]:.4f} (+{highest_pnl_pct:.2%})")
                            
                            # Vérifier si le trailing stop est activé (on a dépassé le seuil d'activation)
                            highest_pnl_pct = (self.highest_prices[symbol] - entry_price) / entry_price
                            if highest_pnl_pct >= self.trailing_stop_activation_pct:
                                # Calculer la distance en pourcentage depuis le plus haut
                                drop_from_high_pct = (self.highest_prices[symbol] - current_price) / self.highest_prices[symbol]
                                
                                # Si on a chuté plus que le pourcentage de trailing stop depuis le plus haut
                                if drop_from_high_pct >= self.trailing_stop_pct:
                                    logger.info(f"{symbol} a déclenché le trailing stop: -{drop_from_high_pct:.2%} depuis le plus haut de ${self.highest_prices[symbol]:.4f}")
                                    self.execute_sell(symbol, current_price, position)
            
        except Exception as e:
            logger.error(f"Erreur de traitement de {symbol}: {e}")
    
    def execute_buy(self, symbol: str, price: float):
        """Exécuter un ordre d'achat"""
        try:
            # Obtenir le cash disponible
            account = self.api.get_account()
            cash = float(account.cash)
            
            # Calculer la taille de l'ordre
            order_value = cash * self.position_size_pct
            order_qty = order_value / price
            
            # Noter que les quantités peuvent être fractionnelles pour les cryptos
            # Arrondir à 6 décimales pour éviter les erreurs de précision
            order_qty = round(order_qty, 6)
            
            MIN_ORDER_VALUE = 1  # Broker minimum notional (e.g. $1)
            if order_value < MIN_ORDER_VALUE:
                logger.warning(f"Ordre non exécuté pour {symbol}: valeur {order_value:.2f} < minimum autorisé {MIN_ORDER_VALUE}")
                return

            if order_qty > 0:
                logger.info(f"Achat de {order_qty:.6f} {symbol} @ ${price:.4f} (valeur: ${order_value:.2f})")
                
                # Exécuter l'ordre
                self.api.submit_order(
                    symbol=symbol,
                    qty=order_qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                
                # Initialiser le tracking du prix le plus élevé pour ce symbole (trailing stop)
                self.highest_prices[symbol] = price
                
                # Enregistrer la transaction dans l'historique
                if not hasattr(self, 'trade_history'):
                    self.trade_history = []
                    
                self.trade_history.append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': order_qty,
                    'price': price,
                    'value': order_value
                })
            else:
                logger.warning(f"Ordre non exécuté pour {symbol}: taille d'ordre insuffisante")
        except Exception as e:
            logger.error(f"Erreur lors de l'achat de {symbol}: {e}")
    
    def execute_sell(self, symbol: str, price: float, position):
        """Exécuter un ordre de vente"""
        try:
            qty = float(position.qty)
            
            if qty <= 0:
                logger.warning(f"Quantité de position invalide pour {symbol}: {qty}")
                return
                
            logger.info(f"SIGNAL DE VENTE: {symbol} à ${price:.4f}, qté: {qty:.6f}")
            
            # Placer un ordre au marché
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            
            if order:
                logger.info(f"Ordre de vente placé pour {symbol}: {order.id}")
                self.trade_history.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'action': 'vente',
                    'quantity': qty,
                    'price': price
                })
            else:
                logger.error(f"Échec du placement de l'ordre de vente pour {symbol}")
                
        except Exception as e:
            logger.error(f"Erreur d'exécution de vente pour {symbol}: {e}")
    
    def update_portfolio_state(self):
        """Mettre à jour la valeur du portefeuille et les positions"""
        try:
            account = self.api.get_account()
            self.portfolio_value = float(account.portfolio_value)
            logger.info(f"Valeur actuelle du portefeuille: ${self.portfolio_value:.2f}")
            
            # Mettre à jour les positions
            try:
                positions = self.api.list_positions()
                crypto_positions = [p for p in positions if '/' in p.symbol]
                
                # Journaliser les positions ouvertes
                if crypto_positions:
                    logger.info(f"Positions ouvertes actuelles: {len(crypto_positions)}")
                    for pos in crypto_positions:
                        entry_price = float(pos.avg_entry_price)
                        current_price = float(pos.current_price)
                        qty = float(pos.qty)
                        market_value = float(pos.market_value)
                        pnl = float(pos.unrealized_pl)
                        pnl_pct = float(pos.unrealized_plpc) * 100
                        
                        logger.info(f"  {pos.symbol}: {qty:.6f} @ ${entry_price:.4f} - Valeur: ${market_value:.2f} - P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                else:
                    logger.info("Pas de positions ouvertes")
            except Exception as e:
                logger.error(f"Erreur de récupération des positions: {e}")
                
        except Exception as e:
            logger.error(f"Erreur de mise à jour de l'état du portefeuille: {e}")
    
    def generate_performance_report(self):
        """Générer un rapport de performance à la fin de la session de trading"""
        # Créer un fichier de rapport séparé
        report_file = f"crypto_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            end_time = datetime.now()
            duration = end_time - self.session_start_time if self.session_start_time else timedelta(0)
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info("===================================================")
            logger.info("RAPPORT DE PERFORMANCE DE LA SESSION DE TRADING CRYPTO")
            logger.info("===================================================")
            logger.info(f"Durée de la session: {hours}h {minutes}m {seconds}s")
            logger.info(f"Heure de début: {self.session_start_time}")
            logger.info(f"Heure de fin: {end_time}")
            
            # Obtenir l'état final du compte
            try:
                account = self.api.get_account()
                final_value = float(account.portfolio_value)
                
                if self.initial_portfolio_value > 0:
                    profit_loss = final_value - self.initial_portfolio_value
                    profit_loss_pct = (profit_loss / self.initial_portfolio_value) * 100
                    logger.info(f"Valeur initiale du portefeuille: ${self.initial_portfolio_value:.2f}")
                    logger.info(f"Valeur finale du portefeuille: ${final_value:.2f}")
                    logger.info(f"Profit/Perte: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            except Exception as e:
                logger.warning(f"Impossible de récupérer les informations finales du compte: {e}")
            
            # Afficher les positions ouvertes
            try:
                positions = self.api.list_positions()
                crypto_positions = [p for p in positions if '/' in p.symbol]
                
                if crypto_positions:
                    logger.info(f"Positions ouvertes à la fin de la session: {len(crypto_positions)}")
                    for pos in crypto_positions:
                        entry_price = float(pos.avg_entry_price)
                        current_price = float(pos.current_price)
                        qty = float(pos.qty)
                        market_value = float(pos.market_value)
                        pnl = float(pos.unrealized_pl)
                        pnl_pct = float(pos.unrealized_plpc) * 100
                        
                        logger.info(f"  {pos.symbol}: {qty:.6f} @ ${entry_price:.4f} - Valeur: ${market_value:.2f} - P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                else:
                    logger.info("Pas de positions ouvertes à la fin de la session")
            except Exception as e:
                logger.warning(f"Impossible de récupérer les informations de position: {e}")
                
            logger.info("===================================================")
            logger.info("SESSION DE TRADING CRYPTO TERMINÉE")
            logger.info("===================================================")
                
        except Exception as e:
            logger.error(f"Erreur de génération du rapport de performance: {e}")

        # Écrire le rapport également dans un fichier séparé
        with open(report_file, 'w') as f:
            f.write("===================================================\n")
            f.write("RAPPORT DE PERFORMANCE DE LA SESSION DE TRADING CRYPTO\n")
            f.write("===================================================\n\n")
            f.write(f"Durée de la session: {hours}h {minutes}m {seconds}s\n")
            f.write(f"Heure de début: {self.session_start_time}\n")
            f.write(f"Heure de fin: {end_time}\n\n")
            
            try:
                if self.initial_portfolio_value > 0:
                    f.write(f"Valeur initiale du portefeuille: ${self.initial_portfolio_value:.2f}\n")
                    f.write(f"Valeur finale du portefeuille: ${final_value:.2f}\n")
                    f.write(f"Profit/Perte: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)\n\n")
            except:
                f.write("Impossible de récupérer les informations finales du compte\n\n")
                
            f.write("Positions ouvertes à la fin de la session:\n")
            try:
                if crypto_positions:
                    for pos in crypto_positions:
                        f.write(f"  {pos.symbol}: {float(pos.qty):.6f} @ ${float(pos.avg_entry_price):.4f} - ")
                        f.write(f"Valeur: ${float(pos.market_value):.2f} - ")
                        f.write(f"P/L: ${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc) * 100:.2f}%)\n")
                else:
                    f.write("Aucune position ouverte\n")
            except:
                f.write("Impossible de récupérer les informations de position\n")
            
            f.write("\n===================================================\n")
            f.write("RÉSUMÉ DES TRANSACTIONS IMPORTANTES\n")
            f.write("===================================================\n")
            if hasattr(self, 'trade_history') and self.trade_history:
                for trade in self.trade_history:
                    f.write(f"{trade['time']} - {trade['symbol']} - {trade['action']} - ")
                    f.write(f"{trade['quantity']:.6f} @ ${trade['price']:.4f} - P/L: ${trade.get('pnl', 0):.2f}\n")
            else:
                f.write("Aucune transaction effectuée\n")
                
        logger.info(f"Rapport détaillé sauvegardé dans {report_file}")

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Système de trading crypto Alpaca")
    parser.add_argument("--duration", type=str, choices=["1h", "4h", "8h", "custom"], default="1h",
                        help="Durée de la session de trading (1h, 4h, 8h, ou custom)")
    parser.add_argument("--custom-seconds", type=int, default=0,
                        help="Durée personnalisée en secondes si --duration=custom")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Niveau de journalisation")
    parser.add_argument("--no-trailing-stop", action="store_true",
                        help="Désactiver le trailing stop-loss")
    parser.add_argument("--trailing-stop-pct", type=float, default=0.02,
                        help="Pourcentage de trailing stop-loss (default: 0.02 soit 2%)")
    parser.add_argument("--trailing-activation-pct", type=float, default=0.015,
                        help="Pourcentage de gain avant activation du trailing stop (default: 0.015 soit 1.5%)")
                        
    args = parser.parse_args()
    
    # Définir le niveau de journalisation
    numeric_level = getattr(logging, args.log_level)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Déterminer la durée de la session
    duration_map = {
        "1h": SessionDuration.ONE_HOUR,
        "4h": SessionDuration.FOUR_HOURS,
        "8h": SessionDuration.EIGHT_HOURS,
        "custom": SessionDuration.CUSTOM
    }
    session_duration = duration_map.get(args.duration, SessionDuration.ONE_HOUR)
    custom_duration = args.custom_seconds if args.duration == "custom" else 0
    
    # Créer le trader
    trader = AlpacaCryptoTrader(session_duration=session_duration)
    
    # Configurer les options de trailing stop
    if args.no_trailing_stop:
        trader.use_trailing_stop = False
    else:
        trader.use_trailing_stop = True
        trader.trailing_stop_pct = args.trailing_stop_pct
        trader.trailing_stop_activation_pct = args.trailing_activation_pct
        logger.info(f"Trailing stop activé: {args.trailing_stop_pct*100}% de baisse depuis le plus haut, après {args.trailing_activation_pct*100}% de gain")
    
    # Enregistrer les gestionnaires de signaux pour une fermeture propre
    def signal_handler(sig, frame):
        global running, session_end_time
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        running = False
        session_end_time = datetime.now()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Exécuter le trader
    try:
        if custom_duration > 0:
            trader.start(custom_duration)
        else:
            trader.start()
    except KeyboardInterrupt:
        logger.info("Interruption clavier reçue, arrêt en cours...")
    except Exception as e:
        logger.error(f"Erreur d'exécution du trader crypto: {e}")
    finally:
        logger.info("Arrêt du trader crypto terminé")

if __name__ == "__main__":
    main()
