"""
Cash Secured Put (CSP) strategy for options trading.

This strategy involves selling a put on an asset while having sufficient cash
to cover the cost of purchasing the shares at the strike price if the option is exercised.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import inspect

from app.core.models.option import OptionContract, OptionPosition, OptionType
from app.strategies.options.base_options_strategy import BaseOptionsStrategy

logger = logging.getLogger(__name__)

class CashSecuredPutStrategy(BaseOptionsStrategy):
    """
    Implementation of the Cash Secured Put strategy.
    
    This strategy is optimal for securities that you want to buy but at a lower price
    than the current market price. It generates premium income but exposes you to buying
    the security if the price falls below the strike price at expiration.
    """
    
    def __init__(
        self,
        underlying_symbol: str,
        account_size: float,
        max_position_size: float = 0.10,
        min_implied_volatility: float = 0.20,
        max_days_to_expiry: int = 45,
        min_days_to_expiry: int = 20,
        target_delta: float = -0.30,
        delta_range: float = 0.05,
        profit_target_pct: float = 0.50,
        stop_loss_pct: float = 0.100,
        roll_when_dte: int = 7,
        use_technical_filters: bool = True,
        **kwargs
    ):
        """
        Initialize the Cash Secured Put strategy.
        
        Args:
            underlying_symbol: Symbol of the underlying asset
            account_size: Account size in USD
            max_position_size: Maximum position size as % of account
            min_implied_volatility: Minimum implied volatility to sell a put
            max_days_to_expiry: Maximum number of days until expiration
            min_days_to_expiry: Minimum number of days until expiration
            target_delta: Target delta for the put (negative because it's a put)
            delta_range: Acceptable range around the target delta
            profit_target_pct: % of premium to reach to take profits
            stop_loss_pct: % of premium to lose to cut the position
            roll_when_dte: Number of days remaining before expiration to roll the position
            use_technical_filters: Use technical filters for entry
        """
        super().__init__(name="Cash Secured Put", description="Vente de puts couverts par du cash")
        
        self.min_implied_volatility = min_implied_volatility
        self.max_days_to_expiry = max_days_to_expiry
        self.min_days_to_expiry = min_days_to_expiry
        self.target_delta = target_delta  # Négatif car put
        self.delta_range = delta_range
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.roll_when_dte = roll_when_dte
        self.use_technical_filters = use_technical_filters
        
        self.current_put: Optional[OptionContract] = None
        self.position_size: int = 0
        self.entry_premium: float = 0
        self.max_drawdown: float = 0
        
    async def should_enter(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if a new Cash Secured Put position should be opened.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if entry is recommended, False otherwise
        """
        if not await self._check_account_requirements():
            return False
            
        # Vérifier si nous avons déjà une position ouverte
        if self.current_put is not None:
            return False
            
        # Vérifier les conditions techniques du sous-jacent
        if self.use_technical_filters and not self._check_technical_filters(market_data):
            logger.info(f"Les filtres techniques ne sont pas satisfaits pour {self.underlying_symbol}")
            return False
            
        # Obtenir le dernier prix du sous-jacent
        last_price = market_data['close'].iloc[-1] if not market_data.empty else None
        
        if not last_price:
            logger.warning(f"Impossible d'obtenir le dernier prix pour {self.underlying_symbol}")
            return False
            
        # Chercher une opportunité de vente de put
        try:
            option_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                option_type=OptionType.PUT
            )
            
            if not option_chain:
                logger.warning(f"Aucune option disponible pour {self.underlying_symbol}")
                return False
                
            # Filtrer les options selon nos critères
            filtered_options = self._filter_options(option_chain, last_price)
            
            if not filtered_options:
                logger.info(f"Aucune option ne correspond à nos critères pour {self.underlying_symbol}")
                return False
                
            # Sélectionner la meilleure option
            self.current_put = self._select_best_option(filtered_options)
            
            if not self.current_put:
                return False
                
            # Vérifier que nous avons suffisamment de liquidités
            required_cash = self.current_put.strike * 100  # Un contrat = 100 actions
            available_cash = self.account_size * self.max_position_size
            
            self.position_size = max(1, int(available_cash // required_cash))
            
            # Limiter la taille de la position
            max_contracts = int((self.account_size * self.max_position_size) // required_cash)
            self.position_size = min(self.position_size, max_contracts)
            
            if self.position_size < 1:
                logger.info(f"Pas assez de liquidités pour ouvrir une position CSP sur {self.underlying_symbol}")
                self.current_put = None
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation d'entrée CSP: {str(e)}")
            return False
            
    async def should_exit(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if the CSP position should be closed.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if exit is recommended, False otherwise
        """
        if not self.current_put:
            return False
            
        try:
            # Récupérer les données actuelles de l'option
            option_data = await self._get_current_option_data()
            
            if not option_data:
                logger.warning(f"Impossible d'obtenir les données actuelles pour l'option {self.current_put.symbol}")
                return False
                
            # Calcul des jours restants jusqu'à l'expiration
            today = datetime.now().date()
            expiry = datetime.strptime(self.current_put.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = (expiry - today).days
            
            # Prix actuel du put
            current_price = option_data.get('last', 0)
            if not current_price and option_data.get('bid', 0) > 0:
                # Utiliser le prix bid si last n'est pas disponible
                current_price = option_data.get('bid', 0)
                
            # Critères de sortie
            
            # 1. Atteinte de l'objectif de profit
            profit_target = self.entry_premium * (1 - self.profit_target_pct)
            if current_price <= profit_target:
                logger.info(f"Sortie CSP: objectif de profit atteint pour {self.current_put.symbol}")
                return True
                
            # 2. Stop loss
            stop_loss = self.entry_premium * (1 + self.stop_loss_pct)
            if current_price >= stop_loss:
                logger.info(f"Sortie CSP: stop loss atteint pour {self.current_put.symbol}")
                return True
                
            # 3. Proche de l'expiration
            if days_to_expiry <= self.roll_when_dte:
                logger.info(f"Sortie CSP: proche de l'expiration ({days_to_expiry} jours)")
                return True
                
            # 4. L'option est profondément en la monnaie
            last_price = market_data['close'].iloc[-1] if not market_data.empty else None
            if last_price and last_price < self.current_put.strike * 0.95:
                logger.info(f"Sortie CSP: option profondément en la monnaie, prix sous {self.current_put.strike * 0.95}")
                return True
                
            # Suivre le drawdown maximal
            if current_price > self.entry_premium:
                drawdown_pct = (current_price - self.entry_premium) / self.entry_premium
                self.max_drawdown = max(self.max_drawdown, drawdown_pct)
                
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de sortie CSP: {str(e)}")
            return False
            
    async def execute_entry(self) -> Dict[str, Any]:
        """
        Executes the entry position for the CSP strategy.
        
        Returns:
            Dict[str, Any]: Result of the entry operation
        """
        if not self.current_put:
            return {"success": False, "error": "Aucune option sélectionnée pour l'entrée CSP"}
            
        try:
            # Vendre des puts
            order_result = await self.broker.place_option_order(
                option_symbol=self.current_put.symbol,
                qty=self.position_size,
                side="sell",  # Vendre des puts
                order_type="limit",
                limit_price=self.current_put.bid,
                time_in_force="day"
            )
            
            if order_result.get("success", False):
                self.entry_premium = self.current_put.bid
                self.max_drawdown = 0
                
                # Enregistrer les détails de la transaction
                order_id = order_result.get("order_id", "unknown")
                logger.info(f"CSP entry exécutée: {self.position_size} contrats de {self.current_put.symbol} vendus à {self.entry_premium}")
                
                # Calculer le collatéral immobilisé
                collateral = self.current_put.strike * 100 * self.position_size
                max_potential_profit = self.entry_premium * 100 * self.position_size
                
                entry_result = {
                    "success": True,
                    "order_id": order_id,
                    "strategy": "Cash Secured Put",
                    "action": "SELL_PUT",
                    "symbol": self.current_put.symbol,
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "premium": self.entry_premium,
                    "strike": self.current_put.strike,
                    "expiry": self.current_put.expiry_date,
                    "collateral": collateral,
                    "max_profit": max_potential_profit,
                    "trade_date": datetime.now().isoformat()
                }
                
                self._log_trade("ENTRY", entry_result)
                return entry_result
            else:
                error_msg = order_result.get("error", "Échec de l'ordre sans message d'erreur")
                logger.error(f"Échec de l'entrée CSP: {error_msg}")
                self.current_put = None
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception lors de l'exécution de l'entrée CSP: {str(e)}")
            self.current_put = None
            return {"success": False, "error": str(e)}
            
    async def execute_exit(self) -> Dict[str, Any]:
        """
        Executes the exit from the CSP position.
        
        Returns:
            Dict[str, Any]: Result of the exit operation
        """
        if not self.current_put:
            return {"success": False, "error": "Aucune position CSP ouverte à fermer"}
            
        try:
            # Récupérer les données actuelles de l'option
            option_data = await self._get_current_option_data()
            
            if not option_data:
                return {
                    "has_position": True,
                    "error": f"Impossible d'obtenir les données pour {self.current_put.symbol}"
                }
                
            # Acheter des puts pour fermer la position (un put vendu est racheté)
            ask_price = option_data.get("ask", 0)
            if ask_price <= 0:
                logger.warning(f"Prix ask invalide ({ask_price}) pour {self.current_put.symbol}, utilisation du dernier prix")
                ask_price = option_data.get("last", 0.05)  # Prix minimal
                
            order_result = await self.broker.place_option_order(
                option_symbol=self.current_put.symbol,
                qty=self.position_size,
                side="buy",  # Racheter les puts
                order_type="limit",
                limit_price=ask_price,
                time_in_force="day"
            )
            
            if order_result.get("success", False):
                exit_premium = ask_price
                profit_loss = (self.entry_premium - exit_premium) * 100 * self.position_size
                
                # Enregistrer les détails de la transaction
                order_id = order_result.get("order_id", "unknown")
                logger.info(f"CSP exit exécutée: {self.position_size} contrats de {self.current_put.symbol} rachetés à {exit_premium}")
                
                exit_result = {
                    "success": True,
                    "order_id": order_id,
                    "strategy": "Cash Secured Put",
                    "action": "BUY_PUT",
                    "symbol": self.current_put.symbol,
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "entry_premium": self.entry_premium,
                    "exit_premium": exit_premium,
                    "profit_loss": profit_loss,
                    "max_drawdown_pct": self.max_drawdown * 100,
                    "strike": self.current_put.strike,
                    "trade_date": datetime.now().isoformat()
                }
                
                # Réinitialiser les variables de position
                self.current_put = None
                self.position_size = 0
                self.entry_premium = 0
                self.max_drawdown = 0
                
                self._log_trade("EXIT", exit_result)
                return exit_result
            else:
                error_msg = order_result.get("error", "Échec de l'ordre sans message d'erreur")
                logger.error(f"Échec de la sortie CSP: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception lors de l'exécution de la sortie CSP: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def get_position_info(self) -> Dict[str, Any]:
        """
        Provides information about the current CSP position.
        
        Returns:
            Dict: Details of the current position
        """
        if not self.current_put:
            return {"has_position": False}
            
        try:
            # Récupérer les données actuelles de l'option
            option_data = await self._get_current_option_data()
            
            if not option_data:
                return {
                    "has_position": True,
                    "error": f"Impossible d'obtenir les données actuelles pour {self.current_put.symbol}"
                }
                
            # Prix actuel du put
            current_price = option_data.get('last', 0)
            if not current_price and option_data.get('bid', 0) > 0:
                # Utiliser le prix bid si last n'est pas disponible
                current_price = option_data.get('bid', 0)
                
            # Calcul des métriques de la position
            unrealized_pl = (self.entry_premium - current_price) * 100 * self.position_size
            unrealized_pl_pct = ((self.entry_premium - current_price) / self.entry_premium) * 100 if self.entry_premium > 0 else 0
            
            # Jours restants jusqu'à l'expiration
            today = datetime.now().date()
            expiry = datetime.strptime(self.current_put.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = (expiry - today).days
            
            return {
                "has_position": True,
                "strategy": "Cash Secured Put",
                "symbol": self.current_put.symbol,
                "underlying": self.underlying_symbol,
                "quantity": self.position_size,
                "strike": self.current_put.strike,
                "expiry": self.current_put.expiry_date,
                "days_to_expiry": days_to_expiry,
                "entry_premium": self.entry_premium,
                "current_premium": current_price,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pct": unrealized_pl_pct,
                "max_drawdown_pct": self.max_drawdown * 100,
                "collateral": self.current_put.strike * 100 * self.position_size
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations de position CSP: {str(e)}")
            return {
                "has_position": True,
                "error": str(e),
                "symbol": self.current_put.symbol if self.current_put else "unknown"
            }
            
    def _filter_options(self, options: List[OptionContract], stock_price: float) -> List[OptionContract]:
        """
        Filters options according to strategy criteria.
        
        Args:
            options: List of available option contracts
            stock_price: Current price of the underlying asset
            
        Returns:
            List[OptionContract]: Filtered options
        """
        filtered = []
        
        for option in options:
            # Vérifier qu'il s'agit bien d'un put
            if option.option_type != OptionType.PUT:
                continue
                
            # Calculer les jours jusqu'à l'expiration
            today = datetime.now().date()
            try:
                expiry = datetime.strptime(option.expiry_date, "%Y-%m-%d").date()
                days_to_expiry = (expiry - today).days
            except:
                # Si le format de date est incorrect, passer à l'option suivante
                continue
                
            # Filtres de base
            if (
                days_to_expiry < self.min_days_to_expiry or
                days_to_expiry > self.max_days_to_expiry or
                option.implied_volatility < self.min_implied_volatility or
                option.bid <= 0.05  # Prix minimal pour éviter les options illiquides
            ):
                continue
                
            # Filtre sur le delta
            if not (self.target_delta - self.delta_range <= option.delta <= self.target_delta + self.delta_range):
                continue
                
            # Vérifier si l'option a suffisamment de liquidité
            if option.volume < 10 or option.open_interest < 100:
                continue
                
            # Vérifier le spread bid-ask (ne pas dépasser 10%)
            if option.ask > 0 and option.bid > 0:
                spread_pct = (option.ask - option.bid) / option.bid
                if spread_pct > 0.10:
                    continue
                    
            # Ajouter l'option aux résultats filtrés
            filtered.append(option)
            
        return filtered
        
    def _select_best_option(self, filtered_options: List[OptionContract]) -> Optional[OptionContract]:
        """
        Selects the best option from the filtered options.
        
        Args:
            filtered_options: List of filtered options
            
        Returns:
            Optional[OptionContract]: Best option or None
        """
        if not filtered_options:
            return None
            
        # Trier par proximité au delta cible
        sorted_by_delta = sorted(filtered_options, key=lambda x: abs(x.delta - self.target_delta))
        
        # Prendre les 3 meilleures options par delta
        top_delta_options = sorted_by_delta[:3] if len(sorted_by_delta) >= 3 else sorted_by_delta
        
        # Parmi ces options, prendre celle avec le meilleur rapport premium/jours
        best_option = None
        best_ratio = 0
        
        for option in top_delta_options:
            # Calculer les jours jusqu'à l'expiration
            today = datetime.now().date()
            expiry = datetime.strptime(option.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = max(1, (expiry - today).days)  # Éviter division par zéro
            
            # Calculer le ratio premium annualisé / capital immobilisé
            premium_per_day = option.bid / days_to_expiry
            capital_required = option.strike
            
            # Annualiser le rendement
            annual_yield = (premium_per_day * 365) / capital_required
            
            if annual_yield > best_ratio:
                best_ratio = annual_yield
                best_option = option
                
        return best_option
        
    def _check_technical_filters(self, market_data: pd.DataFrame) -> bool:
        """
        Checks technical filters for position entry.
        
        Args:
            market_data: Market data
            
        Returns:
            bool: True if filters are satisfied, False otherwise
        """
        if market_data.empty or len(market_data) < 20:
            return False
            
        try:
            # Calculer les moyennes mobiles
            market_data['sma20'] = market_data['close'].rolling(window=20).mean()
            market_data['sma50'] = market_data['close'].rolling(window=50).mean()
            market_data['sma200'] = market_data['close'].rolling(window=200).mean()
            
            # Dernières valeurs
            last_close = market_data['close'].iloc[-1]
            last_sma20 = market_data['sma20'].iloc[-1]
            last_sma50 = market_data['sma50'].iloc[-1]
            last_sma200 = market_data['sma200'].iloc[-1]
            
            # Calculer le RSI sur 14 périodes
            delta = market_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            last_rsi = rsi.iloc[-1]
            
            # Vérifier la tendance globale (vendre des puts sur tendance haussière)
            uptrend = last_close > last_sma50 > last_sma200
            
            # Vérifier le RSI (pas de survente)
            rsi_ok = last_rsi >= 40  # Éviter les marchés en forte baisse
            
            # Vérifier si le prix est dans une zone de support
            near_support = False
            
            # Calculer les niveaux de support à partir des creux locaux récents
            lows = market_data['low'].rolling(window=10, center=True).min()
            local_lows = []
            
            for i in range(5, len(lows) - 5):
                if lows.iloc[i] == market_data['low'].iloc[i] and lows.iloc[i] < lows.iloc[i-5] and lows.iloc[i] < lows.iloc[i+5]:
                    local_lows.append(lows.iloc[i])
            
            # Vérifier si le prix actuel est proche d'un support (à 3% près)
            if local_lows:
                for support_level in local_lows[-5:]:  # Considérer uniquement les 5 derniers niveaux
                    if abs(last_close - support_level) / last_close < 0.03:
                        near_support = True
                        break
            
            # Pour CSP, nous voulons vendre des puts quand le marché est haussier
            # ou stable et que le prix n'est pas en train de chuter
            return uptrend and rsi_ok
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des filtres techniques: {str(e)}")
            return False
            
    async def _get_current_option_data(self) -> Dict[str, Any]:
        """
        Retrieves current data for the current option.
        
        Returns:
            Dict: Current option data
        """
        if not self.current_put:
            return {}
            
        try:
            # Essayer de récupérer les informations de prix actuelles
            option_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                expiration_date=self.current_put.expiry_date,
                option_type=OptionType.PUT
            )
            
            if not option_chain:
                return {}
                
            # Trouver notre option spécifique
            for option in option_chain:
                if (option.symbol == self.current_put.symbol or 
                    (option.strike == self.current_put.strike and 
                     option.expiry_date == self.current_put.expiry_date and
                     option.option_type == OptionType.PUT)):
                    return {
                        "bid": option.bid,
                        "ask": option.ask,
                        "last": option.last,
                        "delta": option.delta,
                        "implied_volatility": option.implied_volatility,
                        "volume": option.volume,
                        "open_interest": option.open_interest
                    }
                    
            return {}
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données d'option actuelles: {str(e)}")
            return {}
            
    async def _check_account_requirements(self) -> bool:
        """
        Checks if the account meets the requirements for this strategy.
        
        Returns:
            bool: True if conditions are met, False otherwise
        """
        try:
            # Fix for broker attribute missing - use broker_adapter if available
            if not hasattr(self, 'broker'):
                if hasattr(self, 'broker_adapter'):
                    self.broker = self.broker_adapter
                    logger.info(f"Fixed broker attribute using broker_adapter")
                elif hasattr(self, 'trading_service') and hasattr(self.trading_service, 'broker'):
                    self.broker = self.trading_service.broker
                    logger.info(f"Fixed broker attribute using trading_service.broker")
                else:
                    logger.warning("No broker available for strategy")
                    return False
            
            # Support both sync and async get_account
            import inspect
            account_info = self.broker.get_account()
            if inspect.isawaitable(account_info):
                account_info = await account_info
            if not account_info:
                logger.warning("Impossible de récupérer les informations du compte")
                return False

            # Support both Account object and dict
            buying_power = getattr(account_info, "buying_power", None)
            if buying_power is None and isinstance(account_info, dict):
                buying_power = account_info.get("buying_power")
            if buying_power is None:
                logger.warning("Impossible de récupérer buying_power depuis le compte")
                return False

            buying_power = float(buying_power)

            if buying_power < 5000:  # Minimum requis pour CSP
                logger.warning(f"Pouvoir d'achat insuffisant pour la stratégie CSP: {buying_power}")
                return False
                
            # Vérifier si le trading d'options est activé
            if not self.broker.enable_options:
                logger.warning("Le trading d'options n'est pas activé")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des exigences du compte: {str(e)}")
            return False
