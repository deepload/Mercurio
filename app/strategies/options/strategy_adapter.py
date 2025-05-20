#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Options Strategy Adapter

Ce module fournit des adaptateurs pour uniformiser l'initialisation
des différentes stratégies d'options qui ont des signatures de constructeurs différentes.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Type

from app.strategies.options.base_options_strategy import BaseOptionsStrategy
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.cash_secured_put import CashSecuredPutStrategy
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy

from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.options_service import OptionsService


logger = logging.getLogger(__name__)


class OptionsStrategyAdapter:
    """
    Adaptateur pour uniformiser l'initialisation des stratégies d'options.
    """

    @staticmethod
    def create_strategy(
        strategy_name: str,
        symbol: str,
        market_data_service: MarketDataService,
        trading_service: TradingService,
        options_service: OptionsService,
        account_size: float,
        max_position_size: float = 0.10,
        days_to_expiration: int = 30,
        delta_target: float = 0.30,
        profit_target_pct: float = 0.50,
        stop_loss_pct: float = 0.20,
        **kwargs
    ) -> BaseOptionsStrategy:
        """
        Crée une instance de stratégie d'options avec l'interface unifiée.
        
        Args:
            strategy_name: Nom de la stratégie (COVERED_CALL, CASH_SECURED_PUT, etc.)
            symbol: Symbole de l'actif sous-jacent
            market_data_service: Service de données de marché
            trading_service: Service de trading
            options_service: Service d'options
            account_size: Taille du compte en USD
            max_position_size: Taille maximale de position en % du compte
            days_to_expiration: Jours jusqu'à l'expiration cible
            delta_target: Delta cible pour les stratégies basées sur les grecques
            profit_target_pct: Cible de profit en pourcentage
            stop_loss_pct: Stop loss en pourcentage
            
        Returns:
            Une instance de stratégie d'options correctement initialisée
        """
        strategy = None
        
        # Paramètres communs pour toutes les stratégies
        common_params = {
            "max_position_size": max_position_size,
            "profit_target_pct": profit_target_pct,
            "stop_loss_pct": stop_loss_pct,
        }
        
        # Spécifiques par stratégie
        if strategy_name == 'COVERED_CALL':
            # Filtre les kwargs spécifiques à cette stratégie pour éviter les arguments inattendus
            covered_call_kwargs = {
                'delta_target': delta_target,
                'dte_range': (days_to_expiration - 5, days_to_expiration + 5),
                'sizing_pct': max_position_size,
                'profit_target': profit_target_pct,
                'stop_loss_pct': stop_loss_pct
            }
            strategy = CoveredCallStrategy(
                ticker=symbol,
                market_data_service=market_data_service,
                trading_service=trading_service,
                options_service=options_service,
                **covered_call_kwargs
            )
        
        elif strategy_name == 'CASH_SECURED_PUT':
            # Filtre les kwargs spécifiques à cette stratégie
            csp_kwargs = {
                'max_position_size': max_position_size,
                'target_delta': -delta_target,  # Négatif pour les puts
                'min_days_to_expiry': days_to_expiration - 5,
                'max_days_to_expiry': days_to_expiration + 5,
                'profit_target_pct': profit_target_pct,
                'stop_loss_pct': stop_loss_pct
            }
            strategy = CashSecuredPutStrategy(
                underlying_symbol=symbol,
                account_size=account_size,
                **csp_kwargs
            )
            
        elif strategy_name == 'LONG_CALL':
            long_call_kwargs = {
                'max_position_size': max_position_size,
                'days_to_expiration': days_to_expiration,
                'profit_target_pct': profit_target_pct,
                'stop_loss_pct': stop_loss_pct
            }
            strategy = LongCallStrategy(
                underlying_symbol=symbol,
                account_size=account_size,
                **long_call_kwargs
            )
            
        elif strategy_name == 'LONG_PUT':
            long_put_kwargs = {
                'max_position_size': max_position_size,
                'days_to_expiration': days_to_expiration,
                'profit_target_pct': profit_target_pct,
                'stop_loss_pct': stop_loss_pct
            }
            strategy = LongPutStrategy(
                underlying_symbol=symbol,
                account_size=account_size,
                **long_put_kwargs
            )
            
        elif strategy_name == 'IRON_CONDOR':
            # Map parameters to match IronCondorStrategy signature
            iron_condor_kwargs = {
                'max_position_size': max_position_size,
                'max_days_to_expiry': days_to_expiration,  # Map to max_days_to_expiry
                'min_days_to_expiry': max(5, days_to_expiration - 10),  # Provide a reasonable min_days_to_expiry
                'profit_target_pct': profit_target_pct,
                'stop_loss_pct': stop_loss_pct
            }
            try:
                strategy = IronCondorStrategy(
                    underlying_symbol=symbol,
                    account_size=account_size,
                    **iron_condor_kwargs
                )
            except TypeError as e:
                # Si la classe IronCondorStrategy ne prend pas ces arguments, essayons une alternative
                logger.warning(f"Adaptation pour IronCondorStrategy: {e}")
                strategy = IronCondorStrategy(
                    symbol=symbol,
                    account_size=account_size,
                    **iron_condor_kwargs
                )
        
        elif strategy_name == 'BUTTERFLY':
            butterfly_kwargs = {
                'max_position_size': max_position_size,
                'days_to_expiration': days_to_expiration,
                'delta_target': delta_target,
                'profit_target_pct': profit_target_pct,
                'stop_loss_pct': stop_loss_pct,
                'option_type': 'call'  # Défaut à call pour la stratégie butterfly
            }
            try:
                strategy = ButterflySpreadStrategy(
                    underlying_symbol=symbol,
                    **butterfly_kwargs
                )
                # Enregistrer explicitement le symbole pour BUTTERFLY
                strategy.symbol = symbol
                strategy.underlying_symbol = symbol
            except TypeError as e:
                # Si ButterflySpreadStrategy ne prend pas ces arguments, essayons une alternative
                logger.warning(f"Adaptation pour ButterflySpreadStrategy: {e}")
                modified_kwargs = butterfly_kwargs.copy()
                if 'days_to_expiration' in modified_kwargs:
                    modified_kwargs['days_to_expiry'] = modified_kwargs.pop('days_to_expiration')
                strategy = ButterflySpreadStrategy(
                    symbol=symbol,
                    **modified_kwargs
                )
                # Enregistrer explicitement le symbole pour BUTTERFLY
                strategy.symbol = symbol
                strategy.underlying_symbol = symbol
            
        else:
            raise ValueError(f"Stratégie non supportée: {strategy_name}")
        
        # Ajoutons des références aux services pour toutes les stratégies
        if hasattr(strategy, 'broker_adapter') and trading_service and hasattr(trading_service, 'broker'):
            strategy.broker_adapter = trading_service.broker
            
        if hasattr(strategy, 'options_service') and options_service:
            strategy.options_service = options_service
            
        if hasattr(strategy, 'market_data_service') and market_data_service:
            strategy.market_data_service = market_data_service
            
        # Assurons-nous que toutes les stratégies ont un attribut symbol ou underlying_symbol
        if not hasattr(strategy, 'symbol') and not hasattr(strategy, 'underlying_symbol'):
            setattr(strategy, 'symbol', symbol)
            setattr(strategy, 'underlying_symbol', symbol)
        
        logger.info(f"Stratégie {strategy_name} initialisée pour {symbol}")
        return strategy
