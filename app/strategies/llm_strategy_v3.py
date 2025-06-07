import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Any, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from app.strategies.base import BaseStrategy
from app.db.models import TradeAction
from app.utils.llm_utils import load_llm_model, call_llm
from app.strategies.sentiment.news_api_agent import NewsAPIAgent
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class LLMStrategyV3(BaseStrategy):
    """
    Stratégie avancée MercurioAI LLM_V3 avec architecture multi-agent Claude :
      - Claude Trader : analyse technique + prise de décision
      - Claude Analyste : analyse des news et médias en temps réel
      - Claude Coordinateur : fusionne et arbitre les signaux pour agir avec un coup d'avance
      - Système d'apprentissage en boucle fermée : les agents apprennent de leurs décisions passées
    """
    def __init__(self,
                 trader_model_name: str = "claude-3-7-sonnet-20240620",
                 analyst_model_name: str = "claude-3-7-sonnet-20240620",
                 coordinator_model_name: str = "claude-3-7-sonnet-20240620",
                 api_key: Optional[str] = None,
                 news_lookback_hours: int = 2,
                 sentiment_weight: float = 0.6,
                 min_confidence: float = 0.65,
                 position_size: float = 0.02,
                 stop_loss: float = 0.03,
                 take_profit: float = 0.06,
                 data_provider: str = "binance",
                 use_local_model: bool = False,
                 local_model_path: Optional[str] = None,
                 memory_size: int = 10,  # Nombre de décisions précédentes à conserver en mémoire
                 feedback_window: int = 24,  # Fenêtre de temps (en heures) pour évaluer les performances
                 memory_path: str = "data/agent_memory",  # Chemin pour stocker la mémoire des agents
                 **kwargs):
        super().__init__(**kwargs)
        self.name = "LLMStrategyV3"
        self.description = "LLM_V3: Claude multi-agent (trader, analyste, coordinateur) avec apprentissage en boucle fermée"
        self.trader_model_name = trader_model_name
        self.analyst_model_name = analyst_model_name
        self.coordinator_model_name = coordinator_model_name
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self.news_lookback_hours = news_lookback_hours
        self.sentiment_weight = sentiment_weight
        self.technical_weight = 1.0 - sentiment_weight
        self.min_confidence = min_confidence
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.data_provider = data_provider
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        
        # Paramètres du système d'apprentissage en boucle fermée
        self.memory_size = memory_size
        self.feedback_window = feedback_window
        self.memory_path = memory_path
        
        # Création du répertoire de mémoire s'il n'existe pas
        Path(self.memory_path).mkdir(parents=True, exist_ok=True)

        # Services
        self.market_data = MarketDataService()
        self.news_agent = NewsAPIAgent(lookback_hours=self.news_lookback_hours)

        # LLM Agents
        self.claude_trader = load_llm_model(
            model_name=self.trader_model_name,
            use_local=self.use_local_model,
            local_path=self.local_model_path,
            api_key=self.api_key
        )
        self.claude_analyst = load_llm_model(
            model_name=self.analyst_model_name,
            use_local=self.use_local_model,
            local_path=self.local_model_path,
            api_key=self.api_key
        )
        self.claude_coordinator = load_llm_model(
            model_name=self.coordinator_model_name,
            use_local=self.use_local_model,
            local_path=self.local_model_path,
            api_key=self.api_key
        )
        
        # Initialisation des mémoires des agents
        self._init_agent_memories()

        logger.info(f"Initialized LLMStrategyV3 with trader model: {self.trader_model_name}, analyst model: {self.analyst_model_name}, coordinator model: {self.coordinator_model_name}")
    
    def _init_agent_memories(self):
        """
        Initialise les fichiers de mémoire pour les agents
        """
        # Créer le répertoire de mémoire s'il n'existe pas
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Créer le répertoire d'archives s'il n'existe pas
        archive_dir = os.path.join(self.memory_path, "archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Créer le répertoire des trades s'il n'existe pas
        trades_dir = os.path.join(self.memory_path, "trades")
        os.makedirs(trades_dir, exist_ok=True)
        
        # Initialiser les fichiers de mémoire pour chaque agent
        self.memory_files = {
            "trader": os.path.join(self.memory_path, "trader_memory.json"),
            "analyst": os.path.join(self.memory_path, "analyst_memory.json"),
            "coordinator": os.path.join(self.memory_path, "coordinator_memory.json"),
            "performance": os.path.join(self.memory_path, "performance_history.json")
        }
        
        # Création des fichiers de mémoire s'ils n'existent pas
        for memory_file in self.memory_files.values():
            if not os.path.exists(memory_file):
                with open(memory_file, 'w') as f:
                    json.dump([], f)
                    
        # Nettoyer et archiver les fichiers de mémoire anciens
        self._cleanup_memory_files()
    
    def _cleanup_memory_files(self):
        """
        Nettoie et archive les fichiers de mémoire anciens
        """
        try:
            # Définir la date limite pour les fichiers à archiver (30 jours)
            cutoff_date = datetime.now() - timedelta(days=30)
            cutoff_time = cutoff_date.isoformat()
            
            # Chemin du répertoire d'archives
            archive_dir = os.path.join(self.memory_path, "archives")
            
            # Vérifier les fichiers dans le répertoire des trades
            trades_dir = os.path.join(self.memory_path, "trades")
            if os.path.exists(trades_dir):
                for filename in os.listdir(trades_dir):
                    filepath = os.path.join(trades_dir, filename)
                    if os.path.isfile(filepath):
                        # Vérifier la date de modification du fichier
                        file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if file_mod_time < cutoff_date:
                            # Archiver le fichier
                            archive_path = os.path.join(archive_dir, f"trades_{datetime.now().strftime('%Y%m%d')}_{filename}")
                            os.replace(filepath, archive_path)
                            logger.info(f"Fichier de trades archivé: {filename}")
            
            # Archiver les performances anciennes
            performance_file = self.memory_files.get("performance")
            if os.path.exists(performance_file):
                try:
                    with open(performance_file, 'r') as f:
                        performances = json.load(f)
                    
                    # Filtrer les performances récentes
                    recent_performances = [p for p in performances if p.get("datetime", "") >= cutoff_time]
                    
                    # Si des performances ont été filtrées, archiver les anciennes
                    if len(recent_performances) < len(performances):
                        # Sauvegarder les performances anciennes dans une archive
                        archive_path = os.path.join(archive_dir, f"performance_history_{datetime.now().strftime('%Y%m%d')}.json")
                        with open(archive_path, 'w') as f:
                            json.dump([p for p in performances if p.get("datetime", "") < cutoff_time], f, indent=2)
                        
                        # Mettre à jour le fichier de performances avec uniquement les récentes
                        with open(performance_file, 'w') as f:
                            json.dump(recent_performances, f, indent=2)
                            
                        logger.info(f"Performances anciennes archivées: {len(performances) - len(recent_performances)} entrées")
                except Exception as e:
                    logger.error(f"Erreur lors de l'archivage des performances: {e}")
            
            # Vérifier les autres fichiers de mémoire
            for agent_type in ["analyst", "trader", "coordinator"]:
                memory_file = self.memory_files.get(agent_type)
                if os.path.exists(memory_file):
                    try:
                        with open(memory_file, 'r') as f:
                            memories = json.load(f)
                        
                        # Filtrer les mémoires récentes
                        recent_memories = [m for m in memories if m.get("datetime", "") >= cutoff_time]
                        
                        # Si des mémoires ont été filtrées, archiver les anciennes
                        if len(recent_memories) < len(memories):
                            # Sauvegarder les mémoires anciennes dans une archive
                            archive_path = os.path.join(archive_dir, f"{agent_type}_memory_{datetime.now().strftime('%Y%m%d')}.json")
                            with open(archive_path, 'w') as f:
                                json.dump([m for m in memories if m.get("datetime", "") < cutoff_time], f, indent=2)
                            
                            # Mettre à jour le fichier de mémoires avec uniquement les récentes
                            with open(memory_file, 'w') as f:
                                json.dump(recent_memories, f, indent=2)
                                
                            logger.info(f"Mémoires anciennes de {agent_type} archivées: {len(memories) - len(recent_memories)} entrées")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'archivage des mémoires de {agent_type}: {e}")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des fichiers de mémoire: {e}")
    
    def get_trade_history(self, symbol: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des trades pour un symbole donné
        
        Args:
            symbol: Le symbole pour lequel récupérer l'historique
            limit: Nombre maximum de trades à récupérer (None pour tous)
            
        Returns:
            Une liste de dictionnaires contenant les détails des trades
        """
        try:
            # Chemin du fichier de trades pour ce symbole
            trades_file = os.path.join(self.memory_path, "trades", f"{symbol.replace('/', '_')}_trades.json")
            
            if not os.path.exists(trades_file):
                return []
                
            # Charger les trades
            with open(trades_file, "r") as f:
                try:
                    trades = json.load(f)
                except json.JSONDecodeError:
                    return []
            
            # Trier par date (plus récent d'abord)
            trades.sort(key=lambda x: x.get("datetime", ""), reverse=True)
            
            # Limiter le nombre de trades si nécessaire
            if limit is not None and limit > 0:
                trades = trades[:limit]
                
            return trades
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique des trades pour {symbol}: {e}")
            return []
            
    def export_performance_stats(self, output_path: str = None, days: int = 30) -> Dict[str, Any]:
        """
        Exporte les statistiques de performance du système d'apprentissage en boucle fermée
        
        Args:
            output_path: Chemin où exporter les statistiques (None pour retourner sans exporter)
            days: Nombre de jours à analyser
            
        Returns:
            Un dictionnaire contenant les statistiques globales et par symbole
        """
        try:
            # Définir la période d'analyse
            cutoff_time = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Récupérer les performances
            with open(self.memory_files["performance"], 'r') as f:
                try:
                    performances = json.load(f)
                except json.JSONDecodeError:
                    performances = []
            
            # Filtrer par période
            recent_performances = [p for p in performances if p.get("datetime", "") >= cutoff_time]
            
            if not recent_performances:
                stats = {
                    "global": {
                        "total_trades": 0,
                        "win_rate": 0,
                        "profit_loss": 0,
                        "trader_accuracy": 0,
                        "analyst_accuracy": 0,
                        "coordinator_accuracy": 0
                    },
                    "symbols": {}
                }
                
                if output_path:
                    with open(output_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                        
                return stats
            
            # Calculer les statistiques globales
            total_trades = len(recent_performances)
            trader_correct = sum(1 for p in recent_performances if p.get("trader_correct", False))
            analyst_correct = sum(1 for p in recent_performances if p.get("analyst_correct", False))
            coordinator_correct = sum(1 for p in recent_performances if p.get("coordinator_correct", False))
            
            profits = [p.get("profit_loss", 0) for p in recent_performances if p.get("profit_loss", 0) > 0]
            losses = [p.get("profit_loss", 0) for p in recent_performances if p.get("profit_loss", 0) < 0]
            
            win_rate = len(profits) / total_trades if total_trades > 0 else 0
            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            total_pnl = sum(p.get("profit_loss", 0) for p in recent_performances)
            
            # Statistiques globales
            global_stats = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_loss": total_pnl,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "trader_accuracy": trader_correct / total_trades if total_trades > 0 else 0,
                "analyst_accuracy": analyst_correct / total_trades if total_trades > 0 else 0,
                "coordinator_accuracy": coordinator_correct / total_trades if total_trades > 0 else 0
            }
            
            # Statistiques par symbole
            symbols = set(p.get("symbol") for p in recent_performances)
            symbol_stats = {}
            
            for symbol in symbols:
                symbol_perfs = [p for p in recent_performances if p.get("symbol") == symbol]
                symbol_total = len(symbol_perfs)
                
                symbol_trader_correct = sum(1 for p in symbol_perfs if p.get("trader_correct", False))
                symbol_analyst_correct = sum(1 for p in symbol_perfs if p.get("analyst_correct", False))
                symbol_coordinator_correct = sum(1 for p in symbol_perfs if p.get("coordinator_correct", False))
                
                symbol_profits = [p.get("profit_loss", 0) for p in symbol_perfs if p.get("profit_loss", 0) > 0]
                symbol_losses = [p.get("profit_loss", 0) for p in symbol_perfs if p.get("profit_loss", 0) < 0]
                
                symbol_win_rate = len(symbol_profits) / symbol_total if symbol_total > 0 else 0
                symbol_avg_profit = sum(symbol_profits) / len(symbol_profits) if symbol_profits else 0
                symbol_avg_loss = sum(symbol_losses) / len(symbol_losses) if symbol_losses else 0
                symbol_pnl = sum(p.get("profit_loss", 0) for p in symbol_perfs)
                
                symbol_stats[symbol] = {
                    "total_trades": symbol_total,
                    "win_rate": symbol_win_rate,
                    "profit_loss": symbol_pnl,
                    "avg_profit": symbol_avg_profit,
                    "avg_loss": symbol_avg_loss,
                    "trader_accuracy": symbol_trader_correct / symbol_total if symbol_total > 0 else 0,
                    "analyst_accuracy": symbol_analyst_correct / symbol_total if symbol_total > 0 else 0,
                    "coordinator_accuracy": symbol_coordinator_correct / symbol_total if symbol_total > 0 else 0
                }
            
            # Assembler les statistiques
            stats = {
                "global": global_stats,
                "symbols": symbol_stats,
                "period": {
                    "start": cutoff_time,
                    "end": datetime.utcnow().isoformat(),
                    "days": days
                }
            }
            
            # Exporter si nécessaire
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"Statistiques de performance exportées vers {output_path}")
            
            return stats
        except Exception as e:
            logger.error(f"Erreur lors de l'export des statistiques de performance: {e}")
            return {"error": str(e)}
            
    def train_and_optimize(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Méthode d'entraînement et d'optimisation du système d'apprentissage en boucle fermée
        Utilisée pendant les périodes d'inactivité du marché pour améliorer les performances
        
        Args:
            symbols: Liste des symboles à analyser (None pour tous)
            
        Returns:
            Un dictionnaire contenant les résultats de l'entraînement
        """
        try:
            logger.info("Démarrage de l'entraînement et de l'optimisation du système LLM_V3")
            
            # Si aucun symbole n'est spécifié, récupérer tous les symboles disponibles
            if not symbols:
                # Récupérer tous les symboles pour lesquels nous avons des données
                trades_dir = os.path.join(self.memory_path, "trades")
                if os.path.exists(trades_dir):
                    trade_files = [f for f in os.listdir(trades_dir) if f.endswith("_trades.json")]
                    symbols = [f.replace("_trades.json", "").replace("_", "/") for f in trade_files]
            
            if not symbols:
                logger.warning("Aucun symbole disponible pour l'entraînement")
                return {"status": "error", "message": "Aucun symbole disponible"}
            
            results = {}
            
            # Pour chaque symbole
            for symbol in symbols:
                logger.info(f"Analyse des performances pour {symbol}")
                
                # Récupérer l'historique des trades
                trades = self.get_trade_history(symbol)
                
                if not trades:
                    logger.warning(f"Aucun trade disponible pour {symbol}")
                    results[symbol] = {"status": "skipped", "reason": "no_trades"}
                    continue
                
                # Analyser les performances
                performance = self.analyze_past_performance(symbol)
                
                # Générer un rapport de synthèse pour améliorer les prompts
                prompt_improvement = self._generate_prompt_improvement(symbol, performance, trades)
                
                # Mettre à jour les prompts des agents si nécessaire
                if prompt_improvement.get("should_update", False):
                    self._update_agent_prompts(symbol, prompt_improvement)
                
                # Nettoyer et archiver les anciennes données
                self._cleanup_memory_files()
                
                results[symbol] = {
                    "status": "success",
                    "trades_analyzed": len(trades),
                    "performance": performance,
                    "prompt_updated": prompt_improvement.get("should_update", False)
                }
            
            # Exporter les statistiques globales
            stats_path = os.path.join(self.memory_path, "performance_stats.json")
            stats = self.export_performance_stats(output_path=stats_path)
            
            logger.info(f"Entraînement et optimisation terminés pour {len(symbols)} symboles")
            
            return {
                "status": "success",
                "symbols_processed": len(symbols),
                "results": results,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement et de l'optimisation: {e}")
            return {"status": "error", "message": str(e)}
            
    def _generate_prompt_improvement(self, symbol: str, performance: Dict[str, Any], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Génère des améliorations pour les prompts des agents basées sur les performances passées
        
        Args:
            symbol: Le symbole concerné
            performance: Les métriques de performance
            trades: L'historique des trades
            
        Returns:
            Un dictionnaire contenant les améliorations suggérées
        """
        # Cette méthode pourrait être améliorée pour utiliser Claude lui-même pour analyser les performances
        # et suggérer des améliorations aux prompts, mais pour l'instant nous utilisons une approche simple
        
        try:
            win_rate = performance.get("win_rate", 0)
            trader_accuracy = float(str(performance.get("trader_accuracy", "0")).replace("N/A", "0"))
            analyst_accuracy = float(str(performance.get("news_accuracy", "0")).replace("N/A", "0"))
            coordinator_accuracy = float(str(performance.get("coordinator_accuracy", "0")).replace("N/A", "0"))
            
            # Déterminer s'il faut mettre à jour les prompts
            should_update = win_rate < 0.5 or trader_accuracy < 0.6 or analyst_accuracy < 0.6 or coordinator_accuracy < 0.6
            
            # Si les performances sont bonnes, pas besoin de mettre à jour
            if not should_update:
                return {"should_update": False}
            
            # Identifier les points faibles
            weaknesses = []
            if trader_accuracy < 0.6:
                weaknesses.append("analyse_technique")
            if analyst_accuracy < 0.6:
                weaknesses.append("analyse_news")
            if coordinator_accuracy < 0.6:
                weaknesses.append("coordination")
                
            # Suggérer des améliorations
            improvements = {
                "analyst": "Mettre plus d'accent sur les signaux forts dans les actualités et moins sur les bruits" if "analyse_news" in weaknesses else "",
                "trader": "Accorder plus d'importance aux tendances à moyen terme et moins aux fluctuations à court terme" if "analyse_technique" in weaknesses else "",
                "coordinator": "Améliorer la pondération entre les signaux techniques et les actualités" if "coordination" in weaknesses else ""
            }
            
            return {
                "should_update": True,
                "weaknesses": weaknesses,
                "improvements": improvements
            }
        except Exception as e:
            logger.error(f"Erreur lors de la génération des améliorations de prompt: {e}")
            return {"should_update": False}
    
    def _update_agent_prompts(self, symbol: str, improvements: Dict[str, Any]) -> None:
        """
        Met à jour les prompts des agents en fonction des améliorations suggérées
        
        Args:
            symbol: Le symbole concerné
            improvements: Les améliorations suggérées
        """
        # Cette méthode pourrait être développée pour modifier dynamiquement les prompts
        # Pour l'instant, nous nous contentons de logger les améliorations suggérées
        
        if not improvements.get("should_update", False):
            return
        
        logger.info(f"Améliorations suggérées pour les prompts de {symbol}:")
        for agent, improvement in improvements.get("improvements", {}).items():
            if improvement:
                logger.info(f"  - {agent}: {improvement}")
        
        # Dans une implémentation future, nous pourrions stocker ces améliorations
        # et les intégrer aux prompts lors de l'analyse
        
    def analyze_past_performance(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyse les performances passées pour un symbole donné
        
        Args:
            symbol: Le symbole à analyser
            days: Nombre de jours à analyser
            
        Returns:
            Un dictionnaire contenant les métriques de performance
        """
        try:
            # Récupérer l'historique des trades
            trades = self.get_trade_history(symbol)
            
            if not trades:
                return {
                    "symbol": symbol,
                    "total_trades": 0,
                    "win_rate": "N/A",
                    "avg_profit": "N/A",
                    "avg_loss": "N/A",
                    "total_pnl": 0,
                    "trader_accuracy": "N/A",
                    "news_accuracy": "N/A",
                    "coordinator_accuracy": "N/A"
                }
            
            # Définir la période d'analyse
            cutoff_time = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Filtrer les trades par période
            recent_trades = [t for t in trades if t.get("datetime", "") >= cutoff_time]
            
            if not recent_trades:
                return {
                    "symbol": symbol,
                    "total_trades": 0,
                    "win_rate": "N/A",
                    "avg_profit": "N/A",
                    "avg_loss": "N/A",
                    "total_pnl": 0,
                    "trader_accuracy": "N/A",
                    "news_accuracy": "N/A",
                    "coordinator_accuracy": "N/A"
                }
            
            # Calculer les métriques de performance
            total_trades = len(recent_trades)
            
            # Compter les trades gagnants et perdants
            winning_trades = [t for t in recent_trades if t.get("profit_loss", 0) > 0]
            losing_trades = [t for t in recent_trades if t.get("profit_loss", 0) < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # Calculer les profits et pertes moyens
            avg_profit = sum(t.get("profit_loss", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get("profit_loss", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            total_pnl = sum(t.get("profit_loss", 0) for t in recent_trades)
            
            # Calculer la précision des agents
            trader_correct = sum(1 for t in recent_trades if t.get("trader_correct", False))
            news_correct = sum(1 for t in recent_trades if t.get("analyst_correct", False))
            coordinator_correct = sum(1 for t in recent_trades if t.get("coordinator_correct", False))
            
            trader_accuracy = trader_correct / total_trades if total_trades > 0 else 0
            news_accuracy = news_correct / total_trades if total_trades > 0 else 0
            coordinator_accuracy = coordinator_correct / total_trades if total_trades > 0 else 0
            
            # Assembler les résultats
            return {
                "symbol": symbol,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "total_pnl": total_pnl,
                "trader_accuracy": trader_accuracy,
                "news_accuracy": news_accuracy,
                "coordinator_accuracy": coordinator_accuracy
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des performances passées pour {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "total_trades": 0,
                "win_rate": "N/A",
                "avg_profit": "N/A",
                "avg_loss": "N/A",
                "total_pnl": 0,
                "trader_accuracy": "N/A",
                "news_accuracy": "N/A",
                "coordinator_accuracy": "N/A"
            }
                     
    def get_decision_history(self, symbol: str, agent_type: str = "all") -> Dict[str, Any]:
        """
        Récupère l'historique des décisions pour un symbole et un type d'agent donnés
        
        Args:
            symbol: Le symbole pour lequel récupérer l'historique
            agent_type: Le type d'agent ("trader", "analyst", "coordinator" ou "all")
            
        Returns:
            Un dictionnaire contenant l'historique des décisions
        """
        result = {}
        
        if agent_type in ["all", "trader"]:
            result["trader_decisions"] = self._get_agent_memory("trader", symbol)
            
        if agent_type in ["all", "analyst"]:
            result["news_analysis"] = self._get_agent_memory("analyst", symbol)
            
        if agent_type in ["all", "coordinator"]:
            result["final_decisions"] = self._get_agent_memory("coordinator", symbol)
            
        return result
    
    def _get_agent_memory(self, agent_type: str, symbol: str) -> List[Dict[str, Any]]:
        """
        Récupère la mémoire d'un agent spécifique pour un symbole donné
        """
        try:
            with open(self.memory_files[agent_type], 'r') as f:
                memories = json.load(f)
                
            # Filtrer par symbole et trier par date (plus récent d'abord)
            symbol_memories = [m for m in memories if m.get("symbol") == symbol]
            symbol_memories.sort(key=lambda x: x.get("datetime", ""), reverse=True)
            
            # Limiter au nombre de mémoires spécifié
            return symbol_memories[:self.memory_size]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Erreur lors de la récupération de la mémoire de l'agent {agent_type}: {e}")
            return []
    
    def save_decision(self, agent_type: str, decision_data: Dict[str, Any]):
        """
        Sauvegarde une décision dans la mémoire d'un agent
        """
        try:
            # Charger les mémoires existantes
            with open(self.memory_files[agent_type], 'r') as f:
                memories = json.load(f)
                
            # Ajouter la nouvelle décision
            decision_data["datetime"] = datetime.utcnow().isoformat()
            memories.append(decision_data)
            
            # Limiter le nombre de mémoires par symbole
            symbol = decision_data.get("symbol")
            symbol_memories = [m for m in memories if m.get("symbol") == symbol]
            if len(symbol_memories) > self.memory_size:
                # Garder uniquement les plus récentes
                symbol_memories.sort(key=lambda x: x.get("datetime", ""), reverse=True)
                to_keep = symbol_memories[:self.memory_size]
                memories = [m for m in memories if m.get("symbol") != symbol] + to_keep
            
            # Sauvegarder les mémoires mises à jour
            with open(self.memory_files[agent_type], 'w') as f:
                json.dump(memories, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la décision dans la mémoire de l'agent {agent_type}: {e}")
    
    def analyze_past_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Analyse les performances des décisions passées pour un symbole donné
        
        Args:
            symbol: Le symbole pour lequel analyser les performances
            
        Returns:
            Un dictionnaire contenant les métriques de performance
        """
        try:
            # Récupérer l'historique des performances
            with open(self.memory_files["performance"], 'r') as f:
                performances = json.load(f)
                
            # Filtrer par symbole et par période (dernières 'feedback_window' heures)
            cutoff_time = (datetime.utcnow() - timedelta(hours=self.feedback_window)).isoformat()
            symbol_performances = [p for p in performances 
                               if p.get("symbol") == symbol and p.get("datetime", "") >= cutoff_time]
            
            if not symbol_performances:
                return {
                    "trader_accuracy": "Pas assez de données",
                    "news_accuracy": "Pas assez de données",
                    "coordinator_accuracy": "Pas assez de données",
                    "profit_loss": 0.0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                    "avg_loss": 0.0
                }
            
            # Calculer les métriques de performance
            trader_correct = sum(1 for p in symbol_performances if p.get("trader_correct", False))
            analyst_correct = sum(1 for p in symbol_performances if p.get("analyst_correct", False))
            coordinator_correct = sum(1 for p in symbol_performances if p.get("coordinator_correct", False))
            
            profits = [p.get("profit_loss", 0) for p in symbol_performances if p.get("profit_loss", 0) > 0]
            losses = [p.get("profit_loss", 0) for p in symbol_performances if p.get("profit_loss", 0) < 0]
            
            total_trades = len(symbol_performances)
            win_rate = len(profits) / total_trades if total_trades > 0 else 0
            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            total_pnl = sum(p.get("profit_loss", 0) for p in symbol_performances)
            
            return {
                "trader_accuracy": f"{trader_correct / total_trades:.2f}" if total_trades > 0 else "N/A",
                "news_accuracy": f"{analyst_correct / total_trades:.2f}" if total_trades > 0 else "N/A",
                "coordinator_accuracy": f"{coordinator_correct / total_trades:.2f}" if total_trades > 0 else "N/A",
                "profit_loss": total_pnl,
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "total_trades": total_trades
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des performances passées: {e}")
            return {
                "trader_accuracy": "Erreur",
                "news_accuracy": "Erreur",
                "coordinator_accuracy": "Erreur",
                "profit_loss": 0.0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0
            }
    
    def record_trade_performance(self, symbol: str, trade_result: Dict[str, Any], actual_price_movement: float):
        """
        Enregistre la performance d'un trade pour l'apprentissage futur
        
        Args:
            symbol: Le symbole du trade
            trade_result: Le résultat du trade (décisions des agents)
            actual_price_movement: Le mouvement réel du prix après la décision
        """
        try:
            # Extraire les décisions des agents
            analyst_sentiment = self._extract_sentiment_score(trade_result.get("news_summary", ""))
            trader_action = self._extract_action(trade_result.get("trader_analysis", ""))
            final_action = self._extract_action(trade_result.get("final_decision", ""))
            
            # Déterminer si les décisions étaient correctes
            trader_correct = self._is_decision_correct(trader_action, actual_price_movement)
            analyst_correct = self._is_sentiment_correct(analyst_sentiment, actual_price_movement)
            coordinator_correct = self._is_decision_correct(final_action, actual_price_movement)
            
            # Calculer le profit/perte théorique
            position_size = self._extract_position_size(trade_result.get("final_decision", ""))
            profit_loss = position_size * actual_price_movement if final_action == "BUY" else \
                         -position_size * actual_price_movement if final_action == "SELL" else 0
            
            # Créer l'enregistrement de performance
            performance_record = {
                "symbol": symbol,
                "datetime": datetime.utcnow().isoformat(),
                "trader_decision": trader_action,
                "analyst_sentiment": analyst_sentiment,
                "final_decision": final_action,
                "actual_movement": actual_price_movement,
                "trader_correct": trader_correct,
                "analyst_correct": analyst_correct,
                "coordinator_correct": coordinator_correct,
                "profit_loss": profit_loss
            }
            
            # Charger les performances existantes
            with open(self.memory_files["performance"], 'r') as f:
                performances = json.load(f)
                
            # Ajouter la nouvelle performance
            performances.append(performance_record)
            
            # Sauvegarder les performances mises à jour
            with open(self.memory_files["performance"], 'w') as f:
                json.dump(performances, f, indent=2)
                
            logger.info(f"Performance enregistrée pour {symbol}: P&L={profit_loss:.4f}, Trader correct={trader_correct}, Analyste correct={analyst_correct}, Coordinateur correct={coordinator_correct}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la performance du trade: {e}")
    
    def _extract_sentiment_score(self, text: str) -> float:
        """
        Extrait le score de sentiment d'un texte
        """
        try:
            # Recherche d'un pattern comme "SCORE: 0.75"
            import re
            match = re.search(r"SCORE:\s*([+-]?\d+\.?\d*)", text)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception:
            return 0.0
    
    def _extract_action(self, text: str) -> str:
        """
        Extrait l'action (BUY, SELL, HOLD) d'un texte
        """
        try:
            import re
            match = re.search(r"ACTION:\s*(BUY|SELL|HOLD)", text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            return "HOLD"
        except Exception:
            return "HOLD"
    
    def _extract_position_size(self, text: str) -> float:
        """
        Extrait la taille de position d'un texte
        """
        try:
            import re
            match = re.search(r"TAILLE POSITION:\s*(\d+\.?\d*)", text)
            if match:
                return float(match.group(1)) / 100.0  # Convertir le pourcentage en décimal
            return self.position_size
        except Exception:
            return self.position_size
    
    def _is_decision_correct(self, action: str, price_movement: float) -> bool:
        """
        Détermine si une décision était correcte en fonction du mouvement de prix réel
        """
        if action == "BUY" and price_movement > 0:
            return True
        if action == "SELL" and price_movement < 0:
            return True
        if action == "HOLD" and abs(price_movement) < 0.005:  # Mouvement de prix inférieur à 0.5%
            return True
        return False
    
    def _is_sentiment_correct(self, sentiment: float, price_movement: float) -> bool:
        """
        Détermine si l'analyse de sentiment était correcte en fonction du mouvement de prix réel
        """
        if sentiment > 0.2 and price_movement > 0:
            return True
        if sentiment < -0.2 and price_movement < 0:
            return True
        if -0.2 <= sentiment <= 0.2 and abs(price_movement) < 0.01:
            return True
        return False
        
    def _format_analyst_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Formate l'historique des analyses de l'Analyste pour le prompt
        """
        if not history:
            return "HISTORIQUE: Aucune analyse précédente disponible."
            
        formatted = "HISTORIQUE DE TES ANALYSES PRÉCÉDENTES:\n"
        for i, item in enumerate(history[:3]):  # Limiter à 3 analyses précédentes
            analysis = item.get("analysis", "")
            date = datetime.fromisoformat(item.get("datetime", datetime.utcnow().isoformat()))
            formatted += f"[{date.strftime('%Y-%m-%d %H:%M')}]\n{analysis[:200]}...\n\n"
            
        return formatted
    
    def _format_trader_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Formate l'historique des décisions du Trader pour le prompt
        """
        if not history:
            return "HISTORIQUE: Aucune décision précédente disponible."
            
        formatted = "HISTORIQUE DE TES DÉCISIONS PRÉCÉDENTES:\n"
        for i, item in enumerate(history[:3]):  # Limiter à 3 décisions précédentes
            analysis = item.get("analysis", "")
            date = datetime.fromisoformat(item.get("datetime", datetime.utcnow().isoformat()))
            formatted += f"[{date.strftime('%Y-%m-%d %H:%M')}]\n{analysis[:200]}...\n\n"
            
        return formatted
    
    def _format_coordinator_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Formate l'historique des décisions du Coordinateur pour le prompt
        """
        if not history:
            return "HISTORIQUE: Aucune décision précédente disponible."
            
        formatted = "HISTORIQUE DE TES DÉCISIONS PRÉCÉDENTES:\n"
        for i, item in enumerate(history[:3]):  # Limiter à 3 décisions précédentes
            analysis = item.get("analysis", "")
            date = datetime.fromisoformat(item.get("datetime", datetime.utcnow().isoformat()))
            action = self._extract_action(analysis)
            confidence = self._extract_confidence(analysis)
            formatted += f"[{date.strftime('%Y-%m-%d %H:%M')}] ACTION: {action}, CONFIANCE: {confidence}\n"
            
        return formatted
    
    def _format_performance_summary(self, performance: Dict[str, Any]) -> Dict[str, str]:
        """
        Formate le résumé des performances pour les prompts des agents
        """
        if not performance or performance.get("total_trades", 0) == 0:
            return {
                "news_performance": "PERFORMANCES: Pas assez de données pour évaluer tes performances.",
                "trader_performance": "PERFORMANCES: Pas assez de données pour évaluer tes performances.",
                "overall_performance": "PERFORMANCES: Pas assez de données pour évaluer les performances."
            }
            
        # Performance de l'Analyste
        news_perf = (
            f"PERFORMANCES DE TES ANALYSES PRÉCÉDENTES:\n"
            f"- Précision: {performance.get('news_accuracy', 'N/A')}\n"
            f"- Nombre total d'analyses: {performance.get('total_trades', 0)}\n"
        )
        
        # Performance du Trader
        trader_perf = (
            f"PERFORMANCES DE TES DÉCISIONS PRÉCÉDENTES:\n"
            f"- Précision: {performance.get('trader_accuracy', 'N/A')}\n"
            f"- Nombre total de décisions: {performance.get('total_trades', 0)}\n"
        )
        
        # Performance globale
        overall_perf = (
            f"PERFORMANCES GLOBALES:\n"
            f"- Précision du Coordinateur: {performance.get('coordinator_accuracy', 'N/A')}\n"
            f"- Taux de réussite: {performance.get('win_rate', 0.0) * 100:.1f}%\n"
            f"- Profit/Perte moyen: {performance.get('profit_loss', 0.0):.4f}\n"
            f"- Gain moyen: {performance.get('avg_profit', 0.0):.4f}\n"
            f"- Perte moyenne: {performance.get('avg_loss', 0.0):.4f}\n"
        )
        
        return {
            "news_performance": news_perf,
            "trader_performance": trader_perf,
            "overall_performance": overall_perf
        }
        
    def _extract_confidence(self, text: str) -> str:
        """
        Extrait la valeur de confiance d'un texte
        """
        try:
            import re
            match = re.search(r"CONFIANCE:\s*(\d+\.?\d*)", text)
            if match:
                return match.group(1)
            return "N/A"
        except Exception:
            return "N/A"

    def analyze(self, data: pd.DataFrame, symbol: str = None, **kwargs) -> Dict[str, Any]:
        """
        Analyse complète avec architecture multi-agent Claude et apprentissage en boucle fermée :
        1. Claude Analyste collecte et résume les news/événements récents en tenant compte de ses analyses précédentes
        2. Claude Trader analyse le marché et propose des actions en tenant compte de ses décisions précédentes
        3. Claude Coordinateur fusionne les analyses, évalue les performances passées et prend la décision finale
        """
        symbol = symbol or kwargs.get("symbol", "BTC/USD")
        now = datetime.utcnow()
        
        # Récupérer l'historique des décisions et les performances passées
        decision_history = self.get_decision_history(symbol)
        past_performance = self.analyze_past_performance(symbol)
        
        # Préparer les résumés d'historique pour les prompts
        analyst_history = self._format_analyst_history(decision_history.get("news_analysis", []))
        trader_history = self._format_trader_history(decision_history.get("trader_decisions", []))
        coordinator_history = self._format_coordinator_history(decision_history.get("final_decisions", []))
        performance_summary = self._format_performance_summary(past_performance)

        # --- 1. Analyse des news par Claude Analyste avec apprentissage ---
        news = self.news_agent.get_news_for_symbol(symbol)
        news_context = "\n".join([f"- {n['title']} ({n.get('source', {}).get('name', '')})" for n in news]) if news else "Aucune news récente."
        
        analyst_prompt = (
            f"Tu es Analyste, expert en analyse d'actualités financières et crypto. \n"
            f"Synthétise l'impact des actualités suivantes sur {symbol} pour un trader professionnel. \n"
            f"Donne un score de sentiment précis (-1.0 à 1.0) et un résumé en 3-4 phrases maximum.\n\n"
            
            # Inclure l'historique des analyses précédentes si disponible
            f"{analyst_history}\n\n"
            
            # Inclure les performances passées si disponibles
            f"{performance_summary.get('news_performance', '')}\n\n"
            
            f"Format attendu:\n"
            f"SCORE: [valeur numérique entre -1.0 et 1.0]\n"
            f"RÉSUMÉ: [ton analyse concise]\n"
            f"AMÉLIORATION: [comment tu as amélioré ton analyse par rapport aux précédentes]\n\n"
            f"Actualités des dernières {self.news_lookback_hours} heures:\n{news_context}"
        )
        analyst_response = call_llm(self.claude_analyst, analyst_prompt, temperature=0.2, max_tokens=512)
        
        # Sauvegarder la décision de l'analyste
        self.save_decision("analyst", {"symbol": symbol, "analysis": analyst_response})

        # --- 2. Analyse technique par Claude Trader avec apprentissage ---
        tech_prompt = (
            f"Tu es Trader, expert en analyse technique et trading crypto. \n"
            f"Analyse les données de marché suivantes pour {symbol} et propose une action précise.\n\n"
            
            # Inclure l'historique des décisions précédentes si disponible
            f"{trader_history}\n\n"
            
            # Inclure les performances passées si disponibles
            f"{performance_summary.get('trader_performance', '')}\n\n"
            
            f"Données de marché récentes:\n{data.tail(10).to_string(index=False)}\n\n"
            f"Format attendu:\n"
            f"ACTION: [BUY, SELL ou HOLD]\n"
            f"CONFIANCE: [valeur entre 0.0 et 1.0]\n"
            f"TAILLE POSITION: [pourcentage recommandé du capital, entre 0.01 et {self.position_size*2}]\n"
            f"STOP LOSS: [pourcentage recommandé, entre 0.01 et {self.stop_loss*2}]\n"
            f"TAKE PROFIT: [pourcentage recommandé, entre 0.01 et {self.take_profit*2}]\n"
            f"ANALYSE: [justification brève de ta décision]\n"
            f"AMÉLIORATION: [comment tu as amélioré ta stratégie par rapport aux décisions précédentes]\n"
        )
        trader_response = call_llm(self.claude_trader, tech_prompt, temperature=0.2, max_tokens=512)
        
        # Sauvegarder la décision du trader
        self.save_decision("trader", {"symbol": symbol, "analysis": trader_response})

        # --- 3. Coordination par Claude Coordinateur avec apprentissage ---
        coordinator_prompt = (
            f"Tu es Coordinateur, expert en fusion de signaux et prise de décision finale pour le trading crypto.\n"
            f"Tu dois prendre une décision finale pour {symbol} en intégrant l'analyse technique et l'analyse des news.\n\n"
            
            # Inclure l'historique des décisions précédentes si disponible
            f"{coordinator_history}\n\n"
            
            # Inclure les performances passées si disponibles
            f"{performance_summary.get('overall_performance', '')}\n\n"
            
            f"DONNÉES DE MARCHÉ:\n{data.tail(5).to_string(index=False)}\n\n"
            f"ANALYSE DES NEWS (Analyste):\n{analyst_response}\n\n"
            f"ANALYSE TECHNIQUE (Trader):\n{trader_response}\n\n"
            f"PARAMÈTRES STRATÉGIQUES:\n"
            f"- Poids sentiment: {self.sentiment_weight}\n"
            f"- Poids technique: {self.technical_weight}\n"
            f"- Confiance minimale requise: {self.min_confidence}\n"
            f"- Taille position par défaut: {self.position_size}\n"
            f"- Stop loss par défaut: {self.stop_loss}\n"
            f"- Take profit par défaut: {self.take_profit}\n\n"
            f"CONTEXTE IMPORTANT:\n"
            f"- Il existe une latence entre les données Binance et l'exécution Alpaca\n"
            f"- Tu dois anticiper cette latence dans ta décision finale\n"
            f"- Tiens compte des performances passées pour améliorer ta décision\n\n"
            f"Format attendu pour ta réponse finale:\n"
            f"ACTION: [BUY, SELL ou HOLD]\n"
            f"CONFIANCE: [valeur entre 0.0 et 1.0]\n"
            f"TAILLE POSITION: [pourcentage du capital]\n"
            f"STOP LOSS: [pourcentage]\n"
            f"TAKE PROFIT: [pourcentage]\n"
            f"JUSTIFICATION: [explication concise de ta décision finale]\n"
            f"AMÉLIORATION: [comment tu as amélioré ta décision par rapport aux précédentes]\n"
        )
        coordinator_response = call_llm(self.claude_coordinator, coordinator_prompt, temperature=0.3, max_tokens=1024)
        
        # Sauvegarder la décision du coordinateur
        self.save_decision("coordinator", {"symbol": symbol, "analysis": coordinator_response})
        
        # Extraction des décisions pour le résultat final
        result = {
            "symbol": symbol,
            "datetime": now.isoformat(),
            "news_summary": analyst_response,
            "trader_analysis": trader_response,
            "final_decision": coordinator_response,
            "raw_news": news,
            "raw_market_data": data.tail(10).to_dict(orient="records"),
            "performance_metrics": past_performance
        }
        
        logger.info(f"LLM_V3 Final Decision for {symbol}: {coordinator_response[:200]}...")
        return result

    def execute_trade(self, analysis_result: Dict[str, Any], symbol: str = None, **kwargs) -> TradeAction:
        """
        Exécute un trade basé sur l'analyse du Claude Coordinateur
        Extrait les paramètres de trading (action, taille, stop loss, take profit) et les applique
        Enregistre les performances du trade pour l'apprentissage futur
        """
        symbol = symbol or analysis_result.get("symbol", "BTC/USD")
        final_decision = analysis_result.get("final_decision", "")
        
        # Extraction des paramètres de trading
        action = self._extract_action(final_decision)
        confidence = self._extract_confidence(final_decision)
        position_size = self._extract_position_size(final_decision) or self.position_size
        stop_loss = self._extract_stop_loss(final_decision) or self.stop_loss
        take_profit = self._extract_take_profit(final_decision) or self.take_profit
        
        # Vérification de la confiance minimale
        confidence_value = float(confidence) if confidence != "N/A" else 0.0
        if confidence_value < self.min_confidence:
            logger.info(f"Confiance trop faible ({confidence_value} < {self.min_confidence}), pas de trade pour {symbol}")
            return TradeAction(symbol=symbol, action="HOLD", quantity=0)
        
        # Préparation de l'action de trading
        quantity = 0
        if action in ["BUY", "SELL"]:
            # Calcul de la quantité en fonction de la taille de position
            quantity = self._calculate_position_quantity(symbol, position_size)
            
            # Ajout des ordres stop loss et take profit si nécessaire
            if action == "BUY":
                self._set_stop_loss_take_profit(symbol, "sell", stop_loss, take_profit)
            elif action == "SELL":
                self._set_stop_loss_take_profit(symbol, "buy", stop_loss, take_profit)
            
            # Enregistrer les détails du trade pour l'évaluation future
            trade_details = {
                "symbol": symbol,
                "datetime": datetime.utcnow().isoformat(),
                "action": action,
                "confidence": confidence_value,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "analyst_response": analysis_result.get("news_summary", ""),
                "trader_response": analysis_result.get("trader_analysis", ""),
                "coordinator_response": final_decision,
                "entry_price": self._get_current_price(symbol),
                "status": "open"
            }
            self.save_trade(symbol, trade_details)
            
            # Planifier une évaluation différée du trade pour fermer la boucle d'apprentissage
            self._schedule_trade_evaluation(symbol, trade_details)
        
        logger.info(f"LLM_V3 Trade pour {symbol}: {action}, quantité: {quantity}, confiance: {confidence}")
        return TradeAction(symbol=symbol, action=action, quantity=quantity)
        
    def _get_current_price(self, symbol: str) -> float:
        """
        Récupère le prix actuel d'un symbole
        """
        try:
            # Utiliser le fournisseur de données pour obtenir le prix actuel
            if self.data_provider == "binance":
                from app.services.market_data.binance_data import BinanceData
                binance = BinanceData()
                ticker = binance.get_ticker(symbol)
                return float(ticker.get("lastPrice", 0.0))
            else:
                # Fallback pour d'autres fournisseurs
                return 0.0
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix pour {symbol}: {str(e)}")
            return 0.0
    
    def _calculate_position_quantity(self, symbol: str, position_size: float) -> float:
        """
        Calcule la quantité à acheter/vendre en fonction de la taille de position
        """
        try:
            # Obtenir le solde du compte
            balance = self._get_account_balance()
            
            # Obtenir le prix actuel
            price = self._get_current_price(symbol)
            
            if price <= 0:
                return 0
                
            # Calculer la quantité en fonction du solde et de la taille de position
            quantity = (balance * position_size) / price
            
            # Arrondir à la précision appropriée pour la crypto
            return round(quantity, 6)
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la quantité pour {symbol}: {str(e)}")
            return 0
    
    def _get_account_balance(self) -> float:
        """
        Récupère le solde du compte
        """
        try:
            # Implémentation fictive pour le moment
            # Dans une implémentation réelle, il faudrait récupérer le solde via l'API Alpaca ou autre
            return 10000.0  # Valeur par défaut pour les tests
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde: {str(e)}")
            return 1000.0  # Valeur par défaut en cas d'erreur
    
    def _set_stop_loss_take_profit(self, symbol: str, side: str, stop_loss: float, take_profit: float):
        """
        Configure les ordres stop loss et take profit
        """
        try:
            # Implémentation fictive pour le moment
            # Dans une implémentation réelle, il faudrait placer les ordres via l'API Alpaca ou autre
            price = self._get_current_price(symbol)
            
            if side == "buy":
                stop_price = price * (1 - stop_loss)
                take_price = price * (1 + take_profit)
            else:  # sell
                stop_price = price * (1 + stop_loss)
                take_price = price * (1 - take_profit)
                
            logger.info(f"Stop loss pour {symbol} configuré à {stop_price:.2f}")
            logger.info(f"Take profit pour {symbol} configuré à {take_price:.2f}")
        except Exception as e:
            logger.error(f"Erreur lors de la configuration des ordres SL/TP pour {symbol}: {str(e)}")
    
    def save_trade(self, symbol: str, trade_details: Dict[str, Any]):
        """
        Enregistre les détails d'un trade dans le fichier de mémoire des trades
        """
        try:
            # Créer le dossier des trades s'il n'existe pas
            trades_dir = os.path.join(self.memory_dir, "trades")
            os.makedirs(trades_dir, exist_ok=True)
            
            # Chemin du fichier de trades pour ce symbole
            trades_file = os.path.join(trades_dir, f"{symbol.replace('/', '_')}_trades.json")
            
            # Charger les trades existants ou créer une liste vide
            trades = []
            if os.path.exists(trades_file):
                with open(trades_file, "r") as f:
                    try:
                        trades = json.load(f)
                    except json.JSONDecodeError:
                        trades = []
            
            # Ajouter le nouveau trade
            trades.append(trade_details)
            
            # Limiter le nombre de trades stockés
            if len(trades) > self.memory_size:
                trades = trades[-self.memory_size:]
            
            # Sauvegarder les trades
            with open(trades_file, "w") as f:
                json.dump(trades, f, indent=2)
                
            logger.info(f"Trade enregistré pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du trade pour {symbol}: {str(e)}")
    
    def _schedule_trade_evaluation(self, symbol: str, trade_details: Dict[str, Any]):
        """
        Planifie une évaluation différée du trade pour fermer la boucle d'apprentissage
        Dans un environnement de production, cela pourrait être implémenté avec un scheduler
        Pour simplifier, nous allons simuler cela avec un thread séparé
        """
        try:
            import threading
            import time
            
            def evaluate_trade_later():
                # Attendre un certain temps (simule le délai pour évaluer le résultat du trade)
                # Dans un environnement réel, cela serait basé sur des événements de marché
                time.sleep(60)  # Attendre 1 minute pour la démo
                
                # Récupérer le prix actuel
                current_price = self._get_current_price(symbol)
                entry_price = trade_details.get("entry_price", current_price)
                
                # Calculer le mouvement de prix
                if entry_price > 0:
                    price_movement = (current_price - entry_price) / entry_price
                else:
                    price_movement = 0
                
                # Inverser le mouvement si c'était une vente
                if trade_details.get("action") == "SELL":
                    price_movement = -price_movement
                
                # Enregistrer la performance du trade
                self.record_trade_performance(
                    symbol=symbol,
                    trade_id=trade_details.get("datetime", ""),
                    price_movement=price_movement,
                    trade_details=trade_details
                )
                
                logger.info(f"Évaluation du trade pour {symbol} terminée, mouvement de prix: {price_movement:.4f}")
            
            # Lancer l'évaluation dans un thread séparé
            thread = threading.Thread(target=evaluate_trade_later)
            thread.daemon = True  # Le thread s'arrêtera quand le programme principal s'arrête
            thread.start()
            
            logger.info(f"Évaluation du trade pour {symbol} planifiée")
        except Exception as e:
            logger.error(f"Erreur lors de la planification de l'évaluation du trade pour {symbol}: {str(e)}")
    
    # Méthode pour le backtest (à implémenter si nécessaire)
    # def backtest(self, historical_data: pd.DataFrame, **kwargs):
    #     pass
