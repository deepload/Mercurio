import os
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Any, Optional, Dict, Tuple
import pandas as pd
from abc import ABC, abstractmethod
from app.strategies.base import BaseStrategy
from app.db.models import TradeAction
from app.utils.llm_utils import load_llm_model, call_llm
from app.strategies.sentiment.news_api_agent import NewsAPIAgent
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

# Créer le dossier logs s'il n'existe pas
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Fonction d'écriture directe dans le fichier de log pour les prompts
def log_prompt(message, symbol=""):
    """Fonction d'écriture directe dans le fichier de log pour les prompts"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(log_dir, 'claude_prompts.log')
        
        # S'assurer que le message est correctement formaté
        if isinstance(message, str):
            formatted_message = message.replace('\r\n', '\n')
        else:
            formatted_message = str(message)
        
        # Ajouter des séparateurs clairs pour faciliter la lecture
        entry = f"[{timestamp}] - {symbol} - {formatted_message}\n"
        entry += "-" * 80 + "\n"
        
        # Écrire dans le fichier avec un encodage explicite
        with open(log_file, 'a', encoding='utf-8', errors='replace') as f:
            f.write(entry)
        
        # Également afficher dans la console (version tronquée si trop long)
        console_message = formatted_message
        if len(console_message) > 500:
            console_message = console_message[:250] + "..." + console_message[-250:]
        print(f"[{timestamp}] - {symbol} - {console_message}")
    except Exception as e:
        print(f"ERREUR de logging: {str(e)}")
        logger.error(f"Erreur lors de l'écriture dans le fichier de log: {str(e)}")

# Test d'écriture dans le fichier de log
log_prompt("===== INITIALISATION DU LOGGER CLAUDE PROMPTS =====\nLe logger est prêt à enregistrer les prompts et réponses Claude\n===========")

class LLMStrategyV3(BaseStrategy):
    """
    Stratégie avancée MercurioAI LLM_V3 avec architecture multi-agent Claude :
      - Claude Trader : analyse technique + prise de décision
      - Claude Analyste : analyse des news et médias en temps réel
      - Claude Coordinateur : fusionne et arbitre les signaux pour agir avec un coup d'avance
    """
    def __init__(self,
                 trader_model_name: str = "claude-3-opus-20240229",
                 analyst_model_name: str = "claude-3-opus-20240229",
                 coordinator_model_name: str = "claude-3-opus-20240229",
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
                 newsapi_key: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = "LLMStrategyV3"
        self.description = "LLM_V3: Claude multi-agent (trader, analyste, coordinateur)"
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
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY")

        # Services
        self.market_data = MarketDataService()
        self.news_agent = NewsAPIAgent(api_key=self.newsapi_key, lookback_hours=self.news_lookback_hours)

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

        logger.info(f"Initialized LLMStrategyV3 with trader model: {self.trader_model_name}, analyst model: {self.analyst_model_name}, coordinator model: {self.coordinator_model_name}")

    def analyze(self, data: pd.DataFrame, symbol: str = None, **kwargs) -> Dict[str, Any]:
        """Analyse multi-agent: Analyste (news) + Trader (technique) + Coordinateur (décision finale)"""
        # Log pour vérifier si la méthode est appelée
        log_prompt(f"===== MÉTHODE ANALYZE APPELÉE POUR {symbol} =====\nDonnées: {len(data)} lignes", symbol)
        logger.info(f"LLMStrategyV3.analyze appelée pour {symbol} avec {len(data)} lignes de données")
        
        # Vérifier si les données sont suffisantes
        if data is None or len(data) < 10:
            logger.warning(f"Données insuffisantes pour l'analyse LLM_V3 de {symbol}")
            return {"final_decision": "HOLD", "confidence": 0.0, "reason": "Données insuffisantes"}
        
        symbol = symbol or kwargs.get("symbol", "BTC/USD")
        now = datetime.utcnow()

        # --- 1. Analyse des news par Claude Analyste ---
        news = self.news_agent.get_news_for_symbol(symbol)
        news_context = "\n".join([f"- {n['title']} ({n.get('source', {}).get('name', '')})" for n in news]) if news else "Aucune news récente."
        analyst_prompt = (
            f"Tu es Analyste, expert en analyse d'actualités financières et crypto. \n"
            f"Synthétise l'impact des actualités suivantes sur {symbol} pour un trader professionnel. \n"
            f"Donne un score de sentiment précis (-1.0 à 1.0) et un résumé en 3-4 phrases maximum.\n"
            f"Format attendu:\n"
            f"SCORE: [valeur numérique entre -1.0 et 1.0]\n"
            f"RÉSUMÉ: [ton analyse concise]\n\n"
            f"Actualités des dernières {self.news_lookback_hours} heures:\n{news_context}"
        )
        # Log du prompt de l'analyste pour débogage
        log_prompt(f"===== PROMPT ANALYSTE =====\n{analyst_prompt}\n===========", symbol)
        try:
            analyst_response = call_llm(self.claude_analyst, analyst_prompt, temperature=0.2, max_tokens=512)
            # Log de la réponse de l'analyste
            log_prompt(f"===== RÉPONSE ANALYSTE =====\n{analyst_response}\n===========", symbol)
        except Exception as e:
            error_msg = f"ERREUR lors de l'appel à l'API Claude (Analyste): {str(e)}"
            logger.error(error_msg)
            log_prompt(f"===== ERREUR ANALYSTE =====\n{error_msg}\n===========", symbol)
            analyst_response = "ERREUR: Impossible d'obtenir une analyse des news"

        # --- 2. Analyse technique par Claude Trader ---
        tech_prompt = (
            f"Tu es Trader, expert en analyse technique et trading crypto. \n"
            f"Analyse les données de marché suivantes pour {symbol} et propose une action précise.\n"
            f"Données de marché récentes:\n{data.tail(10).to_string(index=False)}\n\n"
            f"Format attendu:\n"
            f"ACTION: [BUY, SELL ou HOLD]\n"
            f"CONFIANCE: [valeur entre 0.0 et 1.0]\n"
            f"TAILLE POSITION: [pourcentage recommandé du capital, entre 0.01 et {self.position_size*2}]\n"
            f"STOP LOSS: [pourcentage recommandé, entre 0.01 et {self.stop_loss*2}]\n"
            f"TAKE PROFIT: [pourcentage recommandé, entre 0.01 et {self.take_profit*2}]\n"
            f"ANALYSE: [justification brève de ta décision]\n"
        )
        # Log du prompt du trader pour débogage
        log_prompt(f"===== PROMPT TRADER =====\n{tech_prompt}\n===========", symbol)
        try:
            trader_response = call_llm(self.claude_trader, tech_prompt, temperature=0.2, max_tokens=512)
            # Log de la réponse du trader
            log_prompt(f"===== RÉPONSE TRADER =====\n{trader_response}\n===========", symbol)
        except Exception as e:
            error_msg = f"ERREUR lors de l'appel à l'API Claude (Trader): {str(e)}"
            logger.error(error_msg)
            log_prompt(f"===== ERREUR TRADER =====\n{error_msg}\n===========", symbol)
            trader_response = "ERREUR: Impossible d'obtenir une analyse technique"

        # --- 3. Coordination par Claude Coordinateur ---
        coordinator_prompt = (
            f"Tu es Coordinateur, expert en fusion de signaux et prise de décision finale pour le trading crypto.\n"
            f"Tu dois prendre une décision finale pour {symbol} en intégrant l'analyse technique et l'analyse des news.\n\n"
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
            f"- Tu dois anticiper cette latence dans ta décision finale\n\n"
            f"N'hésite pas à recommander SELL si les conditions sont réunies (tendance baissière, mauvaises nouvelles, etc.)\n"
            f"Équilibre tes décisions entre BUY, SELL et HOLD selon les conditions réelles du marché\n"
            f"Pour les cryptos en baisse, n'hésite pas à vendre pour limiter les pertes\n\n"
            f"Format attendu pour ta réponse finale:\n"
            f"ACTION: [BUY, SELL ou HOLD]\n"
            f"CONFIANCE: [valeur entre 0.0 et 1.0]\n"
            f"TAILLE POSITION: [pourcentage du capital]\n"
            f"STOP LOSS: [pourcentage]\n"
            f"TAKE PROFIT: [pourcentage]\n"
            f"JUSTIFICATION: [explication concise de ta décision finale]\n"
        )
        # Log du prompt du coordinateur pour débogage
        log_prompt(f"===== PROMPT COORDINATEUR =====\n{coordinator_prompt}\n===========", symbol)
        try:
            coordinator_response = call_llm(self.claude_coordinator, coordinator_prompt, temperature=0.3, max_tokens=1024)
            # Log de la réponse du coordinateur
            log_prompt(f"===== RÉPONSE COORDINATEUR =====\n{coordinator_response}\n===========", symbol)
        except Exception as e:
            error_msg = f"ERREUR lors de l'appel à l'API Claude (Coordinateur): {str(e)}"
            logger.error(error_msg)
            log_prompt(f"===== ERREUR COORDINATEUR =====\n{error_msg}\n===========", symbol)
            coordinator_response = "ERREUR: Impossible d'obtenir une décision finale"
        
        # Extraction des décisions pour le résultat final
        result = {
            "symbol": symbol,
            "datetime": now.isoformat(),
            "news_summary": analyst_response,
            "trader_analysis": trader_response,
            "final_decision": coordinator_response,
            "raw_news": news,
            "raw_market_data": data.tail(10).to_dict(orient="records"),
        }
        
        logger.info(f"LLM_V3 Final Decision for {symbol}: {coordinator_response[:200]}...")
        return result

    def execute_trade(self, analysis_result: Dict[str, Any], symbol: str = None, **kwargs) -> TradeAction:
        """
        Exécute un trade basé sur l'analyse du Claude Coordinateur
        Extrait les paramètres de trading (action, taille, stop loss, take profit) et les applique
        """
        symbol = symbol or analysis_result.get("symbol", "BTC/USD")
        final_decision = analysis_result.get("final_decision", "")
        
        # Log de la décision complète pour débogage
        logger.info(f"===== DÉCISION COMPLÈTE POUR {symbol} =====\n{final_decision}\n===================================")
        log_prompt(f"===== DÉCISION FINALE À EXÉCUTER =====\n{final_decision}\n===========", symbol)
        
        # Extraction des paramètres de trading depuis la réponse du coordinateur
        action = "HOLD"  # Valeur par défaut
        confidence = 0.0
        position_size = self.position_size
        stop_loss = self.stop_loss
        take_profit = self.take_profit
        justification = ""
        
        # Parsing de la réponse du coordinateur avec logs détaillés
        for line in final_decision.split("\n"):
            if line.startswith("ACTION:"):
                action_str = line.replace("ACTION:", "").strip()
                logger.info(f"Action détectée dans la réponse: '{action_str}'")
                if action_str.upper() in ["BUY", "SELL", "HOLD"]:
                    action = action_str.upper()
                    logger.info(f"Action validée: {action}")
                    if action == "SELL":
                        log_prompt(f"===== ACTION SELL DÉTECTÉE =====\n"
                                 f"Ligne complète: {line}\n===========", symbol)
                else:
                    logger.warning(f"Action non reconnue: '{action_str}', utilisation de HOLD par défaut")
            elif line.startswith("CONFIANCE:") or line.startswith("CONFIDENCE:"):
                try:
                    conf_line = line.replace("CONFIANCE:", "").replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_line)
                    logger.info(f"Confiance détectée: {confidence}")
                except ValueError as e:
                    logger.error(f"Erreur de conversion de la confiance: {e}, valeur: '{line}'")
            elif line.startswith("TAILLE POSITION:") or line.startswith("POSITION SIZE:"):
                try:
                    size_str = line.replace("TAILLE POSITION:", "").replace("POSITION SIZE:", "").strip().replace("%", "")
                    position_size = float(size_str) / 100  # Conversion en décimal
                    logger.info(f"Taille de position détectée: {position_size}")
                except ValueError as e:
                    logger.error(f"Erreur de conversion de la taille de position: {e}, valeur: '{line}'")
            elif line.startswith("STOP LOSS:"):
                try:
                    sl_str = line.replace("STOP LOSS:", "").strip().replace("%", "")
                    stop_loss = float(sl_str) / 100  # Conversion en décimal
                    logger.info(f"Stop loss détecté: {stop_loss}")
                except ValueError as e:
                    logger.error(f"Erreur de conversion du stop loss: {e}, valeur: '{line}'")
            elif line.startswith("TAKE PROFIT:"):
                try:
                    tp_str = line.replace("TAKE PROFIT:", "").strip().replace("%", "")
                    take_profit = float(tp_str) / 100  # Conversion en décimal
                    logger.info(f"Take profit détecté: {take_profit}")
                except ValueError as e:
                    logger.error(f"Erreur de conversion du take profit: {e}, valeur: '{line}'")
            elif line.startswith("JUSTIFICATION:"):
                justification = line.replace("JUSTIFICATION:", "").strip()
                logger.info(f"Justification: {justification}")
        
        # Vérification de la confiance minimale avec logs détaillés
        original_action = action
        if confidence < self.min_confidence:
            if action == "SELL":
                logger.warning(f"Action SELL convertie en HOLD car confiance ({confidence}) < seuil minimal ({self.min_confidence})")
                log_prompt(f"===== SELL CONVERTI EN HOLD =====\n"
                         f"Confiance: {confidence}, Seuil minimal: {self.min_confidence}\n"
                         f"Justification: {justification}\n===========", symbol)
            elif action == "BUY":
                logger.warning(f"Action BUY convertie en HOLD car confiance ({confidence}) < seuil minimal ({self.min_confidence})")
            else:
                logger.info(f"Action HOLD maintenue, confiance ({confidence}) < seuil minimal ({self.min_confidence})")
            action = "HOLD"
        else:
            logger.info(f"Action {action} validée avec confiance {confidence} >= seuil minimal {self.min_confidence}")
        
        # Création de l'action de trading
        trade_action = TradeAction(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.utcnow(),
            strategy_name=self.name
        )
        
        # Log final de l'action de trading
        logger.info(f"Action finale pour {symbol}: {action} (action originale: {original_action})")
        if action == "SELL":
            logger.info(f"===== ORDRE SELL VALIDÉ POUR {symbol} =====\n"
                       f"Confiance: {confidence}\n"
                       f"Taille: {position_size}\n"
                       f"Stop Loss: {stop_loss}\n"
                       f"Take Profit: {take_profit}\n"
                       f"Justification: {justification}\n===================================")
            log_prompt(f"===== ORDRE SELL VALIDÉ =====\n"
                     f"Confiance: {confidence}, Taille: {position_size}\n"
                     f"Justification: {justification}\n===========", symbol)
        
        return trade_action
    
    async def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Charge les données historiques pour le symbole et la période spécifiés.
        
        Args:
            symbol: Le symbole du marché (ex: 'BTC/USD')
            start_date: Date de début pour le chargement des données
            end_date: Date de fin pour le chargement des données
            
        Returns:
            DataFrame contenant les données historiques
        """
        try:
            # Utiliser le service de données de marché pour charger les données
            data = self.market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider=self.data_provider
            )
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données pour {symbol}: {e}")
            return pd.DataFrame()
    
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données pour l'analyse LLM.
        
        Args:
            data: Données brutes du marché
            
        Returns:
            Données prétraitées
        """
        # Pour LLM_V3, nous n'avons pas besoin de prétraitement complexe
        # car les agents Claude travaillent avec des descriptions textuelles
        if data.empty:
            return data
            
        # Ajouter quelques indicateurs techniques de base qui pourraient être utiles
        try:
            # Calculer les moyennes mobiles
            data['SMA_20'] = data['close'].rolling(window=20).mean()
            data['SMA_50'] = data['close'].rolling(window=50).mean()
            data['SMA_200'] = data['close'].rolling(window=200).mean()
            
            # Calculer RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Supprimer les lignes avec des NaN
            data = data.dropna()
            
            return data
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données: {e}")
            return data
    
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Génère un signal de trading basé sur les données d'entrée.
        
        Args:
            data: Données de marché à analyser
            
        Returns:
            Tuple de (TradeAction, confiance)
        """
        if data.empty:
            return TradeAction.HOLD, 0.0
            
        # Utiliser la méthode analyze pour obtenir la prédiction
        symbol = data.get('symbol', [None])[0] or "BTC/USD"
        result = await self.analyze(data, symbol=symbol)
        
        # Convertir le signal en TradeAction
        signal = result.get('signal', 'neutral').upper()
        confidence = result.get('strength', 0.0)
        
        if signal == 'BUY':
            return TradeAction.BUY, confidence
        elif signal == 'SELL':
            return TradeAction.SELL, confidence
        else:
            return TradeAction.HOLD, confidence
    
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entraîne le modèle de stratégie sur les données historiques.
        Note: Pour LLM_V3, il n'y a pas d'entraînement traditionnel car nous utilisons
        des modèles pré-entraînés avec des prompts spécifiques.
        
        Args:
            data: Données de marché prétraitées
            
        Returns:
            Dictionnaire contenant les métriques d'entraînement
        """
        # Pour LLM_V3, nous n'avons pas d'entraînement traditionnel
        # mais nous pourrions implémenter ici l'optimisation des prompts
        # basée sur les performances passées
        
        logger.info("La stratégie LLM_V3 n'a pas besoin d'entraînement traditionnel")
        self.is_trained = True
        
        return {
            "status": "success",
            "message": "LLM_V3 utilise des modèles pré-entraînés, pas d'entraînement nécessaire"
        }
    
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Effectue un backtest de la stratégie sur des données historiques.
        
        Args:
            data: Données historiques de marché
            initial_capital: Capital initial pour le backtest
            
        Returns:
            Dictionnaire contenant les résultats du backtest
        """
        if data.empty:
            return {"error": "Données insuffisantes pour le backtest"}
            
        try:
            # Variables de suivi pour le backtest
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = []
            
            # Parcourir les données jour par jour
            for i in range(200, len(data)):
                # Extraire les données jusqu'à ce jour
                current_data = data.iloc[:i+1]
                current_price = current_data.iloc[-1]['close']
                current_date = current_data.iloc[-1].name
                
                # Obtenir le signal pour ce jour
                symbol = data.get('symbol', [None])[0] or "BTC/USD"
                result = await self.analyze(current_data.tail(200), symbol=symbol)
                signal = result.get('signal', 'neutral').upper()
                confidence = result.get('strength', 0.0)
                reason = result.get('reason', '')
                
                # Exécuter le signal si la confiance est suffisante
                if signal == 'BUY' and position == 0 and confidence >= self.min_confidence:
                    # Calculer la taille de la position
                    position_size = capital * self.position_size
                    position = position_size / current_price
                    capital -= position_size
                    
                    trades.append({
                        "date": current_date,
                        "type": "BUY",
                        "price": current_price,
                        "position": position,
                        "capital": capital,
                        "reason": reason
                    })
                    
                elif signal == 'SELL' and position > 0 and confidence >= self.min_confidence:
                    # Vendre la position
                    capital += position * current_price
                    position = 0
                    
                    trades.append({
                        "date": current_date,
                        "type": "SELL",
                        "price": current_price,
                        "position": position,
                        "capital": capital,
                        "reason": reason
                    })
                
                # Calculer la valeur totale du portefeuille
                portfolio_value = capital + (position * current_price)
                equity_curve.append({
                    "date": current_date,
                    "value": portfolio_value
                })
            
            # Calculer les métriques de performance
            final_value = capital + (position * data.iloc[-1]['close'])
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            return {
                "initial_capital": initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "total_trades": len(trades),
                "trades": trades,
                "equity_curve": equity_curve
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {e}")
            return {"error": str(e)}
