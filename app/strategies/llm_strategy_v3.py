import os
import logging
from datetime import datetime, timedelta
from typing import List, Any, Optional, Dict
import pandas as pd
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

        logger.info(f"Initialized LLMStrategyV3 with trader model: {self.trader_model_name}, analyst model: {self.analyst_model_name}, coordinator model: {self.coordinator_model_name}")

    def analyze(self, data: pd.DataFrame, symbol: str = None, **kwargs) -> Dict[str, Any]:
        """
        Analyse complète avec architecture multi-agent Claude :
        1. Claude Analyste collecte et résume les news/événements récents
        2. Claude Trader analyse le marché et propose des actions
        3. Claude Coordinateur fusionne les analyses et prend la décision finale
        """
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
        analyst_response = call_llm(self.claude_analyst, analyst_prompt, temperature=0.2, max_tokens=512)

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
        trader_response = call_llm(self.claude_trader, tech_prompt, temperature=0.2, max_tokens=512)

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
            f"Format attendu pour ta réponse finale:\n"
            f"ACTION: [BUY, SELL ou HOLD]\n"
            f"CONFIANCE: [valeur entre 0.0 et 1.0]\n"
            f"TAILLE POSITION: [pourcentage du capital]\n"
            f"STOP LOSS: [pourcentage]\n"
            f"TAKE PROFIT: [pourcentage]\n"
            f"JUSTIFICATION: [explication concise de ta décision finale]\n"
        )
        coordinator_response = call_llm(self.claude_coordinator, coordinator_prompt, temperature=0.3, max_tokens=1024)
        
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
        
        # Extraction des paramètres de trading depuis la réponse du coordinateur
        action = "HOLD"  # Valeur par défaut
        confidence = 0.0
        position_size = self.position_size
        stop_loss = self.stop_loss
        take_profit = self.take_profit
        
        # Parsing de la réponse du coordinateur
        for line in final_decision.split("\n"):
            if line.startswith("ACTION:"):
                action_str = line.replace("ACTION:", "").strip()
                if action_str in ["BUY", "SELL", "HOLD"]:
                    action = action_str
            elif line.startswith("CONFIANCE:"):
                try:
                    confidence = float(line.replace("CONFIANCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("TAILLE POSITION:"):
                try:
                    size_str = line.replace("TAILLE POSITION:", "").strip().replace("%", "")
                    position_size = float(size_str) / 100  # Conversion en décimal
                except ValueError:
                    pass
            elif line.startswith("STOP LOSS:"):
                try:
                    sl_str = line.replace("STOP LOSS:", "").strip().replace("%", "")
                    stop_loss = float(sl_str) / 100  # Conversion en décimal
                except ValueError:
                    pass
            elif line.startswith("TAKE PROFIT:"):
                try:
                    tp_str = line.replace("TAKE PROFIT:", "").strip().replace("%", "")
                    take_profit = float(tp_str) / 100  # Conversion en décimal
                except ValueError:
                    pass
        
        # Vérification de la confiance minimale
        if confidence < self.min_confidence:
            logger.info(f"Confiance insuffisante ({confidence:.2f} < {self.min_confidence}), pas d'action")
            action = "HOLD"
        
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
        
        logger.info(f"LLM_V3 Trade Action: {symbol} {action} (conf: {confidence:.2f}, size: {position_size:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f})")
        return trade_action
        
    # Méthode pour le backtest (à implémenter si nécessaire)
    # def backtest(self, historical_data: pd.DataFrame, **kwargs):
    #     pass
