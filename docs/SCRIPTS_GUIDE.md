# Guide des Scripts Utilitaires de MercurioAI

> [!NOTE]
> **Navigation Rapide:**
> - [🔍 Index de tous les guides](./GUIDES_INDEX.md)
> - [📈 Guide de Day Trading](./day_trading_guide.md)
> - [📊 Guide des Options](./options_trading.md)
> - [🧠 Guide d'Entraînement des Modèles](./MODEL_TRAINING_GUIDE.md)
> - [🔍 Guide du Screener d'Actifs](./ASSET_SCREENER_GUIDE.md)
> - [📔 Documentation Principale](./README.md)

Ce document décrit les scripts utilitaires disponibles dans le dossier `/scripts` de MercurioAI. Ces scripts sont conçus pour faciliter l'utilisation, le test et la démonstration des fonctionnalités de la plateforme.

Chaque script est documenté avec ses paramètres disponibles et des exemples de cas d'utilisation pour vous aider à tirer le meilleur parti de la plateforme Mercurio AI.

## Table des matières

1. [Scripts de test](#scripts-de-test)
2. [Scripts de trading](#scripts-de-trading)
3. [Scripts d'analyse](#scripts-d'analyse)
4. [Scripts de démonstration](#scripts-de-démonstration)
5. [Scripts d'optimisation](#scripts-d'optimisation)
6. [Scripts de visualisation](#scripts-de-visualisation)

## Scripts de test

Ces scripts permettent de vérifier la configuration et le bon fonctionnement des différentes parties du système.

### `test_alpaca.py`

**Fonction** : Teste la connexion à l'API Alpaca en utilisant les informations d'identification configurées dans le fichier `.env`.

**Paramètres** : Ce script n'a pas de paramètres en ligne de commande, mais il est configuré via les variables d'environnement suivantes dans le fichier `.env` :

- `ALPACA_MODE` : Détermine le mode de trading ("paper" ou "live")
- `ALPACA_PAPER_KEY` et `ALPACA_PAPER_SECRET` : Clés API pour le paper trading
- `ALPACA_LIVE_KEY` et `ALPACA_LIVE_SECRET` : Clés API pour le live trading
- `ALPACA_PAPER_URL` et `ALPACA_LIVE_URL` : URLs de base pour les environnements paper et live

**Utilisation** :
```bash
python scripts/test_alpaca.py
```

**Description** : Ce script vérifie que les clés API d'Alpaca sont correctement configurées et que la connexion fonctionne. Il utilise automatiquement les clés appropriées selon le mode configuré dans `ALPACA_MODE` (paper ou live).

**Cas d'utilisation** :
1. **Validation initiale** : Après avoir configuré vos clés API pour la première fois
2. **Dépannage de connexion** : Si vous rencontrez des erreurs avec les API Alpaca
3. **Vérification avant trading live** : Avant de passer du paper trading au live trading

### `test_alpaca_connection.py`

**Fonction** : Version détaillée du test de connexion à Alpaca.

**Utilisation** :
```bash
python scripts/test_alpaca_connection.py
```

**Description** : Ce script fournit des informations plus détaillées sur le compte, y compris le solde, la valeur du portefeuille et le statut du marché.

### `test_api_access.py`

**Fonction** : Teste l'accès à diverses API de données de marché.

**Utilisation** :
```bash
python scripts/test_api_access.py
```

**Description** : Vérifie les connexions à toutes les sources de données configurées (Alpaca, Yahoo Finance, etc.) et affiche des informations sur chaque connexion.

### `test_market_data.py`

**Fonction** : Teste le service de données de marché de MercurioAI.

**Utilisation** :
```bash
python scripts/test_market_data.py
```

**Description** : Vérifie que le service MarketDataService fonctionne correctement et peut récupérer des données historiques et en temps réel.

### `test_stocks_and_crypto.py`

**Fonction** : Teste spécifiquement la récupération de données pour les actions et les cryptomonnaies.

**Utilisation** :
```bash
python scripts/test_stocks_and_crypto.py
```

**Description** : Vérifie que MercurioAI peut récupérer des données pour les symboles d'actions (comme AAPL, MSFT) et de cryptomonnaies (comme BTC-USD, ETH-USD).

## Scripts de trading

Ces scripts permettent d'exécuter différentes stratégies de trading.

### `run_strategy_crypto_trader.py`

**Fonction** : Exécute le trading de cryptomonnaies avec une sélection de stratégies différentes et configurations personnalisables.

**Paramètres** :
- `--strategy <nom>` : Stratégie à utiliser (options: `moving_average`, `momentum`, `mean_reversion`, `breakout`, `statistical_arbitrage`, `transformer`, `llm`, `llm_v2`)
- `--duration <durée>` : Durée de la session de trading (options: `1h`, `4h`, `8h`, `night`)
- `--position-size <taille>` : Taille de position en pourcentage du portefeuille (défaut: 0.02 = 2%)
- `--stop-loss <pourcentage>` : Stop loss en pourcentage (défaut: 0.03 = 3%)
- `--take-profit <pourcentage>` : Take profit en pourcentage (défaut: 0.06 = 6%)
- `--use-custom-symbols` : Utiliser la liste personnalisée de symboles au lieu du filtre automatique
- `--fast-ma <période>` : Période de la moyenne mobile rapide pour la stratégie moving_average
- `--slow-ma <période>` : Période de la moyenne mobile lente pour la stratégie moving_average
- `--momentum-lookback <période>` : Période de lookback pour la stratégie momentum
- `--mean-reversion-lookback <période>` : Période de lookback pour la stratégie mean reversion
- `--breakout-lookback <période>` : Période de lookback pour la stratégie breakout
- `--volatility-lookback <période>` : Période de lookback pour le calcul de la volatilité

**Paramètres spécifiques à la stratégie Transformer** :
- `--sequence-length <longueur>` : Longueur de la séquence d'entrée pour le modèle Transformer (défaut: 60)
- `--prediction-horizon <horizon>` : Horizon de prédiction pour le modèle Transformer (défaut: 1)
- `--d-model <dimension>` : Dimension du modèle Transformer (défaut: 64)
- `--nhead <têtes>` : Nombre de têtes d'attention pour le modèle Transformer (défaut: 4)
- `--num-layers <couches>` : Nombre de couches pour le modèle Transformer (défaut: 2)
- `--dropout <taux>` : Taux de dropout pour le modèle Transformer (défaut: 0.1)
- `--signal-threshold <seuil>` : Seuil de signal pour la stratégie Transformer (défaut: 0.6)
- `--use-gpu` : Utiliser le GPU pour l'entraînement et l'inférence (si disponible)
- `--retrain` : Réentraîner le modèle Transformer même si un modèle entraîné existe déjà

**Paramètres spécifiques à la stratégie LLM** :
- `--model-name <nom>` : Nom du modèle LLM à utiliser (défaut: "mistralai/Mixtral-8x7B-Instruct-v0.1")
- `--use-local-model` : Utiliser un modèle local plutôt qu'une API
- `--local-model-path <chemin>` : Chemin vers le modèle local (si --use-local-model est activé)
- `--api-key <clé>` : Clé API pour le modèle LLM (non nécessaire en mode démo)
- `--sentiment-threshold <seuil>` : Seuil de sentiment pour générer un signal (défaut: 0.7)
- `--news-lookback <heures>` : Nombre d'heures de données d'actualités à analyser (défaut: 24)

**Paramètres spécifiques à la stratégie LLM_V2** :
- `--model-name <nom>` : Nom du modèle LLM principal à utiliser (défaut: "mistralai/Mixtral-8x7B-Instruct-v0.1")
- `--sentiment-model-name <nom>` : Nom du modèle LLM pour l'analyse de sentiment (optionnel)
- `--use-local-model` : Utiliser un modèle local plutôt qu'une API
- `--local-model-path <chemin>` : Chemin vers le modèle local (si --use-local-model est activé)
- `--api-key <clé>` : Clé API pour le modèle LLM (peut être "demo_mode" pour les tests)
- `--use-web-sentiment` : Activer l'analyse de sentiment web (défaut: True)
- `--sentiment-weight <poids>` : Poids donné à l'analyse de sentiment (0 à 1, défaut: 0.5)
- `--min-confidence <seuil>` : Seuil de confiance minimal pour générer un signal (défaut: 0.65)
- `--news-lookback <heures>` : Nombre d'heures de données d'actualités à analyser (défaut: 24)

**Utilisation** :
```bash
# Trading avec la stratégie de momentum
python scripts/run_strategy_crypto_trader.py --strategy momentum --use-custom-symbols

# Trading avec mean reversion sur une session de 4 heures
python scripts/run_strategy_crypto_trader.py --strategy mean_reversion --duration 4h --stop-loss 0.02 --take-profit 0.05

# Trading avec moving average (stratégie par défaut)
python scripts/run_strategy_crypto_trader.py --fast-ma 10 --slow-ma 30

# Trading avec la stratégie Transformer pendant 8 heures
python scripts/run_strategy_crypto_trader.py --strategy transformer --duration 8h --use-custom-symbols

# Trading avec Transformer personnalisé pour les cryptomonnaies à forte volatilité (risque modéré)
python scripts/run_strategy_crypto_trader.py --strategy transformer --sequence-length 120 --d-model 128 --nhead 8 --position-size 0.01 --stop-loss 0.02 --take-profit 0.05 --use-gpu --use-custom-symbols

# Trading avec Transformer personnalisé pour marchés volatils (risque accru)
python scripts/run_strategy_crypto_trader.py --strategy transformer --duration night --sequence-length 120 --d-model 128 --nhead 8 --num-layers 3 --signal-threshold 0.6 --position-size 0.015 --stop-loss 0.02 --take-profit 0.05 --use-gpu --use-custom-symbols --data-provider binance

# Trading en utilisant Yahoo Finance comme source de données de marché
python scripts/run_strategy_crypto_trader.py --strategy moving_average --data-provider yahoo

# Trading avec LLMStrategyV2 et Yahoo Finance pour les données de marché
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --data-provider yahoo --use-custom-symbols

# Trading avec LLMStrategyV2 et Binance pour les données de marché
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --news-lookback 24 --sentiment-weight 0.6 --use-custom-symbols --data-provider binance

# Trading avec LLMStrategyV2 et Binance pour les données de marché sur une session de 48 heures
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --news-lookback 2 --sentiment-weight 0.6 --use-custom-symbols --data-provider binance --duration 48h

#Trading risqué avec LLMStrategyV2 et Binance pour les données de marché sur une session de 48 heures
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --news-lookback 2 --sentiment-weight 0.6 --use-custom-symbols --data-provider binance --duration 48h --position-size 0.05 --stop-loss 0.02 --take-profit 0.05

#Trading avec LLMStrategyV2 et Binance pour les données de marché sur une session de 48 heures avec Claude 3.7
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --model-name claude-3-7-sonnet-20240620 --sentiment-weight 0.7 --min-confidence 0.65 --news-lookback 24 --position-size 0.03 --use-custom-symbols

#Trading avec LLMStrategyV3 et Binance pour les données de marché sur une session de 48 heures avec Claude 3.7
 python scripts/run_strategy_crypto_trader.py --strategy llm_v3 --use-custom-symbols --duration continuous --trader-model-name claude-3-7-sonnet-20240620 --analyst-model-name claude-3-7-sonnet-20240620 --coordinator-model-name claude-3-7-sonnet-20240620 --news-lookback 30 --min-confidence 0.65 --data-provider binance

#Trading avec LLMStrategyV3 et Binance pour les données de marché sur une session de 1 heure avec Claude 3.7
 python scripts/run_strategy_crypto_trader.py --strategy llm_v3 --use-custom-symbols --duration 1h --trader-model-name claude-3-7-sonnet-20240620 --analyst-model-name claude-3-7-sonnet-20240620 --coordinator-model-name claude-3-7-sonnet-20240620 --news-lookback 30 --min-confidence 0.65 --data-provider binance

#Trading en short avec LLMStrategyV2 et Binance pour les données de marché sur une session de 48 heures
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --news-lookback 2 --sentiment-weight 0.6 --use-custom-symbols --data-provider binance --duration 48h --position-size 0.05 --stop-loss 0.02 --take-profit 0.05 --short

#Trading avec LLMStrategyV3 et Binance pour les données de marché sur une session de 48 heures
python scripts/run_llm_v3_integrated.py --duration continuous

#Trading avec LLMStrategyV3 et Binance pour les données de marché sur une session de 
python scripts/run_strategy_crypto_trader.py --strategy llm_v3 --use-custom-symbols --duration continuous --trader-model-name claude-3-7-sonnet-20240620 --analyst-model-name claude-3-7-sonnet-20240620 --coordinator-model-name claude-3-7-sonnet-20240620 --news-lookback 2 --min-confidence 0.65 --data-provider binance

# Trading avec la stratégie LLM de base pour analyse de sentiment
python scripts/run_strategy_crypto_trader.py --strategy llm --news-lookback 48 --sentiment-threshold 0.6 --position-size 0.02 --use-custom-symbols

# Trading avec LLMStrategyV2 en mode démo (pas besoin de clé API)
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --api-key demo_mode --sentiment-weight 0.5

# Trading avec LLMStrategyV2 optimisé pour crypto volatiles
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --sentiment-weight 0.7 --min-confidence 0.75 --position-size 0.01 --stop-loss 0.025 --take-profit 0.05 --use-custom-symbols

# Trading avec LLMStrategyV2 utilisant un modèle local pour les tests
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --use-local-model --local-model-path models/llama-2-7b
```

**Description** : Ce script permet de lancer un trader de cryptomonnaies avec différentes stratégies de trading (moving average, momentum, mean reversion, breakout, statistical arbitrage). Il offre des options détaillées pour configurer chaque stratégie ainsi que les durées de session. Le script prend en charge le paper trading via Alpaca et peut utiliser une liste personnalisée de paires de cryptomonnaies.

**Cas d'utilisation** :
1. **Trading nocturne** : Pour faire tourner un trader de crypto pendant la nuit avec une stratégie optimisée
2. **Test de stratégies** : Pour comparer différentes stratégies de trading sur les cryptomonnaies
3. **Trading adaptatif** : Pour utiliser différentes stratégies selon les conditions de marché

### `auto_trader.py`

**Fonction** : Agent de trading automatisé avancé.

**Paramètres** :
- `--config <chemin>` : Chemin vers le fichier de configuration JSON (défaut: "config/agent_config.json")

**Format du fichier de configuration** :
```json
{
  "symbols": ["AAPL", "MSFT", "BTC-USD", "ETH-USD"],
  "strategies": ["MovingAverageStrategy", "LSTMPredictorStrategy"],
  "strategy_params": {
    "MovingAverageStrategy": {
      "short_window": 10,
      "long_window": 30,
      "use_ml": true
    },
    "LSTMPredictorStrategy": {
      "lookback_periods": 30,
      "prediction_periods": 5
    }
  },
  "risk_level": "moderate",  // "conservative", "moderate", "aggressive"
  "check_interval": 300,     // Intervalle de vérification en secondes
  "max_positions": 5,        // Nombre maximum de positions simultanées
  "position_size_pct": 0.1   // Taille de position en % du portefeuille
}
```

**Utilisation** :
```bash
python scripts/auto_trader.py --config configs/my_trading_config.json
```

**Description** : Un agent de trading complet qui peut sélectionner automatiquement entre différentes stratégies, optimiser les paramètres et prendre des décisions basées sur l'analyse de marché.

**Cas d'utilisation** :
1. **Trading automatisé multi-stratégies** : Pour exécuter simultanément plusieurs stratégies et basculer automatiquement entre elles en fonction des conditions du marché
2. **Trading adapté aux régimes de marché** : L'agent analyse en continu le régime du marché (haussier, baissier, latéral, volatil) et adapte ses stratégies en conséquence
3. **Trading avec gestion des risques avancée** : Intègre des fonctionnalités de gestion des risques comme les stop-loss dynamiques et l'ajustement des positions en fonction de la volatilité

### `enhanced_trader.py`

**Fonction** : Version améliorée du trader standard avec fonctionnalités supplémentaires.

**Paramètres** :
- `--strategy <nom>` : Nom de la stratégie à utiliser (défaut: "MovingAverageStrategy")
- `--symbols <liste>` : Liste de symboles séparés par des virgules (défaut: "AAPL,MSFT,GOOGL")
- `--interval <secondes>` : Intervalle de vérification en secondes (défaut: 300)
- `--risk_limit <ratio>` : Limite de risque par position en pourcentage (défaut: 0.02)
- `--config <chemin>` : Chemin vers un fichier de configuration JSON optionnel

**Utilisation** :
```bash
python scripts/enhanced_trader.py --strategy LSTMPredictorStrategy --symbols AAPL,MSFT,BTC-USD --interval 600 --risk_limit 0.03
```

**Description** : Ajoute des fonctionnalités comme la gestion avancée des risques, les rapports détaillés et une gestion améliorée des erreurs.

**Cas d'utilisation** :
1. **Trading avec paramètres personnalisés** : Pour exécuter une stratégie spécifique avec des symboles et des paramètres personnalisés
2. **Trading avec risque contrôlé** : Pour limiter le risque par position tout en permettant un trading autonome
3. **Surveillance à différents intervalles** : Pour adapter la fréquence de vérification des signaux selon le style de trading (court, moyen ou long terme)

### `trading_agent.py` et `trading_agent_fixed.py`

**Fonction** : Agents de trading qui simulent une session de trading complète.

**Paramètres** :
- `--mode <mode>` : Mode de trading ("paper" ou "backtest", défaut: "paper")
- `--strategy <nom>` : Nom de la stratégie à utiliser (défaut: "MovingAverageStrategy")
- `--symbols <liste>` : Liste de symboles séparés par des virgules (défaut: "AAPL,MSFT,GOOGL")
- `--capital <montant>` : Capital initial en USD (défaut: 10000)
- `--duration <heures>` : Durée de la simulation en heures (défaut: 24)
- `--log_level <niveau>` : Niveau de journalisation ("DEBUG", "INFO", "WARNING", "ERROR", défaut: "INFO")

**Utilisation** :
```bash
# Trading papier de base
python scripts/trading_agent_fixed.py --mode paper --strategy MovingAverageStrategy --symbols AAPL,MSFT,BTC-USD

# Backtest sur une durée spécifique
python scripts/trading_agent_fixed.py --mode backtest --strategy LSTMPredictorStrategy --symbols BTC-USD,ETH-USD --duration 48

# Trading papier avec capital personnalisé
python scripts/trading_agent_fixed.py --strategy MovingAverageStrategy --symbols TSLA --capital 50000
```

**Description** : Le script `trading_agent.py` est la version originale, tandis que `trading_agent_fixed.py` contient des corrections pour les problèmes connus. Les deux implémentent un agent de trading complet qui gère tout le cycle de trading, de l'acquisition des données à l'exécution des ordres.

**Cas d'utilisation** :
1. **Simulation de trading complète** : Pour exécuter une simulation de trading sur une période définie
2. **Comparaison de performances** : Pour tester différentes stratégies sur les mêmes symboles
3. **Tests sur différentes classes d'actifs** : Pour comparer les performances sur les actions vs. les cryptomonnaies

## Scripts d'analyse

Ces scripts permettent d'analyser le marché et d'évaluer les performances des stratégies.

### `get_all_symbols.py`

**Fonction** : Récupère tous les symboles d'actions et de crypto-monnaies disponibles via différentes sources de données (Alpaca, Yahoo Finance) et vérifie leur accessibilité.

**Paramètres** : Ce script n'a pas de paramètres en ligne de commande, il utilise les variables d'environnement configurées dans le fichier `.env`.

**Utilisation** :
```bash
python scripts/get_all_symbols.py
```

**Description** : Ce script interroge plusieurs sources de données pour récupérer la liste complète des symboles d'actions et de crypto-monnaies disponibles. Il vérifie ensuite que les données historiques sont accessibles pour chaque symbole. Les résultats sont sauvegardés dans des fichiers CSV dans le dossier `data` qui peuvent être utilisés pour l'entraînement des modèles.

Les fichiers générés incluent :
- `all_stocks_YYYYMMDD.csv` : Liste complète des actions accessibles
- `all_crypto_YYYYMMDD.csv` : Liste complète des crypto-monnaies accessibles
- `symbols_metadata_YYYYMMDD.json` : Métadonnées sur les symboles

Ce script garantit que tous les actifs majeurs (comme Apple, Microsoft, Berkshire Hathaway, etc.) sont inclus dans la liste, même s'ils ne sont pas disponibles via l'API Alpaca, en utilisant des sources alternatives comme Yahoo Finance.

**Cas d'utilisation** :
1. **Préparation pour l'entraînement** : Obtenir la liste complète des symboles avant d'exécuter `train_all_models.py`
2. **Vérification de l'accès aux données** : Identifier quels symboles sont réellement accessibles via votre abonnement API
3. **Mise à jour périodique** : Maintenir une liste à jour des actifs disponibles pour le trading et l'analyse

### `market_analyzer.py`

**Fonction** : Analyse approfondie du marché pour divers instruments.

**Paramètres** :
- `--symbols <liste>` : Liste de symboles séparés par des virgules à analyser (défaut : "AAPL,MSFT,GOOGL,BTC-USD,ETH-USD")
- `--lookback <jours>` : Période d'analyse en jours (défaut : 90)
- `--output <format>` : Format de sortie ("console", "csv", "json", "html", défaut : "console")
- `--anomaly_scan` : Activer la détection d'anomalies (sans valeur)
- `--sentiment` : Inclure l'analyse de sentiment (sans valeur)
- `--save_path <chemin>` : Chemin où sauvegarder les résultats (défaut : "reports/market_analysis")

**Utilisation** :
```bash
# Analyse de base avec détection d'anomalies
python scripts/market_analyzer.py --symbols AAPL,MSFT,BTC-USD --lookback 90 --anomaly_scan

# Analyse complète avec sentiment et exportation JSON
python scripts/market_analyzer.py --symbols BTC-USD,ETH-USD --output json --sentiment --save_path reports/crypto_analysis

# Génération d'un rapport HTML
python scripts/market_analyzer.py --symbols AAPL,TSLA,AMZN --output html
```

**Description** : Fournit une analyse technique détaillée, des indicateurs de sentiment et des prévisions pour les symboles spécifiés. Le script identifie également les régimes de marché (haussier, baissier, latéral, volatil) et peut détecter des anomalies potentielles.

**Cas d'utilisation** :
1. **Analyse pré-trading** : Pour évaluer les conditions du marché avant de déployer des stratégies
2. **Détection d'anomalies** : Pour identifier des mouvements inhabituels ou des manipulations potentielles
3. **Analyse de régime** : Pour déterminer quel type de stratégie serait le plus efficace dans les conditions actuelles
4. **Analyse de sentiment** : Pour combiner l'analyse technique avec le sentiment du marché

### `best_assets_screener.py`

**Fonction** : Évalue et classe les meilleures actions et cryptomonnaies pour le trading à moyen terme.

**Paramètres** :
- `--top_stocks <nombre>` : Nombre d'actions à inclure dans le rapport final (défaut : 50)
- `--top_crypto <nombre>` : Nombre de cryptomonnaies à inclure dans le rapport final (défaut : 100)
- `--lookback <jours>` : Nombre de jours d'historique à analyser (défaut : 30)
- `--stocks <liste>` : Liste d'actions personnalisée séparée par des virgules (si vide, utilise la liste par défaut)
- `--crypto <liste>` : Liste de cryptomonnaies personnalisée séparée par des virgules (si vide, utilise la liste par défaut)

**Utilisation** :
```bash
# Utilisation de base avec les paramètres par défaut
python scripts/best_assets_screener.py

# Personnalisation du nombre d'actifs à analyser
python scripts/best_assets_screener.py --top_stocks 20 --top_crypto 50 --lookback 60

# Utilisation d'une liste personnalisée d'actifs
python scripts/best_assets_screener.py --stocks AAPL,MSFT,GOOGL,AMZN,TSLA --crypto BTC-USD,ETH-USD,SOL-USD
```

**Description** : Ce script analyse une large liste d'actions et de cryptomonnaies pour identifier les meilleurs actifs pour le trading à moyen terme, en utilisant plusieurs stratégies de MercurioAI. Il génère un rapport détaillé classant les actifs selon un score composite basé sur des indicateurs techniques et des prédictions de différentes stratégies (moyennes mobiles, LSTM, MSI, transformers).

**Cas d'utilisation** :
1. **Sélection d'actifs pour investissement** : Pour identifier les meilleures opportunités d'investissement à moyen terme
2. **Analyse multi-stratégies** : Pour obtenir une vue consensuelle en combinant différentes approches d'analyse
3. **Surveillance périodique du marché** : Pour maintenir une liste d'actifs à surveiller mise à jour régulièrement
4. **Construction de portefeuille** : Pour diversifier les investissements en sélectionnant les meilleurs actifs de différentes classes

### `run_all_strategies.py`

**Fonction** : Exécute toutes les stratégies disponibles sur un ensemble de symboles.

**Utilisation** :
```bash
python scripts/run_all_strategies.py --symbols AAPL,MSFT,BTC-USD
```

**Description** : Permet de comparer rapidement les performances de toutes les stratégies implémentées dans le système.

## Scripts de démonstration

Ces scripts montrent les fonctionnalités de MercurioAI à travers des exemples concrets.

### `first_script.py`

**Fonction** : Script d'introduction pour les nouveaux utilisateurs.

**Utilisation** :
```bash
python scripts/first_script.py
```

**Description** : Un exemple simple qui montre comment obtenir des données et exécuter une stratégie de base.

### `simplified_demo.py`

**Fonction** : Démo simplifiée de MercurioAI.

**Utilisation** :
```bash
python scripts/simplified_demo.py
```

**Description** : Une démonstration épurée qui présente les fonctionnalités essentielles du système.

### `demo_enhanced_architecture.py` et `demo_phase2_enhancements.py`

**Fonction** : Démontrent les améliorations architecturales et les nouvelles fonctionnalités.

**Utilisation** :
```bash
python scripts/demo_enhanced_architecture.py
```

**Description** : Ces scripts illustrent les améliorations apportées à l'architecture et les nouvelles fonctionnalités introduites dans les différentes phases de développement.

## Scripts d'optimisation

Ces scripts permettent d'optimiser les paramètres des stratégies de trading.

### `optimize_moving_average.py`

**Fonction** : Trouve les paramètres optimaux pour la stratégie de moyenne mobile.

**Utilisation** :
```bash
python scripts/optimize_moving_average.py --symbol AAPL --period 90
```

**Description** : Utilise diverses techniques pour trouver les meilleures fenêtres court terme et long terme pour la stratégie de moyenne mobile.

### `optimized_portfolio.py`

**Fonction** : Optimise l'allocation du portefeuille pour un ensemble de symboles.

**Utilisation** :
```bash
python scripts/optimized_portfolio.py --symbols AAPL,MSFT,GOOGL,BTC-USD
```

**Description** : Utilise des techniques d'optimisation de portefeuille pour maximiser le ratio de Sharpe ou minimiser le risque.

## Scripts d'entraînement des modèles

Ces scripts permettent d'entraîner les différents modèles d'intelligence artificielle utilisés par MercurioAI.

### `train_lstm_model.py`

**Fonction** : Entraîne un modèle LSTM pour un actif spécifique.

**Paramètres** :
- `--symbol <symbole>` : Symbole de l'actif à utiliser pour l'entraînement (ex: BTC-USD, AAPL)
- `--lookback <jours>` : Nombre de jours d'historique à utiliser pour l'entraînement (défaut: 180)
- `--sequence_length <nombre>` : Longueur des séquences pour l'entraînement (défaut: 60)
- `--prediction_horizon <nombre>` : Nombre de périodes à prédire (défaut: 5)
- `--epochs <nombre>` : Nombre d'époques d'entraînement (défaut: 50)

**Utilisation** :
```bash
python scripts/train_lstm_model.py --symbol BTC-USD --lookback 180 --epochs 100
```

**Description** : Entraîne un modèle LSTM pour la prédiction de prix d'un actif spécifique. Le modèle entraîné est sauvegardé dans le répertoire `models/lstm/` et peut être utilisé par les stratégies de trading et le screener d'actifs.

**Cas d'utilisation** :
1. **Préparation des stratégies** : Pour améliorer la précision des prédictions de la stratégie LSTM
2. **Préparation du screener** : Pour permettre au screener d'actifs d'utiliser des modèles entraînés
3. **Expérimentation** : Pour tester différentes configurations de modèles sur des actifs spécifiques

### `train_transformer_model.py`

**Fonction** : Entraîne un modèle Transformer sur plusieurs actifs simultanément.

**Paramètres** :
- `--symbols <liste>` : Liste des symboles d'actifs séparés par des virgules (ex: BTC-USD,ETH-USD,AAPL)
- `--lookback <jours>` : Nombre de jours d'historique à utiliser pour l'entraînement (défaut: 180)
- `--epochs <nombre>` : Nombre d'époques d'entraînement (défaut: 50)
- `--use_gpu` : Utiliser le GPU si disponible (sans valeur)

**Utilisation** :
```bash
python scripts/train_transformer_model.py --symbols BTC-USD,ETH-USD,AAPL --epochs 100
```

**Description** : Entraîne un modèle Transformer pour la prédiction de prix sur plusieurs actifs simultanément. Le modèle entraîné est sauvegardé dans le répertoire `models/transformer/` et peut être utilisé par les stratégies de trading et le screener d'actifs.

**Cas d'utilisation** :
1. **Analyse multi-actifs** : Pour capturer les relations entre différents actifs
2. **Généralisation améliorée** : Pour créer un modèle capable de généraliser sur de nouveaux actifs
3. **Préparation du screener** : Pour permettre au screener d'actifs d'utiliser des modèles entraînés

### `train_all_models.py`

**Fonction** : Entraîne tous les modèles d'IA utilisés par MercurioAI en une seule commande.

**Paramètres** :
- `--symbols <liste>` : Liste des symboles d'actifs séparés par des virgules (si vide, utilise des actifs populaires)
- `--days <nombre>` : Nombre de jours d'historique à utiliser pour l'entraînement (défaut: 180)
- `--epochs <nombre>` : Nombre d'époques d'entraînement pour tous les modèles (défaut: 50)
- `--top_assets <nombre>` : Nombre d'actifs populaires à inclure automatiquement (défaut: 10)
- `--include_stocks` : Inclure les actions populaires (sans valeur)
- `--include_crypto` : Inclure les cryptomonnaies populaires (sans valeur)
- `--use_gpu` : Utiliser le GPU si disponible (sans valeur)

**Utilisation** :
```bash
python scripts/train_all_models.py --days 90 --top_assets 20
```

**Description** : Entraîne automatiquement tous les modèles d'IA utilisés par MercurioAI (LSTM et Transformer) sur les actifs spécifiés ou sur une liste d'actifs populaires. Les modèles entraînés sont sauvegardés dans les répertoires `models/lstm/` et `models/transformer/` et peuvent être utilisés par les stratégies de trading et le screener d'actifs.

**Cas d'utilisation** :
1. **Initialisation du système** : Pour préparer tous les modèles en une seule commande
2. **Mise à jour périodique** : Pour rafraîchir tous les modèles avec les données récentes
3. **Nouvelle installation** : Pour configurer rapidement un nouveau système MercurioAI

### `list_trained_models.py`

**Fonction** : Affiche une liste de tous les modèles entraînés disponibles dans le système.

**Utilisation** :
```bash
python scripts/list_trained_models.py
```

**Description** : Affiche une liste détaillée de tous les modèles LSTM et Transformer entraînés disponibles dans le système, avec des informations sur leur état, leur date de création, leur taille et leurs paramètres.

**Cas d'utilisation** :
1. **Inventaire des modèles** : Pour voir quels modèles sont déjà entraînés
2. **Vérification avant utilisation** : Pour vérifier que les modèles nécessaires sont disponibles avant d'exécuter le screener d'actifs
3. **Gestion des modèles** : Pour identifier les modèles obsolètes ou manquants

## Scripts de visualisation

Ces scripts génèrent des visualisations et des tableaux de bord pour suivre les performances de trading.

### `strategy_dashboard.py` et `trading_dashboard.py`

**Fonction** : Tableaux de bord interactifs pour évaluer les stratégies et suivre le trading.

**Utilisation** :
```bash
python scripts/strategy_dashboard.py
```

**Description** : Génèrent des interfaces utilisateur Streamlit qui permettent d'explorer interactivement les performances des stratégies (`strategy_dashboard.py`) ou des activités de trading (`trading_dashboard.py`).

### `comprehensive_dashboard.py`

**Fonction** : Tableau de bord complet qui combine toutes les métriques et visualisations.

**Utilisation** :
```bash
python scripts/comprehensive_dashboard.py
```

**Description** : Une interface utilisateur avancée qui intègre analyse de marché, suivi de portefeuille, performance des stratégies et journaux de trading.

### `generate_strategy_comparison_plot.py`

**Fonction** : Génère des graphiques comparatifs pour différentes stratégies.

**Utilisation** :
```bash
python scripts/generate_strategy_comparison_plot.py --output comparison.png
```

**Description** : Crée des visualisations qui comparent les performances de différentes stratégies sur un même graphique.

## Utilitaires auxiliaires

### `comprehensive_simulation.py`

**Fonction** : Simulation complète de trading sur des données historiques.

**Utilisation** :
```bash
python scripts/comprehensive_simulation.py --start 2022-01-01 --end 2022-12-31
```

**Description** : Effectue une simulation détaillée de toutes les stratégies sur une période historique définie, avec des rapports complets.

### `simulation_utils.py`

**Fonction** : Utilitaires pour les simulations de trading.

**Description** : Ce fichier n'est pas destiné à être exécuté directement, mais contient des fonctions utilisées par d'autres scripts de simulation.

---

## Comment utiliser ces scripts

1. Assurez-vous que votre fichier `.env` est correctement configuré avec les informations d'API nécessaires
2. Activez votre environnement virtuel Python
3. Exécutez les scripts depuis la racine du projet pour garantir que les chemins d'importation fonctionnent correctement

## Résolution des problèmes courants

- **Erreurs d'importation** : Assurez-vous d'exécuter les scripts depuis la racine du projet
- **Erreurs d'API** : Vérifiez vos clés API dans le fichier `.env` avec les scripts de test
- **Mode de trading** : Pour basculer entre paper trading et live trading, modifiez `ALPACA_MODE` dans `.env`

## Exemple de flux de travail

1. Utilisez `test_alpaca_connection.py` pour vérifier votre configuration
2. Exécutez `test_stocks_and_crypto.py` pour confirmer l'accès aux données
3. Utilisez `optimize_moving_average.py` pour optimiser votre stratégie
4. Lancez `strategy_dashboard.py` pour visualiser les performances
5. Démarrez le trading papier avec `auto_trader.py` pour tester en conditions réelles
