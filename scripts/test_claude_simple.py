#!/usr/bin/env python
"""
Script de test simple pour vérifier l'intégration d'Anthropic Claude dans Mercurio
et tester la fonction log_prompt
"""

import os
import sys
import logging
from datetime import datetime

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importer la fonction log_prompt
from app.strategies.llm_strategy_v3 import log_prompt

# Importer le module Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("Module Anthropic disponible: Oui")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.error("Module Anthropic NON disponible")
    sys.exit(1)

def main():
    """Test simple de l'API Anthropic Claude et de la fonction log_prompt"""
    
    # Créer un nouveau fichier de log propre
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'claude_prompts.log')
    
    # Supprimer l'ancien fichier de log s'il existe
    if os.path.exists(log_file):
        os.remove(log_file)
        logger.info(f"Ancien fichier de log supprimé: {log_file}")
    
    # Créer un nouveau fichier de log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("")
    logger.info(f"Nouveau fichier de log créé: {log_file}")
    
    # Initialiser le logger
    log_prompt("===== TEST SIMPLE DE L'API CLAUDE =====", "TEST")
    
    # Récupérer la clé API depuis les variables d'environnement
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Clé API Anthropic non trouvée dans les variables d'environnement")
        return
    
    logger.info(f"Clé API Anthropic trouvée: {api_key[:5]}...{api_key[-5:]}")
    
    # Modèle à tester
    model_name = "claude-3-opus-20240229"  # Claude 3 Opus - modèle confirmé fonctionnel
    logger.info(f"Test du modèle: {model_name}")
    
    # Initialiser le client Anthropic
    try:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Client Anthropic initialisé avec succès: {type(client)}")
        
        # Prompt de test pour l'analyse de sentiment
        prompt = "Analyse le sentiment de cette phrase concernant Bitcoin: 'Le prix du Bitcoin a chuté de 10% aujourd'hui suite à des nouvelles réglementaires négatives.'"
        
        # Log du prompt
        log_prompt(f"===== PROMPT =====\n{prompt}", "TEST")
        
        # Appeler l'API Anthropic
        response = client.messages.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        # Récupérer le texte de la réponse
        response_text = response.content[0].text
        
        # Log de la réponse
        log_prompt(f"===== RÉPONSE =====\n{response_text}", "TEST")
        
        # Afficher la réponse
        logger.info(f"Réponse de {model_name} (premiers 200 caractères):")
        logger.info(f"{response_text[:200]}...")
        
        # Vérifier le contenu du fichier de log
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            logger.info(f"Contenu du fichier de log (longueur: {len(log_content)} caractères)")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API Anthropic: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
