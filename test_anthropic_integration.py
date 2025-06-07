#!/usr/bin/env python
"""
Script de test pour vérifier l'intégration d'Anthropic Claude dans Mercurio
"""

import os
import logging
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importer le module Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    print("Module Anthropic disponible: Oui")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Module Anthropic disponible: Non")

def test_anthropic_direct():
    """Teste directement l'API Anthropic Claude"""
    
    logger.info("Test direct de l'API Anthropic Claude")
    
    # Récupérer la clé API depuis les variables d'environnement
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        logger.error("Clé API Anthropic non trouvée dans les variables d'environnement")
        return
    
    # Modèle à tester
    model_name = "claude-3-opus-20240229"  # Claude 3 Opus - modèle confirmé fonctionnel
    
    logger.info(f"Test du modèle: {model_name}")
    
    # Initialiser le client Anthropic
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print(f"Client Anthropic initialisé avec succès: {type(client)}")
        
        # Prompt de test pour l'analyse de sentiment
        prompt = "Analyse le sentiment de cette phrase concernant Bitcoin: 'Le prix du Bitcoin a chuté de 10% aujourd'hui suite à des nouvelles réglementaires négatives.'"
        
        # Appeler l'API Anthropic
        response = client.messages.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        # Afficher la réponse
        logger.info(f"Réponse de {model_name}:")
        response_text = response.content[0].text
        
        # Enregistrer la réponse dans un fichier
        with open("claude_response.txt", "w", encoding="utf-8") as f:
            f.write(f"Réponse de {model_name}:\n")
            f.write("-" * 80 + "\n")
            f.write(response_text + "\n")
            f.write("-" * 80 + "\n")
            
        print(f"Réponse enregistrée dans le fichier claude_response.txt")
        
        # Afficher les premiers 200 caractères de la réponse
        print("Début de la réponse:")
        print(response_text[:200] + "...")
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API Anthropic: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_anthropic_direct()
