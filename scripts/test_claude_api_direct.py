import os
import sys
import logging
from datetime import datetime

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Importer les fonctions nécessaires
from app.utils.llm_utils import load_llm_model, call_llm
from app.strategies.llm_strategy_v3 import log_prompt

def main():
    """Test direct des appels à l'API Claude et de la journalisation"""
    try:
        # Créer le dossier logs s'il n'existe pas
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Dossier de logs vérifié: {log_dir}")
        
        # Vérifier que le fichier de log existe ou peut être créé
        log_file = os.path.join(log_dir, 'claude_prompts.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - TEST - === DÉBUT DU TEST DIRECT ===\n")
        logger.info(f"Fichier de log vérifié: {log_file}")
        
        # Vérifier la clé API
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.error("Clé API Anthropic non trouvée dans les variables d'environnement")
            return
        
        logger.info(f"Clé API Anthropic trouvée: {api_key[:5]}...{api_key[-5:]}")
        
        # Charger le modèle Claude
        logger.info("Chargement du modèle Claude...")
        model = load_llm_model("claude-3-opus-20240229")  # Utiliser le modèle confirmé fonctionnel
        if model is None:
            logger.error("Échec du chargement du modèle Claude")
            return
        
        logger.info("Modèle Claude chargé avec succès")
        
        # Test simple avec un prompt direct
        prompt = "Bonjour Claude, peux-tu me donner un résumé de la situation actuelle du Bitcoin en 2-3 phrases?"
        
        # Log du prompt
        log_prompt(f"===== TEST PROMPT =====\n{prompt}\n===========", "TEST")
        
        # Appel à l'API Claude avec force_real_llm=True pour éviter le mode démo
        logger.info("Appel à l'API Claude...")
        try:
            response = call_llm(model, prompt, temperature=0.2, max_tokens=512, force_real_llm=True)
            logger.info("Appel à l'API Claude réussi")
            
            # Log de la réponse
            log_prompt(f"===== TEST RÉPONSE =====\n{response}\n===========", "TEST")
            
            # Afficher la réponse
            logger.info(f"Réponse de Claude: {response}")
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API Claude: {e}")
            import traceback
            traceback.print_exc()
        
        # Vérifier le contenu du fichier de log
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Contenu du fichier de log (derniers 500 caractères): {content[-500:] if len(content) > 500 else content}")
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
