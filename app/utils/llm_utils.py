"""
LLM Utility functions for Mercurio AI.

This module provides functions for working with Large Language Models (LLMs)
in the Mercurio trading platform.
"""

import os
import logging
import requests
import json
from typing import Any, Dict, Optional, Union

# Import pour Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    
logger = logging.getLogger(__name__)

def load_llm_model(model_name: str, use_local: bool = False, 
                 local_path: Optional[str] = None,
                 api_key: Optional[str] = None) -> Any:
    """
    Load an LLM model for inference.
    
    This function handles loading different types of LLMs:
    - Local models (if use_local=True)
    - Remote API-based models (if use_local=False)
    
    Args:
        model_name: Name of the model to load
        use_local: Whether to use a local model or API
        local_path: Path to local model files (if use_local=True)
        api_key: API key for remote model access
        
    Returns:
        Model object or API client that can be used for inference
    """
    if use_local:
        return _load_local_model(model_name, local_path)
    else:
        return _initialize_remote_client(model_name, api_key)

def _load_local_model(model_name: str, local_path: Optional[str] = None) -> Any:
    """Load a local LLM model"""
    try:
        # Try to import necessary libraries
        model_type = model_name.lower()
        
        if 'llama' in model_type:
            try:
                from llama_cpp import Llama
                
                model_path = local_path or f"./models/llm/{model_name}.gguf"
                if os.path.exists(model_path):
                    logger.info(f"Loading local Llama model from {model_path}")
                    return Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=4
                    )
                else:
                    logger.error(f"Model file not found at {model_path}")
                    return None
            except ImportError:
                logger.error("llama-cpp-python not installed")
                return None
                
        elif 'mistral' in model_type or 'mixtral' in model_type:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                model_path = local_path or f"./models/llm/{model_name}"
                if os.path.exists(model_path):
                    logger.info(f"Loading local Mistral/Mixtral model from {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    return {"model": model, "tokenizer": tokenizer}
                else:
                    logger.error(f"Model directory not found at {model_path}")
                    return None
            except ImportError:
                logger.error("transformers not installed")
                return None
                
        elif any(x in model_type for x in ['gpt', 'openai']):
            logger.warning("OpenAI models should be used with the API, not locally")
            return None
            
        else:
            # Generic Hugging Face model loading
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                model_path = local_path or f"./models/llm/{model_name}"
                if os.path.exists(model_path):
                    logger.info(f"Loading local Hugging Face model from {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    return {"model": model, "tokenizer": tokenizer}
                else:
                    logger.error(f"Model directory not found at {model_path}")
                    return None
            except ImportError:
                logger.error("transformers not installed")
                return None
                
    except Exception as e:
        logger.error(f"Error loading local model {model_name}: {str(e)}")
        return None

def _initialize_remote_client(model_name: str, api_key: Optional[str] = None) -> Any:
    """Initialize client for remote LLM API"""
    try:
        model_type = model_name.lower()
        
        # Check for Anthropic Claude models
        if any(x in model_type for x in ['claude', 'anthropic']) and ANTHROPIC_AVAILABLE:
            try:
                # Setup API key from parameter or environment
                claude_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
                if not claude_api_key:
                    logger.error("No Anthropic API key provided")
                    return None
                
                # Get base URL from environment if available
                base_url = os.environ.get("ANTHROPIC_BASE_URL")
                
                if base_url:
                    logger.info(f"Initialized Anthropic client for model {model_name} with base_url: {base_url}")
                    client = anthropic.Anthropic(api_key=claude_api_key, base_url=base_url)
                else:
                    logger.info(f"Initialized Anthropic client for model {model_name} with default base_url")
                    client = anthropic.Anthropic(api_key=claude_api_key)
                    
                client.model_name = model_name  # Store the model name for later use
                return client
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {str(e)}")
                return None
                
        # Check for OpenAI models
        elif any(x in model_type for x in ['gpt', 'openai']):
            try:
                import openai
                
                # Setup API key from parameter or environment
                openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
                if not openai.api_key:
                    logger.error("No OpenAI API key provided")
                    return None
                    
                logger.info(f"Initialized OpenAI client for model {model_name}")
                return openai
            except ImportError:
                logger.error("openai package not installed")
                return None
                
        # Check for Hugging Face models
        elif 'huggingface' in model_type or 'hf' in model_type:
            try:
                from huggingface_hub import InferenceClient
                
                # Setup API key
                hf_token = api_key or os.environ.get("HF_TOKEN")
                if not hf_token:
                    logger.warning("No Hugging Face token provided, using default anonymous access")
                    
                logger.info(f"Initialized Hugging Face client for model {model_name}")
                client = InferenceClient(token=hf_token)
                client.model_name = model_name  # Store the model name for later use
                return client
            except ImportError:
                logger.error("huggingface_hub not installed")
                return None
                
        # For specific Hugging Face models
        elif any(x in model_type for x in ['mistral', 'mixtral', 'llama']):
            try:
                from huggingface_hub import InferenceClient
                
                # Setup API key
                hf_token = api_key or os.environ.get("HF_TOKEN")
                
                logger.info(f"Initialized Hugging Face client for model {model_name}")
                client = InferenceClient(token=hf_token)
                client.model_name = model_name  # Store the model name for later use
                return client
            except ImportError:
                logger.error("huggingface_hub not installed")
                return None
        
        # Generic client for other models
        else:
            logger.warning(f"Unknown model type: {model_name}, using fallback generic client")
            return {"model_name": model_name, "api_key": api_key}
            
    except Exception as e:
        logger.error(f"Error initializing remote client for {model_name}: {str(e)}")
        return None

def call_llm(model, prompt: str, temperature: float = 0.1, 
            max_tokens: int = 1024, stop_sequences: list = None, 
            force_real_llm: bool = False, system_prompt: str = None) -> str:
    """
    Generate text from a prompt using the given LLM model.
    
    Args:
        model: LLM model or client returned by load_llm_model
        prompt: Input prompt text
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        stop_sequences: List of stop sequences to end generation
        force_real_llm: If True, bypasses demo mode and uses the real LLM even in demo mode
        
    Returns:
        Generated text response
    """
    # Check for demo mode - runs when LLM_API_KEY=demo_mode in .env
    api_key = os.environ.get("LLM_API_KEY", "")
    if api_key.lower() == "demo_mode" and not force_real_llm:
        # Only use demo mode responses if not forcing real LLM
        logger.info("Running LLM in demo mode with sample responses")
        return _generate_demo_response(prompt)
        
    if model is None:
        logger.error("No model provided")
        return "Error: Model not available. Please check logs."
    
    try:
        # Handle Anthropic API
        if ANTHROPIC_AVAILABLE:
            try:
                logger.info(f"Calling Anthropic API with model: {model.model_name if hasattr(model, 'model_name') else 'claude-3-5-sonnet-20240620'}")
                logger.info(f"Anthropic API version: {anthropic.__version__ if hasattr(anthropic, '__version__') else 'unknown'}")
                
                # Déterminer le nom du modèle à utiliser
                model_name = model.model_name if hasattr(model, "model_name") else "claude-3-5-sonnet-20240620"
                logger.info(f"Using model: {model_name}")
                
                # Vérifier la version de l'API Anthropic
                anthropic_version = getattr(anthropic, "__version__", "0.0.0")
                logger.info(f"Detected Anthropic version: {anthropic_version}")
                
                # Compatibilité avec différentes versions de l'API Anthropic
                if hasattr(model, "messages"):
                    # Format pour API v0.4.0+
                    logger.info("Using Anthropic API v0.4.0+ format with messages.create()")
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    logger.info(f"Messages structure: {messages}")
                    
                    response = model.messages.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop_sequences=stop_sequences
                    )
                    logger.info("Anthropic API call successful using messages.create()")
                    return response.content[0].text
                elif hasattr(model, "completions"):
                    # Format pour API v0.3.x
                    logger.info("Using Anthropic API v0.3.x format with completions.create()")
                    response = model.completions.create(
                        model=model_name,
                        prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                        temperature=temperature,
                        max_tokens_to_sample=max_tokens,
                        stop_sequences=stop_sequences or ["\n\nHuman:"]
                    )
                    logger.info("Anthropic API call successful using completions.create()")
                    return response.completion
                else:
                    # Format pour API v0.5.x+
                    logger.info("Using Anthropic API v0.5.x+ format with direct client.messages()")
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    # Pour Anthropic v0.52.2
                    try:
                        response = model.messages.create(
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        logger.info("Anthropic API v0.52.2 call successful")
                        if hasattr(response, 'content') and len(response.content) > 0:
                            if hasattr(response.content[0], 'text'):
                                return response.content[0].text
                            elif hasattr(response.content[0], 'value'):
                                return response.content[0].value
                        return str(response)
                    except Exception as e:
                        logger.warning(f"Error with Anthropic API v0.52.2 format: {e}, trying direct call")
                        # Dernier recours - appel direct
                        response = model(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        logger.info("Anthropic API call successful using direct call")
                        return str(response)
            except Exception as e:
                logger.error(f"Error with Anthropic API: {str(e)}")
                return f"Error with Anthropic API: {str(e)}"
                
        # Handle OpenAI API
        elif hasattr(model, "ChatCompletion") or (isinstance(model, dict) and "openai" in str(model).lower()):
            openai_model = model.ChatCompletion if hasattr(model, "ChatCompletion") else model
            
            response = openai_model.create(
                model=model.model_name if hasattr(model, "model_name") else "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            
            return response.choices[0].message.content
            
        # Handle Hugging Face Hub InferenceClient
        elif hasattr(model, "text_generation"):
            response = model.text_generation(
                prompt,
                temperature=temperature,
                max_new_tokens=max_tokens,
                stop_sequences=stop_sequences
            )
            
            return response
            
        # Handle local Llama models
        elif hasattr(model, "generate") and hasattr(model, "detokenize"):
            # This is likely a llama-cpp model
            output = model.generate(
                prompt.encode(), 
                top_k=50,
                top_p=0.95,
                temp=temperature,
                repeat_penalty=1.1,
                max_tokens=max_tokens
            )
            
            return model.detokenize(output)
            
        # Handle local Hugging Face models
        elif isinstance(model, dict) and "model" in model and "tokenizer" in model:
            import torch
            
            tokenizer = model["tokenizer"]
            hf_model = model["model"]
            
            inputs = tokenizer(prompt, return_tensors="pt")
            attention_mask = inputs.attention_mask
            
            with torch.no_grad():
                output_ids = hf_model.generate(
                    inputs.input_ids,
                    attention_mask=attention_mask,
                    max_length=len(inputs.input_ids[0]) + max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            output = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            return output
            
        # Fallback for unknown model types
        else:
            logger.warning(f"Unknown model type: {type(model)}, using fallback generation")
            return f"ERROR: Unsupported model type: {type(model)}"
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error during generation: {str(e)}"

def _generate_demo_response(prompt: str) -> str:
    """
    Generate sample responses for demo mode when using LLM_API_KEY=demo_mode.
    This allows testing the strategy without an actual LLM API key.
    
    Args:
        prompt: The input prompt text
        
    Returns:
        A realistic sample response based on prompt content
    """
    # Check if it's a trading signal prompt requesting JSON output
    if "trading signal" in prompt.lower() and "action" in prompt.lower() and "JSON" in prompt:
        # Generate different responses based on key terms in the prompt
        if "macd: bullish" in prompt.lower() or \
           "rsi" in prompt.lower() and "oversold" in prompt.lower() or \
           "sma is above" in prompt.lower():
            return """{
  "action": "BUY",
  "confidence": 0.75,
  "justification": "Technical indicators show strong bullish momentum with positive MACD crossover and support at current price levels."
}""" 
        elif "macd: bearish" in prompt.lower() or \
             "rsi" in prompt.lower() and "overbought" in prompt.lower() or \
             "sma is below" in prompt.lower():
            return """{
  "action": "SELL",
  "confidence": 0.82,
  "justification": "Multiple bearish signals detected with negative MACD divergence and price breaking below support levels."
}"""  
        else:
            return """{
  "action": "HOLD",
  "confidence": 0.60,
  "justification": "Mixed signals in the market data with no clear directional trend. Waiting for more decisive price action."
}"""
    
    # If it's a sentiment analysis prompt
    elif "sentiment" in prompt.lower() and ("news" in prompt.lower() or "article" in prompt.lower()):
        if "positive" in prompt.lower() or "growth" in prompt.lower() or "increase" in prompt.lower():
            return """{
  "sentiment": "positive",
  "score": 0.78,
  "keywords": ["growth", "earnings", "bullish", "outperform"],
  "summary": "The overall sentiment is positive with strong indicators of continued growth and market optimism."
}"""
        elif "negative" in prompt.lower() or "decline" in prompt.lower() or "decrease" in prompt.lower():
            return """{
  "sentiment": "negative",
  "score": 0.67,
  "keywords": ["decline", "missed expectations", "bearish", "underperform"],
  "summary": "The overall sentiment is negative with concerns about performance and market position."
}"""
        else:
            return """{
  "sentiment": "neutral",
  "score": 0.51,
  "keywords": ["steady", "stable", "unchanged", "meeting expectations"],
  "summary": "The overall sentiment is neutral with balanced positive and negative factors."
}"""
    
    # Generic fallback response
    else:
        return "This is a demo response from the LLM strategy. In production, this would use the actual LLM API. To use a real LLM, set LLM_API_KEY to your API key in the .env file."


def analyze_sentiment(text: str, model_name: str = "finbert") -> Dict[str, float]:
    """
    Analyze sentiment in text using NLP models.
    
    Args:
        text: Text to analyze
        model_name: Name of sentiment model to use
        
    Returns:
        Dictionary with sentiment scores
    """
    try:
        if model_name == "finbert":
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                
                tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                labels = ["negative", "neutral", "positive"]
                
                return {labels[i]: float(predictions[0][i]) for i in range(len(labels))}
            except ImportError:
                logger.warning("transformers not installed, using fallback")
                
        # Simple fallback keyword-based sentiment
        positive_words = ["buy", "bullish", "up", "positive", "growth", "gain", "profit", "rally"]
        negative_words = ["sell", "bearish", "down", "negative", "loss", "fall", "drop", "decline"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total = pos_count + neg_count
        
        if total == 0:
            return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            
        pos_score = pos_count / total
        neg_score = neg_count / total
        neutral_score = 1.0 - (pos_score + neg_score)
        
        return {
            "positive": pos_score,
            "neutral": neutral_score,
            "negative": neg_score
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
