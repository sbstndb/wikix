"""Gestion des fournisseurs de modèles de langage (LLM).

Ce module fournit une abstraction pour interagir avec différents
fournisseurs de LLM comme OpenAI et Google Gemini.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator

import openai
import google.generativeai as genai
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Configuration des clés d'API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Initialisation des clients ---
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# --- Classe de base pour les fournisseurs LLM ---
class LLMProvider(ABC):
    """Classe de base abstraite pour un fournisseur de modèles de langage."""

    @abstractmethod
    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        """Génère du texte en streaming à partir d'un prompt."""
        pass

    def generate_with_context_stream(
        self,
        subject: str,
        context: str,
        template: str,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> Iterator[str]:
        """Génère du texte en streaming avec un contexte."""
        prompt = template.replace("{sujet}", subject).replace("{contexte}", context)
        yield from self.generate_stream(prompt, temperature, model)


# --- Implémentation pour OpenAI ---
class OpenAIProvider(LLMProvider):
    """Fournisseur pour les modèles OpenAI."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("La clé d'API OpenAI est requise pour ce fournisseur.")
        self.client = openai.OpenAI()
        self.default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        selected_model = model or self.default_model
        params = {"model": selected_model, "messages": [{"role": "user", "content": prompt}], "stream": True}
        if not (selected_model.startswith("o3") or selected_model.startswith("o4")):
            params["temperature"] = temperature
        
        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


# --- Implémentation pour OpenRouter ---
class OpenRouterProvider(LLMProvider):
    """Fournisseur utilisant l'API OpenRouter avec contrainte Cerebras."""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("La clé d'API OpenRouter est requise pour ce fournisseur.")
        self.base_url = "https://openrouter.ai/api/v1"
        # Modèle par défaut : GPT-OSS-120B (Cerebras)
        self.default_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")

    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        import json
        selected_model = model or self.default_model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": temperature,
            "provider": {"only": ["Groq"]},
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Erreur OpenRouter: {response.status_code} - {response.text}")

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue


# --- Implémentation pour Gemini ---
class GeminiProvider(LLMProvider):
    """Fournisseur pour les modèles Google Gemini."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("La clé d'API Gemini est requise pour ce fournisseur.")
        self.default_model = "gemini-1.5-flash"

    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        selected_model = model or self.default_model
        model_instance = genai.GenerativeModel(selected_model)
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature
        )
        
        responses = model_instance.generate_content(
            prompt,
            stream=True,
            generation_config=generation_config
        )
        
        for response in responses:
            yield response.text


# --- Implémentation pour Cerebras ---
class CerebrasProvider(LLMProvider):
    """Fournisseur pour les modèles Cerebras via Cloudflare AI Gateway."""

    def __init__(self):
        if not CEREBRAS_API_KEY:
            raise ValueError("La clé d'API Cerebras est requise pour ce fournisseur.")
        
        self.api_key = CEREBRAS_API_KEY
        self.base_url = "https://api.cerebras.ai/v1"
        self.default_model = "llama3.1-8b"

    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        selected_model = model or self.default_model
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Erreur Cerebras: {response.status_code} - {response.text}")
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Enlever 'data: '
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        import json
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue


# --- Fonctions dépréciées (à supprimer après refactoring complet) ---
def generate_fiche_stream(subject: str, template: str, temperature: float = 0.7, model: str = None):
    prompt = template.replace("{sujet}", subject)
    provider = get_provider(model)
    return provider.generate_stream(prompt, temperature, model)

def generate_fiche_with_context_stream(subject: str, context: str, template: str, temperature: float = 0.7, model: str = None):
    provider = get_provider(model)
    return provider.generate_with_context_stream(subject, context, template, temperature, model)


# --- Sélecteur de fournisseur ---
PROVIDER_MAPPING = {
    "gpt": OpenAIProvider,
    "o3": OpenAIProvider,
    "o4": OpenAIProvider,
    "gemini": GeminiProvider,
    "cerebras": CerebrasProvider,
    "llama": CerebrasProvider,
    "openrouter": OpenRouterProvider,
    "openai/gpt-oss": OpenRouterProvider,
}

def get_provider(model_name: str) -> LLMProvider:
    """Retourne le fournisseur approprié en fonction du nom du modèle."""
    if not model_name:
        return OpenAIProvider()
        
    for prefix, provider_class in PROVIDER_MAPPING.items():
        if model_name.startswith(prefix):
            return provider_class()
    
    raise ValueError(f"Aucun fournisseur trouvé pour le modèle '{model_name}'.")


# --- Fonctions pour la compatibilité descendante ---
def generate_fiche(subject: str, template: str, temperature: float = 0.7, model: str = None) -> str:
    """Génère une fiche simple (non-stream)."""
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    provider = get_provider(selected_model)
    stream = provider.generate_stream(template.replace("{sujet}", subject), temperature, selected_model)
    return "".join(stream)

def generate_fiche_with_context(subject: str, context: str, template: str, temperature: float = 0.7, model: str = None) -> str:
    """Génère une fiche avec contexte (non-stream)."""
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    provider = get_provider(selected_model)
    stream = provider.generate_with_context_stream(subject, context, template, temperature, selected_model)
    return "".join(stream)
