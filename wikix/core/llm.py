"""Gestion des fournisseurs de modèles de langage (LLM).

Ce module fournit une abstraction propre et extensible pour interagir avec
différents fournisseurs de LLM (OpenAI, Google Gemini, Cerebras, OpenRouter).

Objectifs de conception (niveau senior):
- Imports paresseux et optionnels pour éviter les erreurs d'environnement
  lors de l'import du module si certaines dépendances ne sont pas installées.
- API cohérente: streaming et agrégation non-stream sur la même interface.
- Gestion d'erreurs explicite et messages d'action (clé manquante, paquet non
  installé, réponse HTTP invalide).
- Paramètres réseau avec timeouts configurables.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from importlib import import_module

# --- Chargement .env optionnel ---
def _safe_load_dotenv() -> None:
    try:
        dotenv_mod = import_module("dotenv")
        getattr(dotenv_mod, "load_dotenv")()
    except (ModuleNotFoundError, AttributeError):
        # dotenv non présent ou API inattendue: ignorer silencieusement
        return


_safe_load_dotenv()


# --- Importeurs paresseux (sans dépendances statiques) ---
def _require_module(module_name: str, install_hint: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            f"Le paquet requis '{module_name}' est manquant. Installez-le avec: {install_hint}"
        ) from exc


def _require_openai():
    return _require_module("openai", "pip install openai")


def _require_requests():
    return _require_module("requests", "pip install requests")


def _require_genai():
    # Le paquet s'installe sous le nom 'google-generativeai' mais le module est 'google.generativeai'
    return _require_module("google.generativeai", "pip install google-generativeai")

# --- Configuration des clés d'API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Paramètres réseau ---
DEFAULT_HTTP_TIMEOUT_S: float = float(os.getenv("LLM_HTTP_TIMEOUT_S", "60"))


# --- Classe de base pour les fournisseurs LLM ---
class LLMProvider(ABC):
    """Classe de base abstraite pour un fournisseur de modèles de langage."""

    @abstractmethod
    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        """Génère du texte en streaming à partir d'un prompt."""
        raise NotImplementedError

    # API non-stream standardisée (agrégation par défaut)
    def generate(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> str:
        return "".join(self.generate_stream(prompt, temperature, model))

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
        openai = _require_openai()
        if not OPENAI_API_KEY:
            raise ValueError("La clé d'API OpenAI est requise pour ce fournisseur.")
        # Configuration client isolée dans l'instance
        openai.api_key = OPENAI_API_KEY
        self.client = openai.OpenAI()
        self.default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        selected_model = model or self.default_model
        params = {"model": selected_model, "messages": [{"role": "user", "content": prompt}], "stream": True}
        # Certains modèles (o3/o4/gpt-5) n'acceptent pas temperature != 1
        if not (selected_model.startswith("o3") or selected_model.startswith("o4") or selected_model.startswith("gpt-5")):
            params["temperature"] = temperature

        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


# --- Implémentation pour OpenRouter ---
class OpenRouterProvider(LLMProvider):
    """Fournisseur utilisant l'API OpenRouter avec contrainte Cerebras."""

    def __init__(self):
        self._requests = _require_requests()
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
            # Optionnel: piloter le provider sous-jacent (commenté par défaut)
            # "provider": {"only": ["Groq"]},
        }

        response = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=DEFAULT_HTTP_TIMEOUT_S,
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
                            content_piece = delta.get("content")
                            if content_piece:
                                yield content_piece
                    except json.JSONDecodeError:
                        continue


# --- Implémentation pour Gemini ---
class GeminiProvider(LLMProvider):
    """Fournisseur pour les modèles Google Gemini."""

    def __init__(self):
        self._genai = _require_genai()
        if not GEMINI_API_KEY:
            raise ValueError("La clé d'API Gemini est requise pour ce fournisseur.")
        self.default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        # Configuration du client
        self._genai.configure(api_key=GEMINI_API_KEY)

    def generate_stream(
        self, prompt: str, temperature: float = 0.7, model: str | None = None
    ) -> Iterator[str]:
        import time
        selected_model = model or self.default_model
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                model_instance = self._genai.GenerativeModel(selected_model)
                generation_config = self._genai.types.GenerationConfig(temperature=temperature)

                responses = model_instance.generate_content(
                    prompt,
                    stream=True,
                    generation_config=generation_config
                )

                for response in responses:
                    # Certains chunks peuvent ne pas contenir de texte
                    text_part = getattr(response, "text", None)
                    if text_part:
                        yield text_part
                return  # Succès, sortir de la boucle de retry
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Erreurs de quota - ne pas retry, lever immédiatement
                if any(phrase in error_msg for phrase in [
                    "quota", "limit exceeded", "billing", "insufficient", "permission denied"
                ]):
                    raise LLMError(f"Quota API Gemini épuisé ou problème de facturation. "
                                 f"Vérifiez votre quota sur Google AI Studio ou passez à un autre provider (OpenAI, etc.). "
                                 f"Détail: {str(e)}") from e
                
                # Erreurs réseau/temporaires - retry avec backoff
                if attempt < max_retries - 1 and any(phrase in error_msg for phrase in [
                    "timeout", "connection", "network", "unavailable", "internal error", "rate limit"
                ]):
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                
                # Autres erreurs - lever immédiatement
                raise LLMError(f"Erreur Gemini (tentative {attempt + 1}/{max_retries}): {str(e)}") from e


# --- Implémentation pour Cerebras ---
class CerebrasProvider(LLMProvider):
    """Fournisseur pour les modèles Cerebras via Cloudflare AI Gateway."""

    def __init__(self):
        self._requests = _require_requests()
        if not CEREBRAS_API_KEY:
            raise ValueError("La clé d'API Cerebras est requise pour ce fournisseur.")

        self.api_key = CEREBRAS_API_KEY
        self.base_url = "https://api.cerebras.ai/v1"
        self.default_model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")

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

        response = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            stream=True,
            timeout=DEFAULT_HTTP_TIMEOUT_S,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Erreur Cerebras: {response.status_code} - {response.text}")

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]  # Enlever 'data: '
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content_piece = delta.get("content")
                            if content_piece:
                                yield content_piece
                    except json.JSONDecodeError:
                        continue


# --- Fonctions d'aide ---
class LLMError(RuntimeError):
    """Erreur générique pour les fournisseurs LLM."""


# --- Fonctions dépréciées (à supprimer après refactoring complet) ---
def generate_fiche_stream(
    subject: str,
    template: str,
    temperature: float = 0.7,
    model: str | None = None,
    provider: str | None = None,
):
    prompt = template.replace("{sujet}", subject)
    provider_impl = select_provider_with_fallback(provider, model)
    return provider_impl.generate_stream(prompt, temperature, model)

def generate_fiche_with_context_stream(
    subject: str,
    context: str,
    template: str,
    temperature: float = 0.7,
    model: str | None = None,
    provider: str | None = None,
):
    provider_impl = select_provider_with_fallback(provider, model)
    return provider_impl.generate_with_context_stream(subject, context, template, temperature, model)


# --- Sélecteur de fournisseur ---
PROVIDER_MAPPING: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "gpt": OpenAIProvider,
    "o3": OpenAIProvider,
    "o4": OpenAIProvider,
    "gpt-": OpenAIProvider,
    "gemini": GeminiProvider,
    "cerebras": CerebrasProvider,
    "llama": CerebrasProvider,
    "openrouter": OpenRouterProvider,
    # Quelques préfixes OpenRouter populaires
    "openrouter/anthropic": OpenRouterProvider,
    "openrouter/openai": OpenRouterProvider,
    "openrouter/google": OpenRouterProvider,
    "openrouter/deepseek": OpenRouterProvider,
}

def get_provider(model_name: str | None) -> LLMProvider:
    """Retourne le fournisseur approprié en fonction du nom du modèle."""
    if not model_name:
        # Si aucun modèle n'est fourni, tomber sur un modèle OpenAI raisonnable
        return OpenAIProvider()

    for prefix, provider_class in PROVIDER_MAPPING.items():
        if model_name.startswith(prefix):
            return provider_class()

    raise ValueError(f"Aucun fournisseur trouvé pour le modèle '{model_name}'.")


def select_provider(provider_name: str | None, model_name: str | None) -> LLMProvider:
    """Sélectionne un fournisseur explicitement ou déduit depuis le modèle."""
    if provider_name:
        normalized = provider_name.lower()
        for prefix, provider_class in PROVIDER_MAPPING.items():
            if normalized.startswith(prefix):
                try:
                    return provider_class()
                except Exception as e:
                    raise LLMError(f"Impossible d'initialiser le provider '{provider_name}': {str(e)}") from e
        valid = ", ".join(sorted(set(k.split("/")[0] for k in PROVIDER_MAPPING.keys())))
        raise ValueError(f"Provider inconnu '{provider_name}'. Providers valides: {valid}")
    return get_provider(model_name)


def select_provider_with_fallback(provider_name: str | None, model_name: str | None) -> LLMProvider:
    """Sélectionne un fournisseur avec fallback automatique en cas d'erreur de quota."""
    # Ordre de fallback préféré
    fallback_order = [
        ("openai", "gpt-4o-mini"),
        ("openrouter", "openai/gpt-oss-120b"), 
        ("cerebras", "llama3.1-8b"),
        ("gemini", "gemini-2.5-flash")
    ]
    
    # Essayer d'abord le provider/modèle demandé
    if provider_name or model_name:
        try:
            return select_provider(provider_name, model_name)
        except LLMError as e:
            # Si c'est une erreur de quota/billing, essayer un fallback
            if any(phrase in str(e).lower() for phrase in ["quota", "billing", "limit exceeded"]):
                print(f"⚠️ Erreur {provider_name or 'détecté'}: {str(e)}")
                print("🔄 Tentative de fallback vers un autre provider...")
            else:
                raise  # Reraise si ce n'est pas une erreur de quota
    
    # Essayer les providers dans l'ordre de fallback
    for fallback_provider, fallback_model in fallback_order:
        try:
            # Éviter de réessayer le même provider qui a échoué
            if provider_name and provider_name.lower() == fallback_provider:
                continue
                
            provider = select_provider(fallback_provider, fallback_model)
            print(f"✅ Fallback réussi vers {fallback_provider} ({fallback_model})")
            return provider
        except Exception:
            continue  # Essayer le suivant
    
    # Si tous les fallbacks échouent
    raise LLMError("Tous les providers disponibles ont échoué. Vérifiez vos clés d'API et quotas.")


# --- Fonctions pour la compatibilité descendante ---
def generate_fiche(
    subject: str,
    template: str,
    temperature: float = 0.7,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """Génère une fiche simple (non-stream)."""
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    provider_impl = select_provider_with_fallback(provider, selected_model)
    prompt = template.replace("{sujet}", subject)
    return provider_impl.generate(prompt, temperature, selected_model)

def generate_fiche_with_context(
    subject: str,
    context: str,
    template: str,
    temperature: float = 0.7,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """Génère une fiche avec contexte (non-stream)."""
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    provider_impl = select_provider_with_fallback(provider, selected_model)
    prompt = template.replace("{sujet}", subject).replace("{contexte}", context)
    return provider_impl.generate(prompt, temperature, selected_model)
