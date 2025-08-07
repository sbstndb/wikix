"""Wrapper autour de l'API OpenAI pour génération de fiches.

Pour garder le projet léger, on lit la clé dans la variable d'environnement
OPENAI_API_KEY ou depuis un fichier .env à la racine.
"""
from __future__ import annotations

import os
from typing import List

import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("La variable d'environnement OPENAI_API_KEY est requise.")

client = openai.OpenAI()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def generate_fiche(subject: str, template: str, temperature: float = 0.7) -> str:
    """Génère une fiche pour *subject* en utilisant *template* comme prompt.

    Le *template* doit contenir le placeholder {sujet}.
    """
    prompt = template.replace("{sujet}", subject)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_fiche_with_context(subject: str, context: str, template: str, temperature: float = 0.7) -> str:
    """Génère une fiche tenant compte d'un contexte précédent."""
    prompt = template.replace("{sujet}", subject).replace("{contexte}", context)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_fiche_stream(subject: str, template: str, temperature: float = 0.7, model: str = None):
    """Génère une fiche en streaming (générateur qui yield les mots au fur et à mesure)."""
    prompt = template.replace("{sujet}", subject)
    selected_model = model or MODEL_NAME
    
    # Adapter les paramètres selon le modèle
    params = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    
    # Certains modèles (comme o3-mini et o4-mini) ne supportent pas le paramètre temperature
    if not (selected_model.startswith("o3") or selected_model.startswith("o4")):
        params["temperature"] = temperature
    
    stream = client.chat.completions.create(**params)
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def generate_fiche_with_context_stream(subject: str, context: str, template: str, temperature: float = 0.7, model: str = None):
    """Génère une fiche avec contexte en streaming."""
    prompt = template.replace("{sujet}", subject).replace("{contexte}", context)
    selected_model = model or MODEL_NAME
    
    # Adapter les paramètres selon le modèle
    params = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    
    # Certains modèles (comme o3-mini et o4-mini) ne supportent pas le paramètre temperature
    if not (selected_model.startswith("o3") or selected_model.startswith("o4")):
        params["temperature"] = temperature
    
    stream = client.chat.completions.create(**params)
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content



