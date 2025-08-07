"""Configuration centrale pour Wikix.

Ce module contient les chemins, les constantes et les templates
utilisés à travers l'application pour éviter la duplication.
"""
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = BASE_DIR / "prompts"
GENERATED_DIR = BASE_DIR / "generated"

# Création du dossier pour les fiches générées
GENERATED_DIR.mkdir(exist_ok=True)

# Templates de prompts
try:
    TEMPLATE_GENERAL = (PROMPT_DIR / "fiche_generale.txt").read_text(encoding="utf-8")
    TEMPLATE_CONTEXT = (PROMPT_DIR / "fiche_contexte.txt").read_text(encoding="utf-8")
except FileNotFoundError:
    print("⚠️ Attention : Fichiers de prompt non trouvés. Certaines fonctionnalités pourraient être limitées.")
    TEMPLATE_GENERAL = "Parle-moi de {sujet}."
    TEMPLATE_CONTEXT = "En te basant sur ce texte:\n{contexte}\n\nParle-moi de {sujet}."

# Support multilingue pour les prompts
LANG_TEMPLATES = {
    "fr": {
        "general": TEMPLATE_GENERAL,
        "context": TEMPLATE_CONTEXT,
    },
    "en": {
        "general": (PROMPT_DIR / "fiche_generale_en.txt").read_text(encoding="utf-8"),
        "context": (PROMPT_DIR / "fiche_contexte_en.txt").read_text(encoding="utf-8"),
    },
    "es": {
        "general": (PROMPT_DIR / "fiche_generale_es.txt").read_text(encoding="utf-8"),
        "context": (PROMPT_DIR / "fiche_contexte_es.txt").read_text(encoding="utf-8"),
    },
}

def get_template_for_lang(lang: str, with_context: bool = False) -> str:
    """Récupère le template approprié pour une langue donnée."""
    lang_code = lang if lang in LANG_TEMPLATES else "fr"
    template_type = "context" if with_context else "general"
    return LANG_TEMPLATES[lang_code][template_type]

