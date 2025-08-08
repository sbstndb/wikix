from __future__ import annotations

import os
import curses # Garder pour la compatibilité des couleurs si on l'importe dans d'autres contextes

class TUIState:
    """Classe pour centraliser les configurations UI (thèmes, modèles, langues)."""

    def __init__(self):
        self.current_theme: str = "dark"
        self.themes: dict = {
            "dark": {"id": "dark", "name": "🌙 Sombre", "bg": curses.COLOR_BLACK, "text": curses.COLOR_WHITE, "title": curses.COLOR_CYAN,
                     "instructions": curses.COLOR_YELLOW, "status": curses.COLOR_GREEN, "border": curses.COLOR_BLUE,
                     "selection": curses.COLOR_MAGENTA, "cursor_bg": curses.COLOR_WHITE, "cursor_text": curses.COLOR_BLACK},
            "light": {"id": "light", "name": "☀️ Clair", "bg": curses.COLOR_WHITE, "text": curses.COLOR_BLACK, "title": curses.COLOR_BLUE,
                      "instructions": curses.COLOR_RED, "status": curses.COLOR_GREEN, "border": curses.COLOR_BLUE,
                      "selection": curses.COLOR_MAGENTA, "cursor_bg": curses.COLOR_BLACK, "cursor_text": curses.COLOR_WHITE},
            "ocean": {"id": "ocean", "name": "🌊 Océan", "bg": curses.COLOR_BLUE, "text": curses.COLOR_WHITE, "title": curses.COLOR_CYAN,
                      "instructions": curses.COLOR_YELLOW, "status": curses.COLOR_GREEN, "border": curses.COLOR_CYAN,
                      "selection": curses.COLOR_MAGENTA, "cursor_bg": curses.COLOR_WHITE, "cursor_text": curses.COLOR_BLUE},
            "matrix": {"id": "matrix", "name": "📟 Matrix", "bg": curses.COLOR_BLACK, "text": curses.COLOR_GREEN, "title": curses.COLOR_GREEN,
                       "instructions": curses.COLOR_GREEN, "status": curses.COLOR_GREEN, "border": curses.COLOR_GREEN,
                       "selection": curses.COLOR_WHITE, "cursor_bg": curses.COLOR_GREEN, "cursor_text": curses.COLOR_BLACK},
            "pure_black": {"id": "pure_black", "name": "🖤 Noir Pur", "bg": curses.COLOR_BLACK, "text": curses.COLOR_WHITE, "title": curses.COLOR_WHITE,
                           "instructions": curses.COLOR_WHITE, "status": curses.COLOR_WHITE, "border": curses.COLOR_WHITE,
                           "selection": curses.COLOR_BLACK, "cursor_bg": curses.COLOR_WHITE, "cursor_text": curses.COLOR_BLACK},
            "pure_white": {"id": "pure_white", "name": "🤍 Blanc Pur", "bg": curses.COLOR_WHITE, "text": curses.COLOR_BLACK, "title": curses.COLOR_BLACK,
                           "instructions": curses.COLOR_BLACK, "status": curses.COLOR_BLACK, "border": curses.COLOR_BLACK,
                           "selection": curses.COLOR_WHITE, "cursor_bg": curses.COLOR_BLACK, "cursor_text": curses.COLOR_WHITE}
        }

        self.current_language: str = "fr"
        self.languages: dict = {
            "fr": {"name": "Français", "flag": "FR"},
            "en": {"name": "English", "flag": "EN"},
            "es": {"name": "Español", "flag": "ES"}
        }

        self.current_model: str = "gpt-4o-mini"
        self.current_provider: str | None = None
        self.available_models: dict = {
            "gpt-4o-mini": {"name": "4o-mini", "symbol": "🔹", "features": "standard"},
            "gpt-4o": {"name": "4o", "symbol": "🔷", "features": "standard"},
            "o3-mini": {"name": "o3-mini", "symbol": "🟦", "features": "no_temp"},
            "o4-mini": {"name": "o4-mini", "symbol": "🟪", "features": "future"},
            "gemini-2.5-flash": {"name": "gemini-2.5-flash", "symbol": "⚡️", "features": "standard"},
            "gemini-2.5-flash-lite": {"name": "gemini-2.5-lite", "symbol": "💡", "features": "fast"},
            "gemini-2.5-pro": {"name": "gemini-2.5-pro", "symbol": "✨", "features": "advanced"},
            "llama3.1-8b": {"name": "llama-8b", "symbol": "🦙", "features": "ultra-fast"},
            "llama3.1-70b": {"name": "llama-70b", "symbol": "🚀", "features": "ultra-fast"},
            "anthropic/claude-3.5-sonnet": {"name": "claude-3.5-sonnet", "symbol": "🧠", "features": "reasoning"},
            "openai/gpt-4o": {"name": "gpt-4o (OR)", "symbol": "🔷", "features": "popular"},
            "google/gemini-2.5-flash": {"name": "gemini-2.5-flash (OR)", "symbol": "⚡️", "features": "popular"},
        }
        self.provider_models: dict = {
            None: list(self.available_models.keys()),
            "openai": ["gpt-4o-mini", "gpt-4o", "o3-mini", "o4-mini"],
            "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
            "cerebras": ["llama3.1-8b", "llama3.1-70b"],
            "openrouter": [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o",
                "google/gemini-2.5-flash",
            ],
        }

# Fallback minimal si curses n'est pas disponible (environnement web)
try:
    import curses # Tenter l'importation de curses
except ImportError:
    # Si curses n'est pas là, définir un mock pour éviter les erreurs
    class MockCurses:
        COLOR_BLACK = 0
        COLOR_WHITE = 1
        COLOR_CYAN = 2
        COLOR_YELLOW = 3
        COLOR_GREEN = 4
        COLOR_BLUE = 5
        COLOR_MAGENTA = 6

    curses = MockCurses()

