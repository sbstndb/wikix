"""Interface TUI simple basée sur curses pour navigation dans le texte.

Améliorations niveau senior:
- Regex de mots robuste (accents, apostrophes typographiques, tirets, chiffres)
- Pagination clavier (PgUp/PgDn, Home/End, g/G) et molette souris
- Compatibilité provider/modèle clarifiée
- Constantes d'UI centralisées; robustesse écriture de fichier
- Détection fiable de l'état « en génération »
"""
from __future__ import annotations

import contextlib
import curses
import re
import textwrap
import threading
import time

from wikix.core.config import (
    GENERATED_DIR,
    get_template_for_lang,
)
from wikix.core.llm import generate_fiche_stream, generate_fiche_with_context_stream


# --- Constantes UI ---
HEADER_HEIGHT = 4
TITLE_TO_CONTENT_PADDING = 3  # lignes vides après le titre de la fiche avant le texte
HEADER_TOP_PADDING = 2           # lignes vides au-dessus du logo WIKIX
HEADER_BETWEEN_PADDING = 2       # lignes vides entre WIKIX et le sujet encadré
INSTRUCTIONS_HEIGHT = 1
SEPARATOR_HEIGHT = 1
STATUS_HEIGHT = 1

# Préfixe de message de génération (utilisé pour éviter de traiter ce placeholder comme un vrai contexte)
GENERATING_MSG_PREFIX = "🔄 Génération de '"

# --- Regex & wrapping ---
# Mots: lettres (ASCII + étendu), chiffres, avec apostrophes/tirets internes (ex: aujourd’hui, Jean-Pierre, 4G)
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[’'\-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*")

# Défilement via molette
SCROLL_WHEEL_LINES = 3


# --- Mesures d'affichage robustes (emoji, accents) ---
try:
    from wcwidth import wcswidth as _wcswidth
except ImportError:  # pragma: no cover - fallback si wcwidth non dispo
    _wcswidth = None


def _display_width(text: str) -> int:
    """Retourne la largeur d'affichage (colonnes) d'une chaîne.
    Utilise wcwidth si disponible, sinon longueur naïve.
    """
    if _wcswidth is None:
        return len(text)
    width = _wcswidth(text)
    if width is not None and width >= 0:
        return width
    # Fallback caractère par caractère
    total = 0
    for ch in text:
        w = _wcswidth(ch) if _wcswidth else 1
        if w is None or w < 0:
            w = 1
        total += w
    return total


def _truncate_display(text: str, max_columns: int) -> str:
    """Tronque la chaîne pour ne pas dépasser max_columns en colonnes d'affichage."""
    if max_columns <= 0:
        return ""
    if _wcswidth is None:
        return text[: max(0, max_columns)]
    acc = 0
    out_chars: list[str] = []
    for ch in text:
        w = _wcswidth(ch)
        if w is None or w < 0:
            w = 1
        if acc + w > max_columns:
            break
        out_chars.append(ch)
        acc += w
    return "".join(out_chars)


def _wrap_text_display(text: str, max_columns: int) -> list[str]:
    """Découpe le texte en lignes n'excédant pas max_columns colonnes d'affichage.
    Essaye de couper par mots; en cas de mot trop long, coupe par caractères.
    """
    if max_columns <= 0:
        return [""]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if _display_width(candidate) <= max_columns:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        # Mot trop long: couper par caractères
        remaining = word
        while remaining:
            take = _truncate_display(remaining, max_columns)
            if not take:
                break
            lines.append(take)
            remaining = remaining[len(take):]
    if current:
        lines.append(current)
    return lines or [""]


# ASCII art multi-lignes pour "wikix"
WIKIX_ASCII_ART: list[str] = [
    "W   W  III  K  K  III  X   X",
    "W   W   I   K K    I    X X ",
    "W W W   I   KK     I     X  ",
    "WW WW   I   K K    I    X X ",
    "W   W  III  K  K  III  X   X",
]


class TUIState:
    """Encapsule l'état de l'interface TUI pour une meilleure gestion."""
    def __init__(self, initial_subject: str = "Wikipédia"):
        self.current_subject: str = initial_subject
        self.current_text: str = ""
        self.words: list[tuple[str, int, int]] = []
        self.wrapped_lines: list[str] = []
        self.cursor_word_index: int = 0
        self.history: list[str] = []
        self.full_history: list[str] = [initial_subject]
        self.history_cursor: int = 0
        self.history_mode: bool = False
        self.history_scroll: int = 0
        self.subject_texts: dict[str, str] = {initial_subject: ""}

        self.scroll_offset: int = 0
        self.is_generating: bool = False
        self.streaming_thread: threading.Thread | None = None
        self.stop_streaming: bool = False
        self.text_lock = threading.Lock()
        self.last_text_hash: int = 0
        self.force_redraw: bool = False

        self.selection_mode: bool = False
        self.selection_start: int = -1
        self.selection_end: int = -1

        self.current_language: str = "fr"
        self.languages: dict = {
            "fr": {"name": "Français", "flag": "FR"},
            "en": {"name": "English", "flag": "EN"},
            "es": {"name": "Español", "flag": "ES"}
        }

        self.input_mode: bool = False
        self.input_text: str = ""
        self.input_prompt: str = "Entrez un sujet :"

        self.current_model: str = "gpt-4o-mini"
        self.current_provider: str | None = None
        # Modèles supportés
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
            # OpenRouter: 3 modèles populaires (slugs valides OpenRouter)
            "anthropic/claude-3.5-sonnet": {"name": "claude-3.5-sonnet", "symbol": "🧠", "features": "reasoning"},
            "openai/gpt-4o": {"name": "gpt-4o (OR)", "symbol": "🔷", "features": "popular"},
            "google/gemini-2.5-flash": {"name": "gemini-2.5-flash (OR)", "symbol": "⚡️", "features": "popular"},
        }
        # Filtrage par provider
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

        self.current_theme: str = "dark"
        self.themes: dict = {
            "dark": {
                "bg": curses.COLOR_BLACK, "text": curses.COLOR_WHITE, "title": curses.COLOR_CYAN,
                "instructions": curses.COLOR_YELLOW, "status": curses.COLOR_GREEN, "border": curses.COLOR_BLUE,
                "selection": curses.COLOR_MAGENTA, "cursor_bg": curses.COLOR_WHITE, "cursor_text": curses.COLOR_BLACK
            },
            "light": {
                "bg": curses.COLOR_WHITE, "text": curses.COLOR_BLACK, "title": curses.COLOR_BLUE,
                "instructions": curses.COLOR_RED, "status": curses.COLOR_GREEN, "border": curses.COLOR_BLUE,
                "selection": curses.COLOR_MAGENTA, "cursor_bg": curses.COLOR_BLACK, "cursor_text": curses.COLOR_WHITE
            },
            "ocean": {
                "bg": curses.COLOR_BLUE, "text": curses.COLOR_WHITE, "title": curses.COLOR_CYAN,
                "instructions": curses.COLOR_YELLOW, "status": curses.COLOR_GREEN, "border": curses.COLOR_CYAN,
                "selection": curses.COLOR_MAGENTA, "cursor_bg": curses.COLOR_WHITE, "cursor_text": curses.COLOR_BLUE
            },
            "matrix": {
                "bg": curses.COLOR_BLACK, "text": curses.COLOR_GREEN, "title": curses.COLOR_GREEN,
                "instructions": curses.COLOR_GREEN, "status": curses.COLOR_GREEN, "border": curses.COLOR_GREEN,
                "selection": curses.COLOR_WHITE, "cursor_bg": curses.COLOR_GREEN, "cursor_text": curses.COLOR_BLACK
            },
            "pure_black": {
                "bg": curses.COLOR_BLACK, "text": curses.COLOR_WHITE, "title": curses.COLOR_WHITE,
                "instructions": curses.COLOR_WHITE, "status": curses.COLOR_WHITE, "border": curses.COLOR_WHITE,
                "selection": curses.COLOR_BLACK, "cursor_bg": curses.COLOR_WHITE, "cursor_text": curses.COLOR_BLACK
            },
            "pure_white": {
                "bg": curses.COLOR_WHITE, "text": curses.COLOR_BLACK, "title": curses.COLOR_BLACK,
                "instructions": curses.COLOR_BLACK, "status": curses.COLOR_BLACK, "border": curses.COLOR_BLACK,
                "selection": curses.COLOR_WHITE, "cursor_bg": curses.COLOR_BLACK, "cursor_text": curses.COLOR_WHITE
            }
        }

        # Nombre de lignes physiques utilisées pour les instructions (centrage plein écran)
        self.instructions_lines_count: int = 1


class SimpleTUI:
    def __init__(self, initial_subject: str = "Wikipédia"):
        self.state = TUIState(initial_subject)
        # Hauteurs dynamiques mémorisées
        self._last_header_height: int = HEADER_HEIGHT

    def _draw_header(self, stdscr, screen_width, start_y: int):
        """Dessine l'en-tête ASCII art 'WIKIX' + sujet encadré à partir de start_y, centrés plein écran.
        Retourne la hauteur réellement dessinée (nb de lignes).
        """
        def centered_x(s: str) -> int:
            return max(0, (screen_width - _display_width(s)) // 2)

        current_y = start_y
        # Espace au-dessus du logo WIKIX
        for _ in range(HEADER_TOP_PADDING):
            with contextlib.suppress(curses.error):
                stdscr.addstr(current_y, 0, "")
            current_y += 1
        # 1) ASCII ART WIKIX
        for line in WIKIX_ASCII_ART:
            draw_line = _truncate_display(line, max_columns=screen_width)
            try:
                stdscr.addstr(current_y, centered_x(draw_line), draw_line, curses.color_pair(1) | curses.A_BOLD)
            except curses.error:
                with contextlib.suppress(curses.error):
                    stdscr.addstr(current_y, centered_x(draw_line), draw_line, curses.A_BOLD)
            current_y += 1

        # Espace entre le logo et le sujet encadré
        for _ in range(HEADER_BETWEEN_PADDING):
            with contextlib.suppress(curses.error):
                stdscr.addstr(current_y, 0, "")
            current_y += 1

        # 2) Sujet encadré
        subject = self.state.current_subject.upper()
        border_top = "┌" + "─" * (len(subject) + 2) + "┐"
        border_mid = f"│ {subject} │"
        border_bot = "└" + "─" * (len(subject) + 2) + "┘"
        for line in (border_top, border_mid, border_bot):
            display_line = _truncate_display(line, max_columns=screen_width)
            try:
                stdscr.addstr(current_y, centered_x(display_line), display_line, curses.color_pair(1) | curses.A_BOLD)
            except curses.error:
                with contextlib.suppress(curses.error):
                    stdscr.addstr(current_y, centered_x(display_line), display_line, curses.A_BOLD)
            current_y += 1

        # Padding supplémentaire entre le titre de la fiche et le texte
        for _ in range(TITLE_TO_CONTENT_PADDING):
            with contextlib.suppress(curses.error):
                stdscr.addstr(current_y, 0, "")
            current_y += 1

        return current_y - start_y

    def _draw_history_panel(self, stdscr, height, separator_line, left_panel_width):
        """Dessine le panneau de l'historique."""
        panel_height = height - (separator_line + 1) - 1
        panel_start_line = separator_line + 1
        # Suppression de la barre verticale pour un design plus épuré
        # (anciennes barres verticales supprimées)
        title = "Historique"
        try:
            stdscr.addstr(panel_start_line, 1, title[:left_panel_width-2], curses.color_pair(3) | curses.A_BOLD)
        except curses.error:
            stdscr.addstr(panel_start_line, 1, title[:left_panel_width-2], curses.A_BOLD)

        visible_items = self.state.full_history
        max_visible = panel_height - 2
        if len(visible_items) > max_visible:
            if self.state.history_cursor < self.state.history_scroll:
                self.state.history_scroll = self.state.history_cursor
            elif self.state.history_cursor >= self.state.history_scroll + max_visible:
                self.state.history_scroll = self.state.history_cursor - max_visible + 1
            items_slice = visible_items[self.state.history_scroll:self.state.history_scroll + max_visible]
        else:
            self.state.history_scroll = 0
            items_slice = visible_items
        for idx, subj in enumerate(items_slice):
            line_no = panel_start_line + 1 + idx
            is_current = (subj == self.state.current_subject)
            is_selected = (self.state.history_mode and (self.state.history_scroll + idx) == self.state.history_cursor)
            display_text = subj[:left_panel_width-2].ljust(left_panel_width-2)
            try:
                if is_selected:
                    stdscr.addstr(line_no, 1, display_text, curses.color_pair(2))
                elif is_current:
                    stdscr.addstr(line_no, 1, display_text, curses.color_pair(7))
                else:
                    stdscr.addstr(line_no, 1, display_text, curses.color_pair(1))
            except curses.error:
                attr = curses.A_REVERSE if is_selected else (curses.A_BOLD if is_current else curses.A_NORMAL)
                stdscr.addstr(line_no, 1, display_text, attr)

    def _draw_content(self, stdscr, height, screen_width, separator_line, left_panel_width, current_text_copy):
        """Dessine le contenu principal de la fiche en itérant sur les mots."""
        # Les calculs se basent sur la largeur totale de l'écran et tiennent compte du panneau historique
        content_left_margin, text_width = self.calculate_text_margins(screen_width, left_panel_width)

        words, wrapped_lines = self.extract_words(current_text_copy, text_width)
        self.state.words = words
        self.state.wrapped_lines = wrapped_lines
        # Assurer que le défilement reste dans les bornes
        # Désactiver totalement l'auto-scroll pour éviter les sauts quand on appuie une touche
        # (le scroll ne change que via input explicite)

        _, display_height = self.calculate_content_area(height)
        content_start_y = separator_line + 1
        # Pendant le streaming, ancrer en haut; sinon, respecter l'offset utilisateur
        scroll_start_line = 0 if self.state.is_generating else self.state.scroll_offset

        visible_lines = wrapped_lines[scroll_start_line : scroll_start_line + display_height]
        # Toujours ancrer le texte en haut de la zone de contenu (pas de centrage vertical)
        vertical_offset = 0

        # Dessiner le texte de fond
        for i, line_text in enumerate(visible_lines):
            screen_y = i + content_start_y + vertical_offset
            screen_x = content_left_margin
            # Éviter d'écrire sous le panneau historique (clipping à gauche)
            draw_x = max(screen_x, left_panel_width)
            if draw_x >= screen_width:
                continue
            clip_start = max(0, draw_x - screen_x)
            max_cols = max(0, screen_width - draw_x)
            visible_text = line_text[clip_start: clip_start + max_cols]
            with contextlib.suppress(curses.error):
                stdscr.addstr(screen_y, draw_x, visible_text, curses.color_pair(1))

        # Déterminer la plage de sélection
        selection_range = set()
        if self.state.selection_mode and self.state.selection_start != -1 and self.state.selection_end != -1:
            start_idx = min(self.state.selection_start, self.state.selection_end)
            end_idx = max(self.state.selection_start, self.state.selection_end)
            selection_range = set(range(start_idx, end_idx + 1))

        # Redessiner les mots avec un style spécial
        # Autoriser la surbrillance pendant le streaming, mais pas pendant le placeholder "Génération de ..."
        placeholder_active = current_text_copy.startswith(GENERATING_MSG_PREFIX)
        if not placeholder_active:
            for i, (text, line_idx, col_start) in enumerate(self.state.words):
                if scroll_start_line <= line_idx < scroll_start_line + display_height:
                    style = None
                    if i == self.state.cursor_word_index:
                        style = curses.color_pair(2)
                    elif i in selection_range:
                        style = curses.color_pair(7) | curses.A_BOLD

                    if style:
                        screen_y = (line_idx - scroll_start_line) + content_start_y + vertical_offset
                        base_x = content_left_margin + col_start
                        # Clipper si le mot commence sous le panneau historique
                        draw_x = max(base_x, left_panel_width)
                        if draw_x < screen_width:
                            clip_start = max(0, draw_x - base_x)
                            max_cols = max(0, screen_width - draw_x)
                            visible_text = text[clip_start: clip_start + max_cols]
                            if visible_text:
                                with contextlib.suppress(curses.error):
                                    stdscr.addstr(screen_y, draw_x, visible_text, style)

    def calculate_text_margins(self, width: int, left_panel_width: int = 0) -> tuple[int, int]:
        """Calcule des marges équilibrées et une largeur de texte maximale raisonnable.

        Si un panneau d'historique occupe la gauche (left_panel_width > 0), on réduit
        la largeur de texte pour que le bloc reste CENTRÉ sur l'écran tout en évitant
        de passer sous le panneau.
        """
        # Largeur de texte cible (80 colonnes si possible)
        max_text_width = 80
        # Limite dure: garder 4 colonnes de marge de chaque côté
        hard_limit = max(20, width - 8)
        text_width = min(max_text_width, hard_limit)

        if left_panel_width > 0:
            # Pour rester centré sur l'écran sans chevaucher le panneau gauche, il faut:
            # start_x = (width - text_width) // 2 >= left_panel_width
            #  => text_width <= width - 2*left_panel_width
            max_centerable = max(10, width - 2 * left_panel_width)
            text_width = min(text_width, max_centerable)

        # Marges gauche/droite identiques (centrage global sur l'écran)
        left_margin = max(0, (width - text_width) // 2)
        return left_margin, text_width

    def extract_words(self, text: str, text_width: int) -> tuple[list[tuple[str, int, int]], list[str]]:
        """Extrait les mots avec leurs positions après wrapping du texte."""
        words = []
        lines = text.split("\n")
        wrapped_lines = []

        for line in lines:
            if line.strip():
                # Utiliser text_width directement pour le wrapping
                wrapped = textwrap.fill(line, width=text_width)
                wrapped_lines.extend(wrapped.split("\n"))
            else:
                wrapped_lines.append("")

        for line_num, line in enumerate(wrapped_lines):
            for match in WORD_PATTERN.finditer(line):
                word = match.group()
                # La position (colonne) est relative au début de la ligne wrappée
                words.append((word, line_num, match.start()))

        return words, wrapped_lines

    def get_selected_text(self) -> str:
        """Retourne le texte sélectionné (groupe de mots)."""
        if not self.state.selection_mode or self.state.selection_start == -1 or self.state.selection_end == -1:
            if self.state.words and self.state.cursor_word_index < len(self.state.words):
                return self.state.words[self.state.cursor_word_index][0]
            return ""

        start_idx = min(self.state.selection_start, self.state.selection_end)
        end_idx = max(self.state.selection_start, self.state.selection_end)

        selected_words = [self.state.words[i][0] for i in range(start_idx, end_idx + 1) if i < len(self.state.words)]

        return " ".join(selected_words)

    def clear_selection(self):
        """Efface la sélection actuelle."""
        self.state.selection_mode = False
        self.state.selection_start = -1
        self.state.selection_end = -1
        self.state.force_redraw = True

    def start_selection(self):
        """Démarre une nouvelle sélection au curseur actuel."""
        self.state.selection_mode = True
        self.state.selection_start = self.state.cursor_word_index
        self.state.selection_end = self.state.cursor_word_index
        self.state.force_redraw = True

    def update_selection_end(self):
        """Met à jour la fin de la sélection au curseur actuel."""
        if self.state.selection_mode:
            self.state.selection_end = self.state.cursor_word_index
            self.state.force_redraw = True



    def toggle_theme(self):
        """Cycle entre les thèmes disponibles."""
        theme_keys = list(self.state.themes.keys())
        current_index = theme_keys.index(self.state.current_theme)
        next_index = (current_index + 1) % len(theme_keys)
        self.state.current_theme = theme_keys[next_index]
        self.state.force_redraw = True

    def cycle_language(self):
        """Cycle entre les langues disponibles."""
        lang_keys = list(self.state.languages.keys())
        current_index = lang_keys.index(self.state.current_language)
        next_index = (current_index + 1) % len(lang_keys)
        self.state.current_language = lang_keys[next_index]
        self.state.force_redraw = True

    def get_models_for_current_provider(self) -> list[str]:
        return self.state.provider_models.get(self.state.current_provider, list(self.state.available_models.keys()))

    def _reconcile_model_with_provider(self):
        allowed = self.get_models_for_current_provider()
        if self.state.current_model not in allowed:
            self.state.current_model = allowed[0]

    def cycle_model(self):
        """Cycle entre les modèles disponibles selon le provider courant."""
        allowed = self.get_models_for_current_provider()
        if self.state.current_model not in allowed:
            self.state.current_model = allowed[0]
            self.state.force_redraw = True
            return
        current_index = allowed.index(self.state.current_model)
        next_index = (current_index + 1) % len(allowed)
        self.state.current_model = allowed[next_index]
        self.state.force_redraw = True

    def cycle_provider(self):
        """Cycle entre les providers disponibles (None -> openai -> openrouter -> gemini -> cerebras -> None)."""
        providers = [None, "openai", "openrouter", "gemini", "cerebras"]
        idx = providers.index(self.state.current_provider) if self.state.current_provider in providers else 0
        self.state.current_provider = providers[(idx + 1) % len(providers)]
        self._reconcile_model_with_provider()
        self.state.force_redraw = True

    def start_input_mode(self):
        """Démarre le mode de saisie de texte."""
        self.state.input_mode = True
        self.state.input_text = ""
        if self.state.current_language == "en":
            self.state.input_prompt = "Enter a subject:"
        elif self.state.current_language == "es":
            self.state.input_prompt = "Ingrese un tema:"
        else:
            self.state.input_prompt = "Entrez un sujet :"
        self.state.force_redraw = True

    def cancel_input_mode(self):
        """Annule le mode de saisie."""
        self.state.input_mode = False
        self.state.input_text = ""
        self.state.force_redraw = True

    def handle_input_char(self, key):
        """Gère la saisie de caractères en mode input."""
        if key == 27:
            self.cancel_input_mode()
        elif key in (ord("\n"), ord("\r")):
            if self.state.input_text.strip():
                return self.state.input_text.strip()
            self.cancel_input_mode()
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if self.state.input_text:
                self.state.input_text = self.state.input_text[:-1]
                self.state.force_redraw = True
        elif 32 <= key <= 126:
            self.state.input_text += chr(key)
            self.state.force_redraw = True
        return None

    def init_colors(self):
        """Initialise les couleurs selon le thème actuel."""
        if not curses.has_colors():
            return

        try:
            curses.start_color()
            curses.use_default_colors()
            theme = self.state.themes[self.state.current_theme]

            # Paires de couleurs standard
            curses.init_pair(1, theme["text"], theme["bg"])
            curses.init_pair(2, theme["cursor_text"], theme["cursor_bg"])
            curses.init_pair(3, theme["title"], theme["bg"])
            curses.init_pair(4, theme["instructions"], theme["bg"])
            curses.init_pair(5, theme["status"], theme["bg"])
            curses.init_pair(6, theme["border"], theme["bg"])

            # Gestion spéciale de la sélection pour les thèmes purs
            if self.state.current_theme in ["pure_black", "pure_white"]:
                # Inverser le texte et le fond pour la sélection
                curses.init_pair(7, theme["bg"], theme["text"])
            else:
                curses.init_pair(7, theme["selection"], theme["bg"])

        except curses.error:
            # Fallback en cas d'erreur
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(7, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

    def get_templates_for_language(self):
        """Retourne les templates appropriés pour la langue actuelle."""
        return get_template_for_lang(self.state.current_language, with_context=False), \
               get_template_for_lang(self.state.current_language, with_context=True)

    def generate_fiche_streaming_thread(self, subject: str, previous_text: str):
        """Génère une fiche en streaming dans un thread séparé."""
        try:
            general_template, context_template = self.get_templates_for_language()

            if previous_text and subject != self.state.current_subject:
                stream_gen = generate_fiche_with_context_stream(
                    subject,
                    previous_text,
                    context_template,
                    model=self.state.current_model,
                    provider=self.state.current_provider,
                )
            else:
                stream_gen = generate_fiche_stream(
                    subject,
                    general_template,
                    model=self.state.current_model,
                    provider=self.state.current_provider,
                )

            full_content = ""
            for chunk in stream_gen:
                if self.state.stop_streaming:
                    break
                # Ajouter le chunk, mais afficher sans déclencher de scroll automatique
                full_content += chunk
                with self.state.text_lock:
                    self.state.current_text = full_content
                    self.state.force_redraw = True
                time.sleep(0.06)

            if not self.state.stop_streaming:
                output_path = GENERATED_DIR / f"{subject.lower().replace(' ', '_')}.md"
                # Écriture robuste: s'assurer que le dossier existe et écrire de façon atomique
                GENERATED_DIR.mkdir(parents=True, exist_ok=True)
                tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                tmp_path.write_text(full_content, encoding="utf-8")
                tmp_path.replace(output_path)

            # Sauvegarder le texte pour le sujet généré (pas forcément le sujet courant si l'utilisateur a navigué)
            with self.state.text_lock:
                self.state.subject_texts[subject] = full_content
                if subject in self.state.full_history:
                    self.state.full_history.remove(subject)
                self.state.full_history.insert(0, subject)
            self.state.is_generating = False
        except (ValueError, RuntimeError, OSError, ImportError) as e:
            with self.state.text_lock:
                self.state.current_text = f"Erreur lors de la génération : {str(e)}"
            self.state.is_generating = False

    def draw_screen(self, stdscr):
        """Dessine l'écran principal."""
        with self.state.text_lock:
            current_text_copy = self.state.current_text
            current_hash = hash(current_text_copy + str(self.state.cursor_word_index) + str(self.state.scroll_offset))
            needs_redraw = (current_hash != self.state.last_text_hash) or self.state.force_redraw

            if not needs_redraw:
                return

            self.state.last_text_hash = current_hash
            self.state.force_redraw = False

        stdscr.erase()
        height, width = stdscr.getmaxyx()

        show_history_panel = width >= 60
        left_panel_width = 24 if show_history_panel else 0

        self.init_colors()

        with contextlib.suppress(curses.error):
            stdscr.bkgd(" ", curses.color_pair(1))

        theme_map = {
            "dark": "🌙 Sombre", "light": "☀️ Clair", "ocean": "🌊 Océan",
            "matrix": "📟 Matrix", "pure_black": "🖤 Noir Pur", "pure_white": "🤍 Blanc Pur"
        }
        theme_name = theme_map.get(self.state.current_theme, self.state.current_theme.title())
        lang_info = self.state.languages[self.state.current_language]
        model_info = self.state.available_models[self.state.current_model]
        if self.state.history_mode:
            instructions = "🗂️ Historique | ↑↓: Parcourir | Entrée: Ouvrir | Esc/h: Fermer"
        elif self.state.input_mode:
            instructions = "✏️ Mode saisie | Tapez votre sujet puis Entrée | Esc: Annuler"
        elif self.state.is_generating and self.state.selection_mode:
            instructions = f"🔄📝 Génération + Sélection | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 'p': {self.state.current_provider or 'auto'} | 's': Arrêter"
        elif self.state.is_generating:
            instructions = f"🔄 En génération | ↑↓←→: Naviguer | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 'p': {self.state.current_provider or 'auto'} | 's': Arrêter"
        elif self.state.selection_mode:
            instructions = f"📝 Mode sélection | ↑↓←→: Étendre | Entrée: Générer | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 'p': {self.state.current_provider or 'auto'} | Esc: Annuler"
        else:
            instructions = f"↑↓←→: Naviguer | Entrée: Générer | Space: Sélection | 'r': Régénérer | 'i': Saisie | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 'p': {self.state.current_provider or 'auto'} | 'h': Hist. | 'b': Retour | 'q': Quitter"

        # Afficher le panneau d'infos (instructions) TOUT EN HAUT
        instructions_line = 0
        # Instructions sur UNE ligne fixe, centrées plein écran (pas de wrap pour éviter les sauts de layout)
        display_instr = _truncate_display(instructions, max_columns=max(0, width - 2))
        x_instr = max(0, (width - _display_width(display_instr)) // 2)
        try:
            stdscr.addstr(instructions_line, x_instr, display_instr, curses.color_pair(4))
        except curses.error:
            stdscr.addstr(instructions_line, x_instr, display_instr, curses.A_DIM)
        # Hauteur d'instructions fixe à 1 pour stabilité du layout
        self.state.instructions_lines_count = 1

        # Dessiner ensuite l'en-tête ASCII art sous les instructions
        header_start_y = instructions_line + self.state.instructions_lines_count
        header_height = self._draw_header(stdscr, width, header_start_y)
        # Mémoriser la hauteur du header pour calcul de zone contenu
        self._last_header_height = header_height

        input_line = header_start_y + header_height
        if self.state.input_mode:
            prompt_text = f"{self.state.input_prompt} {self.state.input_text}"
            # Pas d'indicateur clignotant pendant la génération, pour éviter l'illusion de sélection
            cursor_indicator = " " if self.state.is_generating else ("█" if len(self.state.input_text) % 2 == 0 else " ")
            display_text = f"{prompt_text}{cursor_indicator}"
            # Centrage sur l'écran pour l'invite, wcwidth-safe
            disp = _truncate_display(display_text, max_columns=max(0, width - 2))
            x_input = max(0, (width - _display_width(disp)) // 2)
            try:
                stdscr.addstr(input_line, x_input, disp, curses.color_pair(5) | curses.A_BOLD)
            except curses.error:
                with contextlib.suppress(curses.error):
                    stdscr.addstr(input_line, x_input, disp, curses.A_BOLD)
            input_line += 2

        separator_line = input_line
        # Séparateur horizontal retiré pour un design plus épuré
        try:
            stdscr.addstr(separator_line, 0, " " * width, curses.color_pair(6))
        except curses.error:
            stdscr.addstr(separator_line, 0, " " * width)

        if show_history_panel:
            self._draw_history_panel(stdscr, height, separator_line, left_panel_width)

        if current_text_copy:
            self._draw_content(stdscr, height, width, separator_line, left_panel_width, current_text_copy)
        else:
            msg = "Génération de la fiche en cours..."
            disp_msg = _truncate_display(msg, max_columns=max(0, width - 2))
            x_msg = max(0, (width - _display_width(disp_msg)) // 2)
            with contextlib.suppress(curses.error):
                stdscr.addstr(height // 2, x_msg, disp_msg)

        if self.state.words:
            if self.state.selection_mode and self.state.selection_start != -1 and self.state.selection_end != -1:
                selected_text = self.get_selected_text()
                if len(selected_text) > 25:
                    selected_text = selected_text[:22] + "..."
                status = f'📝 Sélection: "{selected_text}" ({abs(self.state.selection_end - self.state.selection_start) + 1} mots)'
            else:
                current_word = self.state.words[self.state.cursor_word_index][0] if self.state.cursor_word_index < len(self.state.words) else ""
                status = f"📍 Mot {self.state.cursor_word_index + 1}/{len(self.state.words)}: {current_word}"

            theme_indicator = "🌙" if self.state.current_theme == "dark" else "☀️"
            lang_indicator = self.state.languages[self.state.current_language]["flag"]
            model_indicator = f"{self.state.available_models[self.state.current_model]['symbol']}{self.state.available_models[self.state.current_model]['name']}"
            provider_indicator = f"🏷️ {self.state.current_provider}" if self.state.current_provider else "🏷️ auto"

            if current_text_copy:
                total_lines = len(self.state.wrapped_lines)
                _, available_height = self.calculate_content_area(height)

                if total_lines > available_height:
                    visible_start = self.state.scroll_offset + 1
                    visible_end = min(self.state.scroll_offset + available_height, total_lines)
                    scroll_info = f"({visible_start}-{visible_end}/{total_lines})"
                    status = f"{status} | {scroll_info}"

            status_parts = [status, theme_indicator, lang_indicator, model_indicator, provider_indicator]
            status = " | ".join(status_parts)

            # Centrer la barre de statut sur la même zone de texte que le contenu
            # Centrer le statut sur toute la largeur écran (wcwidth-safe)
            status = _truncate_display(status, max_columns=max(0, width - 2))
            status_pos = max(0, (width - _display_width(status)) // 2)
            try:
                color = curses.color_pair(7) if self.state.selection_mode else curses.color_pair(5)
                stdscr.addstr(height - 1, status_pos, status, color | curses.A_BOLD)
            except curses.error:
                attr = curses.A_REVERSE if self.state.selection_mode else curses.A_BOLD
                stdscr.addstr(height - 1, status_pos, status, attr)

        if width > 40 and current_text_copy:
            # Recalcule des marges (sans variables inutilisées)
            _left_margin, _text_width = self.calculate_text_margins(width, left_panel_width)
            # Barres verticales supprimées pour un design plus épuré

        stdscr.refresh()

    def handle_input(self, stdscr, key):
        """Gère les entrées clavier."""
        height, width = stdscr.getmaxyx()

        if self.state.history_mode:
            if key in (27, ord("h")):
                self.state.history_mode = False
                self.state.force_redraw = True
                return False
            if key == curses.KEY_UP:
                if self.state.history_cursor > 0:
                    self.state.history_cursor -= 1
                    self.state.force_redraw = True
                return False
            if key == curses.KEY_DOWN:
                if self.state.history_cursor < len(self.state.full_history) - 1:
                    self.state.history_cursor += 1
                    self.state.force_redraw = True
                return False
            if key in (ord("\n"), ord("\r")):
                if 0 <= self.state.history_cursor < len(self.state.full_history):
                    selected_subject = self.state.full_history[self.state.history_cursor]
                    if self.state.is_generating:
                        self.state.stop_streaming = True
                        if self.state.streaming_thread:
                            self.state.streaming_thread.join(timeout=1)
                        self.state.is_generating = False
                    with self.state.text_lock:
                        self.state.subject_texts[self.state.current_subject] = self.state.current_text
                    self.state.history_mode = False
                    self.state.history_cursor = 0
                    if selected_subject in self.state.subject_texts:
                        self.state.current_subject = selected_subject
                        with self.state.text_lock:
                            self.state.current_text = self.state.subject_texts.get(selected_subject, "") or f"🔄 Génération de '{selected_subject}' en cours..."
                            self.state.force_redraw = True
                        if not self.state.current_text.startswith("🔄 Génération") and self.state.current_text:
                            self.state.is_generating = False
                        else:
                            self.load_subject_streaming(selected_subject, stdscr)
                    else:
                        self.load_subject_streaming(selected_subject, stdscr)
                    return False
            else:
                return False

        if self.state.input_mode:
            result = self.handle_input_char(key)
            if result:
                self.state.input_mode = False
                self.state.history.append(self.state.current_subject)
                self.load_subject_streaming(result, stdscr)
            return False

        if key == ord("q"):
            if self.state.is_generating:
                self.state.stop_streaming = True
                if self.state.streaming_thread:
                    self.state.streaming_thread.join(timeout=1)
            return True
        if key == ord("s") and self.state.is_generating:
            self.state.stop_streaming = True
            if self.state.streaming_thread:
                self.state.streaming_thread.join(timeout=1)
            self.state.is_generating = False
            self.state.force_redraw = True
        elif key == ord("r") and not self.state.is_generating:
            self.load_subject_streaming(self.state.current_subject, stdscr)
        elif key == ord("h"):
            self.state.history_mode = not self.state.history_mode
            self.state.force_redraw = True
        elif key == ord("t"):
            self.toggle_theme()
        elif key == ord("l"):
            self.cycle_language()
        elif key == ord("m"):
            self.cycle_model()
        elif key == ord("p"):
            self.cycle_provider()
        elif key == ord("i") and not self.state.is_generating:
            self.start_input_mode()
        elif key == ord("b") and self.state.history:
            prev_subject = self.state.history.pop()
            self.load_subject_streaming(prev_subject, stdscr)
        elif key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, bstate = curses.getmouse()
            except curses.error:
                bstate = 0
            # Molette bas/haut
            if bstate & getattr(curses, "BUTTON5_PRESSED", 0):
                self.scroll_by_lines(SCROLL_WHEEL_LINES, height)
                self.state.force_redraw = True
                return False
            if bstate & getattr(curses, "BUTTON4_PRESSED", 0):
                self.scroll_by_lines(-SCROLL_WHEEL_LINES, height)
                self.state.force_redraw = True
                return False
            if bstate & curses.BUTTON1_PRESSED:
                word_idx = self.word_index_at_screen(mx, my, width, height)
                if word_idx != -1:
                    self.state.cursor_word_index = word_idx
                    if self.state.selection_mode:
                        self.update_selection_end()
                    self.adjust_scroll(height)
                    self.state.force_redraw = True
            return False
        elif key == ord(" "):
            # Autoriser la sélection pendant le streaming après le placeholder initial
            if self.state.is_generating and self.state.current_text.startswith(GENERATING_MSG_PREFIX):
                return False
            if not self.state.selection_mode:
                self.start_selection()
            else:
                self.clear_selection()
        elif key == 27:
            if self.state.selection_mode:
                self.clear_selection()
        elif key == curses.KEY_UP:
            new_index = self.find_word_above()
            if new_index != -1:
                self.state.cursor_word_index = new_index
                if self.state.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.state.force_redraw = True
        elif key == curses.KEY_DOWN:
            new_index = self.find_word_below()
            if new_index != -1:
                self.state.cursor_word_index = new_index
                if self.state.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.state.force_redraw = True
        elif key == curses.KEY_LEFT:
            new_index = self.find_word_left()
            if new_index != -1:
                self.state.cursor_word_index = new_index
                if self.state.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.state.force_redraw = True
        elif key == curses.KEY_RIGHT:
            new_index = self.find_word_right()
            if new_index != -1:
                self.state.cursor_word_index = new_index
                if self.state.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.state.force_redraw = True
        elif key == curses.KEY_NPAGE:
            self.scroll_page_down(height)
            self.state.force_redraw = True
        elif key == curses.KEY_PPAGE:
            self.scroll_page_up(height)
            self.state.force_redraw = True
        elif key == curses.KEY_HOME or key == ord("g"):
            self.scroll_to_top()
            self.state.force_redraw = True
        elif key == curses.KEY_END or key == ord("G"):
            self.scroll_to_bottom(height)
            self.state.force_redraw = True
        elif key in (ord("\n"), ord("\r")):
            # Pendant le streaming: autoriser Entrée si on n'est plus sur le placeholder initial
            if self.state.is_generating and self.state.current_text.startswith(GENERATING_MSG_PREFIX):
                return False
            if self.state.words:
                selected_text = self.get_selected_text()
                if selected_text:
                    self.state.history.append(self.state.current_subject)
                    if self.state.selection_mode:
                        self.clear_selection()
                    self.load_subject_streaming(selected_text, stdscr)

        return False

    def find_word_above(self) -> int:
        if not self.state.words or self.state.cursor_word_index >= len(self.state.words):
            return -1

        _, current_line, current_col = self.state.words[self.state.cursor_word_index]

        best_index = -1
        best_distance = float("inf")

        for i, (_, line, col) in enumerate(self.state.words):
            if line < current_line:
                distance = abs(col - current_col)
                total_distance = (current_line - line) * 1000 + distance
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_index = i
        return best_index

    def find_word_below(self) -> int:
        if not self.state.words or self.state.cursor_word_index >= len(self.state.words):
            return -1

        _, current_line, current_col = self.state.words[self.state.cursor_word_index]

        best_index = -1
        best_distance = float("inf")

        for i, (_, line, col) in enumerate(self.state.words):
            if line > current_line:
                distance = abs(col - current_col)
                total_distance = (line - current_line) * 1000 + distance
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_index = i
        return best_index

    def find_word_left(self) -> int:
        if not self.state.words or self.state.cursor_word_index >= len(self.state.words):
            return -1

        _, current_line, current_col = self.state.words[self.state.cursor_word_index]

        best_index = -1
        for i, (_, line, col) in enumerate(self.state.words):
            if line == current_line and col < current_col and (
                best_index == -1 or col > self.state.words[best_index][2]
            ):
                best_index = i

        if best_index == -1:
            for i, (_, line, col) in enumerate(self.state.words):
                if line == current_line - 1 and (
                    best_index == -1 or col > self.state.words[best_index][2]
                ):
                    best_index = i
        return best_index

    def find_word_right(self) -> int:
        if not self.state.words or self.state.cursor_word_index >= len(self.state.words):
            return -1

        _, current_line, current_col = self.state.words[self.state.cursor_word_index]

        best_index = -1
        for i, (_, line, col) in enumerate(self.state.words):
            if line == current_line and col > current_col and (
                best_index == -1 or col < self.state.words[best_index][2]
            ):
                best_index = i

        if best_index == -1:
            for i, (_, line, col) in enumerate(self.state.words):
                if line == current_line + 1 and (
                    best_index == -1 or col < self.state.words[best_index][2]
                ):
                    best_index = i
        return best_index

    def word_index_at_screen(self, x: int, y: int, term_width: int, term_height: int) -> int:
        """Retourne l'index du mot situé aux coordonnées écran (x,y) ou -1."""
        if not self.state.words or not self.state.wrapped_lines:
            return -1

        # --- Logique de calcul de position (miroir de _draw_content) ---
        # Centrage global: basé sur la largeur totale (on estime le panneau gauche s'il existe)
        # On reproduit la règle used dans draw_screen: left_panel si width>=60
        simulated_left_panel = 24 if term_width >= 60 else 0
        content_left_margin, _text_width = self.calculate_text_margins(term_width, simulated_left_panel)

        _, display_height = self.calculate_content_area(term_height)
        content_start_y = self.calculate_content_area(term_height)[0]

        scroll_start_line = self.state.scroll_offset
        # S'assurer d'utiliser les wrapped_lines déjà calculées
        visible_lines = self.state.wrapped_lines[scroll_start_line : scroll_start_line + display_height]
        vertical_offset = max(0, (display_height - len(visible_lines)) // 2)
        # --- Fin de la logique miroir ---

        # Déterminer la ligne de texte cliquée
        clicked_text_line_index = -1
        for i in range(len(visible_lines)):
            screen_y = i + content_start_y + vertical_offset
            if y == screen_y:
                clicked_text_line_index = scroll_start_line + i
                break

        if clicked_text_line_index == -1:
            return -1

        # Déterminer la colonne de texte cliquée
        clicked_text_col = x - content_left_margin
        if clicked_text_col < 0:
            return -1

        # Trouver le mot qui correspond aux coordonnées
        for idx, (word_text, word_line_idx, word_col_start) in enumerate(self.state.words):
            if word_line_idx == clicked_text_line_index and (
                word_col_start <= clicked_text_col < word_col_start + len(word_text)
            ):
                return idx

        return -1

        # --- Fin navigation souris ---

    def calculate_content_area(self, height):
        # Calcule la zone de contenu en tenant compte des hauteurs dynamiques (instructions + header)
        header_height = getattr(self, "_last_header_height", HEADER_HEIGHT)
        instructions_height = max(INSTRUCTIONS_HEIGHT, getattr(self.state, "instructions_lines_count", 1))
        input_height = 2 if self.state.input_mode else 0
        separator_height = SEPARATOR_HEIGHT
        status_height = STATUS_HEIGHT

        ui_total = header_height + instructions_height + input_height + separator_height + status_height
        content_start = ui_total - status_height
        content_height = height - ui_total

        return content_start, max(1, content_height)

    def adjust_scroll(self, height):
        if not self.state.words:
            return

        word_line = self.state.words[self.state.cursor_word_index][1]
        _, display_height = self.calculate_content_area(height)

        if word_line < self.state.scroll_offset:
            self.state.scroll_offset = word_line
        elif word_line >= self.state.scroll_offset + display_height:
            self.state.scroll_offset = word_line - display_height + 1

    def clamp_scroll(self, height):
        """Borne l'offset de défilement à [0, max]."""
        total_lines = len(self.state.wrapped_lines) if self.state.wrapped_lines else 0
        _, display_height = self.calculate_content_area(height)
        max_offset = max(0, total_lines - display_height)
        if self.state.scroll_offset < 0:
            self.state.scroll_offset = 0
        elif self.state.scroll_offset > max_offset:
            self.state.scroll_offset = max_offset

    def scroll_by_lines(self, delta: int, height: int):
        if not self.state.wrapped_lines:
            return
        self.state.scroll_offset += delta
        self.clamp_scroll(height)

    def scroll_page_down(self, height: int):
        _, display_height = self.calculate_content_area(height)
        self.scroll_by_lines(display_height - 1, height)

    def scroll_page_up(self, height: int):
        _, display_height = self.calculate_content_area(height)
        self.scroll_by_lines(-(display_height - 1), height)

    def scroll_to_top(self):
        self.state.scroll_offset = 0

    def scroll_to_bottom(self, height: int):
        total_lines = len(self.state.wrapped_lines) if self.state.wrapped_lines else 0
        _, display_height = self.calculate_content_area(height)
        self.state.scroll_offset = max(0, total_lines - display_height)

    def load_subject_streaming(self, subject: str, stdscr):
        if self.state.is_generating:
            self.state.stop_streaming = True
            if self.state.streaming_thread and self.state.streaming_thread.is_alive():
                self.state.streaming_thread.join(timeout=0.5)

        with self.state.text_lock:
            # Ne pas utiliser le placeholder de génération comme contexte précédent
            if self.state.current_text.startswith(GENERATING_MSG_PREFIX):
                previous_text = ""
            else:
                previous_text = self.state.current_text
        self.state.subject_texts[self.state.current_subject] = previous_text

        if subject not in self.state.full_history:
            self.state.full_history.insert(0, subject)
        self.state.history_cursor = 0
        self.state.current_subject = subject
        with self.state.text_lock:
            self.state.current_text = f"{GENERATING_MSG_PREFIX}{subject}' en cours..."
            self.state.force_redraw = True
        self.state.cursor_word_index = 0
        self.state.scroll_offset = 0
        self.state.is_generating = True
        self.state.stop_streaming = False

        self.draw_screen(stdscr)
        stdscr.refresh()

        self.state.streaming_thread = threading.Thread(
            target=self.generate_fiche_streaming_thread,
            args=(subject, previous_text),
            daemon=True
        )
        self.state.streaming_thread.start()

    def run(self, stdscr):
        """Boucle principale de l'interface."""
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.timeout(100)
        # Activation du support souris
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            curses.mouseinterval(0)
        except curses.error:
            # Souris non supportée: on continue sans
            pass

        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(7, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
            stdscr.bkgd(" ", curses.color_pair(1))
        except curses.error:
            # En cas d'erreur, continuer sans couleurs pour garder l'UI fonctionnelle
            pass

        self.load_subject_streaming(self.state.current_subject, stdscr)

        running = True
        while running:
            self.draw_screen(stdscr)

            key = stdscr.getch()
            if key != -1:
                should_quit = self.handle_input(stdscr, key)
                if should_quit:
                    running = False

        if self.state.is_generating:
            self.state.stop_streaming = True
            if self.state.streaming_thread:
                self.state.streaming_thread.join(timeout=2)


def run_simple_tui(initial_subject: str = "Lyon"):
    """Lance l'interface TUI simple."""
    tui = SimpleTUI(initial_subject)
    curses.wrapper(tui.run)
