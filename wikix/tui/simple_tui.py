"""Interface TUI simple basée sur curses pour navigation dans le texte."""
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
        self.available_models: dict = {
            "gpt-4o-mini": {"name": "4o-mini", "symbol": "🔹", "features": "standard"},
            "gpt-4o": {"name": "4o", "symbol": "🔷", "features": "standard"},
            "gemini-2.5-flash": {"name": "gemini-flash", "symbol": "⚡️", "features": "standard"},
            "gemini-2.5-flash-light": {"name": "gemini-flash-light", "symbol": "💡", "features": "standard"},
            "gemini-2.5-pro": {"name": "gemini-pro", "symbol": "✨", "features": "standard"},
            "llama3.1-8b": {"name": "llama-8b", "symbol": "🦙", "features": "ultra-fast"},
            "llama3.1-70b": {"name": "llama-70b", "symbol": "🚀", "features": "ultra-fast"},
            "qwen-3-32b": {"name": "qwen-32b", "symbol": "💻", "features": "coding"},
            "o3-mini": {"name": "o3-mini", "symbol": "🟦", "features": "no_temp"},
            "o4-mini": {"name": "o4-mini", "symbol": "🟪", "features": "future"},
            "openai/gpt-oss-120b": {"name": "gpt-oss-120b", "symbol": "🧠", "features": "cerebras"}
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


class SimpleTUI:
    def __init__(self, initial_subject: str = "Wikipédia"):
        self.state = TUIState(initial_subject)

    def _draw_header(self, stdscr, width):
        """Dessine l'en-tête avec logo et sujet encadré."""
        title_line1 = "WIKIX"
        subject = self.state.current_subject.upper()

        # Ligne du logo
        try:
            stdscr.addstr(0, (width - len(title_line1)) // 2, title_line1, curses.color_pair(1) | curses.A_BOLD)
        except Exception:
            stdscr.addstr(0, (width - len(title_line1)) // 2, title_line1, curses.A_BOLD)

        # Construction de l'encadrement du sujet
        border_top = "┌" + "─" * (len(subject) + 2) + "┐"
        border_mid = f"│ {subject} │"
        border_bot = "└" + "─" * (len(subject) + 2) + "┘"

        lines = [border_top, border_mid, border_bot]
        for idx, line in enumerate(lines, start=1):
            x = (width - len(line)) // 2
            try:
                stdscr.addstr(idx, x, line, curses.color_pair(1) | curses.A_BOLD)
            except Exception:
                stdscr.addstr(idx, x, line, curses.A_BOLD)

        return 4

    def _draw_history_panel(self, stdscr, height, separator_line, left_panel_width):
        """Dessine le panneau de l'historique."""
        panel_height = height - (separator_line + 1) - 1
        panel_start_line = separator_line + 1
        # Suppression de la barre verticale pour un design plus épuré
        # (anciennes barres verticales supprimées)
        title = "Historique"
        try:
            stdscr.addstr(panel_start_line, 1, title[:left_panel_width-2], curses.color_pair(3) | curses.A_BOLD)
        except Exception:
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
            except Exception:
                attr = curses.A_REVERSE if is_selected else (curses.A_BOLD if is_current else curses.A_NORMAL)
                stdscr.addstr(line_no, 1, display_text, attr)

    def _draw_content(self, stdscr, height, right_width, separator_line, left_panel_width, current_text_copy):
        """Dessine le contenu principal de la fiche en itérant sur les mots."""
        # Les calculs se basent sur la 'right_width' (zone de contenu)
        content_left_margin, text_width = self.calculate_text_margins(right_width)

        words, wrapped_lines = self.extract_words(current_text_copy, text_width)
        self.state.words = words
        self.state.wrapped_lines = wrapped_lines

        _, display_height = self.calculate_content_area(height)
        content_start_y = separator_line + 1
        scroll_start_line = self.state.scroll_offset

        visible_lines = wrapped_lines[scroll_start_line : scroll_start_line + display_height]
        vertical_offset = max(0, (display_height - len(visible_lines)) // 2)

        # Dessiner le texte de fond
        for i, line_text in enumerate(visible_lines):
            screen_y = i + content_start_y + vertical_offset
            screen_x = left_panel_width + content_left_margin
            stdscr.addstr(screen_y, screen_x, line_text, curses.color_pair(1))

        # Déterminer la plage de sélection
        selection_range = set()
        if self.state.selection_mode and self.state.selection_start != -1 and self.state.selection_end != -1:
            start_idx = min(self.state.selection_start, self.state.selection_end)
            end_idx = max(self.state.selection_start, self.state.selection_end)
            selection_range = set(range(start_idx, end_idx + 1))

        # Redessiner les mots avec un style spécial
        for i, (text, line_idx, col_start) in enumerate(self.state.words):
            if scroll_start_line <= line_idx < scroll_start_line + display_height:
                style = None
                if i == self.state.cursor_word_index:
                    style = curses.color_pair(2)
                elif i in selection_range:
                    style = curses.color_pair(7) | curses.A_BOLD

                if style:
                    screen_y = (line_idx - scroll_start_line) + content_start_y + vertical_offset
                    screen_x = left_panel_width + content_left_margin + col_start
                    with contextlib.suppress(curses.error):
                        stdscr.addstr(screen_y, screen_x, text, style)

    def calculate_text_margins(self, width: int) -> tuple[int, int]:
        """Calcule des marges équilibrées et une largeur de texte maximale raisonnable."""
        # Largeur de texte cible (80 colonnes si possible)
        max_text_width = 80
        text_width = min(max_text_width, max(20, width - 8))
        # Si le terminal est très étroit, on utilise tout l'espace disponible moins 4 col.
        if text_width > width - 8:
            text_width = width - 8
        # Marges gauche/droite identiques
        left_margin = max(4, (width - text_width) // 2)
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
            for match in re.finditer(r"\b[A-Za-zÀ-ÿ]+\b", line):
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

    def cycle_model(self):
        """Cycle entre les modèles disponibles."""
        model_keys = list(self.state.available_models.keys())
        current_index = model_keys.index(self.state.current_model)
        next_index = (current_index + 1) % len(model_keys)
        self.state.current_model = model_keys[next_index]
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

        except Exception:
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
                stream_gen = generate_fiche_with_context_stream(subject, previous_text, context_template, model=self.state.current_model)
            else:
                stream_gen = generate_fiche_stream(subject, general_template, model=self.state.current_model)

            full_content = ""
            for chunk in stream_gen:
                if self.state.stop_streaming:
                    break
                full_content += chunk
                with self.state.text_lock:
                    self.state.current_text = full_content
                    self.state.force_redraw = True
                time.sleep(0.08)

            if not self.state.stop_streaming:
                output_path = GENERATED_DIR / f"{subject.lower().replace(' ', '_')}.md"
                output_path.write_text(full_content, encoding="utf-8")

            self.state.subject_texts[self.state.current_subject] = full_content
            if self.state.current_subject in self.state.full_history:
                self.state.full_history.remove(self.state.current_subject)
            self.state.full_history.insert(0, self.state.current_subject)
            self.state.is_generating = False
        except Exception as e:
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
        right_width = width - left_panel_width

        self.init_colors()

        with contextlib.suppress(Exception):
            stdscr.bkgd(" ", curses.color_pair(1))

        header_height = self._draw_header(stdscr, width)


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
            instructions = f"🔄📝 Génération + Sélection | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 's': Arrêter"
        elif self.state.is_generating:
            instructions = f"🔄 En génération | ↑↓←→: Naviguer | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 's': Arrêter"
        elif self.state.selection_mode:
            instructions = f"📝 Mode sélection | ↑↓←→: Étendre | Entrée: Générer | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | Esc: Annuler"
        else:
            instructions = f"↑↓←→: Naviguer | Entrée: Générer | Space: Sélection | 'r': Régénérer | 'i': Saisie | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 'h': Hist. | 'b': Retour | 'q': Quitter"

        instructions_line = header_height
        display_instr = instructions[:max(0, width - 4)]
        x_instr = max(0, (width - len(display_instr)) // 2)
        try:
            stdscr.addstr(instructions_line, x_instr, display_instr, curses.color_pair(4))
        except Exception:
            stdscr.addstr(instructions_line, x_instr, display_instr, curses.A_DIM)

        input_line = instructions_line + 1
        if self.state.input_mode:
            prompt_text = f"{self.state.input_prompt} {self.state.input_text}"
            cursor_indicator = "█" if len(self.state.input_text) % 2 == 0 else " "
            display_text = f"{prompt_text}{cursor_indicator}"
            try:
                stdscr.addstr(input_line, (width - len(display_text)) // 2, display_text, curses.color_pair(5) | curses.A_BOLD)
            except Exception:
                stdscr.addstr(input_line, (width - len(display_text)) // 2, display_text, curses.A_BOLD)
            input_line += 2

        separator_line = input_line
        # Séparateur horizontal retiré pour un design plus épuré
        try:
            stdscr.addstr(separator_line, 0, " " * width, curses.color_pair(6))
        except Exception:
            stdscr.addstr(separator_line, 0, " " * width)

        if show_history_panel:
            self._draw_history_panel(stdscr, height, separator_line, left_panel_width)

        if current_text_copy:
            self._draw_content(stdscr, height, right_width, separator_line, left_panel_width, current_text_copy)
        else:
            msg = "Génération de la fiche en cours..."
            stdscr.addstr(height // 2, (width - len(msg)) // 2, msg)

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

            if current_text_copy:
                total_lines = len(self.state.wrapped_lines)
                _, available_height = self.calculate_content_area(height)

                if total_lines > available_height:
                    visible_start = self.state.scroll_offset + 1
                    visible_end = min(self.state.scroll_offset + available_height, total_lines)
                    scroll_info = f"({visible_start}-{visible_end}/{total_lines})"
                    status = f"{status} | {scroll_info}"

            status_parts = [status, theme_indicator, lang_indicator, model_indicator]
            status = " | ".join(status_parts)

            status_pos = max(2, (width - len(status)) // 2)
            try:
                color = curses.color_pair(7) if self.state.selection_mode else curses.color_pair(5)
                stdscr.addstr(height - 1, status_pos, status[:width-4], color | curses.A_BOLD)
            except Exception:
                attr = curses.A_REVERSE if self.state.selection_mode else curses.A_BOLD
                stdscr.addstr(height - 1, status_pos, status[:width-4], attr)

        if width > 40 and current_text_copy:
            left_margin, text_width = self.calculate_text_margins(right_width)
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
        elif key == ord("i") and not self.state.is_generating:
            self.start_input_mode()
        elif key == ord("b") and self.state.history:
            prev_subject = self.state.history.pop()
            self.load_subject_streaming(prev_subject, stdscr)
        elif key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, bstate = curses.getmouse()
            except Exception:
                bstate = 0
            if bstate & curses.BUTTON1_PRESSED:
                show_history_panel = width >= 60
                left_panel_width = 24 if show_history_panel else 0
                word_idx = self.word_index_at_screen(mx, my, width, height, left_panel_width)
                if word_idx != -1:
                    self.state.cursor_word_index = word_idx
                    if self.state.selection_mode:
                        self.update_selection_end()
                    self.adjust_scroll(height)
                    self.state.force_redraw = True
            return False
        elif key == ord(" "):
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
        elif key in (ord("\n"), ord("\r")):
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

    def word_index_at_screen(self, x: int, y: int, term_width: int, term_height: int, left_panel_width: int) -> int:
        """Retourne l'index du mot situé aux coordonnées écran (x,y) ou -1."""
        if not self.state.words or not self.state.wrapped_lines:
            return -1

        # --- Logique de calcul de position (miroir de _draw_content) ---
        right_width = term_width - left_panel_width
        content_left_margin, text_width = self.calculate_text_margins(right_width)

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
        clicked_text_col = x - left_panel_width - content_left_margin
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
        header_height = 4
        instructions_height = 1
        input_height = 2 if self.state.input_mode else 0
        separator_height = 1
        status_height = 1

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

    def load_subject_streaming(self, subject: str, stdscr):
        if self.state.is_generating:
            self.state.stop_streaming = True
            if self.state.streaming_thread and self.state.streaming_thread.is_alive():
                self.state.streaming_thread.join(timeout=0.5)

        with self.state.text_lock:
            previous_text = self.state.current_text if not self.state.current_text.startswith("Génération en cours") else ""
        self.state.subject_texts[self.state.current_subject] = previous_text

        if subject not in self.state.full_history:
            self.state.full_history.insert(0, subject)
        self.state.history_cursor = 0
        self.state.current_subject = subject
        with self.state.text_lock:
            self.state.current_text = f"🔄 Génération de '{subject}' en cours..."
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
        except Exception:
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
        except Exception:
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
