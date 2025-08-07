"""Interface TUI simple basée sur curses pour navigation dans le texte."""
from __future__ import annotations

import curses
import re
import textwrap
import threading
import time
from pathlib import Path
from typing import List, Tuple

from .llm import generate_fiche, generate_fiche_with_context, generate_fiche_stream, generate_fiche_with_context_stream

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
TEMPLATE_GENERAL = (PROMPT_DIR / "fiche_generale.txt").read_text(encoding="utf-8")
TEMPLATE_CONTEXT = (PROMPT_DIR / "fiche_contexte.txt").read_text(encoding="utf-8")
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


class SimpleTUI:
    def __init__(self, initial_subject: str = "Wikipédia"):
        self.current_subject = initial_subject
        self.current_text = ""
        self.words: List[Tuple[str, int, int]] = []  # (word, line, col)
        self.cursor_word_index = 0
        # Pile de navigation arrière (retour)
        self.history: List[str] = []
        # Historique complet pour l'affichage latéral (dernier sujet en premier)
        self.full_history: List[str] = [initial_subject]
        self.history_cursor: int = 0  # Curseur dans le panneau historique
        self.history_mode: bool = False  # Mode navigation dans l'historique
        self.history_scroll: int = 0  # Décalage de scroll pour l'historique
        # Cache des textes déjà générés (ou partiellement générés)
        self.subject_texts: dict[str, str] = {initial_subject: ""}
        
        self.scroll_offset = 0
        self.is_generating = False
        self.streaming_thread = None
        self.stop_streaming = False
        self.text_lock = threading.Lock()
        self.last_text_hash = 0  # Pour éviter les redraws inutiles
        self.force_redraw = False
        
        # Système de sélection de groupes de mots
        self.selection_mode = False  # Mode sélection activé/désactivé
        self.selection_start = -1    # Index du premier mot sélectionné
        self.selection_end = -1      # Index du dernier mot sélectionné
        
        # Système de langues
        self.current_language = "fr"  # "fr", "en", "es"
        self.languages = {
            "fr": {"name": "Français", "flag": "FR"},
            "en": {"name": "English", "flag": "EN"},
            "es": {"name": "Español", "flag": "ES"}
        }
        
        # Système de saisie de texte
        self.input_mode = False
        self.input_text = ""
        self.input_prompt = "Entrez un sujet :"
        

        
        # Système de sélection de modèle
        self.current_model = "gpt-4o-mini"  # Modèle par défaut
        self.available_models = {
            "gpt-4o-mini": {"name": "4o-mini", "symbol": "🔹", "features": "standard"},
            "gpt-4o": {"name": "4o", "symbol": "🔷", "features": "standard"},
            "o3-mini": {"name": "o3-mini", "symbol": "🟦", "features": "no_temp"},
            "o4-mini": {"name": "o4-mini", "symbol": "🟪", "features": "future"}  # Future model 😉
        }
        
        # Système de thèmes
        self.current_theme = "dark"  # "dark" ou "light"
        self.themes = {
            "dark": {
                "bg": curses.COLOR_BLACK,
                "text": curses.COLOR_WHITE,
                "title": curses.COLOR_CYAN,
                "instructions": curses.COLOR_YELLOW,
                "status": curses.COLOR_GREEN,
                "border": curses.COLOR_BLUE,
                "selection": curses.COLOR_MAGENTA,
                "cursor_bg": curses.COLOR_WHITE,
                "cursor_text": curses.COLOR_BLACK
            },
            "light": {
                "bg": curses.COLOR_WHITE,
                "text": curses.COLOR_BLACK,
                "title": curses.COLOR_BLUE,
                "instructions": curses.COLOR_RED,
                "status": curses.COLOR_GREEN,
                "border": curses.COLOR_BLUE,
                "selection": curses.COLOR_MAGENTA,
                "cursor_bg": curses.COLOR_BLACK,
                "cursor_text": curses.COLOR_WHITE
            }
        }
        
    def calculate_text_margins(self, width: int) -> tuple[int, int]:
        """Calcule les marges pour centrer le texte avec style."""
        # Largeur de texte idéale selon la taille d'écran
        if width < 60:
            text_width = width - 8  # Petits écrans : marges minimales
        elif width < 100:
            text_width = min(width - 16, 70)  # Écrans moyens : max 70 chars
        else:
            text_width = min(width - 24, 80)  # Grands écrans : max 80 chars
        
        left_margin = (width - text_width) // 2
        return left_margin, text_width
    
    def extract_words(self, text: str, width: int) -> List[Tuple[str, int, int]]:
        """Extrait les mots avec leurs positions après wrapping du texte."""
        words = []
        lines = text.split('\n')
        wrapped_lines = []
        
        # Calculer les marges pour le style
        left_margin, text_width = self.calculate_text_margins(width)
        
        # Wrapper le texte pour la largeur centrée
        for line in lines:
            if line.strip():
                wrapped = textwrap.fill(line, width=text_width)
                wrapped_lines.extend(wrapped.split('\n'))
            else:
                wrapped_lines.append('')
        
        # Extraire les mots avec positions (ajustées pour les marges)
        for line_num, line in enumerate(wrapped_lines):
            # Trouve tous les mots dans la ligne
            for match in re.finditer(r'\b[A-Za-zÀ-ÿ]+\b', line):
                word = match.group()
                # Inclure tous les mots, même ceux de 1-2 lettres pour la sélection
                # Position ajustée avec la marge gauche
                words.append((word, line_num, match.start() + left_margin))
        
        return words, wrapped_lines
    
    def get_selected_text(self) -> str:
        """Retourne le texte sélectionné (groupe de mots)."""
        if not self.selection_mode or self.selection_start == -1 or self.selection_end == -1:
            # Pas de sélection : retourner le mot actuel
            if self.words and self.cursor_word_index < len(self.words):
                return self.words[self.cursor_word_index][0]
            return ""
        
        # Assurer que start <= end
        start_idx = min(self.selection_start, self.selection_end)
        end_idx = max(self.selection_start, self.selection_end)
        
        # Extraire les mots sélectionnés
        selected_words = []
        for i in range(start_idx, end_idx + 1):
            if i < len(self.words):
                selected_words.append(self.words[i][0])
        
        return " ".join(selected_words)
    
    def clear_selection(self):
        """Efface la sélection actuelle."""
        self.selection_mode = False
        self.selection_start = -1
        self.selection_end = -1
        self.force_redraw = True
    
    def start_selection(self):
        """Démarre une nouvelle sélection au curseur actuel."""
        self.selection_mode = True
        self.selection_start = self.cursor_word_index
        self.selection_end = self.cursor_word_index
        self.force_redraw = True
    
    def update_selection_end(self):
        """Met à jour la fin de la sélection au curseur actuel."""
        if self.selection_mode:
            self.selection_end = self.cursor_word_index
            self.force_redraw = True
    
    def get_char_style(self, col: int, line: int) -> str:
        """Détermine le style d'affichage pour un caractère donné."""
        if not self.words:
            return "normal"
        
        # Vérifier si le caractère fait partie du mot sous le curseur
        if self.cursor_word_index < len(self.words):
            cursor_word, cursor_line, cursor_col = self.words[self.cursor_word_index]
            if (cursor_line == line and 
                cursor_col <= col < cursor_col + len(cursor_word)):
                return "cursor"
        
        # Vérifier si le caractère fait partie de la sélection
        if self.selection_mode and self.selection_start != -1 and self.selection_end != -1:
            start_idx = min(self.selection_start, self.selection_end)
            end_idx = max(self.selection_start, self.selection_end)
            
            for i in range(start_idx, end_idx + 1):
                if i < len(self.words):
                    word, word_line, word_col = self.words[i]
                    if (word_line == line and 
                        word_col <= col < word_col + len(word)):
                        return "selection"
        
        return "normal"
    
    def toggle_theme(self):
        """Bascule entre les thèmes clair et sombre."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.force_redraw = True
    
    def cycle_language(self):
        """Cycle entre les langues disponibles."""
        lang_keys = list(self.languages.keys())
        current_index = lang_keys.index(self.current_language)
        next_index = (current_index + 1) % len(lang_keys)
        self.current_language = lang_keys[next_index]
        self.force_redraw = True
    
    def cycle_model(self):
        """Cycle entre les modèles disponibles."""
        model_keys = list(self.available_models.keys())
        current_index = model_keys.index(self.current_model)
        next_index = (current_index + 1) % len(model_keys)
        self.current_model = model_keys[next_index]
        self.force_redraw = True
    
    def start_input_mode(self):
        """Démarre le mode de saisie de texte."""
        self.input_mode = True
        self.input_text = ""
        # Adapter le prompt selon la langue
        if self.current_language == "en":
            self.input_prompt = "Enter a subject:"
        elif self.current_language == "es":
            self.input_prompt = "Ingrese un tema:"
        else:
            self.input_prompt = "Entrez un sujet :"
        self.force_redraw = True
    
    def cancel_input_mode(self):
        """Annule le mode de saisie."""
        self.input_mode = False
        self.input_text = ""
        self.force_redraw = True
    
    def handle_input_char(self, key):
        """Gère la saisie de caractères en mode input."""
        if key == 27:  # Escape
            self.cancel_input_mode()
        elif key == ord('\n') or key == ord('\r'):  # Entrée
            if self.input_text.strip():
                return self.input_text.strip()
            else:
                self.cancel_input_mode()
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
            # Backspace
            if self.input_text:
                self.input_text = self.input_text[:-1]
                self.force_redraw = True
        elif 32 <= key <= 126:  # Caractères imprimables
            self.input_text += chr(key)
            self.force_redraw = True
        
        return None
    
    def init_colors(self):
        """Initialise les couleurs selon le thème actuel."""
        if not curses.has_colors():
            return
        
        try:
            curses.start_color()
            curses.use_default_colors()
            
            theme = self.themes[self.current_theme]
            
            # Paires de couleurs : (id, foreground, background)
            curses.init_pair(1, theme["text"], theme["bg"])        # Texte normal
            curses.init_pair(2, theme["cursor_text"], theme["cursor_bg"])  # Curseur
            curses.init_pair(3, theme["title"], theme["bg"])       # Titre
            curses.init_pair(4, theme["instructions"], theme["bg"]) # Instructions
            curses.init_pair(5, theme["status"], theme["bg"])      # Statut
            curses.init_pair(6, theme["border"], theme["bg"])      # Bordures
            curses.init_pair(7, theme["selection"], theme["bg"])   # Sélection
            
        except:
            # Fallback si les couleurs ne sont pas supportées
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(7, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    
    def get_templates_for_language(self):
        """Retourne les templates appropriés pour la langue actuelle."""
        if self.current_language == "en":
            general_path = BASE_DIR / "prompts" / "fiche_generale_en.txt"
            context_path = BASE_DIR / "prompts" / "fiche_contexte_en.txt"
        elif self.current_language == "es":
            general_path = BASE_DIR / "prompts" / "fiche_generale_es.txt"
            context_path = BASE_DIR / "prompts" / "fiche_contexte_es.txt"
        else:  # français par défaut
            general_path = BASE_DIR / "prompts" / "fiche_generale.txt"
            context_path = BASE_DIR / "prompts" / "fiche_contexte.txt"
        
        general_template = general_path.read_text(encoding="utf-8") if general_path.exists() else TEMPLATE_GENERAL
        context_template = context_path.read_text(encoding="utf-8") if context_path.exists() else TEMPLATE_CONTEXT
        
        return general_template, context_template
    

    
    def generate_fiche_streaming_thread(self, subject: str, previous_text: str):
        """Génère une fiche en streaming dans un thread séparé."""
        try:
            # Obtenir les templates pour la langue actuelle
            general_template, context_template = self.get_templates_for_language()
            
            # Détermine le type de génération avec le modèle sélectionné
            if previous_text and subject != self.current_subject:
                stream_gen = generate_fiche_with_context_stream(subject, previous_text, context_template, model=self.current_model)
            else:
                stream_gen = generate_fiche_stream(subject, general_template, model=self.current_model)
            
            # Accumule le contenu progressivement
            full_content = ""
            chunk_count = 0
            for chunk in stream_gen:
                if self.stop_streaming:
                    break
                    
                full_content += chunk
                chunk_count += 1
                
                # Mise à jour thread-safe du texte
                with self.text_lock:
                    self.current_text = full_content
                    self.force_redraw = True  # Marquer pour redraw
                
                # Vérification d'interruption plus fréquente (toutes les 3 chunks)
                if chunk_count % 3 == 0 and self.stop_streaming:
                    break
                
                # Pause plus longue pour réduire le scintillement
                time.sleep(0.08)  # 80ms pour être un peu plus réactif
            
            # Sauvegarde finale
            if not self.stop_streaming:
                filename = f"{subject.lower().replace(' ', '_')}.md"
                output_path = GENERATED_DIR / filename
                output_path.write_text(full_content, encoding="utf-8")
            
            # Mettre à jour le cache avec le contenu généré (même partiel)
            self.subject_texts[self.current_subject] = full_content
            
            # Mettre à jour l'ordre historique : sujet en tête car (re)généré
            if self.current_subject in self.full_history:
                self.full_history.remove(self.current_subject)
            self.full_history.insert(0, self.current_subject)
            
            self.is_generating = False
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération : {str(e)}"
            with self.text_lock:
                self.current_text = error_msg
            self.is_generating = False
    
    def draw_screen(self, stdscr):
        """Dessine l'écran principal."""
        # Vérifier si un redraw est nécessaire
        with self.text_lock:
            current_text_copy = self.current_text
            current_hash = hash(current_text_copy + str(self.cursor_word_index) + str(self.scroll_offset))
            needs_redraw = (current_hash != self.last_text_hash) or self.force_redraw
            
            if not needs_redraw:
                return
                
            self.last_text_hash = current_hash
            self.force_redraw = False
        
        # Effacer et redessiner seulement si nécessaire
        stdscr.erase()  # erase() au lieu de clear() pour moins de scintillement
        height, width = stdscr.getmaxyx()
        
        # Détermination du panneau historique
        show_history_panel = width >= 60  # Afficher seulement si écran suffisamment large
        left_panel_width = 24 if show_history_panel else 0  # Largeur fixe du panneau
        right_width = width - left_panel_width  # Largeur disponible pour le contenu principal
        
        # Initialiser les couleurs selon le thème
        self.init_colors()
        
        # Appliquer l'arrière-plan selon le thème
        try:
            stdscr.bkgd(' ', curses.color_pair(1))
        except:
            pass
        
        # Titre en grand sur toute la largeur
        title_line1 = "WIKIX"
        title_line2 = self.current_subject.upper()
        
        # Ligne 1 : WIKIX centré avec couleur
        try:
            stdscr.addstr(0, (width - len(title_line1)) // 2, title_line1, 
                         curses.color_pair(3) | curses.A_BOLD)
        except:
            stdscr.addstr(0, (width - len(title_line1)) // 2, title_line1, curses.A_BOLD)
        
        # Ligne 2 : Sujet en gros avec bordure colorée
        if len(title_line2) < width - 8:
            # Créer une bordure stylée autour du sujet
            border = "═" * (len(title_line2) + 4)
            try:
                stdscr.addstr(1, (width - len(border)) // 2, border, curses.color_pair(6))
                stdscr.addstr(2, (width - len(title_line2) - 4) // 2, f"  {title_line2}  ", 
                             curses.color_pair(7) | curses.A_BOLD)
                stdscr.addstr(3, (width - len(border)) // 2, border, curses.color_pair(6))
            except:
                stdscr.addstr(1, (width - len(border)) // 2, border, curses.A_DIM)
                stdscr.addstr(2, (width - len(title_line2) - 4) // 2, f"  {title_line2}  ", 
                             curses.A_BOLD | curses.A_REVERSE)
                stdscr.addstr(3, (width - len(border)) // 2, border, curses.A_DIM)
            header_height = 4
        else:
            # Si le titre est trop long, affichage simple
            try:
                stdscr.addstr(1, (width - len(title_line2)) // 2, title_line2, 
                             curses.color_pair(3) | curses.A_BOLD)
            except:
                stdscr.addstr(1, (width - len(title_line2)) // 2, title_line2, curses.A_BOLD)
            header_height = 2
        
        # Instructions sous le titre
        theme_name = "🌙 Sombre" if self.current_theme == "dark" else "☀️ Clair"
        lang_info = self.languages[self.current_language]
        model_info = self.available_models[self.current_model]
        if self.history_mode:
            instructions = "🗂️ Historique | ↑↓: Parcourir | Entrée: Ouvrir | Esc/h: Fermer"
        elif self.input_mode:
            instructions = f"✏️ Mode saisie | Tapez votre sujet puis Entrée | Esc: Annuler"
        elif self.is_generating and self.selection_mode:
            instructions = f"🔄📝 Génération + Sélection | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 's': Arrêter"
        elif self.is_generating:
            instructions = f"🔄 En génération | ↑↓←→: Naviguer | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 's': Arrêter"
        elif self.selection_mode:
            instructions = f"📝 Mode sélection | ↑↓←→: Étendre | Entrée: Générer | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | Esc: Annuler"
        else:
            instructions = f"↑↓←→: Naviguer | Entrée: Générer | Space: Sélection | 'r': Régénérer | 'i': Saisie | 't': {theme_name} | 'l': {lang_info['flag']} | 'm': {model_info['name']} | 'h': Hist. | 'b': Retour | 'q': Quitter"
        
        instructions_line = header_height
        display_instr = instructions if len(instructions) <= width - 4 else instructions[: max(0, width - 4)]
        x_instr = max(0, (width - len(display_instr)) // 2)
        try:
            stdscr.addstr(instructions_line, x_instr, display_instr, curses.color_pair(4))
        except:
            stdscr.addstr(instructions_line, x_instr, display_instr, curses.A_DIM)
        
        # Zone de saisie si en mode input
        input_line = instructions_line + 1
        if self.input_mode:
            # Afficher le prompt et la zone de saisie
            prompt_text = f"{self.input_prompt} {self.input_text}"
            cursor_indicator = "█" if len(self.input_text) % 2 == 0 else " "  # Curseur clignotant simple
            display_text = f"{prompt_text}{cursor_indicator}"
            
            # Centrer la zone de saisie
            try:
                stdscr.addstr(input_line, (width - len(display_text)) // 2, display_text, 
                             curses.color_pair(5) | curses.A_BOLD)
            except:
                stdscr.addstr(input_line, (width - len(display_text)) // 2, display_text, curses.A_BOLD)
            
            input_line += 2
        
        # Ligne de séparation élégante
        separator_line = input_line
        try:
            stdscr.addstr(separator_line, 0, "─" * width, curses.color_pair(6))
        except:
            stdscr.addstr(separator_line, 0, "─" * width, curses.A_DIM)
        
        # -----------------------------------------------------------------
        # Affichage du panneau historique à gauche
        # -----------------------------------------------------------------
        if show_history_panel:
            panel_height = height - (separator_line + 1) - 1  # Laisser la barre de statut
            panel_start_line = separator_line + 1
            # Bordure verticale droite du panneau
            for y in range(panel_start_line, panel_start_line + panel_height):
                try:
                    stdscr.addstr(y, left_panel_width - 1, "│", curses.color_pair(6))
                except:
                    stdscr.addstr(y, left_panel_width - 1, "│", curses.A_DIM)
            # Titre du panneau
            title = "Historique"
            try:
                stdscr.addstr(panel_start_line, 1, title[:left_panel_width-2], curses.color_pair(3) | curses.A_BOLD)
            except:
                stdscr.addstr(panel_start_line, 1, title[:left_panel_width-2], curses.A_BOLD)
            # Lignes de sujets
            visible_items = self.full_history
            max_visible = panel_height - 2  # Une ligne pour le titre, une pour le padding
            if len(visible_items) > max_visible:
                # Ajuster le scroll de l'historique
                if self.history_cursor < self.history_scroll:
                    self.history_scroll = self.history_cursor
                elif self.history_cursor >= self.history_scroll + max_visible:
                    self.history_scroll = self.history_cursor - max_visible + 1
                items_slice = visible_items[self.history_scroll:self.history_scroll + max_visible]
            else:
                self.history_scroll = 0
                items_slice = visible_items
            for idx, subj in enumerate(items_slice):
                line_no = panel_start_line + 1 + idx
                is_current = (subj == self.current_subject)
                is_selected = (self.history_mode and (self.history_scroll + idx) == self.history_cursor)
                display_text = subj[:left_panel_width-2].ljust(left_panel_width-2)
                try:
                    if is_selected:
                        stdscr.addstr(line_no, 1, display_text, curses.color_pair(2))
                    elif is_current:
                        stdscr.addstr(line_no, 1, display_text, curses.color_pair(7))
                    else:
                        stdscr.addstr(line_no, 1, display_text, curses.color_pair(1))
                except:
                    attr = curses.A_REVERSE if is_selected else (curses.A_BOLD if is_current else curses.A_NORMAL)
                    stdscr.addstr(line_no, 1, display_text, attr)
        
        # Affichage du texte avec mots surlignés
        if current_text_copy:  # Utilise la copie déjà récupérée
            # Extraire les mots avec la largeur complète
            words, wrapped_lines = self.extract_words(current_text_copy, right_width)
            self.words = words
            
            # Calculer les marges pour l'affichage
            left_margin, text_width = self.calculate_text_margins(right_width)
            
            # Calcul de la zone d'affichage avec méthode dynamique
            calculated_start, calculated_height = self.calculate_content_area(height)
            content_start = separator_line + 1  # Position réelle du début de contenu
            display_height = calculated_height  # Utiliser la hauteur calculée dynamiquement
            start_line = max(0, self.scroll_offset)
            end_line = min(len(wrapped_lines), start_line + display_height)
            
            # Affichage ligne par ligne avec marges
            content_end = height - 1  # La barre de statut prend la dernière ligne
            for i, line in enumerate(wrapped_lines[start_line:end_line]):
                screen_line = i + content_start
                if screen_line >= content_end:  # Ne pas écrire sur la barre de statut
                    break
                
                # Afficher la ligne avec marge gauche
                col = left_margin
                for j, char in enumerate(line):
                    if col >= right_width - left_margin:
                        break
                    
                    # Vérifier le type de surlignage pour ce caractère
                    char_style = self.get_char_style(col, start_line + i)
                    
                    # Afficher avec le style approprié
                    try:
                        if char_style == "cursor":
                            stdscr.addstr(screen_line, col + left_panel_width, char, curses.color_pair(2))  # Curseur actuel
                        elif char_style == "selection":
                            stdscr.addstr(screen_line, col + left_panel_width, char, curses.color_pair(7) | curses.A_BOLD)  # Zone sélectionnée
                        else:
                            stdscr.addstr(screen_line, col + left_panel_width, char, curses.color_pair(1))  # Texte normal
                    except:
                        if char_style == "cursor":
                            stdscr.addstr(screen_line, col + left_panel_width, char, curses.A_REVERSE)
                        elif char_style == "selection":
                            stdscr.addstr(screen_line, col + left_panel_width, char, curses.A_BOLD)
                        else:
                            stdscr.addstr(screen_line, col + left_panel_width, char)
                    col += 1

        else:
            # Message d'attente centré
            msg = "Génération de la fiche en cours..."
            stdscr.addstr(height // 2, (width - len(msg)) // 2, msg)
        
        # Barre de statut avec style et couleur
        if self.words:
            if self.selection_mode and self.selection_start != -1 and self.selection_end != -1:
                selected_text = self.get_selected_text()
                # Limiter la longueur du texte affiché
                if len(selected_text) > 25:
                    selected_text = selected_text[:22] + "..."
                status = f"📝 Sélection: \"{selected_text}\" ({abs(self.selection_end - self.selection_start) + 1} mots)"
            else:
                current_word = self.words[self.cursor_word_index][0] if self.cursor_word_index < len(self.words) else ""
                status = f"📍 Mot {self.cursor_word_index + 1}/{len(self.words)}: {current_word}"
            
            # Ajouter les indicateurs de thème, langue et modèle
            theme_indicator = "🌙" if self.current_theme == "dark" else "☀️"
            lang_indicator = self.languages[self.current_language]["flag"]
            model_indicator = f"{self.available_models[self.current_model]['symbol']}{self.available_models[self.current_model]['name']}"
            
            # Ajouter un indicateur de scroll si nécessaire
            if current_text_copy:
                words, wrapped_lines = self.extract_words(current_text_copy, right_width)
                total_lines = len(wrapped_lines)
                content_start, available_height = self.calculate_content_area(height)
                
                if total_lines > available_height:
                    visible_start = self.scroll_offset + 1
                    visible_end = min(self.scroll_offset + available_height, total_lines)
                    scroll_info = f"({visible_start}-{visible_end}/{total_lines})"
                    status = f"{status} | {scroll_info}"
            
            # Assembler la barre de statut finale
            status_parts = [status, theme_indicator, lang_indicator, model_indicator]
            status = " | ".join(status_parts)
            
            # Centrer la barre de statut
            status_pos = max(2, (width - len(status)) // 2)
            try:
                color = curses.color_pair(7) if self.selection_mode else curses.color_pair(5)
                stdscr.addstr(height - 1, status_pos, status[:width-4], color | curses.A_BOLD)
            except:
                attr = curses.A_REVERSE if self.selection_mode else curses.A_BOLD
                stdscr.addstr(height - 1, status_pos, status[:width-4], attr)
        
        # Ajouter des bordures subtiles pour le contenu seulement
        if width > 40 and current_text_copy:  # Seulement sur les écrans assez larges et avec du contenu
            left_margin, text_width = self.calculate_text_margins(right_width)
            content_start_border = separator_line + 1
            if left_margin > 3:
                # Bordures verticales subtiles pour le contenu avec couleur
                for y in range(content_start_border, height - 1):
                    try:
                        stdscr.addstr(y, left_margin - 2 + left_panel_width, "│", curses.color_pair(6))
                        stdscr.addstr(y, left_margin + text_width + 1 + left_panel_width, "│", curses.color_pair(6))
                    except:
                        try:
                            stdscr.addstr(y, left_margin - 2 + left_panel_width, "│", curses.A_DIM)
                            stdscr.addstr(y, left_margin + text_width + 1 + left_panel_width, "│", curses.A_DIM)
                        except:
                            pass  # Ignore si hors écran
        
        stdscr.refresh()
    
    def handle_input(self, stdscr, key):
        """Gère les entrées clavier."""
        height, width = stdscr.getmaxyx()
        
        # Gestion du mode historique
        if self.history_mode:
            if key in (27, ord('h')):  # Escape ou 'h' pour quitter le mode historique
                self.history_mode = False
                self.force_redraw = True
                return False
            elif key == curses.KEY_UP:
                if self.history_cursor > 0:
                    self.history_cursor -= 1
                    self.force_redraw = True
                return False
            elif key == curses.KEY_DOWN:
                if self.history_cursor < len(self.full_history) - 1:
                    self.history_cursor += 1
                    self.force_redraw = True
                return False
            elif key == ord('\n') or key == ord('\r'):
                # Sélectionner le sujet courant du curseur historique
                if 0 <= self.history_cursor < len(self.full_history):
                    selected_subject = self.full_history[self.history_cursor]
                    # Si on quitte le sujet actuel, arrêter éventuellement la génération
                    if self.is_generating:
                        self.stop_streaming = True
                        if self.streaming_thread:
                            self.streaming_thread.join(timeout=1)
                        self.is_generating = False
                    # Sauvegarder le texte actuel dans le cache
                    with self.text_lock:
                        self.subject_texts[self.current_subject] = self.current_text
                    self.history_mode = False
                    # Ouvrir le sujet sélectionné : si déjà généré, afficher, sinon générer
                    # La position dans l'historique reste basée sur la date de génération
                    self.history_cursor = 0
                    if selected_subject in self.subject_texts:
                        self.current_subject = selected_subject
                        with self.text_lock:
                            self.current_text = self.subject_texts.get(selected_subject, "") or f"🔄 Génération de '{selected_subject}' en cours..."
                            self.force_redraw = True
                        if not self.current_text.startswith("🔄 Génération") and self.current_text:
                            # Contenu existant : rien à faire
                            self.is_generating = False
                        else:
                            # Pas de contenu : générer
                            self.load_subject_streaming(selected_subject, stdscr)
                    else:
                        # Sujet inconnu : lancer une génération
                        self.load_subject_streaming(selected_subject, stdscr)
                    return False
            else:
                # Ignorer autres touches
                return False
        
        # Si en mode saisie, gérer différemment
        if self.input_mode:
            result = self.handle_input_char(key)
            if result:  # L'utilisateur a validé sa saisie
                self.input_mode = False
                self.history.append(self.current_subject)
                self.load_subject_streaming(result, stdscr)
            return False
        
        if key == ord('q'):
            # Arrêter le streaming si en cours
            if self.is_generating:
                self.stop_streaming = True
                if self.streaming_thread:
                    self.streaming_thread.join(timeout=1)
            return True
        elif key == ord('s') and self.is_generating:
            # Arrêter la génération en cours
            self.stop_streaming = True
            if self.streaming_thread:
                self.streaming_thread.join(timeout=1)
            self.is_generating = False
            self.force_redraw = True
        elif key == ord('r') and not self.is_generating:
            # Régénérer la fiche actuelle
            self.load_subject_streaming(self.current_subject, stdscr)
        elif key == ord('h'):
            # Basculer le mode historique
            self.history_mode = not self.history_mode
            self.force_redraw = True
        elif key == ord('t'):
            # Basculer entre les thèmes
            self.toggle_theme()
        elif key == ord('l'):
            # Changer de langue
            self.cycle_language()
        elif key == ord('m'):
            # Changer de modèle
            self.cycle_model()
        elif key == ord('i') and not self.is_generating:
            # Démarrer le mode de saisie
            self.start_input_mode()
        elif key == ord('b') and self.history:
            # Retour en arrière (interrompt le streaming en cours si nécessaire)
            prev_subject = self.history.pop()
            self.load_subject_streaming(prev_subject, stdscr)
        elif key == ord(' '):
            # Barre d'espace : démarrer/arrêter la sélection (même pendant le streaming)
            if not self.selection_mode:
                self.start_selection()
            else:
                self.clear_selection()
        elif key == 27:  # Touche Escape
            # Annuler la sélection
            if self.selection_mode:
                self.clear_selection()
        elif key == curses.KEY_UP:
            # Navigation possible même pendant le streaming
            new_index = self.find_word_above()
            if new_index != -1:
                self.cursor_word_index = new_index
                if self.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.force_redraw = True
        elif key == curses.KEY_DOWN:
            # Navigation possible même pendant le streaming
            new_index = self.find_word_below()
            if new_index != -1:
                self.cursor_word_index = new_index
                if self.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.force_redraw = True
        elif key == curses.KEY_LEFT:
            # Navigation possible même pendant le streaming
            new_index = self.find_word_left()
            if new_index != -1:
                self.cursor_word_index = new_index
                if self.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.force_redraw = True
        elif key == curses.KEY_RIGHT:
            # Navigation possible même pendant le streaming
            new_index = self.find_word_right()
            if new_index != -1:
                self.cursor_word_index = new_index
                if self.selection_mode:
                    self.update_selection_end()
                self.adjust_scroll(height)
                self.force_redraw = True
        elif key == ord('\n') or key == ord('\r'):
            # Générer une fiche pour le texte sélectionné (interrompt le streaming en cours si nécessaire)
            if self.words:
                selected_text = self.get_selected_text()
                if selected_text:
                    self.history.append(self.current_subject)
                    # Effacer la sélection après génération
                    if self.selection_mode:
                        self.clear_selection()
                    self.load_subject_streaming(selected_text, stdscr)
        
        return False
    
    def find_word_above(self) -> int:
        """Trouve le mot le plus proche au-dessus du mot actuel."""
        if not self.words or self.cursor_word_index >= len(self.words):
            return -1
        
        current_word, current_line, current_col = self.words[self.cursor_word_index]
        
        # Chercher le mot le plus proche sur une ligne au-dessus
        best_index = -1
        best_distance = float('inf')
        
        for i, (word, line, col) in enumerate(self.words):
            if line < current_line:  # Ligne au-dessus
                # Distance horizontale entre les mots
                distance = abs(col - current_col)
                # Préférer les lignes plus proches et les positions horizontales plus proches
                total_distance = (current_line - line) * 1000 + distance
                
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_index = i
        
        return best_index
    
    def find_word_below(self) -> int:
        """Trouve le mot le plus proche en-dessous du mot actuel."""
        if not self.words or self.cursor_word_index >= len(self.words):
            return -1
        
        current_word, current_line, current_col = self.words[self.cursor_word_index]
        
        # Chercher le mot le plus proche sur une ligne en-dessous
        best_index = -1
        best_distance = float('inf')
        
        for i, (word, line, col) in enumerate(self.words):
            if line > current_line:  # Ligne en-dessous
                # Distance horizontale entre les mots
                distance = abs(col - current_col)
                # Préférer les lignes plus proches et les positions horizontales plus proches
                total_distance = (line - current_line) * 1000 + distance
                
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_index = i
        
        return best_index
    
    def find_word_left(self) -> int:
        """Trouve le mot le plus proche à gauche du mot actuel."""
        if not self.words or self.cursor_word_index >= len(self.words):
            return -1
        
        current_word, current_line, current_col = self.words[self.cursor_word_index]
        
        # Chercher le mot le plus proche à gauche sur la même ligne
        best_index = -1
        
        for i, (word, line, col) in enumerate(self.words):
            if line == current_line and col < current_col:  # Même ligne, à gauche
                if best_index == -1 or col > self.words[best_index][2]:
                    best_index = i
        
        # Si pas de mot à gauche sur la même ligne, aller au dernier mot de la ligne précédente
        if best_index == -1:
            for i, (word, line, col) in enumerate(self.words):
                if line == current_line - 1:
                    if best_index == -1 or col > self.words[best_index][2]:
                        best_index = i
        
        return best_index
    
    def find_word_right(self) -> int:
        """Trouve le mot le plus proche à droite du mot actuel."""
        if not self.words or self.cursor_word_index >= len(self.words):
            return -1
        
        current_word, current_line, current_col = self.words[self.cursor_word_index]
        
        # Chercher le mot le plus proche à droite sur la même ligne
        best_index = -1
        
        for i, (word, line, col) in enumerate(self.words):
            if line == current_line and col > current_col:  # Même ligne, à droite
                if best_index == -1 or col < self.words[best_index][2]:
                    best_index = i
        
        # Si pas de mot à droite sur la même ligne, aller au premier mot de la ligne suivante
        if best_index == -1:
            for i, (word, line, col) in enumerate(self.words):
                if line == current_line + 1:
                    if best_index == -1 or col < self.words[best_index][2]:
                        best_index = i
        
        return best_index
    
    def calculate_content_area(self, height):
        """Calcule la zone de contenu disponible selon l'état actuel de l'interface."""
        # Structure dynamique de l'écran :
        header_height = 4 if len(self.current_subject) > 20 else 2  # Titre sur 1 ou 2 lignes
        instructions_height = 1
        input_height = 2 if self.input_mode else 0  # Zone de saisie si active
        separator_height = 1
        status_height = 1
        
        ui_total = header_height + instructions_height + input_height + separator_height + status_height
        content_start = ui_total - status_height  # Début de la zone de contenu
        content_height = height - ui_total  # Hauteur disponible pour le contenu
        
        return content_start, max(1, content_height)
    
    def adjust_scroll(self, height):
        """Ajuste le scroll pour garder le mot sélectionné visible."""
        if not self.words:
            return
        
        word_line = self.words[self.cursor_word_index][1]
        content_start, display_height = self.calculate_content_area(height)
        
        if word_line < self.scroll_offset:
            self.scroll_offset = word_line
        elif word_line >= self.scroll_offset + display_height:
            self.scroll_offset = word_line - display_height + 1
    
    def load_subject_streaming(self, subject: str, stdscr):
        """Charge un nouveau sujet avec génération en streaming."""
        # Arrêter toute génération en cours
        if self.is_generating:
            self.stop_streaming = True
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=0.5)  # Timeout plus court
        
        # Sauvegarder le contexte actuel (seulement si ce n'est pas un message d'attente)
        with self.text_lock:
            previous_text = self.current_text if not self.current_text.startswith("Génération en cours") else ""
        # Sauvegarder le texte courant dans le cache
        self.subject_texts[self.current_subject] = previous_text
        
        # Ajouter au panneau historique uniquement si nouveau sujet
        if subject not in self.full_history:
            self.full_history.insert(0, subject)
        self.history_cursor = 0
        # Réinitialiser l'état
        self.current_subject = subject
        with self.text_lock:
            self.current_text = f"🔄 Génération de '{subject}' en cours..."
            self.force_redraw = True
        self.cursor_word_index = 0
        self.scroll_offset = 0
        self.is_generating = True
        self.stop_streaming = False
        
        # Affichage initial immédiat
        self.draw_screen(stdscr)
        stdscr.refresh()
        
        # Lancer la génération en streaming dans un thread
        self.streaming_thread = threading.Thread(
            target=self.generate_fiche_streaming_thread,
            args=(subject, previous_text),
            daemon=True  # Thread daemon pour nettoyage automatique
        )
        self.streaming_thread.start()
    
    def run(self, stdscr):
        """Boucle principale de l'interface."""
        curses.curs_set(0)  # Cache le curseur
        stdscr.nodelay(1)   # Non-bloquant
        stdscr.timeout(100)  # Timeout plus long pour réduire le scintillement
        
        # Configuration du thème sombre
        try:
            curses.start_color()
            curses.use_default_colors()
            
            # Définir les paires de couleurs pour le thème sombre
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)    # Texte normal
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)    # Texte sélectionné (inverse)
            curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)     # Titre
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)   # Instructions
            curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)    # Statut
            curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)     # Bordures
            curses.init_pair(7, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Accent
            
            # Forcer l'arrière-plan noir pour tout l'écran
            stdscr.bkgd(' ', curses.color_pair(1))
        except:
            pass
        
        # Génération initiale
        self.load_subject_streaming(self.current_subject, stdscr)
        
        running = True
        while running:
            self.draw_screen(stdscr)
            
            key = stdscr.getch()
            if key != -1:  # Si une touche a été pressée
                should_quit = self.handle_input(stdscr, key)
                if should_quit:
                    running = False
        
        # Nettoyage à la sortie
        if self.is_generating:
            self.stop_streaming = True
            if self.streaming_thread:
                self.streaming_thread.join(timeout=2)


def run_simple_tui(initial_subject: str = "Lyon"):
    """Lance l'interface TUI simple."""
    tui = SimpleTUI(initial_subject)
    curses.wrapper(tui.run)
