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
    def __init__(self, initial_subject: str = "Lyon"):
        self.current_subject = initial_subject
        self.current_text = ""
        self.words: List[Tuple[str, int, int]] = []  # (word, line, col)
        self.cursor_word_index = 0
        self.history: List[str] = []
        self.scroll_offset = 0
        self.is_generating = False
        self.streaming_thread = None
        self.stop_streaming = False
        self.text_lock = threading.Lock()
        
    def extract_words(self, text: str, width: int) -> List[Tuple[str, int, int]]:
        """Extrait les mots avec leurs positions après wrapping du texte."""
        words = []
        lines = text.split('\n')
        wrapped_lines = []
        
        # Wrapper le texte pour la largeur du terminal
        for line in lines:
            if line.strip():
                wrapped = textwrap.fill(line, width=width-4)
                wrapped_lines.extend(wrapped.split('\n'))
            else:
                wrapped_lines.append('')
        
        # Extraire les mots avec positions
        for line_num, line in enumerate(wrapped_lines):
            # Trouve tous les mots dans la ligne
            for match in re.finditer(r'\b[A-Za-zÀ-ÿ]+\b', line):
                word = match.group()
                if len(word) > 2:  # Ignore les mots trop courts
                    words.append((word, line_num, match.start()))
        
        return words, wrapped_lines
    
    def generate_fiche_streaming_thread(self, subject: str, previous_text: str):
        """Génère une fiche en streaming dans un thread séparé."""
        try:
            # Détermine le type de génération
            if previous_text and subject != self.current_subject:
                stream_gen = generate_fiche_with_context_stream(subject, previous_text, TEMPLATE_CONTEXT)
            else:
                stream_gen = generate_fiche_stream(subject, TEMPLATE_GENERAL)
            
            # Accumule le contenu progressivement
            full_content = ""
            for chunk in stream_gen:
                if self.stop_streaming:
                    break
                    
                full_content += chunk
                
                # Mise à jour thread-safe du texte
                with self.text_lock:
                    self.current_text = full_content
                
                # Petite pause pour rendre l'effet visible
                time.sleep(0.03)  # 30ms
            
            # Sauvegarde finale
            if not self.stop_streaming:
                filename = f"{subject.lower().replace(' ', '_')}.md"
                output_path = GENERATED_DIR / filename
                output_path.write_text(full_content, encoding="utf-8")
            
            self.is_generating = False
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération : {str(e)}"
            with self.text_lock:
                self.current_text = error_msg
            self.is_generating = False
    
    def draw_screen(self, stdscr):
        """Dessine l'écran principal."""
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Titre
        title = f"Wikix - {self.current_subject}"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
        
        # Instructions
        if self.is_generating:
            instructions = "Génération en cours... | ↑↓←→: Naviguer | 'q': Quitter | 's': Arrêter génération"
        else:
            instructions = "↑↓←→: Naviguer | Entrée: Générer fiche | 'q': Quitter | 'b': Retour"
        stdscr.addstr(1, 2, instructions[:width-4], curses.A_DIM)
        
        # Ligne de séparation
        stdscr.addstr(2, 0, "─" * width)
        
        # Affichage du texte avec mots surlignés
        if self.current_text:
            # Lecture thread-safe du texte
            with self.text_lock:
                current_text_copy = self.current_text
            
            words, wrapped_lines = self.extract_words(current_text_copy, width)
            self.words = words
            
            # Calcul de la zone d'affichage
            display_height = height - 4
            start_line = max(0, self.scroll_offset)
            end_line = min(len(wrapped_lines), start_line + display_height)
            
            # Affichage ligne par ligne
            for i, line in enumerate(wrapped_lines[start_line:end_line]):
                screen_line = i + 3
                if screen_line >= height:
                    break
                
                # Afficher la ligne caractère par caractère pour surligner les mots
                col = 2
                for j, char in enumerate(line):
                    if col >= width - 2:
                        break
                    
                    # Vérifier si on est sur un mot sélectionné
                    is_selected_word = False
                    if self.cursor_word_index < len(self.words):
                        word, word_line, word_col = self.words[self.cursor_word_index]
                        if (word_line == start_line + i and 
                            word_col <= j < word_col + len(word)):
                            is_selected_word = True
                    
                    # Afficher avec ou sans surlignage
                    if is_selected_word:
                        stdscr.addstr(screen_line, col, char, curses.A_REVERSE)
                    else:
                        stdscr.addstr(screen_line, col, char)
                    col += 1
        else:
            # Message d'attente
            msg = "Génération de la fiche en cours..."
            stdscr.addstr(height // 2, (width - len(msg)) // 2, msg)
        
        # Barre de statut
        if self.words:
            status = f"Mot {self.cursor_word_index + 1}/{len(self.words)}: {self.words[self.cursor_word_index][0]}"
            stdscr.addstr(height - 1, 2, status[:width-4], curses.A_BOLD)
        
        stdscr.refresh()
    
    def handle_input(self, stdscr, key):
        """Gère les entrées clavier."""
        height, width = stdscr.getmaxyx()
        
        if key == ord('q'):
            # Arrêter le streaming si en cours
            if self.is_generating:
                self.stop_streaming = True
                if self.streaming_thread:
                    self.streaming_thread.join(timeout=1)
            return False
        elif key == ord('s') and self.is_generating:
            # Arrêter la génération en cours
            self.stop_streaming = True
            if self.streaming_thread:
                self.streaming_thread.join(timeout=1)
            return True
        elif key == ord('b') and self.history and not self.is_generating:
            # Retour en arrière
            prev_subject = self.history.pop()
            self.load_subject_streaming(prev_subject, stdscr)
        elif key == curses.KEY_UP and self.cursor_word_index > 0:
            self.cursor_word_index -= 1
            self.adjust_scroll(height)
        elif key == curses.KEY_DOWN and self.cursor_word_index < len(self.words) - 1:
            self.cursor_word_index += 1
            self.adjust_scroll(height)
        elif key == curses.KEY_LEFT and self.cursor_word_index > 0:
            self.cursor_word_index -= 1
            self.adjust_scroll(height)
        elif key == curses.KEY_RIGHT and self.cursor_word_index < len(self.words) - 1:
            self.cursor_word_index += 1
            self.adjust_scroll(height)
        elif key == ord('\n') or key == ord('\r'):
            # Générer une fiche pour le mot sélectionné (seulement si pas déjà en génération)
            if self.words and self.cursor_word_index < len(self.words) and not self.is_generating:
                selected_word = self.words[self.cursor_word_index][0]
                self.history.append(self.current_subject)
                self.load_subject_streaming(selected_word, stdscr)
        
        return True
    
    def adjust_scroll(self, height):
        """Ajuste le scroll pour garder le mot sélectionné visible."""
        if not self.words:
            return
        
        word_line = self.words[self.cursor_word_index][1]
        display_height = height - 4
        
        if word_line < self.scroll_offset:
            self.scroll_offset = word_line
        elif word_line >= self.scroll_offset + display_height:
            self.scroll_offset = word_line - display_height + 1
    
    def load_subject_streaming(self, subject: str, stdscr):
        """Charge un nouveau sujet avec génération en streaming."""
        # Arrêter toute génération en cours
        if self.is_generating:
            self.stop_streaming = True
            if self.streaming_thread:
                self.streaming_thread.join(timeout=1)
        
        # Sauvegarder le contexte actuel
        previous_text = self.current_text
        
        # Réinitialiser l'état
        self.current_subject = subject
        with self.text_lock:
            self.current_text = "Génération en cours..."
        self.cursor_word_index = 0
        self.scroll_offset = 0
        self.is_generating = True
        self.stop_streaming = False
        
        # Affichage initial
        self.draw_screen(stdscr)
        stdscr.refresh()
        
        # Lancer la génération en streaming dans un thread
        self.streaming_thread = threading.Thread(
            target=self.generate_fiche_streaming_thread,
            args=(subject, previous_text)
        )
        self.streaming_thread.start()
    
    def run(self, stdscr):
        """Boucle principale de l'interface."""
        curses.curs_set(0)  # Cache le curseur
        stdscr.nodelay(1)   # Non-bloquant
        stdscr.timeout(50)  # Timeout réduit pour une meilleure réactivité
        
        # Génération initiale
        self.load_subject_streaming(self.current_subject, stdscr)
        
        running = True
        while running:
            self.draw_screen(stdscr)
            
            key = stdscr.getch()
            if key != -1:  # Si une touche a été pressée
                running = self.handle_input(stdscr, key)
        
        # Nettoyage à la sortie
        if self.is_generating:
            self.stop_streaming = True
            if self.streaming_thread:
                self.streaming_thread.join(timeout=2)


def run_simple_tui(initial_subject: str = "Lyon"):
    """Lance l'interface TUI simple."""
    tui = SimpleTUI(initial_subject)
    curses.wrapper(tui.run)
