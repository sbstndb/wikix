"""Interface TUI basée sur Textual."""
from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input
from textual.containers import Horizontal
from textual.reactive import reactive
from rich.markdown import Markdown

from llm import generate_fiche, generate_fiche_with_context

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
TEMPLATE_GENERAL = (PROMPT_DIR / "fiche_generale.txt").read_text(encoding="utf-8")
TEMPLATE_CONTEXT = (PROMPT_DIR / "fiche_contexte.txt").read_text(encoding="utf-8")
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


class WikiPane(Static):
    """Widget qui affiche le contenu Markdown d'une fiche."""

    def update_content(self, md_text: str):
        self.update(Markdown(md_text))


class WikiApp(App):
    BINDINGS = [
        ("q", "quit", "Quitter"),
    ]

    subject: reactive[str | None] = reactive(None)
    context_text: reactive[str | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            self.pane = WikiPane()
            yield self.pane
        yield Input(placeholder="Sujet ou mot-clé → Entrée pour générer", id="input")
        yield Footer()

    async def on_mount(self):
        self.query_one("#input", Input).focus()

    async def on_input_submitted(self, message: Input.Submitted):
        subject = message.value.strip()
        if not subject:
            return
        self.subject = subject
        input_widget = self.query_one("#input", Input)
        input_widget.value = ""
        await self.generate_and_display(subject)

    async def generate_and_display(self, subject: str):
        self.pane.update_content("⏳ Génération de la fiche en cours...")
        if self.context_text:
            md = await asyncio.get_event_loop().run_in_executor(
                None,
                generate_fiche_with_context,
                subject,
                self.context_text,
                TEMPLATE_CONTEXT,
            )
        else:
            md = await asyncio.get_event_loop().run_in_executor(
                None, generate_fiche, subject, TEMPLATE_GENERAL
            )
        self.context_text = md  # mise à jour du contexte
        self.pane.update_content(md)
        # Sauvegarde
        out_path = GENERATED_DIR / f"{subject.lower().replace(' ', '_')}.md"
        out_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    WikiApp().run()

