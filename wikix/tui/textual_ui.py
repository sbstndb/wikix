"""Interface TUI basée sur Textual."""
from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input
from textual.containers import Horizontal
from textual.reactive import reactive
from rich.markdown import Markdown

from wikix.core.llm import generate_fiche, generate_fiche_with_context
from wikix.core.config import GENERATED_DIR, TEMPLATE_GENERAL, TEMPLATE_CONTEXT


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

