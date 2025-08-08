"""Interface CLI améliorée pour Wikix.

Améliorations principales (niveau senior):
- Typage explicite et docstrings claires
- Slugification robuste des noms de fiches (accents, ponctuation)
- Options communes factorisées: --model, --provider, --temperature, --lang, --output, --overwrite, --quiet
- Support multilingue via get_template_for_lang
- Gestion d'erreurs et messages plus informatifs
"""
from __future__ import annotations

import argparse
import sys
import re
import unicodedata
from pathlib import Path
from typing import Optional

from wikix.core.config import GENERATED_DIR, get_template_for_lang
from wikix.core.llm import generate_fiche, generate_fiche_with_context


def _slugify_filename(text: str) -> str:
    """Convertit un texte libre en nom de fichier sûr (ASCII, tirets).

    - Normalise en NFKD pour retirer les accents
    - Garde [a-z0-9-_.]
    - Remplace les espaces/ponctuations par '-'
    - Évite les doublons de tirets et strip en bord.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    # Remplacer tout ce qui n'est pas alphanumérique par des tirets
    ascii_text = re.sub(r"[^a-z0-9._]+", "-", ascii_text)
    # Compacter les tirets
    ascii_text = re.sub(r"-+", "-", ascii_text).strip("-._")
    return ascii_text or "fiche"


def _resolve_output_path(
    sujet: str,
    output_arg: Optional[str],
    default_suffix: str = ".md",
) -> Path:
    """Calcule le chemin de sortie en respectant --output (fichier ou dossier).

    - Si output_arg est un dossier, on crée <dossier>/<slug>.md
    - Si output_arg est un fichier, on l'utilise (en ajoutant .md si manquant)
    - Sinon, on utilise GENERATED_DIR/<slug>.md
    """
    slug = _slugify_filename(sujet)
    if output_arg:
        candidate = Path(output_arg)
        if candidate.is_dir() or str(output_arg).endswith(("/", "\\")):
            return candidate / f"{slug}{default_suffix}"
        if candidate.suffix.lower() != default_suffix:
            candidate = candidate.with_suffix(default_suffix)
        return candidate
    return GENERATED_DIR / f"{slug}{default_suffix}"


def _safe_write_text(path: Path, content: str, overwrite: bool = False) -> None:
    """Écrit le fichier en créant les dossiers si besoin, protège contre l'écrasement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Le fichier existe déjà: {path}. Utilisez --overwrite pour écraser.")
    path.write_text(content, encoding="utf-8")


def _print_if(text: str, enabled: bool) -> None:
    if enabled:
        print(text)


def generate_command(args: argparse.Namespace) -> None:
    """Commande 'generate' : génère une fiche simple."""
    sujet: str = args.sujet
    model: Optional[str] = getattr(args, "model", None)
    provider: Optional[str] = getattr(args, "provider", None)
    temperature: float = float(getattr(args, "temperature", 0.7))
    lang: str = getattr(args, "lang", "fr")
    output_arg: Optional[str] = getattr(args, "output", None)
    overwrite: bool = bool(getattr(args, "overwrite", False))
    quiet: bool = bool(getattr(args, "quiet", False))

    output_file = _resolve_output_path(sujet, output_arg)

    _print_if(f"🔍 Génération de la fiche pour '{sujet}'...", not quiet)
    if model or provider:
        if model:
            _print_if(f"🤖 Modèle : {model}", not quiet)
        if provider:
            _print_if(f"🏷️ Provider : {provider}", not quiet)

    try:
        template = get_template_for_lang(lang, with_context=False)

        md = generate_fiche(sujet, template, temperature=temperature, model=model, provider=provider)
        _safe_write_text(output_file, md, overwrite=overwrite)
        _print_if(f"✅ Fiche sauvegardée dans {output_file}", not quiet)
        if getattr(args, "show", False):
            print("\n" + "=" * 50)
            print(md)
    except FileExistsError as fe:
        print(f"❌ {fe}")
        sys.exit(1)
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"❌ Erreur lors de la génération: {e}")
        sys.exit(2)


def link_command(args: argparse.Namespace) -> None:
    """Commande 'link' : génère une fiche avec contexte d'une autre fiche."""
    context_file = Path(args.context_file)
    if not context_file.exists():
        print(f"❌ Fichier de contexte introuvable : {context_file}")
        sys.exit(1)

    try:
        context = context_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"❌ Impossible de lire le contexte: {e}")
        sys.exit(1)

    sujet: str = args.sujet
    model: Optional[str] = getattr(args, "model", None)
    provider: Optional[str] = getattr(args, "provider", None)
    temperature: float = float(getattr(args, "temperature", 0.7))
    lang: str = getattr(args, "lang", "fr")
    output_arg: Optional[str] = getattr(args, "output", None)
    overwrite: bool = bool(getattr(args, "overwrite", False))
    quiet: bool = bool(getattr(args, "quiet", False))

    # Par défaut: <slug>-from-<context_stem>.md (via slugification)
    output_file = _resolve_output_path(f"{sujet} from {context_file.stem}", output_arg)

    _print_if(
        f"🔗 Génération de la fiche pour '{sujet}' avec contexte '{context_file.name}'...",
        not quiet,
    )
    if model or provider:
        if model:
            _print_if(f"🤖 Modèle : {model}", not quiet)
        if provider:
            _print_if(f"🏷️ Provider : {provider}", not quiet)

    try:
        template = get_template_for_lang(lang, with_context=True)

        md = generate_fiche_with_context(
            sujet,
            context,
            template,
            temperature=temperature,
            model=model,
            provider=provider,
        )
        _safe_write_text(output_file, md, overwrite=overwrite)
        _print_if(f"✅ Fiche contextuelle sauvegardée dans {output_file}", not quiet)
        if getattr(args, "show", False):
            print("\n" + "=" * 50)
            print(md)
    except FileExistsError as fe:
        print(f"❌ {fe}")
        sys.exit(1)
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"❌ Erreur lors de la génération contextuelle: {e}")
        sys.exit(2)


def _human_size(num_bytes: int) -> str:
    size: float = float(num_bytes)
    for unit in ["o", "Ko", "Mo", "Go", "To"]:
        if size < 1024 or unit == "To":
            if unit == "o":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} To"


def list_command(_args: argparse.Namespace) -> None:
    """Commande 'list' : liste toutes les fiches générées."""
    fiches = sorted(GENERATED_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not fiches:
        print("📂 Aucune fiche générée pour l'instant.")
        return

    print(f"📚 {len(fiches)} fiche(s) générée(s) :")
    for fiche in fiches:
        size = fiche.stat().st_size
        print(f"  • {fiche.name} ({_human_size(size)})")


def show_command(args: argparse.Namespace) -> None:
    """Commande 'show' : affiche une fiche existante."""
    fiche_name: str = args.fiche_name
    if not fiche_name.endswith(".md"):
        fiche_name += ".md"

    fiche_path = GENERATED_DIR / fiche_name
    if not fiche_path.exists():
        print(f"❌ Fiche introuvable : {fiche_name}")
        print("💡 Utilisez 'wikix list' pour voir les fiches disponibles.")
        return

    content = fiche_path.read_text(encoding="utf-8")
    print(f"📖 Contenu de {fiche_name} :")
    print("=" * 50)
    print(content)


def interactive_command(args: argparse.Namespace | None) -> None:
    """Commande 'interactive' : lance l'interface TUI."""
    # Textual a été retiré; on redirige vers la TUI simple
    from wikix.tui.simple_tui import run_simple_tui
    sujet: str = getattr(args, "sujet", "Wikipédia") if args is not None else "Wikipédia"
    run_simple_tui(sujet)


def simple_command(args: argparse.Namespace) -> None:
    """Commande 'simple' : lance l'interface TUI simple."""
    from wikix.tui.simple_tui import run_simple_tui
    initial_subject: str = getattr(args, "sujet", "Wikipédia")
    run_simple_tui(initial_subject)


def create_parser() -> argparse.ArgumentParser:
    """Crée le parser d'arguments CLI."""
    parser = argparse.ArgumentParser(
        prog="wikix",
        description="Générateur de fiches encyclopédiques par IA",
        epilog=(
            "Exemples :\n"
            "  wikix generate Lyon\n"
            "  wikix generate 'São Paulo' --lang en --temperature 0.6 --show\n"
            "  wikix link Rhône --context generated/lyon.md --output out/ --overwrite\n"
            "  wikix list\n"
            "  wikix interactive"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Parser parent pour les options communes aux commandes de génération
    common_gen = argparse.ArgumentParser(add_help=False)
    common_gen.add_argument("--show", action="store_true", help="Affiche la fiche après génération")
    common_gen.add_argument("--model", help="Modèle à utiliser (ex: openai/gpt-oss-120b, gpt-4o-mini, gemini-2.5-flash)")
    common_gen.add_argument("--provider", help="Provider à utiliser (ex: openai, openrouter, gemini, cerebras)")
    common_gen.add_argument("--temperature", type=float, default=0.7, help="Température de génération (0.0-1.0)")
    common_gen.add_argument("--lang", choices=["fr", "en", "es"], default="fr", help="Langue du template")
    common_gen.add_argument("--output", help="Fichier ou dossier de sortie")
    common_gen.add_argument("--overwrite", action="store_true", help="Écrase le fichier s'il existe")
    common_gen.add_argument("--quiet", action="store_true", help="Réduit les sorties console")

    # Commande 'generate'
    gen_parser = subparsers.add_parser("generate", parents=[common_gen], help="Génère une fiche simple")
    gen_parser.add_argument("sujet", help="Sujet de la fiche à générer")
    gen_parser.set_defaults(func=generate_command)

    # Commande 'link'
    link_parser = subparsers.add_parser("link", parents=[common_gen], help="Génère une fiche avec contexte")
    link_parser.add_argument("sujet", help="Sujet de la fiche à générer")
    link_parser.add_argument("--context", dest="context_file", required=True, help="Fichier de contexte (.md)")
    link_parser.set_defaults(func=link_command)

    # Commande 'list'
    list_parser = subparsers.add_parser("list", help="Liste les fiches générées")
    list_parser.set_defaults(func=list_command)

    # Commande 'show'
    show_parser = subparsers.add_parser("show", help="Affiche une fiche existante")
    show_parser.add_argument("fiche_name", help="Nom de la fiche (avec ou sans .md)")
    show_parser.set_defaults(func=show_command)

    # Commande 'interactive'
    interactive_parser = subparsers.add_parser("interactive", help="Lance l'interface TUI")
    interactive_parser.set_defaults(func=interactive_command)

    # Commande 'simple'
    simple_parser = subparsers.add_parser("simple", help="Lance l'interface TUI simple")
    simple_parser.add_argument("sujet", nargs="?", default="Wikipédia", help="Sujet initial (défaut: Wikipédia)")
    simple_parser.set_defaults(func=simple_command)

    return parser


def main() -> None:
    """Point d'entrée principal du CLI."""
    parser = create_parser()

    # Si aucun argument, comportement par défaut
    if len(sys.argv) == 1:
        # Lance directement la TUI simple par défaut
        interactive_command(argparse.Namespace(sujet="Wikipédia"))
        return

    # Si un seul argument et que ce n'est pas une sous-commande connue, génération directe
    valid_commands = {"generate", "link", "list", "show", "interactive", "simple"}
    if (
        len(sys.argv) == 2
        and not sys.argv[1].startswith("-")
        and sys.argv[1] not in valid_commands
    ):
        generate_command(
            argparse.Namespace(
                sujet=sys.argv[1],
                show=False,
                model=None,
                provider=None,
                temperature=0.7,
                lang="fr",
                output=None,
                overwrite=False,
                quiet=False,
            )
        )
        return

    # Parsing normal des sous-commandes
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
