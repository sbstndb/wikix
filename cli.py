"""Interface CLI améliorée pour Wikix."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm import generate_fiche, generate_fiche_with_context

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
TEMPLATE_GENERAL = (PROMPT_DIR / "fiche_generale.txt").read_text(encoding="utf-8")
TEMPLATE_CONTEXT = (PROMPT_DIR / "fiche_contexte.txt").read_text(encoding="utf-8")
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


def generate_command(args):
    """Commande 'generate' : génère une fiche simple."""
    sujet = args.sujet
    output_file = GENERATED_DIR / f"{sujet.lower().replace(' ', '_')}.md"
    
    print(f"🔍 Génération de la fiche pour '{sujet}'...")
    md = generate_fiche(sujet, TEMPLATE_GENERAL)
    
    output_file.write_text(md, encoding="utf-8")
    print(f"✅ Fiche sauvegardée dans {output_file}")
    
    if args.show:
        print("\n" + "="*50)
        print(md)


def link_command(args):
    """Commande 'link' : génère une fiche avec contexte d'une autre fiche."""
    context_file = Path(args.context_file)
    if not context_file.exists():
        print(f"❌ Fichier de contexte introuvable : {context_file}")
        return
    
    context = context_file.read_text(encoding="utf-8")
    sujet = args.sujet
    output_file = GENERATED_DIR / f"{sujet.lower().replace(' ', '_')}_from_{context_file.stem}.md"
    
    print(f"🔗 Génération de la fiche pour '{sujet}' avec contexte '{context_file.name}'...")
    md = generate_fiche_with_context(sujet, context, TEMPLATE_CONTEXT)
    
    output_file.write_text(md, encoding="utf-8")
    print(f"✅ Fiche contextuelle sauvegardée dans {output_file}")
    
    if args.show:
        print("\n" + "="*50)
        print(md)


def list_command(args):
    """Commande 'list' : liste toutes les fiches générées."""
    fiches = list(GENERATED_DIR.glob("*.md"))
    if not fiches:
        print("📂 Aucune fiche générée pour l'instant.")
        return
    
    print(f"📚 {len(fiches)} fiche(s) générée(s) :")
    for fiche in sorted(fiches):
        size = fiche.stat().st_size
        print(f"  • {fiche.name} ({size} octets)")


def show_command(args):
    """Commande 'show' : affiche une fiche existante."""
    fiche_name = args.fiche_name
    if not fiche_name.endswith('.md'):
        fiche_name += '.md'
    
    fiche_path = GENERATED_DIR / fiche_name
    if not fiche_path.exists():
        print(f"❌ Fiche introuvable : {fiche_name}")
        print("💡 Utilisez 'wikix list' pour voir les fiches disponibles.")
        return
    
    content = fiche_path.read_text(encoding="utf-8")
    print(f"📖 Contenu de {fiche_name} :")
    print("=" * 50)
    print(content)


def interactive_command(args):
    """Commande 'interactive' : lance l'interface TUI."""
    from .ui import WikiApp
    WikiApp().run()


def simple_command(args):
    """Commande 'simple' : lance l'interface TUI simple."""
    from .simple_tui import run_simple_tui
    initial_subject = getattr(args, 'sujet', 'Lyon')
    run_simple_tui(initial_subject)


def create_parser():
    """Crée le parser d'arguments CLI."""
    parser = argparse.ArgumentParser(
        prog="wikix",
        description="Générateur de fiches encyclopédiques par IA",
        epilog="Exemples :\n"
               "  wikix generate Lyon\n"
               "  wikix link Rhône --context generated/lyon.md\n"
               "  wikix list\n"
               "  wikix interactive",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")
    
    # Commande 'generate'
    gen_parser = subparsers.add_parser("generate", help="Génère une fiche simple")
    gen_parser.add_argument("sujet", help="Sujet de la fiche à générer")
    gen_parser.add_argument("--show", action="store_true", help="Affiche la fiche après génération")
    gen_parser.set_defaults(func=generate_command)
    
    # Commande 'link'
    link_parser = subparsers.add_parser("link", help="Génère une fiche avec contexte")
    link_parser.add_argument("sujet", help="Sujet de la fiche à générer")
    link_parser.add_argument("--context", dest="context_file", required=True, 
                           help="Fichier de contexte (.md)")
    link_parser.add_argument("--show", action="store_true", help="Affiche la fiche après génération")
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
    simple_parser.add_argument("sujet", nargs="?", default="Lyon", help="Sujet initial (défaut: Lyon)")
    simple_parser.set_defaults(func=simple_command)
    
    return parser


def main():
    """Point d'entrée principal du CLI."""
    parser = create_parser()
    
    # Si aucun argument, comportement par défaut
    if len(sys.argv) == 1:
        # Lance l'interface interactive
        interactive_command(None)
        return
    
    # Si un seul argument et que ce n'est pas une sous-commande connue, génération directe
    valid_commands = {'generate', 'link', 'list', 'show', 'interactive', 'simple'}
    if (len(sys.argv) == 2 and 
        not sys.argv[1].startswith('-') and 
        sys.argv[1] not in valid_commands):
        class SimpleArgs:
            sujet = sys.argv[1]
            show = False
        generate_command(SimpleArgs())
        return
    
    # Parsing normal des sous-commandes
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
