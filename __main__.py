"""Point d'entrée CLI : `python -m wikix <sujet>` ou juste `wikix` si installé.
"""
from __future__ import annotations

from wikix.commands.main import main

if __name__ == "__main__":
    main()

