#!/bin/bash
# Script helper pour lancer wikix facilement

# Vérifie si OPENAI_API_KEY est définie
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Erreur: La variable OPENAI_API_KEY n'est pas définie."
    echo "💡 Définissez-la avec: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# Lance wikix avec les arguments passés
cd /home/sbstndbs/sbstndbs && python -m wikix "$@"
