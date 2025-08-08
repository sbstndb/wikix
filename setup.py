#!/usr/bin/env python3
"""Setup script pour installer wikix comme commande système."""

from setuptools import find_packages, setup

setup(
    name="wikix",
    version="0.1.0",
    description="Générateur de fiches encyclopédiques par IA",
    packages=find_packages(),
    python_requires=">=3.10",
            install_requires=[
            "openai>=1.12.0",
            "python-dotenv>=1.0.0",
            "textual>=0.51.0",
            "rich>=13.7.0",
            "google-generativeai>=0.3.0",
            "requests>=2.31.0",
        ],
    entry_points={
        "console_scripts": [
            "wikix=wikix.commands.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wikix": ["prompts/*.txt"],
    },
)
