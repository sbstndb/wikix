"""Association sujet → emoji via embeddings MiniLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emoji_db.json"


class EmojiModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if not DATA_PATH.exists():
            # Fallback minimal DB
            self.emoji_db = [
                {"emoji": "🦁", "desc": "lion animal sauvage force"},
                {"emoji": "🗼", "desc": "tour eiffel paris monument"},
                {"emoji": "🍽️", "desc": "cuisine gastronomie repas"},
                {"emoji": "📜", "desc": "histoire document ancien"},
                {"emoji": "🎵", "desc": "musique note son"},
            ]
        else:
            self.emoji_db = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        self._prepare_embeddings()

    def _prepare_embeddings(self):
        self.desc_embeddings = self.model.encode([e["desc"] for e in self.emoji_db])

    def get_emoji(self, subject: str) -> str:
        subj_vec = self.model.encode([subject])
        sims = cosine_similarity(subj_vec, self.desc_embeddings)[0]
        best_idx = sims.argmax()
        return self.emoji_db[best_idx]["emoji"]


emoji_model_singleton: EmojiModel | None = None

def get_emoji_model() -> EmojiModel:
    global emoji_model_singleton
    if emoji_model_singleton is None:
        emoji_model_singleton = EmojiModel()
    return emoji_model_singleton
