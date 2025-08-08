from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import re
from pathlib import Path

from wikix.core.llm import generate_fiche_stream, generate_fiche_with_context_stream
from wikix.core.config import get_template_for_lang, GENERATED_DIR
from wikix.core.ui_config import TUIState

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- État global (mémoire, simple) ---
state = {
    "current_subject": "Wikipedia",
    "subject_texts": {"Wikipedia": ""},
    "full_history": ["Wikipedia"],
    "is_generating": False,
    "stop_streaming": False,
    "text_lock": threading.Lock(),
}


def slugify(text: str) -> str:
    return re.sub(r"[\W_]+", "-", text.lower()).strip("-")


# --- Routes Flask ---
@app.route("/")
def index():
    return render_template("index.html", cache_buster=int(time.time()))


@app.route("/api/config")
def api_config():
    ui = TUIState()
    # Normaliser les structures pour le front
    provider_models = {("auto" if k is None else k): v for k, v in ui.provider_models.items()}
    available_models = {k: {"name": v.get("name", k), "symbol": v.get("symbol", "")}
                        for k, v in ui.available_models.items()}
    themes = {
        "dark": {"name": "Dark"},
        "light": {"name": "Light"},
        "ocean": {"name": "Ocean"},
        "matrix": {"name": "Matrix"},
        "pure_black": {"name": "Pure Black"},
        "pure_white": {"name": "Pure White"},
    }
    return jsonify({
        "languages": ui.languages,
        "themes": themes,
        "available_models": available_models,
        "provider_models": provider_models,
    })


# --- Socket.IO ---
@socketio.on("connect")
def on_connect():
    with state["text_lock"]:
        emit(
            "initial_data",
            {
                "subject": state["current_subject"],
                "text": state["subject_texts"].get(state["current_subject"], ""),
                "history": state["full_history"],
            },
        )


@socketio.on("generate_subject")
def on_generate_subject(data):
    # Convertir 'auto' en None pour la logique backend
    if data.get("provider") == "auto":
        data["provider"] = None

    if state["is_generating"]:
        state["stop_streaming"] = True
        time.sleep(0.1)

    state["stop_streaming"] = False
    state["is_generating"] = True

    subject = data.get("subject", "Wikipédia")
    state["current_subject"] = subject

    with state["text_lock"]:
        if subject in state["full_history"]:
            state["full_history"].remove(subject)
        state["full_history"].insert(0, subject)
        emit("history_update", {"history": state["full_history"]})

    # Lancer la génération en tâche de fond (compat Socket.IO)
    socketio.start_background_task(generation_thread, data)


def generation_thread(data):
    subject = data.get("subject", "")
    temperature = data.get("temperature", 0.7)

    # Ne pas réutiliser un placeholder comme contexte
    with state["text_lock"]:
        prev = state["subject_texts"].get(subject, "")
    previous_text = prev if prev and not prev.startswith("🔄") else ""

    # Récupération correcte des templates
    lang = data.get("lang", "en")
    general_template = get_template_for_lang(lang, with_context=False)
    context_template = get_template_for_lang(lang, with_context=True)

    if previous_text:
        stream = generate_fiche_with_context_stream(
            subject,
            previous_text,
            context_template,
            model=data.get("model"),
            provider=data.get("provider"),
            temperature=temperature,
        )
    else:
        stream = generate_fiche_stream(
            subject,
            general_template,
            model=data.get("model"),
            provider=data.get("provider"),
            temperature=temperature,
        )

    full_content = ""
    for chunk in stream:
        if state["stop_streaming"]:
            break
        full_content += chunk
        socketio.emit("text_update", {"text": full_content, "subject": subject})
        # Céder la main pour permettre l'envoi progressif (streaming)
        socketio.sleep(0)

    # Sauvegarde et fin
    with state["text_lock"]:
        state["subject_texts"][subject] = full_content
        if not state["stop_streaming"]:
            out = GENERATED_DIR / f"{slugify(subject)}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(full_content, encoding="utf-8")

    state["is_generating"] = False
    socketio.emit("generation_complete", {"subject": subject})


@socketio.on("load_subject")
def on_load_subject(data):
    subject = data.get("subject")
    if not subject:
        return

    if state["is_generating"]:
        state["stop_streaming"] = True
        time.sleep(0.1)

    state["current_subject"] = subject
    with state["text_lock"]:
        text = state["subject_texts"].get(subject, "")

    if text:
        emit("text_update", {"text": text, "subject": subject})
    else:
        on_generate_subject({
            "subject": subject,
            "lang": "en",
            "model": "gpt-4o-mini",
            "provider": None,
        })


@socketio.on("stop_generation")
def on_stop_generation():
    state["stop_streaming"] = True


if __name__ == "__main__":
    socketio.run(app, debug=True)
