document.addEventListener('DOMContentLoaded', () => {

    const WIKIX_ASCII_ART = [
        "W   W   III   K  K   III   X   X", "W   W    I    K K     I     X X ",
        "W W W    I    KK      I      X  ", "WW WW    I    K K     I     X X ",
        "W   W   III   K  K   III   X   X",
    ];

    // --- DOM Elements ---
    const dom = {
        app: document.getElementById('app-container'),
        statusBar: document.getElementById('status-bar'),
        logo: document.getElementById('wikix-logo'),
        subject: document.getElementById('current-subject'),
        textDisplay: document.getElementById('text-display'),
        historyPanel: document.getElementById('history-panel'),
        historyList: document.getElementById('history-list'),
        inputArea: document.getElementById('input-area'),
        inputPrompt: document.getElementById('input-prompt'),
        subjectInput: document.getElementById('subject-input'),
    };

    // --- State ---
    const state = {
        currentSubject: 'Wikipédia',
        isGenerating: false,
        selectionMode: false,
        historyMode: false,
        inputMode: false,
        history: [],
        historyCursor: 0,
        words: [],
        cursorWordIndex: 0,
        selectionStart: -1,
        // Config from server
        config: {
            languages: {}, themes: {}, available_models: {}, provider_models: {}
        },
        // User choices
        ui: {
            theme: 'dark',
            language: 'en',
            model: 'gpt-4o-mini',
            provider: 'auto',
        }
    };

    // --- Socket.IO Connection ---
    const socket = io();

    // Intercepter 'i' en capture pour ouvrir la saisie sans insérer le caractère
    window.addEventListener('keydown', (e) => {
        const key = (e.key || '').toLowerCase();
        if (key === 'i' && !state.inputMode) {
            e.preventDefault();
            e.stopPropagation();
            actions.toggleInput(true);
            // S'assurer que l'input est vide et focus après le repaint
            setTimeout(() => {
                if (dom.subjectInput) {
                    dom.subjectInput.value = '';
                    dom.subjectInput.focus();
                }
            }, 0);
        }
    }, true);

    // --- Rendering ---
    const render = {
        all() {
            this.header();
            this.statusBar();
            this.textContent();
            this.history();
            this.input();
        },
        header() {
            dom.logo.textContent = WIKIX_ASCII_ART.join('\n');
            dom.subject.textContent = state.currentSubject.toUpperCase();
        },
        statusBar() {
            const modelInfo = state.config.available_models[state.ui.model] || { name: 'N/A' };
            const langInfo = state.config.languages[state.ui.language] || { flag: 'N/A' };
            const themeInfo = state.config.themes[state.ui.theme] || { name: 'N/A' };

            let status = `↑↓←→: Navigate | Enter: Generate | Space: Select | r: Regenerate | i: Input | h: History`;
            if (state.isGenerating) status = `🔄 Generating... | s: Stop`;
            if (state.selectionMode) status = `📝 Selecting... | Enter: Generate | Esc: Cancel`;
            if (state.historyMode) status = `🗂️ History | ↑↓/Tab: Navigate | Enter: Open | Esc/h: Close`;
            if (state.inputMode) status = `✏️ Input... | Enter: Submit | Esc: Cancel`;

            dom.statusBar.textContent = [
                status, `🎨 ${themeInfo.name}`, `🗣️ ${langInfo.flag}`,
                `🤖 ${modelInfo.name}`, `☁️ ${state.ui.provider}`
            ].join(' | ');
        },
        textContent() {
            // Basic text rendering
            dom.textDisplay.textContent = state.currentText;
            // More complex rendering with spans for words would go here
        },
        history() {
            dom.historyList.innerHTML = '';
            state.history.forEach((item, index) => {
                const li = document.createElement('li');
                li.textContent = item;
                if (item === state.currentSubject) li.classList.add('current');
                if (state.historyMode && index === state.historyCursor) li.classList.add('selected');
                li.onclick = () => {
                    socket.emit('load_subject', { subject: item });
                    state.historyMode = false;
                    render.history();
                };
                dom.historyList.appendChild(li);
            });
            dom.historyPanel.classList.toggle('hidden', !state.historyMode);
        },
        input() {
            dom.inputArea.classList.toggle('hidden', !state.inputMode);
            if (state.inputMode) dom.subjectInput.focus();
        },
        theme() {
            document.body.className = `theme-${state.ui.theme}`;
        }
    };

    // --- Actions ---
    const actions = {
        cycle(key, list) {
            const currentIndex = list.indexOf(state.ui[key]);
            const nextIndex = (currentIndex + 1) % list.length;
            state.ui[key] = list[nextIndex];
            render.statusBar();
        },
        generate(subject) {
            state.currentSubject = subject;
            // Pass all UI settings to the backend
            socket.emit('generate_subject', { subject, ...state.ui });
        },
        toggleInput(force = !state.inputMode) {
            state.inputMode = force;
            if (state.inputMode) {
                state.historyMode = false;
                dom.subjectInput.value = '';
            }
            render.all();
        },
        toggleHistory(force = !state.historyMode) {
            state.historyMode = force;
            if (state.historyMode) state.inputMode = false;
            render.all();
        },
        toggleTheme() {
            this.cycle('theme', Object.keys(state.config.themes));
            render.theme();
        }
    };

    // --- Keydown Handler ---
    document.addEventListener('keydown', (e) => {
        if (state.inputMode) {
            if (e.key === 'Enter') {
                e.preventDefault();
                const subject = dom.subjectInput.value.trim();
                if (subject) actions.generate(subject);
                actions.toggleInput(false);
            } else if (e.key === 'Escape') {
                e.preventDefault();
                actions.toggleInput(false);
            }
            return;
        }

        if (state.historyMode) {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                state.historyCursor = Math.max(0, state.historyCursor - 1);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                state.historyCursor = Math.min(state.history.length - 1, state.historyCursor + 1);
            } else if (e.key === 'Tab') {
                e.preventDefault();
                if (state.history.length > 0) {
                    if (e.shiftKey) {
                        state.historyCursor = (state.historyCursor - 1 + state.history.length) % state.history.length;
                    } else {
                        state.historyCursor = (state.historyCursor + 1) % state.history.length;
                    }
                }
            } else if (e.key === 'Enter') {
                e.preventDefault();
                const subject = state.history[state.historyCursor];
                socket.emit('load_subject', { subject });
                state.historyMode = false;
            } else if (e.key === 'Escape' || e.key.toLowerCase() === 'h') {
                e.preventDefault();
                state.historyMode = false;
            }
            render.history();
            return;
        }

        switch (e.key.toLowerCase()) {
            case 'i': actions.toggleInput(); break;
            case 'h': actions.toggleHistory(); break;
            case 's': if (state.isGenerating) socket.emit('stop_generation'); break;
            case 'r': actions.generate(state.currentSubject); break;
            case 't': actions.toggleTheme(); break;
            case 'l': actions.cycle('language', Object.keys(state.config.languages)); break;
            case 'm':
                const providerModels = state.config.provider_models[state.ui.provider] || [];
                if (providerModels.length > 0) actions.cycle('model', providerModels);
                break;
            case 'p':
                actions.cycle('provider', Object.keys(state.config.provider_models));
                const newProviderModels = state.config.provider_models[state.ui.provider] || [];
                if (!newProviderModels.includes(state.ui.model)) {
                    state.ui.model = newProviderModels[0] || '';
                }
                render.statusBar();
                break;
        }
    });

    // --- Socket Event Listeners ---
    socket.on('initial_data', (data) => {
        state.currentSubject = data.subject;
        state.currentText = data.text;
        state.history = data.history;
        if (!data.text) actions.generate(data.subject);
        render.all();
    });

    socket.on('history_update', (data) => {
        state.history = data.history;
        render.history();
    });

    socket.on('text_update', (data) => {
        state.isGenerating = true;
        state.currentText = data.text;
        if (data.subject) state.currentSubject = data.subject; // Update subject if provided
        render.all();
    });

    socket.on('generation_complete', (data) => {
        state.isGenerating = false;
        if (data.subject) state.currentSubject = data.subject;
        render.all();
    });

    // --- Init ---
    async function init() {
        try {
            const response = await fetch('/api/config');
            state.config = await response.json();
            render.theme();
            render.all();
        } catch (e) {
            console.error("Failed to load config from server", e);
            dom.statusBar.textContent = "Server connection error.";
        }
    }

    init();
});

