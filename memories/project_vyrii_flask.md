---
name: project-vyrii-flask
description: "vyrii live repo is C:\\Project\\vyrii — default mode is Flask (pure Python, no Rust); gradio/fastapi are optional extras; Android APK path via Chaquopy"
metadata: 
  node_type: memory
  type: project
  originSessionId: ff23801d-2489-455b-8556-235ea88360dd
---

# vyrii — Flask за замовчуванням, жива копія окремо від сабмодуля

- **Жива копія: `C:\Project\vyrii`**. Загальне правило: ВСІ інструменти (1bcoder, vyrii, yasna, radogast, svitovyd, simargl) живуть у `C:\Project\` як окремі репозиторії; сабмодулі в simrgl — застарілі знімки. Шляху `C:\Projects\` (з "s") не існує — користувач іноді називає теку так по пам'яті.
- **Дефолтний запуск `vyrii` = Flask** (порт 5000): `flask_api.py` (повний паритет з api.py, OpenAI-сумісні + /vyrii/* ендпоінти, basic auth) + статичний HTML/JS UI у `vyrii/ui/`. Коміти: `5a5a96f` "create flask api to be able to run vyrii without rust", `1e16ad1` "prod server, fix auth" (waitress).
- **Базові залежності (v0.1.8) — чистий Python:** requests, flask, flask-cors, apscheduler, waitress. Gradio (порт 4896) і fastapi — опціональні extras: `vyrii[gradio]`, `vyrii[api]`, `vyrii[full]`.

## Android-план (обговорено 2026-06-12)
Завдяки чистому Python варіант "vyrii в APK" реальний і Play-легальний:
- Chaquopy (Python у звичайному Android-додатку) + `pip install vyrii` — без wheel-битв;
- llama-server (llama.cpp, NDK-збірка) у jniLibs як `lib*.so` — exec з nativeLibraryDir дозволений (read-only тека); ollama НЕ потрібен — llama-server має OpenAI-сумісний API, vyrii вміє openai:// хости;
- WebView → localhost:5000 (той самий HTML UI);
- нюанси: HOME → filesDir перед імпортом; lxml optional (Chaquopy має wheel); 16KB page alignment (NDK r27+) для Play.
- Оцінка: feasibility 1–2 вечори, робочий APK 1–2 тижні.
- Альтернативи без APK: bootstrap-скрипт у Termux (`pkg install ollama` є в офіційному репо з 2025); Google Play для termux-шляху закритий (W^X, заборона завантаження виконуваного коду).

Пов'язано: [[project-external-supervision-concept]] (vyrii = черги HITL + нічна консолідація через apscheduler — може жити і на телефоні).
