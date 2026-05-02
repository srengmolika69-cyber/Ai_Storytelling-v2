import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Load .env when running locally
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import google.generativeai as genai
from groq import Groq

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GROQ_KEY   = os.getenv("GROQ_API_KEY", "")

if not TELEGRAM_TOKEN:
    logger.critical("TELEGRAM_TOKEN is not set!")
    sys.exit(1)

# ─── Persistent storage ───────────────────────────────────────────────────────
_DATA_DIR    = Path("/data") if Path("/data").exists() else Path(__file__).parent
KEYS_FILE    = _DATA_DIR / "user_keys.json"
HISTORY_FILE = _DATA_DIR / "story_history.json"

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

USER_KEYS:    dict = _load_json(KEYS_FILE)
STORY_HISTORY: dict = _load_json(HISTORY_FILE)

# Key helpers
def get_user_gemini_key(uid: int) -> str:
    return USER_KEYS.get(str(uid), {}).get("gemini", DEFAULT_GEMINI_KEY)

def get_user_groq_key(uid: int) -> str:
    return USER_KEYS.get(str(uid), {}).get("groq", DEFAULT_GROQ_KEY)

def get_user_provider(uid: int) -> str:
    stored = USER_KEYS.get(str(uid), {}).get("provider")
    if stored:
        return stored
    # Auto-detect: prefer groq if key exists, else gemini, else groq as placeholder
    if USER_KEYS.get(str(uid), {}).get("groq") or DEFAULT_GROQ_KEY:
        return "groq"
    if USER_KEYS.get(str(uid), {}).get("gemini") or DEFAULT_GEMINI_KEY:
        return "gemini"
    return "groq"

def get_user_model(uid: int) -> str:
    defaults = {"groq": "llama-3.3-70b-versatile", "gemini": "gemini-1.5-flash"}
    provider = get_user_provider(uid)
    return USER_KEYS.get(str(uid), {}).get("model", defaults.get(provider, "llama-3.3-70b-versatile"))

def set_user_data(uid: int, **kwargs) -> None:
    key = str(uid)
    if key not in USER_KEYS:
        USER_KEYS[key] = {}
    USER_KEYS[key].update(kwargs)
    _save_json(KEYS_FILE, USER_KEYS)

def delete_user_key(uid: int, provider: str) -> bool:
    key = str(uid)
    if key in USER_KEYS and provider in USER_KEYS[key]:
        del USER_KEYS[key][provider]
        _save_json(KEYS_FILE, USER_KEYS)
        return True
    return False

# History helpers
MAX_HISTORY = 20

def add_to_history(uid: int, genre: str, topic: str, story: str) -> None:
    key = str(uid)
    if key not in STORY_HISTORY:
        STORY_HISTORY[key] = []
    STORY_HISTORY[key].insert(0, {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "genre": genre,
        "topic": topic,
        "story": story[:500],  # store first 500 chars
    })
    STORY_HISTORY[key] = STORY_HISTORY[key][:MAX_HISTORY]
    _save_json(HISTORY_FILE, STORY_HISTORY)

def get_history(uid: int) -> list:
    return STORY_HISTORY.get(str(uid), [])

# ─── AI Models Config ─────────────────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "⚡ Llama 3.3 70B (Best)",
    "llama-3.1-8b-instant":    "🚀 Llama 3.1 8B (Fastest)",
    "llama3-70b-8192":         "🎯 Llama 3 70B",
    "gemma2-9b-it":            "💎 Gemma 2 9B",
}

GEMINI_MODELS = {
    "gemini-1.5-flash":   "⚡ Gemini 1.5 Flash (Fast)",
    "gemini-1.5-pro":     "🧠 Gemini 1.5 Pro (Best)",
    "gemini-2.0-flash":   "🚀 Gemini 2.0 Flash",
}

# ─── Conversation States ──────────────────────────────────────────────────────
(
    CHOOSING_GENRE,
    TYPING_TOPIC,
    READING_STORY,
    WAITING_FOR_KEY,
    CHOOSING_PROVIDER,
    CHOOSING_MODEL,
    VIEWING_HISTORY,
) = range(7)

WAITING_KEY_TYPE = "waiting_key_type"  # stored in context.user_data

# ─── Story Genres ─────────────────────────────────────────────────────────────
GENRES = {
    "folk":      ("🏮 រឿងនិទាន",     "Khmer folk tale / រឿងនិទានខ្មែរ"),
    "ghost":     ("👻 រឿងខ្មោច",     "Khmer ghost / horror story"),
    "love":      ("💕 រឿងស្នេហ៍",    "Khmer romantic love story"),
    "adventure": ("⚔️ រឿងផ្សងព្រេង", "Khmer adventure / hero story"),
    "fable":     ("🐘 រឿងសត្វ",      "Khmer animal fable with moral"),
    "legend":    ("🌟 រឿងព្រេង",     "Khmer legend / mythology"),
    "modern":    ("🏙️ រឿងទំនើប",    "Modern Khmer daily life story"),
    "children":  ("🌈 រឿងកុមារ",    "Khmer children bedtime story"),
    "comedy":    ("😄 រឿងកំប្លែង",  "Khmer comedy / funny story"),
    "mystery":   ("🔍 រឿងអាថ៌កំបាំង", "Khmer mystery / detective story"),
}

SYSTEM_PROMPT = """អ្នកជាអ្នកនិទានរឿងខ្មែរដ៏ពូកែ និងជំនាញ។
សូមបង្កើតរឿងខ្មែរដែលមានគុណភាពខ្ពស់ ដោយប្រើភាសាខ្មែរសុទ្ធ វប្បធម៌ខ្មែរ
និងរចនាប័ទ្មនិទានរឿងប្រពៃណីខ្មែរ។

ក្បួននិទានរឿង:
- ចាប់ផ្តើមរឿងដោយបែបទាក់ទាញ
- ប្រើភាសាខ្មែរស្អាត ងាយយល់
- បន្ថែមស្មារតីខ្មែរ ទំនៀមទម្លាប់ ឬជំនឿ
- រឿងគួរមានអំណានពី ១៥០-២៥០ ពាក្យ
- បញ្ចប់ដោយសាររឿង ឬអត្ថន័យស្រស់ស្អាត
- គ្រប់ "ថ្នាក់" ទាំងអស់ត្រូវសរសេរជាទម្រង់ paragraph
"""

# ─── Keyboards ────────────────────────────────────────────────────────────────
def genre_keyboard() -> InlineKeyboardMarkup:
    keys = []
    items = list(GENRES.items())
    for i in range(0, len(items), 2):
        row = [InlineKeyboardButton(items[i][1][0], callback_data=f"genre:{items[i][0]}")]
        if i + 1 < len(items):
            row.append(InlineKeyboardButton(items[i+1][1][0], callback_data=f"genre:{items[i+1][0]}"))
        keys.append(row)
    return InlineKeyboardMarkup(keys)

def action_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 រឿងថ្មីទៀត",  callback_data="action:new"),
            InlineKeyboardButton("📖 ប្រភេទផ្សេង", callback_data="action:genres"),
        ],
        [
            InlineKeyboardButton("📚 ប្រវត្តិ",    callback_data="action:history"),
            InlineKeyboardButton("🏠 ទំព័រដើម",    callback_data="action:home"),
        ],
    ])

def provider_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = get_user_provider(uid)
    groq_mark   = "✅ " if current == "groq"   else ""
    gemini_mark = "✅ " if current == "gemini" else ""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"{groq_mark}⚡ Groq (បញ្ចូល)", callback_data="prov:groq"),
            InlineKeyboardButton(f"{gemini_mark}🤖 Gemini",       callback_data="prov:gemini"),
        ],
        [InlineKeyboardButton("◀️ ត្រឡប់", callback_data="prov:back")],
    ])

def groq_model_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = get_user_model(uid)
    rows = []
    for m, label in GROQ_MODELS.items():
        mark = "✅ " if current == m else ""
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"model:{m}")])
    rows.append([InlineKeyboardButton("◀️ ត្រឡប់", callback_data="model:back")])
    return InlineKeyboardMarkup(rows)

def gemini_model_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = get_user_model(uid)
    rows = []
    for m, label in GEMINI_MODELS.items():
        mark = "✅ " if current == m else ""
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"model:{m}")])
    rows.append([InlineKeyboardButton("◀️ ត្រឡប់", callback_data="model:back")])
    return InlineKeyboardMarkup(rows)

def settings_keyboard(uid: int) -> InlineKeyboardMarkup:
    provider = get_user_provider(uid)
    model    = get_user_model(uid)
    groq_key   = get_user_groq_key(uid)
    gemini_key = get_user_gemini_key(uid)
    groq_status   = "✅" if groq_key   else "❌"
    gemini_status = "✅" if gemini_key else "❌"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🔀 AI Provider: {provider.title()}", callback_data="settings:provider")],
        [InlineKeyboardButton(f"🧠 Model: {model[:25]}",              callback_data="settings:model")],
        [
            InlineKeyboardButton(f"{groq_status} Groq Key",   callback_data="settings:set_groq"),
            InlineKeyboardButton(f"{gemini_status} Gemini Key", callback_data="settings:set_gemini"),
        ],
        [
            InlineKeyboardButton("🗑️ លុប Groq Key",   callback_data="settings:del_groq"),
            InlineKeyboardButton("🗑️ លុប Gemini Key", callback_data="settings:del_gemini"),
        ],
        [InlineKeyboardButton("◀️ ទំព័រដើម", callback_data="settings:home")],
    ])

# ─── AI Providers ─────────────────────────────────────────────────────────────
def _call_groq(api_key: str, model: str, prompt: str) -> str:
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.85,
    )
    return resp.choices[0].message.content

def _call_gemini(api_key: str, model: str, prompt: str) -> str:
    import google.generativeai as g
    g.configure(api_key=api_key)
    m = g.GenerativeModel(model)
    return m.generate_content(SYSTEM_PROMPT + "\n\n" + prompt).text

async def generate_story(uid: int, genre_desc: str, topic: str) -> tuple[str, str]:
    """Returns (story_text, provider_used)."""
    provider  = get_user_provider(uid)
    model     = get_user_model(uid)
    groq_key  = get_user_groq_key(uid)
    gemini_key = get_user_gemini_key(uid)

    prompt = (
        f"ប្រភេទរឿង: {genre_desc}\n"
        f"ប្រធានបទ: {topic}\n\n"
        f"សូមសរសេររឿងខ្មែរមួយពីប្រធានបទខាងលើ:"
    )

    loop = asyncio.get_running_loop()

    # Try primary provider first
    primary_fn   = _call_groq   if provider == "groq"   else _call_gemini
    primary_key  = groq_key     if provider == "groq"   else gemini_key
    fallback_fn  = _call_gemini if provider == "groq"   else _call_groq
    fallback_key = gemini_key   if provider == "groq"   else groq_key

    # Default models for fallback
    fallback_model = "gemini-1.5-flash" if provider == "groq" else "llama-3.3-70b-versatile"

    if primary_key:
        try:
            text = await loop.run_in_executor(None, primary_fn, primary_key, model, prompt)
            return text, provider
        except Exception as e:
            logger.warning(f"Primary ({provider}) failed: {e}. Trying fallback.")

    # Try fallback provider
    if fallback_key:
        try:
            text = await loop.run_in_executor(None, fallback_fn, fallback_key, fallback_model, prompt)
            fallback_name = "gemini" if provider == "groq" else "groq"
            return text, fallback_name
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")

    return "⚠️ *មិនមាន API Key!*\nសូមប្រើ /settings ដើម្បីដាក់ Groq ឬ Gemini Key។", "none"

# ─── /start ───────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    uid  = user.id
    has_key = get_user_groq_key(uid) or get_user_gemini_key(uid)

    if not has_key:
        await update.message.reply_text(
            f"🙏 សួស្តី *{user.first_name}*!\n\n"
            "🎭 *ស្វាគមន៍មកកាន់ Bot និទានរឿងខ្មែរ AI v2!*\n\n"
            "✨ *ថ្មី:* គាំទ្រ Groq AI (លឿនជាង 10x) + Gemini!\n\n"
            "⚡ *Groq API Key (FREE & FAST):*\n"
            "👉 [console.groq.com/keys](https://console.groq.com/keys)\n\n"
            "🤖 *Gemini API Key (FREE):*\n"
            "👉 [aistudio.google.com](https://aistudio.google.com/app/apikey)\n\n"
            "ប្រើ /settings ដើម្បីដាក់ key 🔑",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return ConversationHandler.END

    await update.message.reply_text(
        f"🙏 សួស្តី *{user.first_name}*!\n\n"
        "🎭 *Bot និទានរឿងខ្មែរ AI v2*\n\n"
        "សូមជ្រើសរើសប្រភេទរឿង ⬇️",
        parse_mode="Markdown",
        reply_markup=genre_keyboard(),
    )
    return CHOOSING_GENRE

# ─── /settings ────────────────────────────────────────────────────────────────
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    provider = get_user_provider(uid)
    model    = get_user_model(uid)
    await update.message.reply_text(
        f"⚙️ *ការកំណត់*\n\n"
        f"🔀 Provider: *{provider.title()}*\n"
        f"🧠 Model: `{model}`\n\n"
        "ជ្រើសរើសសកម្មភាព:",
        parse_mode="Markdown",
        reply_markup=settings_keyboard(uid),
    )
    # Stay in WAITING_FOR_KEY so the ConversationHandler keeps handling
    # settings:* callbacks and key-text messages that follow.
    return WAITING_FOR_KEY

async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    action = query.data.split(":", 1)[1]
    uid    = query.from_user.id

    if action == "provider":
        await query.edit_message_text(
            "🔀 *ជ្រើសរើស AI Provider:*\n\n"
            "⚡ *Groq* — លឿនណាស់ (< 1 វិនាទី), ឥតគិតថ្លៃ\n"
            "🤖 *Gemini* — Google AI, ឥតគិតថ្លៃ\n\n"
            "ប្រសិនបើ provider ដំបូងរអាក, bot នឹងប្តូរ provider ទីពីរដោយស្វ័យប្រវត្ត!",
            parse_mode="Markdown",
            reply_markup=provider_keyboard(uid),
        )
        return WAITING_FOR_KEY  # stay active so prov: callbacks are caught

    elif action == "model":
        provider = get_user_provider(uid)
        kb = groq_model_keyboard(uid) if provider == "groq" else gemini_model_keyboard(uid)
        await query.edit_message_text(
            f"🧠 *ជ្រើសរើស Model ({provider.title()}):*",
            parse_mode="Markdown",
            reply_markup=kb,
        )
        return CHOOSING_MODEL

    elif action == "set_groq":
        context.user_data[WAITING_KEY_TYPE] = "groq"
        await query.edit_message_text(
            "🔑 *ដាក់ Groq API Key*\n\n"
            "ទទួល key ឥតគិតថ្លៃ: [console.groq.com/keys](https://console.groq.com/keys)\n\n"
            "Key ចាប់ផ្តើមដោយ `gsk_...`\n\n"
            "⚠️ Bot នឹងលុប message ភ្លាម ដើម្បីសុវត្ថិភាព\n\n"
            "/cancel ដើម្បីបោះបង់",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return WAITING_FOR_KEY

    elif action == "set_gemini":
        context.user_data[WAITING_KEY_TYPE] = "gemini"
        await query.edit_message_text(
            "🔑 *ដាក់ Gemini API Key*\n\n"
            "ទទួល key ឥតគិតថ្លៃ: [aistudio.google.com](https://aistudio.google.com/app/apikey)\n\n"
            "Key ចាប់ផ្តើមដោយ `AIzaSy...`\n\n"
            "⚠️ Bot នឹងលុប message ភ្លាម ដើម្បីសុវត្ថិភាព\n\n"
            "/cancel ដើម្បីបោះបង់",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return WAITING_FOR_KEY

    elif action == "del_groq":
        deleted = delete_user_key(uid, "groq")
        msg = "🗑️ *Groq Key ត្រូវបានលុប!*\n\n" if deleted else "⚠️ គ្មាន Groq Key ដើម្បីលុប។\n\n"
        provider = get_user_provider(uid)
        model    = get_user_model(uid)
        await query.edit_message_text(
            msg + f"🔀 Provider: *{provider.title()}*\n🧠 Model: `{model}`\n\nជ្រើសរើសសកម្មភាព:",
            parse_mode="Markdown",
            reply_markup=settings_keyboard(uid),
        )
        return WAITING_FOR_KEY

    elif action == "del_gemini":
        deleted = delete_user_key(uid, "gemini")
        msg = "🗑️ *Gemini Key ត្រូវបានលុប!*\n\n" if deleted else "⚠️ គ្មាន Gemini Key ដើម្បីលុប។\n\n"
        provider = get_user_provider(uid)
        model    = get_user_model(uid)
        await query.edit_message_text(
            msg + f"🔀 Provider: *{provider.title()}*\n🧠 Model: `{model}`\n\nជ្រើសរើសសកម្មភាព:",
            parse_mode="Markdown",
            reply_markup=settings_keyboard(uid),
        )
        return WAITING_FOR_KEY

    elif action == "home":
        await query.edit_message_text(
            "🏠 *ទំព័រដើម*\n\n"
            "• /story — បង្កើតរឿង\n"
            "• /settings — ការកំណត់ / Keys\n"
            "• /history — ប្រវត្តិរឿង\n"
            "• /help — ជំនួយ",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    return ConversationHandler.END

# ─── Provider selection callback ─────────────────────────────────────────────
async def provider_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    choice = query.data.split(":", 1)[1]
    uid    = query.from_user.id

    if choice == "back":
        provider = get_user_provider(uid)
        model    = get_user_model(uid)
        await query.edit_message_text(
            f"⚙️ *ការកំណត់*\n\n"
            f"🔀 Provider: *{provider.title()}*\n"
            f"🧠 Model: `{model}`\n\n"
            "ជ្រើសរើសសកម្មភាព:",
            parse_mode="Markdown",
            reply_markup=settings_keyboard(uid),
        )
        return WAITING_FOR_KEY

    set_user_data(uid, provider=choice)
    # Reset to default model for the new provider
    default_model = "llama-3.3-70b-versatile" if choice == "groq" else "gemini-1.5-flash"
    set_user_data(uid, model=default_model)

    await query.edit_message_text(
        f"✅ *ប្ដូរទៅ {choice.title()} រួចហើយ!*\n"
        f"🧠 Model: `{default_model}`\n\n"
        "ប្រើ /story ដើម្បីចាប់ផ្តើម ឬ /settings ដើម្បីប្តូរ model។",
        parse_mode="Markdown",
    )
    return ConversationHandler.END

# ─── Model selection callback ─────────────────────────────────────────────────
async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    choice = query.data.split(":", 1)[1]
    uid    = query.from_user.id

    if choice == "back":
        provider = get_user_provider(uid)
        model    = get_user_model(uid)
        await query.edit_message_text(
            f"⚙️ *ការកំណត់*\n\n"
            f"🔀 Provider: *{provider.title()}*\n"
            f"🧠 Model: `{model}`\n\n"
            "ជ្រើសរើសសកម្មភាព:",
            parse_mode="Markdown",
            reply_markup=settings_keyboard(uid),
        )
        return WAITING_FOR_KEY

    set_user_data(uid, model=choice)
    all_models = {**GROQ_MODELS, **GEMINI_MODELS}
    label = all_models.get(choice, choice)

    await query.edit_message_text(
        f"✅ *ប្ដូរ Model: {label}*\n\nប្រើ /story ដើម្បីចាប់ផ្តើម!",
        parse_mode="Markdown",
    )
    return ConversationHandler.END

# ─── Receive API key ──────────────────────────────────────────────────────────
async def receive_key(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    key      = update.message.text.strip()
    uid      = update.effective_user.id
    key_type = context.user_data.get(WAITING_KEY_TYPE, "groq")

    try:
        await update.message.delete()
    except Exception:
        pass

    # Validate format
    if key_type == "groq" and (not key.startswith("gsk_") or len(key) < 20):
        await update.message.reply_text(
            "❌ *Groq Key មិនត្រឹមត្រូវ!*\n\nKey ត្រូវចាប់ផ្តើមដោយ `gsk_`\n"
            "ព្យាយាមម្តងទៀត ឬ /cancel",
            parse_mode="Markdown",
        )
        return WAITING_FOR_KEY

    if key_type == "gemini" and (not key.startswith("AIza") or len(key) < 30):
        await update.message.reply_text(
            "❌ *Gemini Key មិនត្រឹមត្រូវ!*\n\nKey ត្រូវចាប់ផ្តើមដោយ `AIzaSy`\n"
            "ព្យាយាមម្តងទៀត ឬ /cancel",
            parse_mode="Markdown",
        )
        return WAITING_FOR_KEY

    validating_msg = await update.message.reply_text(
        f"⏳ *កំពុងត្រួតពិនិត្យ {key_type.title()} Key...*", parse_mode="Markdown"
    )

    # Validate against live API
    loop = asyncio.get_running_loop()
    try:
        if key_type == "groq":
            await loop.run_in_executor(
                None, _call_groq, key, "llama-3.1-8b-instant", "Say 'ok' in one word."
            )
        else:
            await loop.run_in_executor(
                None, _call_gemini, key, "gemini-1.5-flash", "Say 'ok' in one word."
            )

        set_user_data(uid, **{key_type: key})
        # Auto-set provider to the one they just added
        set_user_data(uid, provider=key_type)

        await validating_msg.edit_text(
            f"✅ *{key_type.title()} Key ត្រឹមត្រូវ! រក្សាទុករួចហើយ।*\n\n"
            f"🔀 Provider ប្ដូរទៅ *{key_type.title()}* ដោយស្វ័យប្រវត្ត\n\n"
            "ប្រើ /story ដើម្បីចាប់ផ្តើម 🎭",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    except Exception as e:
        err = str(e).lower()
        logger.warning(f"Key validation error ({key_type}): {e}")
        if any(w in err for w in ["api_key", "permission", "invalid", "credential", "auth", "unauthorized"]):
            await validating_msg.edit_text(
                f"❌ *{key_type.title()} Key ខុស!*\n\nសូមពិនិត្យ key ម្តងទៀត ឬ /cancel",
                parse_mode="Markdown",
            )
            return WAITING_FOR_KEY

        # Network/unknown — save anyway
        set_user_data(uid, **{key_type: key})
        set_user_data(uid, provider=key_type)
        await validating_msg.edit_text(
            f"⚠️ *មិនអាចត្រួតពិនិត្យបាន — {key_type.title()} Key ត្រូវបានរក្សាទុក।*\n\n"
            "ប្រើ /story ដើម្បីសាកល្បង។",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

# ─── /history ─────────────────────────────────────────────────────────────────
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid     = update.effective_user.id
    history = get_history(uid)

    if not history:
        await update.message.reply_text(
            "📚 *គ្មានប្រវត្តិរឿង*\n\nប្រើ /story ដើម្បីបង្កើតរឿងដំបូង!",
            parse_mode="Markdown",
        )
        return

    lines = ["📚 *ប្រវត្តិរឿង (ចុងក្រោយ)*\n"]
    for i, h in enumerate(history[:10], 1):
        lines.append(f"*{i}.* {h['genre']} — {h['topic']}\n   🕒 {h['date']}\n")

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
    )

# ─── Story flow ───────────────────────────────────────────────────────────────
async def story_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    if not (get_user_groq_key(uid) or get_user_gemini_key(uid)):
        await update.message.reply_text(
            "⚠️ *អ្នកមិនទាន់មាន API Key ទេ!*\n\n"
            "ប្រើ /settings ដើម្បីដាក់ Groq ឬ Gemini API Key\n\n"
            "⚡ *Groq (FREE & FAST):* [console.groq.com/keys](https://console.groq.com/keys)\n"
            "🤖 *Gemini (FREE):* [aistudio.google.com](https://aistudio.google.com/app/apikey)",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "📖 *ជ្រើសរើសប្រភេទរឿង:*",
        parse_mode="Markdown",
        reply_markup=genre_keyboard(),
    )
    return CHOOSING_GENRE

async def genre_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    genre_key = query.data.split(":")[1]
    genre_label, genre_desc = GENRES[genre_key]
    context.user_data["genre_label"] = genre_label
    context.user_data["genre_desc"]  = genre_desc

    await query.edit_message_text(
        f"✅ ប្រភេទ: *{genre_label}*\n\n"
        "✍️ សូមវាយ *ប្រធានបទ* ឬ *ឈ្មោះតួអង្គ*:\n\n"
        "_ឧទាហរណ៍: កញ្ញាក្នុងព្រៃ, ក្មេងជិតទន្លេ, ស្នេហ៍ក្នុងភ្នំ..._",
        parse_mode="Markdown",
    )
    return TYPING_TOPIC

async def topic_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    topic       = update.message.text.strip()
    uid         = update.effective_user.id
    genre_label = context.user_data.get("genre_label", "")
    genre_desc  = context.user_data.get("genre_desc",  "")
    context.user_data["topic"] = topic

    provider  = get_user_provider(uid)
    model     = get_user_model(uid)
    prov_icon = "⚡" if provider == "groq" else "🤖"

    thinking_msg = await update.message.reply_text(
        f"🪄 *AI កំពុងនិទានរឿង...*\n\n"
        f"📖 ប្រភេទ: {genre_label}\n"
        f"🏷️ ប្រធានបទ: {topic}\n"
        f"{prov_icon} {provider.title()} · `{model}`\n\n"
        "_សូមរង់ចាំ..._",
        parse_mode="Markdown",
    )

    story, used_provider = await generate_story(uid, genre_desc, topic)
    await thinking_msg.delete()

    # If both providers failed, show a clean error (don't wrap error in story template)
    if used_provider == "none":
        await update.message.reply_text(
            "❌ *មិនអាចបង្កើតរឿងបាន!*\n\n"
            "សូមពិនិត្យ API Key:\n"
            "• ប្រើ /settings → ដាក់ Groq ឬ Gemini Key\n\n"
            "⚡ Groq FREE: [console.groq.com/keys](https://console.groq.com/keys)\n"
            "🤖 Gemini FREE: [aistudio.google.com](https://aistudio.google.com/app/apikey)",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return ConversationHandler.END

    prov_badge    = "⚡ Groq" if used_provider == "groq" else "🤖 Gemini"
    fallback_note = " _(fallback)_" if used_provider != provider else ""

    add_to_history(uid, genre_label, topic, story)

    await update.message.reply_text(
        f"📜 *រឿង: {topic}*\n"
        f"🏷️ {genre_label}  {prov_badge}{fallback_note}\n"
        f"{'─'*28}\n\n"
        f"{story}\n\n"
        f"{'─'*28}\n"
        f"_✨ បង្កើតដោយ AI Khmer Storyteller v2_",
        parse_mode="Markdown",
        reply_markup=action_keyboard(),
    )
    return READING_STORY

async def action_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    action = query.data.split(":")[1]
    uid    = query.from_user.id

    if action == "new":
        genre_label = context.user_data.get("genre_label", "")
        genre_desc  = context.user_data.get("genre_desc", "")
        topic       = context.user_data.get("topic", "")
        provider    = get_user_provider(uid)
        prov_icon   = "⚡" if provider == "groq" else "🤖"

        thinking_msg = await query.message.reply_text(
            f"🪄 *AI កំពុងបង្កើតរឿងថ្មី...*\n{prov_icon} {provider.title()}",
            parse_mode="Markdown",
        )
        story, used_provider = await generate_story(uid, genre_desc, topic)
        await thinking_msg.delete()

        if used_provider == "none":
            await query.message.reply_text(
                "❌ *មិនអាចបង្កើតរឿងបាន!*\n\nប្រើ /settings → ពិនិត្យ API Key។",
                parse_mode="Markdown",
            )
            return READING_STORY

        add_to_history(uid, genre_label, topic, story)
        prov_badge    = "⚡ Groq" if used_provider == "groq" else "🤖 Gemini"
        fallback_note = " _(fallback)_" if used_provider != provider else ""

        await query.message.reply_text(
            f"📜 *រឿងថ្មី: {topic}*\n"
            f"🏷️ {genre_label}  {prov_badge}{fallback_note}\n"
            f"{'─'*28}\n\n"
            f"{story}\n\n"
            f"{'─'*28}\n"
            f"_✨ បង្កើតដោយ AI Khmer Storyteller v2_",
            parse_mode="Markdown",
            reply_markup=action_keyboard(),
        )
        return READING_STORY

    elif action == "genres":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "📖 *ជ្រើសរើសប្រភេទរឿង:*",
            parse_mode="Markdown",
            reply_markup=genre_keyboard(),
        )
        return CHOOSING_GENRE

    elif action == "history":
        history = get_history(uid)
        if not history:
            await query.message.reply_text(
                "📚 *គ្មានប្រវត្តិ*\nសូមបង្កើតរឿងបន្ថែម!",
                parse_mode="Markdown",
            )
        else:
            lines = ["📚 *ប្រវត្តិរឿង (ចុងក្រោយ ១០)*\n"]
            for i, h in enumerate(history[:10], 1):
                lines.append(f"*{i}.* {h['genre']} — {h['topic']}\n   🕒 {h['date']}\n")
            await query.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return READING_STORY

    elif action == "home":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "🏠 *ទំព័រដើម*\n\n"
            "• /story — បង្កើតរឿង\n"
            "• /settings — ការកំណត់ / Keys\n"
            "• /history — ប្រវត្តិរឿង\n"
            "• /help — ជំនួយ",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    return READING_STORY

# ─── /help & /cancel ──────────────────────────────────────────────────────────
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📚 *របៀបប្រើ Bot v2*\n\n"
        "*ជំហាន:*\n"
        "1️⃣ ទទួល API Key ឥតគិតថ្លៃ:\n"
        "   ⚡ *Groq (លឿណាស់):* [console.groq.com/keys](https://console.groq.com/keys)\n"
        "   🤖 *Gemini:* [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
        "2️⃣ /settings → ដាក់ key\n"
        "3️⃣ /story → ជ្រើសប្រភេទ → វាយប្រធានបទ\n"
        "4️⃣ ទទួលរឿង 🎉\n\n"
        "*ពាក្យបញ្ជា:*\n"
        "/start — ចាប់ផ្តើម\n"
        "/story — បង្កើតរឿង\n"
        "/settings — ការកំណត់ / Keys / Model\n"
        "/history — ប្រវត្តិរឿង\n"
        "/help — ជំនួយ\n"
        "/cancel — បោះបង់\n\n"
        "*✨ ថ្មីក្នុង v2:*\n"
        "⚡ Groq AI (លឿនជាង 10x)\n"
        "🔄 Auto-fallback (ប្ដូរ provider ស្វ័យប្រវត្ត)\n"
        "🧠 ជ្រើស model ដែលចូលចិត្ត\n"
        "📚 ប្រវត្តិរឿង (ចំងាយ ២០ រឿង)\n"
        "😄 Genre ថ្មី: Comedy + Mystery",
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 បានបោះបង់! ប្រើ /start ឬ /story ដើម្បីបន្ត។")
    return ConversationHandler.END

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "❓ ខ្ញុំមិនយល់ពាក្យបញ្ជានេះ។\nប្រើ /help ដើម្បីមើលការណែនាំ។"
    )

# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Settings conversation (handles key input)
    settings_conv = ConversationHandler(
        entry_points=[
            CommandHandler("settings", settings_command),
            CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
        ],
        states={
            WAITING_FOR_KEY: [
                # Allow /settings and /cancel to escape the waiting state
                CommandHandler("settings", settings_command),
                CommandHandler("cancel",   cancel),
                # Handle settings button callbacks while waiting (e.g. user switches action)
                CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
                CallbackQueryHandler(model_callback,    pattern=r"^model:"),
                CallbackQueryHandler(provider_callback, pattern=r"^prov:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_key),
            ],
            CHOOSING_MODEL: [
                CommandHandler("settings", settings_command),
                CallbackQueryHandler(model_callback,    pattern=r"^model:"),
                CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
            ],
        },
        fallbacks=[
            CommandHandler("cancel",   cancel),
            CommandHandler("settings", settings_command),
            CommandHandler("start",    start),
        ],
        allow_reentry=True,
    )

    # Story conversation
    story_conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("story", story_command),
        ],
        states={
            CHOOSING_GENRE: [
                CommandHandler("settings", settings_command),
                CallbackQueryHandler(genre_chosen, pattern=r"^genre:"),
            ],
            TYPING_TOPIC: [
                CommandHandler("settings", settings_command),
                MessageHandler(filters.TEXT & ~filters.COMMAND, topic_received),
            ],
            READING_STORY: [
                CommandHandler("settings", settings_command),
                CallbackQueryHandler(action_handler, pattern=r"^action:"),
            ],
        },
        fallbacks=[
            CommandHandler("cancel",   cancel),
            CommandHandler("settings", settings_command),
            CommandHandler("start",    start),
        ],
        allow_reentry=True,
    )

    app.add_handler(settings_conv)
    app.add_handler(story_conv)
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("help",    help_command))
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    logger.info("Khmer Storytelling Bot v2 started! (Groq + Gemini)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
