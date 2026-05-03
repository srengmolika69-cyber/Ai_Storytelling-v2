import os
import sys
import json
import logging
import asyncio
import random
from pathlib import Path
from datetime import datetime

# Load .env when running locally
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import google.generativeai as genai

# ─── httpx compatibility patch ────────────────────────────────────────────────
# groq < 1.0.0 passes `proxies` to httpx.Client, removed in httpx 0.28.
try:
    import httpx
    _orig_client_init = httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        _orig_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_client_init

    _orig_async_init = httpx.AsyncClient.__init__
    def _patched_async_init(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        _orig_async_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = _patched_async_init
except Exception:
    pass

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

USER_KEYS:     dict = _load_json(KEYS_FILE)
STORY_HISTORY: dict = _load_json(HISTORY_FILE)

# ─── User data helpers ────────────────────────────────────────────────────────
def get_user_gemini_key(uid: int) -> str:
    return USER_KEYS.get(str(uid), {}).get("gemini", DEFAULT_GEMINI_KEY)

def get_user_groq_key(uid: int) -> str:
    return USER_KEYS.get(str(uid), {}).get("groq", DEFAULT_GROQ_KEY)

def get_user_provider(uid: int) -> str:
    stored = USER_KEYS.get(str(uid), {}).get("provider")
    if stored:
        return stored
    if USER_KEYS.get(str(uid), {}).get("groq") or DEFAULT_GROQ_KEY:
        return "groq"
    if USER_KEYS.get(str(uid), {}).get("gemini") or DEFAULT_GEMINI_KEY:
        return "gemini"
    return "groq"

def get_user_model(uid: int) -> str:
    defaults = {"groq": "llama-3.3-70b-versatile", "gemini": "gemini-2.5-flash"}
    provider = get_user_provider(uid)
    return USER_KEYS.get(str(uid), {}).get("model", defaults.get(provider, "llama-3.3-70b-versatile"))

def get_user_length(uid: int) -> str:
    return USER_KEYS.get(str(uid), {}).get("length", "medium")

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

# ─── History helpers ──────────────────────────────────────────────────────────
MAX_HISTORY = 30

def add_to_history(uid: int, genre: str, topic: str, story: str, length: str) -> None:
    key = str(uid)
    if key not in STORY_HISTORY:
        STORY_HISTORY[key] = []
    STORY_HISTORY[key].insert(0, {
        "date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "genre":  genre,
        "topic":  topic,
        "length": length,
        "story":  story,
        "rating": 0,
    })
    STORY_HISTORY[key] = STORY_HISTORY[key][:MAX_HISTORY]
    _save_json(HISTORY_FILE, STORY_HISTORY)

def rate_story(uid: int, index: int, stars: int) -> bool:
    key = str(uid)
    history = STORY_HISTORY.get(key, [])
    if 0 <= index < len(history):
        history[index]["rating"] = stars
        _save_json(HISTORY_FILE, STORY_HISTORY)
        return True
    return False

def get_history(uid: int) -> list:
    return STORY_HISTORY.get(str(uid), [])

def get_user_stats(uid: int) -> dict:
    history = get_history(uid)
    if not history:
        return {}
    total   = len(history)
    rated   = [h for h in history if h.get("rating", 0) > 0]
    avg_rating = round(sum(h["rating"] for h in rated) / len(rated), 1) if rated else 0
    genre_count: dict = {}
    for h in history:
        g = h.get("genre", "")
        genre_count[g] = genre_count.get(g, 0) + 1
    top_genre = max(genre_count, key=genre_count.get) if genre_count else "—"
    length_count: dict = {}
    for h in history:
        l = h.get("length", "medium")
        length_count[l] = length_count.get(l, 0) + 1
    top_length = max(length_count, key=length_count.get) if length_count else "medium"
    return {
        "total":      total,
        "rated":      len(rated),
        "avg_rating": avg_rating,
        "top_genre":  top_genre,
        "top_length": top_length,
        "genre_count": genre_count,
    }

# ─── AI Models ────────────────────────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "⚡ Llama 3.3 70B (Best)",
    "llama-3.1-8b-instant":    "🚀 Llama 3.1 8B (Fastest)",
    "llama3-70b-8192":         "🎯 Llama 3 70B",
    "gemma2-9b-it":            "💎 Gemma 2 9B",
}

GEMINI_MODELS = {
    "gemini-2.5-flash":      "⚡ Gemini 2.5 Flash (Fast)",
    "gemini-2.5-pro":        "🧠 Gemini 2.5 Pro (Best)",
    "gemini-2.5-flash-lite": "🚀 Gemini 2.5 Flash-Lite (Cheapest)",
}

LENGTH_LABELS = {
    "short":  "📄 ខ្លី (~១៥០ ពាក្យ)",
    "medium": "📃 មធ្យម (~២៥០ ពាក្យ)",
    "long":   "📜 វែង (~៤០០ ពាក្យ)",
}
LENGTH_WORDS  = {"short": "150", "medium": "250", "long": "400"}
LENGTH_ICONS  = {"short": "📄", "medium": "📃", "long": "📜"}

# ─── Conversation States ──────────────────────────────────────────────────────
(
    CHOOSING_GENRE,
    TYPING_TOPIC,
    READING_STORY,
    WAITING_FOR_KEY,
    CHOOSING_PROVIDER,
    CHOOSING_MODEL,
    VIEWING_HISTORY,
    CHOOSING_LENGTH,
    RATING_STORY,
    CONTINUING_STORY,
) = range(10)

WAITING_KEY_TYPE = "waiting_key_type"

# ─── Genres ───────────────────────────────────────────────────────────────────
GENRES = {
    "folk":      ("🏮 រឿងនិទាន",      "Khmer folk tale / រឿងនិទានខ្មែរប្រពៃណី"),
    "ghost":     ("👻 រឿងខ្មោច",      "Khmer ghost / horror / supernatural story"),
    "love":      ("💕 រឿងស្នេហ៍",     "Khmer romantic love story"),
    "adventure": ("⚔️ រឿងផ្សងព្រេង",  "Khmer adventure / hero / warrior story"),
    "fable":     ("🐘 រឿងសត្វ",       "Khmer animal fable with a moral lesson"),
    "legend":    ("🌟 រឿងព្រេង",      "Khmer ancient legend / mythology / Angkor era"),
    "modern":    ("🏙️ រឿងទំនើប",     "Modern Khmer city / daily life / social story"),
    "children":  ("🌈 រឿងកុមារ",     "Khmer children bedtime / educational story"),
    "comedy":    ("😄 រឿងកំប្លែង",   "Khmer comedy / witty / funny story"),
    "mystery":   ("🔍 រឿងអាថ៌កំបាំង","Khmer mystery / detective / thriller story"),
    "wisdom":    ("🧘 រឿងប្រាជ្ញា",  "Khmer wisdom / Buddhist moral / life lesson story"),
    "horror":    ("😱 រឿងភ័យខ្លាច",  "Dark Khmer horror / suspense story for adults"),
}

GENRE_HINTS = {
    "folk":      "ចាប់ផ្ដើមដោយ 'កាលពីបុរាណ...' ឬ 'នៅជំនាន់មួយ...' ប្រើសំដីនិទានជំនាន់ដើម",
    "ghost":     "បង្កើតបរិយាកាសងងឹតខ្មៅ ស្ងៀម ភ័យ — ពិពណ៌នាសញ្ញា ស្លាប ខ្យល់ ឬក្លិន",
    "love":      "ពិពណ៌នាអារម្មណ៍ ការប្រទះ ចេតនា ស្នេហ៍ — ភ្ជាប់ទំនាក់ទំនងឱ្យដូចជីវិតពិត",
    "adventure": "ចរិតក្លាហ៊ាន ការប្រយុទ្ធ ឬដំណើរ — ពើងផ្ដួចផ្ដើមសកម្មភាព",
    "fable":     "សត្វជាតួ — មេរៀនដ៏ច្បាស់នៅចុងរឿង ដោយប្រើភាសាសាមញ្ញ",
    "legend":    "ប្រវត្តិ អង្គរ ទេវតា ឬព្រះពុទ្ធ — ភាសាគួរឱ្យគោរព ថ្ងឹក",
    "modern":    "ជីវិតក្រុង បញ្ហាសង្គម អារម្មណ៍ — ប្រើការសន្ទនា ឬព្រឹត្តិការណ៍ប្រចាំថ្ងៃ",
    "children":  "ភាសាសាមញ្ញ ចរិតគួរស្រឡាញ់ ជ័យជំនះ ឬ មេរៀន — បញ្ចប់ជាសង្ឃឹម",
    "comedy":    "ការយល់ច្រឡំ ស្ថានភាពគួរសើច — ប្រើចរិតមានខ្លឹម មិនត្រឹមតែហ្ហាស",
    "mystery":   "ដំណើរស៊ើបអង្កេត — បើករហ័ស ទ្រទ្រង់ការចង់ដឹង បិទទ្វារដោយការរុករក",
    "wisdom":    "មេរៀនពុទ្ធ ឬប្រាជ្ញា — ពិភពខ្នាតតូច ចរិតជ្រៅ សារដ៏ស៊ីជម្រៅ",
    "horror":    "ភ័យ ភ្ញាក់ ស្ញើប — ពីការប្រទះ ឬអ្វីដែលមើលមិនឃើញ ក្នុងស្ថានភាពកំបាំង",
}


# ─── Random topics per genre ──────────────────────────────────────────────────
RANDOM_TOPICS = {
    "folk":      ["កញ្ញាចិញ្ចឹមត្រី","ក្មេងកំព្រានិងយក្ស","ស្ត្រីទទែហ្លុយ","ពូជស្រូវមាស","ព្រះពួនព្រៃ"],
    "ghost":     ["ផ្ទះទំនេរចុងភូមិ","សសរទ្រព្យលាក់","ស្រែទំនេររាត្រី","ស្រីស្លៀកស",  "ច្រកទ្វារបិទតូច"],
    "love":      ["ស្នេហ៍ក្បែរទន្លេ","ស្នេហ៍ពេលភ្លៀង","អ្នកលក់ត្រីនិងអ្នកហែក","ស្នេហ៍ព្រែកថ្ម","ព្រៃស្នេហ៍"],
    "adventure": ["អ្នកព្រានតែម្នាក់","ដំណើរភ្នំក្រវាញ","ចោរចំការ","នាយកទ័ពក្មេង","ដំណើររំដោះ"],
    "fable":     ["ក្អែកនិងពស់","ដំរីចោររៀន","ត្រីនិងត្រីអណ្ដើក","ក្ដាននិងសត្វបក្សី","ខ្លានិងទន្សាយ"],
    "legend":    ["ប្រាសាទស្រែច","ព្រះខ័នព្រ័ត្រ","ព្រះនាងខ្មៅ","ទន្លេកំពង់ស្ពឺ","ព្រះថោង"],
    "modern":    ["ជីវិតកម្មករ","ក្មេងបម្រើការ","ស្ត្រីលក់នំ","ការងារក្រុង","ចំណាកស្រុក"],
    "children":  ["ក្មេងប្រុសសែន","ផ្ការីករៀន","ផ្ការីកក្ដិ","សត្វពស់착ល","ព្រៃថ្លឹងថ្លៃ"],
    "comedy":    ["ចៅកម្មការ","ការចម្អិនមិនជំនាញ","ភ្លេចផ្ទះ","ការហ្វឹកហ្វឺន","ប្ដីប្រពន្ធថ្មី"],
    "mystery":   ["ករណីអភ័យ","ហិបបាត់","ទូរស័ព្ទប្រហោង","ព្រៃស្ងាត់","ភ្ញៀវមករាប់"],
    "wisdom":    ["ចៅហ្វាយជ្រៅ","ព្រះសង្ឃក្មេង","ជនក្រីក្រចេះចិត្ត","ទន្លេបង្រៀន","ភ្ជួររដូវ"],
    "horror":    ["ផ្ទះមានម្ចាស់","ខ្សែស្រឡាយ","ខ្ញុំបានឃើញ","ការលាក់ ","ម្ហូបចុងក្រោយ"],
}

SYSTEM_PROMPT_BASE = """អ្នកជាអ្នកនិទានរឿងខ្មែរជំនាញ ដែលស្ទាត់ជំនាញក្នុងការសរសេររឿងខ្មែរគ្រប់ប្រភេទ។

ច្បាប់ (ត្រូវអនុវត្តតឹងរ៉ឹង):
១. សរសេរភាសាខ្មែរតែប៉ុណ្ណោះ — ហាមអក្សរឡាតាំង ឬអង់គ្លេស លើកលែងឈ្មោះតួអង្គបរទេស
២. ចាប់ផ្ដើមភ្លាមដោយប្រយោគទាក់ទាញ — ហាមចំណងជើង ហាម "ឧទ្ទេស" ឬ "ស្លាក"
៣. ពិពណ៌នារូបភាព អារម្មណ៍ ឬចរិតឲ្យជ្រៅ — កុំព្រៀងៗ
៤. ចូលខ្នាត paragraph គ្រប់ ២-៣ ប្រយោគ
៥. បញ្ចប់ដោយប្រយោគដ៏ប្រណីត — ផ្ដល់អារម្មណ៍ ឬបន្ទះ — ហាមចប់ "ហើយ​..."
៦. ហាមមានការបន្ថែម "ចំណុច" "ស្ថានភាព" ឬ "ឈ្មោះ" ដំបូង
"""

def build_prompt(genre_desc: str, genre_key: str, topic: str, length: str) -> tuple[str, str]:
    word_count = LENGTH_WORDS.get(length, "250")
    hint       = GENRE_HINTS.get(genre_key, "")
    style_note = f"\nរចនាប័ទ្មពិសេស: {hint}" if hint else ""
    pov = random.choice([
        "ទស្សនៈបុគ្គលទី ៣ (គេ / នាង / ពួកគេ)",
        "ទស្សនៈបុគ្គលទី ១ ដូចជាអ្នកនិទានបង្ហើប",
        "ទស្សនៈបុគ្គលទី ៣ ជ្រៅ — ចូលក្នុងចិត្តតួអង្គ",
    ])
    system = (
        SYSTEM_PROMPT_BASE
        + f"\nប្រវែងរឿង: ប្រហែល {word_count} ពាក្យ — វាស់ឱ្យចំ"
        + style_note
    )
    user = (
        f"ប្រភេទ: {genre_desc}\n"
        f"ប្រធានបទ / តួអង្គ: {topic}\n"
        f"ទស្សនៈ: {pov}\n\n"
        "សរសេររឿងខ្មែរ ចាប់ផ្ដើមភ្លាម:"
    )
    return system, user


def build_continue_prompt(story: str, genre_desc: str, genre_key: str,
                           topic: str, chapter: int, length: str) -> tuple[str, str]:
    word_count = LENGTH_WORDS.get(length, "250")
    hint       = GENRE_HINTS.get(genre_key, "")
    style_note = f"\nរចនាប័ទ្ម: {hint}" if hint else ""
    system = (
        SYSTEM_PROMPT_BASE
        + f"\nប្រវែង: ប្រហែល {word_count} ពាក្យ"
        + style_note
    )
    user = (
        f"ប្រភេទ: {genre_desc}\n"
        f"ប្រធានបទ: {topic}\n"
        f"នេះជា ជំពូក {chapter} — បន្តពីរឿងដែលនៅខាងក្រោម:\n\n"
        f"--- រឿងមុន ---\n{story[-800:]}\n--- ចប់ ---\n\n"
        f"សូមសរសេរ ជំពូក {chapter} ដោយ:\n"
        f"- បន្តឡើងពីដំណើររឿង\n"
        f"- នាំយកព្រឹត្តិការណ៍ ឬការបែករើសថ្មី\n"
        f"- ចាប់ផ្ដើមភ្លាម កុំសរសេរ 'ជំពូក...' ឬចំណងជើង:"
    )
    return system, user

# ─── Keyboards ────────────────────────────────────────────────────────────────
def genre_keyboard() -> InlineKeyboardMarkup:
    keys  = []
    items = list(GENRES.items())
    for i in range(0, len(items), 2):
        row = [InlineKeyboardButton(items[i][1][0], callback_data=f"genre:{items[i][0]}")]
        if i + 1 < len(items):
            row.append(InlineKeyboardButton(items[i+1][1][0], callback_data=f"genre:{items[i+1][0]}"))
        keys.append(row)
    return InlineKeyboardMarkup(keys)

def length_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = get_user_length(uid)
    rows = [
        [InlineKeyboardButton(("✅ " if current == k else "") + lbl, callback_data=f"length:{k}")]
        for k, lbl in LENGTH_LABELS.items()
    ]
    rows.append([InlineKeyboardButton("◀️ ត្រឡប់", callback_data="length:back")])
    return InlineKeyboardMarkup(rows)

def action_keyboard(story_index: int = 0) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 រឿងថ្មីទៀត",      callback_data="action:new"),
            InlineKeyboardButton("🎲 Random",            callback_data="action:random"),
        ],
        [
            InlineKeyboardButton("📖 ប្រភេទផ្សេង",     callback_data="action:genres"),
            InlineKeyboardButton("⏭️ បន្តរឿង",          callback_data=f"action:continue:{story_index}"),
        ],
        [
            InlineKeyboardButton("📏 ប្រែប្រួលប្រវែង",  callback_data="action:length"),
            InlineKeyboardButton(f"⭐ វាយតម្លៃ",        callback_data=f"action:rate:{story_index}"),
        ],
        [
            InlineKeyboardButton("📊 Stats",             callback_data="action:stats"),
            InlineKeyboardButton("📚 ប្រវត្តិ",          callback_data="action:history"),
        ],
        [InlineKeyboardButton("🏠 ទំព័រដើម",            callback_data="action:home")],
    ])

def rating_keyboard(story_index: int) -> InlineKeyboardMarkup:
    stars = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
    rows  = [
        [InlineKeyboardButton(s, callback_data=f"rate:{story_index}:{i+1}")]
        for i, s in enumerate(stars)
    ]
    rows.append([InlineKeyboardButton("◀️ ត្រឡប់", callback_data="rate:back:0")])
    return InlineKeyboardMarkup(rows)

def history_keyboard(history: list) -> InlineKeyboardMarkup:
    rows = []
    for i, h in enumerate(history[:10]):
        rating_str = "⭐" * h.get("rating", 0) if h.get("rating") else ""
        short_topic = h["topic"][:16] + "…" if len(h["topic"]) > 16 else h["topic"]
        label = f"{i+1}. {h['genre']} {short_topic} {rating_str}"
        rows.append([InlineKeyboardButton(label, callback_data=f"hist:read:{i}")])
    rows.append([InlineKeyboardButton("◀️ ត្រឡប់", callback_data="hist:back:0")])
    return InlineKeyboardMarkup(rows)

def provider_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = get_user_provider(uid)
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(("✅ " if current=="groq"   else "")+"⚡ Groq",   callback_data="prov:groq"),
            InlineKeyboardButton(("✅ " if current=="gemini" else "")+"🤖 Gemini", callback_data="prov:gemini"),
        ],
        [InlineKeyboardButton("◀️ ត្រឡប់", callback_data="prov:back")],
    ])

def groq_model_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = get_user_model(uid)
    rows = [
        [InlineKeyboardButton(("✅ " if current==m else "")+lbl, callback_data=f"model:{m}")]
        for m, lbl in GROQ_MODELS.items()
    ]
    rows.append([InlineKeyboardButton("◀️ ត្រឡប់", callback_data="model:back")])
    return InlineKeyboardMarkup(rows)

def gemini_model_keyboard(uid: int) -> InlineKeyboardMarkup:
    current = get_user_model(uid)
    rows = [
        [InlineKeyboardButton(("✅ " if current==m else "")+lbl, callback_data=f"model:{m}")]
        for m, lbl in GEMINI_MODELS.items()
    ]
    rows.append([InlineKeyboardButton("◀️ ត្រឡប់", callback_data="model:back")])
    return InlineKeyboardMarkup(rows)

def settings_keyboard(uid: int) -> InlineKeyboardMarkup:
    provider      = get_user_provider(uid)
    model         = get_user_model(uid)
    length        = get_user_length(uid)
    groq_status   = "✅" if get_user_groq_key(uid)   else "❌"
    gemini_status = "✅" if get_user_gemini_key(uid) else "❌"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🔀 Provider: {provider.title()}",        callback_data="settings:provider")],
        [InlineKeyboardButton(f"🧠 Model: {model[:28]}",                 callback_data="settings:model")],
        [InlineKeyboardButton(f"📏 ប្រវែង: {LENGTH_LABELS[length]}",    callback_data="settings:length")],
        [
            InlineKeyboardButton(f"{groq_status} Groq Key",    callback_data="settings:set_groq"),
            InlineKeyboardButton(f"{gemini_status} Gemini Key", callback_data="settings:set_gemini"),
        ],
        [
            InlineKeyboardButton("🗑️ លុប Groq",    callback_data="settings:del_groq"),
            InlineKeyboardButton("🗑️ លុប Gemini",  callback_data="settings:del_gemini"),
        ],
        [InlineKeyboardButton("◀️ ទំព័រដើម", callback_data="settings:home")],
    ])

# ─── AI Providers ─────────────────────────────────────────────────────────────
def _call_groq(api_key: str, model: str, system: str, prompt: str) -> str:
    client = Groq(api_key=api_key)
    resp   = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1500,
        temperature=0.92,
        top_p=0.95,
    )
    return resp.choices[0].message.content.strip()

def _call_gemini(api_key: str, model: str, system: str, prompt: str) -> str:
    import google.generativeai as g
    g.configure(api_key=api_key)
    m      = g.GenerativeModel(model, system_instruction=system)
    result = m.generate_content(
        prompt,
        generation_config=g.types.GenerationConfig(
            temperature=0.92,
            top_p=0.95,
            max_output_tokens=1500,
        ),
    )
    return result.text.strip()

async def generate_story(uid: int, genre_desc: str, genre_key: str, topic: str) -> tuple[str, str]:
    provider   = get_user_provider(uid)
    model      = get_user_model(uid)
    length     = get_user_length(uid)
    groq_key   = get_user_groq_key(uid)
    gemini_key = get_user_gemini_key(uid)

    system, prompt = build_prompt(genre_desc, genre_key, topic, length)
    loop = asyncio.get_running_loop()

    primary_fn    = _call_groq   if provider == "groq"   else _call_gemini
    primary_key   = groq_key     if provider == "groq"   else gemini_key
    fallback_fn   = _call_gemini if provider == "groq"   else _call_groq
    fallback_key  = gemini_key   if provider == "groq"   else groq_key
    fallback_model = "gemini-2.5-flash" if provider == "groq" else "llama-3.3-70b-versatile"

    if primary_key:
        try:
            text = await loop.run_in_executor(None, primary_fn, primary_key, model, system, prompt)
            return text, provider
        except Exception as e:
            logger.warning(f"Primary ({provider}) failed: {e}. Trying fallback.")

    if fallback_key:
        try:
            text = await loop.run_in_executor(None, fallback_fn, fallback_key, fallback_model, system, prompt)
            return text, "gemini" if provider == "groq" else "groq"
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")

    return "", "none"

# ─── Shared helpers ───────────────────────────────────────────────────────────
def _settings_text(uid: int) -> str:
    return (
        f"⚙️ *ការកំណត់*\n\n"
        f"🔀 Provider: *{get_user_provider(uid).title()}*\n"
        f"🧠 Model: `{get_user_model(uid)}`\n"
        f"📏 ប្រវែង: {LENGTH_LABELS[get_user_length(uid)]}\n\n"
        "ជ្រើសសកម្មភាព:"
    )

def _story_message(topic: str, genre_label: str, prov_badge: str,
                   fallback_note: str, story: str, length: str) -> str:
    li = LENGTH_ICONS.get(length, "📃")
    return (
        f"📜 *{topic}*\n"
        f"🏷️ {genre_label}  {prov_badge}{fallback_note}  {li}\n"
        f"{'─'*30}\n\n"
        f"{story}\n\n"
        f"{'─'*30}\n"
        f"_✨ AI Khmer Storyteller v3_"
    )

# ─── /start ───────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user    = update.effective_user
    uid     = user.id
    has_key = get_user_groq_key(uid) or get_user_gemini_key(uid)

    if not has_key:
        await update.message.reply_text(
            f"🙏 សួស្ដី *{user.first_name}*!\n\n"
            "🎭 *Bot និទានរឿងខ្មែរ AI v3*\n\n"
            "✨ *ថ្មីក្នុង v3:*\n"
            "• ១២ ប្រភេទរឿង (+ ប្រាជ្ញា + ភ័យខ្លាច)\n"
            "• ជ្រើសប្រវែង ខ្លី / មធ្យម / វែង\n"
            "• វាយតម្លៃ ⭐ និងអានរឿងចាស់ m្ដងទៀត\n"
            "• AI Prompt ថ្មី — រឿងខ្ញាក់ and ស្រស់ជាងមុន!\n\n"
            "⚡ *Groq (FREE):* [console.groq.com/keys](https://console.groq.com/keys)\n"
            "🤖 *Gemini (FREE):* [aistudio.google.com](https://aistudio.google.com/app/apikey)\n\n"
            "ប្រើ /settings ដើម្បីដាក់ key 🔑",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return ConversationHandler.END

    await update.message.reply_text(
        f"🙏 សួស្ដី *{user.first_name}*!\n\n"
        "🎭 *Bot និទានរឿងខ្មែរ AI v3*\n\n"
        "ជ្រើសប្រភេទរឿង ⬇️",
        parse_mode="Markdown",
        reply_markup=genre_keyboard(),
    )
    return CHOOSING_GENRE

# ─── /settings ────────────────────────────────────────────────────────────────
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    await update.message.reply_text(
        _settings_text(uid),
        parse_mode="Markdown",
        reply_markup=settings_keyboard(uid),
    )
    return WAITING_FOR_KEY

async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    action = query.data.split(":", 1)[1]
    uid    = query.from_user.id

    if action == "provider":
        await query.edit_message_text(
            "🔀 *ជ្រើស AI Provider:*\n\n"
            "⚡ *Groq* — លឿន (<1s), ឥតគិតថ្លៃ\n"
            "🤖 *Gemini* — Google AI, ឥតគិតថ្លៃ\n\n"
            "_Bot ប្ដូរ provider ស្វ័យប្រវត្ត ប្រសិនបើ primary រអាក_",
            parse_mode="Markdown",
            reply_markup=provider_keyboard(uid),
        )
        return WAITING_FOR_KEY

    elif action == "model":
        provider = get_user_provider(uid)
        kb = groq_model_keyboard(uid) if provider == "groq" else gemini_model_keyboard(uid)
        await query.edit_message_text(
            f"🧠 *ជ្រើស Model ({provider.title()}):*",
            parse_mode="Markdown",
            reply_markup=kb,
        )
        return CHOOSING_MODEL

    elif action == "length":
        await query.edit_message_text(
            "📏 *ជ្រើសប្រវែងរឿង:*\n\n"
            "📄 ខ្លី ~១៥០ ពាក្យ\n"
            "📃 មធ្យម ~២៥០ ពាក្យ\n"
            "📜 វែង ~៤០០ ពាក្យ",
            parse_mode="Markdown",
            reply_markup=length_keyboard(uid),
        )
        return WAITING_FOR_KEY

    elif action == "set_groq":
        context.user_data[WAITING_KEY_TYPE] = "groq"
        await query.edit_message_text(
            "🔑 *ដាក់ Groq API Key*\n\n"
            "ទទួល key: [console.groq.com/keys](https://console.groq.com/keys)\n\n"
            "Key ចាប់ផ្ដើម `gsk_...`\n\n"
            "⚠️ Bot លុប message ភ្លាម ដើម្បីសុវត្ថិភាព\n"
            "/cancel ដើម្បីបោះបង់",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return WAITING_FOR_KEY

    elif action == "set_gemini":
        context.user_data[WAITING_KEY_TYPE] = "gemini"
        await query.edit_message_text(
            "🔑 *ដាក់ Gemini API Key*\n\n"
            "ទទួល key: [aistudio.google.com](https://aistudio.google.com/app/apikey)\n\n"
            "Key ចាប់ផ្ដើម `AIzaSy...`\n\n"
            "⚠️ Bot លុប message ភ្លាម ដើម្បីសុវត្ថិភាព\n"
            "/cancel ដើម្បីបោះបង់",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return WAITING_FOR_KEY

    elif action == "del_groq":
        deleted = delete_user_key(uid, "groq")
        prefix  = "🗑️ *Groq Key លុបហើយ!*\n\n" if deleted else "⚠️ គ្មាន Groq Key ។\n\n"
        await query.edit_message_text(
            prefix + _settings_text(uid),
            parse_mode="Markdown",
            reply_markup=settings_keyboard(uid),
        )
        return WAITING_FOR_KEY

    elif action == "del_gemini":
        deleted = delete_user_key(uid, "gemini")
        prefix  = "🗑️ *Gemini Key លុបហើយ!*\n\n" if deleted else "⚠️ គ្មាន Gemini Key ។\n\n"
        await query.edit_message_text(
            prefix + _settings_text(uid),
            parse_mode="Markdown",
            reply_markup=settings_keyboard(uid),
        )
        return WAITING_FOR_KEY

    elif action == "home":
        await query.edit_message_text(
            "🏠 *ទំព័រដើម*\n\n/story /settings /history /help",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    return WAITING_FOR_KEY

# ─── Provider / Model / Length callbacks ─────────────────────────────────────
async def provider_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    choice = query.data.split(":", 1)[1]
    uid    = query.from_user.id

    if choice == "back":
        await query.edit_message_text(
            _settings_text(uid), parse_mode="Markdown", reply_markup=settings_keyboard(uid)
        )
        return WAITING_FOR_KEY

    default_model = "llama-3.3-70b-versatile" if choice == "groq" else "gemini-2.5-flash"
    set_user_data(uid, provider=choice, model=default_model)
    await query.edit_message_text(
        f"✅ *ប្ដូរទៅ {choice.title()}!*\n🧠 Model: `{default_model}`\n\nប្រើ /story ។",
        parse_mode="Markdown",
    )
    return ConversationHandler.END

async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    choice = query.data.split(":", 1)[1]
    uid    = query.from_user.id

    if choice == "back":
        await query.edit_message_text(
            _settings_text(uid), parse_mode="Markdown", reply_markup=settings_keyboard(uid)
        )
        return WAITING_FOR_KEY

    set_user_data(uid, model=choice)
    label = {**GROQ_MODELS, **GEMINI_MODELS}.get(choice, choice)
    await query.edit_message_text(
        f"✅ *Model: {label}*\n\nប្រើ /story ។", parse_mode="Markdown"
    )
    return ConversationHandler.END

async def length_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    parts  = query.data.split(":", 1)
    choice = parts[1]
    uid    = query.from_user.id

    if choice == "back":
        await query.edit_message_text(
            _settings_text(uid), parse_mode="Markdown", reply_markup=settings_keyboard(uid)
        )
        return WAITING_FOR_KEY

    set_user_data(uid, length=choice)
    await query.edit_message_text(
        f"✅ *ប្រវែងរឿង: {LENGTH_LABELS[choice]}*\n\nប្រើ /story ។",
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

    if key_type == "groq" and (not key.startswith("gsk_") or len(key) < 20):
        await update.message.reply_text(
            "❌ *Groq Key ខុស!*\n\nKey ចាប់ផ្ដើម `gsk_`\nព្យាយាមម្ដងទៀត ឬ /cancel",
            parse_mode="Markdown",
        )
        return WAITING_FOR_KEY

    if key_type == "gemini" and (not key.startswith("AIza") or len(key) < 30):
        await update.message.reply_text(
            "❌ *Gemini Key ខុស!*\n\nKey ចាប់ផ្ដើម `AIzaSy`\nព្យាយាមម្ដងទៀត ឬ /cancel",
            parse_mode="Markdown",
        )
        return WAITING_FOR_KEY

    validating_msg = await update.message.reply_text(
        f"⏳ *ត្រួតពិនិត្យ {key_type.title()} Key...*", parse_mode="Markdown"
    )
    loop = asyncio.get_running_loop()
    try:
        if key_type == "groq":
            await loop.run_in_executor(
                None, _call_groq, key, "llama-3.1-8b-instant",
                "You are a helpful assistant.", "Say ok."
            )
        else:
            await loop.run_in_executor(
                None, _call_gemini, key, "gemini-2.5-flash",
                "You are a helpful assistant.", "Say ok."
            )

        set_user_data(uid, **{key_type: key}, provider=key_type)
        await validating_msg.edit_text(
            f"✅ *{key_type.title()} Key ត្រឹមត្រូវ!*\n\n"
            f"Provider ប្ដូរទៅ *{key_type.title()}* ស្វ័យប្រវត្ត\n\n"
            "ប្រើ /story ដើម្បីចាប់ផ្ដើម 🎭",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    except Exception as e:
        err = str(e).lower()
        logger.warning(f"Key validation ({key_type}): {e}")
        if any(w in err for w in ["api_key","permission","invalid","credential","auth","unauthorized"]):
            await validating_msg.edit_text(
                f"❌ *{key_type.title()} Key ខុស!*\n\nព្យាយាមម្ដងទៀត ឬ /cancel",
                parse_mode="Markdown",
            )
            return WAITING_FOR_KEY

        set_user_data(uid, **{key_type: key}, provider=key_type)
        await validating_msg.edit_text(
            f"⚠️ *មិនអាចត្រួតពិនិត្យ — Key រក្សាទុករួច*\n\nប្រើ /story ដើម្បីសាកល្បង។",
            parse_mode="Markdown",
        )
        return ConversationHandler.END


# ─── /random ──────────────────────────────────────────────────────────────────
async def random_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    if not (get_user_groq_key(uid) or get_user_gemini_key(uid)):
        await update.message.reply_text(
            "⚠️ *មិនទាន់មាន API Key!*\n\nប្រើ /settings ដើម្បីដាក់ key ។",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    genre_key             = random.choice(list(GENRES.keys()))
    genre_label, genre_desc = GENRES[genre_key]
    topic                 = random.choice(RANDOM_TOPICS.get(genre_key, ["ជីវិតខ្មែរ"]))
    context.user_data.update(genre_key=genre_key, genre_label=genre_label,
                              genre_desc=genre_desc, topic=topic)

    provider  = get_user_provider(uid)
    length    = get_user_length(uid)
    prov_icon = "⚡" if provider == "groq" else "🤖"
    li        = LENGTH_ICONS.get(length, "📃")

    thinking_msg = await update.message.reply_text(
        f"🎲 *AI Random កំពុងនិទានរឿង...*\n\n"
        f"📖 {genre_label}  {li}\n"
        f"🏷️ {topic}\n"
        f"{prov_icon} {provider.title()}\n\n"
        "_សូមរង់ចាំ..._",
        parse_mode="Markdown",
    )
    story, used_provider = await generate_story(uid, genre_desc, genre_key, topic)
    await thinking_msg.delete()

    if used_provider == "none":
        await update.message.reply_text("❌ *មិនអាចបង្កើតរឿង!*\n\nប្រើ /settings ។", parse_mode="Markdown")
        return ConversationHandler.END

    prov_badge    = "⚡ Groq" if used_provider == "groq" else "🤖 Gemini"
    fallback_note = " _(fallback)_" if used_provider != provider else ""
    add_to_history(uid, genre_label, topic, story, length)

    msg_text = _story_message(topic, genre_label, prov_badge, fallback_note, story, length)
    if len(msg_text) > 4000:
        msg_text = msg_text[:3990] + "…"
    await update.message.reply_text(msg_text, parse_mode="Markdown", reply_markup=action_keyboard(0))
    return READING_STORY

# ─── /stats ───────────────────────────────────────────────────────────────────
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid   = update.effective_user.id
    stats = get_user_stats(uid)
    if not stats:
        await update.message.reply_text(
            "📊 *គ្មានស្ថិតិ*\n\nប្រើ /story ដើម្បីចាប់ផ្ដើម!",
            parse_mode="Markdown",
        )
        return

    genre_lines = ""
    for g, count in sorted(stats["genre_count"].items(), key=lambda x: -x[1])[:5]:
        bar = "█" * count + "░" * max(0, 5 - count)
        genre_lines += f"  {g} {bar} {count}\n"

    stars = "⭐" * round(stats["avg_rating"]) if stats["avg_rating"] else "—"
    await update.message.reply_text(
        f"📊 *ស្ថិតិរបស់អ្នក*\n\n"
        f"📚 រឿងសរុប: *{stats['total']}*\n"
        f"⭐ វាយតម្លៃ: *{stats['rated']}* រឿង  {stars}\n"
        f"🏆 ពិន្ទុ​មធ្យម: *{stats['avg_rating']}* / 5\n"
        f"🎭 Genre ពេញនិយម: {stats['top_genre']}\n"
        f"📏 ប្រវែងចូលចិត្ត: {LENGTH_LABELS.get(stats['top_length'], stats['top_length'])}\n\n"
        f"*Top Genres:*\n{genre_lines}",
        parse_mode="Markdown",
    )

# ─── /history ─────────────────────────────────────────────────────────────────
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid     = update.effective_user.id
    history = get_history(uid)
    if not history:
        await update.message.reply_text(
            "📚 *គ្មានប្រវត្តិ*\n\nប្រើ /story ដើម្បីបង្កើតរឿងដំបូង!",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "📚 *ប្រវត្តិរឿង* — ចុចដើម្បីអានម្ដងទៀត:",
        parse_mode="Markdown",
        reply_markup=history_keyboard(history),
    )
    return VIEWING_HISTORY

async def history_nav_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    parts = query.data.split(":")   # hist:read:N | hist:back:0 | hist:list:0
    uid   = query.from_user.id

    if parts[1] in ("back", "list"):
        history = get_history(uid)
        if not history:
            await query.edit_message_text("📚 គ្មានប្រវត្តិ ប្រើ /story ។")
            return ConversationHandler.END
        await query.edit_message_text(
            "📚 *ប្រវត្តិរឿង* — ចុចដើម្បីអានម្ដងទៀត:",
            parse_mode="Markdown",
            reply_markup=history_keyboard(history),
        )
        return VIEWING_HISTORY

    if parts[1] == "read":
        index   = int(parts[2])
        history = get_history(uid)
        if index >= len(history):
            await query.answer("រឿងមិនមានទៀតទេ។", show_alert=True)
            return VIEWING_HISTORY

        h          = history[index]
        rating_str = ("⭐" * h["rating"]) if h.get("rating") else "_មិនទាន់វាយ_"
        text = (
            f"📜 *{h['topic']}*\n"
            f"🏷️ {h['genre']}  🕒 {h['date']}  {rating_str}\n"
            f"{'─'*30}\n\n"
            f"{h['story']}\n\n"
            f"{'─'*30}\n"
            f"_✨ AI Khmer Storyteller v3_"
        )
        if len(text) > 4000:
            text = text[:3990] + "…"
        await query.edit_message_text(
            text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("⭐ វាយតម្លៃ",  callback_data=f"rate:{index}:0"),
                    InlineKeyboardButton("◀️ ប្រវត្តិ", callback_data="hist:list:0"),
                ],
            ]),
        )
        return VIEWING_HISTORY

    return VIEWING_HISTORY

# ─── Rating callback ──────────────────────────────────────────────────────────
async def rating_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    parts = query.data.split(":")   # rate:INDEX:STARS | rate:back:0
    uid   = query.from_user.id

    if parts[1] == "back":
        await query.edit_message_text("◀️ ប្រើ /story ឬ /history ។")
        return ConversationHandler.END

    index = int(parts[1])
    stars = int(parts[2]) if len(parts) > 2 else 0

    if stars == 0:
        await query.message.reply_text(
            "⭐ *ជ្រើសការវាយតម្លៃ:*",
            parse_mode="Markdown",
            reply_markup=rating_keyboard(index),
        )
        return READING_STORY

    rate_story(uid, index, stars)
    await query.edit_message_text(
        f"✅ *អរគុណ!* {'⭐' * stars}\n\nប្រើ /story ដើម្បីបង្កើតរឿងថ្មី។",
        parse_mode="Markdown",
    )
    return ConversationHandler.END

# ─── Story flow ───────────────────────────────────────────────────────────────
async def story_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    if not (get_user_groq_key(uid) or get_user_gemini_key(uid)):
        await update.message.reply_text(
            "⚠️ *មិនទាន់មាន API Key!*\n\n"
            "ប្រើ /settings ដើម្បីដាក់ key\n\n"
            "⚡ [console.groq.com/keys](https://console.groq.com/keys)\n"
            "🤖 [aistudio.google.com](https://aistudio.google.com/app/apikey)",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "📖 *ជ្រើសប្រភេទរឿង:*",
        parse_mode="Markdown",
        reply_markup=genre_keyboard(),
    )
    return CHOOSING_GENRE

async def genre_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query     = update.callback_query
    await query.answer()
    genre_key = query.data.split(":")[1]
    genre_label, genre_desc = GENRES[genre_key]
    context.user_data.update(genre_key=genre_key, genre_label=genre_label, genre_desc=genre_desc)

    await query.edit_message_text(
        f"✅ *{genre_label}*\n\n"
        "✍️ វាយ *ប្រធានបទ* ឬ *ឈ្មោះតួអង្គ*:\n\n"
        "_ឧទាហរណ៍: ក្មេងស្រីក្នុងព្រៃ, ចោរភ្នំ, ស្នេហ៍ក្បែរទន្លេ..._",
        parse_mode="Markdown",
    )
    return TYPING_TOPIC

async def topic_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    topic       = update.message.text.strip()
    uid         = update.effective_user.id
    genre_key   = context.user_data.get("genre_key",   "folk")
    genre_label = context.user_data.get("genre_label", "")
    genre_desc  = context.user_data.get("genre_desc",  "")
    context.user_data["topic"] = topic

    provider  = get_user_provider(uid)
    model     = get_user_model(uid)
    length    = get_user_length(uid)
    prov_icon = "⚡" if provider == "groq" else "🤖"
    li        = LENGTH_ICONS.get(length, "📃")

    thinking_msg = await update.message.reply_text(
        f"🪄 *AI កំពុងនិទានរឿង...*\n\n"
        f"📖 {genre_label}  {li} {LENGTH_LABELS[length]}\n"
        f"🏷️ {topic}\n"
        f"{prov_icon} {provider.title()} · `{model}`\n\n"
        "_សូមរង់ចាំ..._",
        parse_mode="Markdown",
    )

    story, used_provider = await generate_story(uid, genre_desc, genre_key, topic)
    await thinking_msg.delete()

    if used_provider == "none":
        await update.message.reply_text(
            "❌ *មិនអាចបង្កើតរឿង!*\n\n"
            "ប្រើ /settings → ពិនិត្យ API Key\n\n"
            "⚡ [console.groq.com/keys](https://console.groq.com/keys)\n"
            "🤖 [aistudio.google.com](https://aistudio.google.com/app/apikey)",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return ConversationHandler.END

    prov_badge    = "⚡ Groq" if used_provider == "groq" else "🤖 Gemini"
    fallback_note = " _(fallback)_" if used_provider != provider else ""
    add_to_history(uid, genre_label, topic, story, length)

    msg_text = _story_message(topic, genre_label, prov_badge, fallback_note, story, length)
    if len(msg_text) > 4000:
        msg_text = msg_text[:3990] + "…"

    await update.message.reply_text(
        msg_text,
        parse_mode="Markdown",
        reply_markup=action_keyboard(0),
    )
    return READING_STORY

async def action_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query  = update.callback_query
    await query.answer()
    parts  = query.data.split(":")
    action = parts[1]
    uid    = query.from_user.id

    if action == "new":
        genre_key   = context.user_data.get("genre_key",   "folk")
        genre_label = context.user_data.get("genre_label", "")
        genre_desc  = context.user_data.get("genre_desc",  "")
        topic       = context.user_data.get("topic",       "")
        provider    = get_user_provider(uid)
        length      = get_user_length(uid)
        li          = LENGTH_ICONS.get(length, "📃")
        prov_icon   = "⚡" if provider == "groq" else "🤖"

        thinking_msg = await query.message.reply_text(
            f"🪄 *AI កំពុងនិទានរឿងថ្មី...*\n{prov_icon} {provider.title()}  {li}",
            parse_mode="Markdown",
        )
        story, used_provider = await generate_story(uid, genre_desc, genre_key, topic)
        await thinking_msg.delete()

        if used_provider == "none":
            await query.message.reply_text(
                "❌ *មិនអាចបង្កើតរឿង!*\nប្រើ /settings → ពិនិត្យ API Key ។",
                parse_mode="Markdown",
            )
            return READING_STORY

        prov_badge    = "⚡ Groq" if used_provider == "groq" else "🤖 Gemini"
        fallback_note = " _(fallback)_" if used_provider != provider else ""
        add_to_history(uid, genre_label, topic, story, length)

        msg_text = _story_message(topic, genre_label, prov_badge, fallback_note, story, length)
        if len(msg_text) > 4000:
            msg_text = msg_text[:3990] + "…"

        await query.message.reply_text(
            msg_text, parse_mode="Markdown", reply_markup=action_keyboard(0)
        )
        return READING_STORY


    elif action == "random":
        genre_key             = random.choice(list(GENRES.keys()))
        genre_label, genre_desc = GENRES[genre_key]
        topic                 = random.choice(RANDOM_TOPICS.get(genre_key, ["ជីវិតខ្មែរ"]))
        context.user_data.update(genre_key=genre_key, genre_label=genre_label,
                                  genre_desc=genre_desc, topic=topic)
        provider  = get_user_provider(uid)
        length    = get_user_length(uid)
        li        = LENGTH_ICONS.get(length, "📃")
        prov_icon = "⚡" if provider == "groq" else "🤖"

        thinking_msg = await query.message.reply_text(
            f"🎲 *AI Random...*\n{prov_icon} {provider.title()}  {li}",
            parse_mode="Markdown",
        )
        story, used_provider = await generate_story(uid, genre_desc, genre_key, topic)
        await thinking_msg.delete()

        if used_provider == "none":
            await query.message.reply_text("❌ *មិនអាចបង្កើតរឿង!*\nប្រើ /settings ។", parse_mode="Markdown")
            return READING_STORY

        prov_badge    = "⚡ Groq" if used_provider == "groq" else "🤖 Gemini"
        fallback_note = " _(fallback)_" if used_provider != provider else ""
        add_to_history(uid, genre_label, topic, story, length)
        msg_text = _story_message(topic, genre_label, prov_badge, fallback_note, story, length)
        if len(msg_text) > 4000:
            msg_text = msg_text[:3990] + "…"
        await query.message.reply_text(msg_text, parse_mode="Markdown", reply_markup=action_keyboard(0))
        return READING_STORY

    elif action == "continue":
        story_index = int(parts[2]) if len(parts) > 2 else 0
        history     = get_history(uid)
        if story_index >= len(history):
            await query.answer("រឿងមិនឃើញទៀតទេ។", show_alert=True)
            return READING_STORY

        h           = history[story_index]
        prev_story  = h["story"]
        genre_key   = context.user_data.get("genre_key", "folk")
        genre_label = h.get("genre", context.user_data.get("genre_label",""))
        genre_desc  = GENRES.get(genre_key, ("",""))[1]
        topic       = h.get("topic", context.user_data.get("topic",""))
        length      = get_user_length(uid)

        # determine chapter number
        chapter_num = context.user_data.get("chapter", 1) + 1
        context.user_data["chapter"]    = chapter_num
        context.user_data["prev_story"] = prev_story
        context.user_data["genre_key"]  = genre_key
        context.user_data["topic"]      = topic

        provider  = get_user_provider(uid)
        prov_icon = "⚡" if provider == "groq" else "🤖"
        li        = LENGTH_ICONS.get(length, "📃")

        thinking_msg = await query.message.reply_text(
            f"⏭️ *AI កំពុងបន្ត ជំពូក {chapter_num}...*\n{prov_icon} {provider.title()}  {li}",
            parse_mode="Markdown",
        )

        system, prompt = build_continue_prompt(prev_story, genre_desc, genre_key, topic, chapter_num, length)
        loop = asyncio.get_running_loop()
        groq_key   = get_user_groq_key(uid)
        gemini_key = get_user_gemini_key(uid)
        model      = get_user_model(uid)
        fallback_model = "gemini-2.5-flash" if provider == "groq" else "llama-3.3-70b-versatile"

        story      = ""
        used_provider = "none"
        primary_fn  = _call_groq   if provider == "groq" else _call_gemini
        primary_key = groq_key     if provider == "groq" else gemini_key
        fallback_fn = _call_gemini if provider == "groq" else _call_groq
        fallback_key= gemini_key   if provider == "groq" else groq_key

        if primary_key:
            try:
                story = await loop.run_in_executor(None, primary_fn, primary_key, model, system, prompt)
                used_provider = provider
            except Exception as e:
                logger.warning(f"Continue primary failed: {e}")
        if not story and fallback_key:
            try:
                story = await loop.run_in_executor(None, fallback_fn, fallback_key, fallback_model, system, prompt)
                used_provider = "gemini" if provider == "groq" else "groq"
            except Exception as e:
                logger.error(f"Continue fallback failed: {e}")

        await thinking_msg.delete()

        if not story:
            await query.message.reply_text("❌ *មិនអាចបន្តរឿង!*\nប្រើ /settings → ពិនិត្យ API Key ។", parse_mode="Markdown")
            return READING_STORY

        prov_badge    = "⚡ Groq" if used_provider == "groq" else "🤖 Gemini"
        fallback_note = " _(fallback)_" if used_provider != provider else ""
        chapter_topic = f"{topic} — ជំពូក {chapter_num}"
        add_to_history(uid, genre_label, chapter_topic, story, length)

        msg_text = _story_message(chapter_topic, genre_label, prov_badge, fallback_note, story, length)
        if len(msg_text) > 4000:
            msg_text = msg_text[:3990] + "…"
        await query.message.reply_text(msg_text, parse_mode="Markdown", reply_markup=action_keyboard(0))
        return READING_STORY

    elif action == "stats":
        stats = get_user_stats(uid)
        if not stats:
            await query.message.reply_text("📊 *គ្មានស្ថិតិ* — ប្រើ /story !", parse_mode="Markdown")
            return READING_STORY
        genre_lines = ""
        for g, count in sorted(stats["genre_count"].items(), key=lambda x: -x[1])[:5]:
            bar = "█" * count + "░" * max(0, 5 - count)
            genre_lines += f"  {g} {bar} {count}\n"
        stars = "⭐" * round(stats["avg_rating"]) if stats["avg_rating"] else "—"
        await query.message.reply_text(
            f"📊 *ស្ថិតិរបស់អ្នក*\n\n"
            f"📚 រឿងសរុប: *{stats['total']}*\n"
            f"⭐ វាយតម្លៃ: *{stats['rated']}* រឿង  {stars}\n"
            f"🏆 ពិន្ទុ​មធ្យម: *{stats['avg_rating']}* / 5\n"
            f"🎭 Genre ពេញនិយម: {stats['top_genre']}\n"
            f"📏 ប្រវែងចូលចិត្ត: {LENGTH_LABELS.get(stats['top_length'], stats['top_length'])}\n\n"
            f"*Top Genres:*\n{genre_lines}",
            parse_mode="Markdown",
        )
        return READING_STORY

    elif action == "genres":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "📖 *ជ្រើសប្រភេទរឿង:*",
            parse_mode="Markdown",
            reply_markup=genre_keyboard(),
        )
        return CHOOSING_GENRE

    elif action == "length":
        await query.message.reply_text(
            "📏 *ជ្រើសប្រវែងរឿង:*",
            parse_mode="Markdown",
            reply_markup=length_keyboard(uid),
        )
        return READING_STORY

    elif action == "rate":
        story_index = int(parts[2]) if len(parts) > 2 else 0
        await query.message.reply_text(
            "⭐ *ជ្រើសការវាយតម្លៃ:*",
            parse_mode="Markdown",
            reply_markup=rating_keyboard(story_index),
        )
        return READING_STORY

    elif action == "history":
        history = get_history(uid)
        if not history:
            await query.message.reply_text("📚 *គ្មានប្រវត្តិ* — ប្រើ /story !", parse_mode="Markdown")
        else:
            await query.message.reply_text(
                "📚 *ប្រវត្តិរឿង* — ចុចដើម្បីអានម្ដងទៀត:",
                parse_mode="Markdown",
                reply_markup=history_keyboard(history),
            )
        return READING_STORY

    elif action == "home":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "🏠 *ទំព័រដើម*\n\n/story /settings /history /help",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    return READING_STORY

# ─── /help & /cancel ──────────────────────────────────────────────────────────
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📚 *Bot និទានរឿងខ្មែរ AI v3*\n\n"
        "*ជំហានប្រើ:*\n"
        "1️⃣ ទទួល API Key ឥតគិតថ្លៃ\n"
        "   ⚡ [console.groq.com/keys](https://console.groq.com/keys)\n"
        "   🤖 [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
        "2️⃣ /settings → ដាក់ key\n"
        "3️⃣ /story → ប្រភេទ → ប្រធានបទ → ទទួលរឿង\n"
        "4️⃣ វាយតម្លៃ ⭐ ឬ 🔄 ជំហ្វានថ្មី\n\n"
        "*ពាក្យបញ្ជា:*\n"
        "/start — ចាប់ផ្ដើម\n"
        "/story — បង្កើតរឿង\n"
        "/settings — ការកំណត់\n"
        "/history — ប្រវត្តិ + អានរឿងចាស់\n"
        "/help — ជំនួយ\n"
        "/random — Random រឿងភ្លាម\n"
        "/stats — ស្ថិតិ\n"
        "/cancel — បោះបង់\n\n"
        "*✨ ថ្មីក្នុង v3 + v4:*\n"
        "• ១២ ប្រភេទ (+ ប្រាជ្ញា + ភ័យខ្លាច)\n"
        "• ជ្រើសប្រវែង ខ្លី / មធ្យម / វែង\n"
        "• វាយតម្លៃ ⭐ + អានរឿងចាស់ m្ដងទៀត\n"
        "• AI Prompt ថ្មី — រឿងស្រស់ ជ្រៅ និងខុសគ្នារៀងរាល់ជំហ្វាន!\n"
        "• 🎲 /random — Random ប្រភេទ + ប្រធានបទដោយស្វ័យប្រវត្ត\n"
        "• ⏭️ បន្តរឿង — Chapter 2, 3... ក្នុងរឿងដដែល\n"
        "• 📊 /stats — ស្ថិតិ genre ពេញនិយម + ពិន្ទុ",
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 បានបោះបង់! ប្រើ /start ឬ /story ។")
    return ConversationHandler.END

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("❓ ប្រើ /help ដើម្បីមើលការណែនាំ។")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    settings_conv = ConversationHandler(
        entry_points=[
            CommandHandler("settings", settings_command),
            CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
        ],
        states={
            WAITING_FOR_KEY: [
                CommandHandler("settings", settings_command),
                CommandHandler("cancel",   cancel),
                CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
                CallbackQueryHandler(model_callback,    pattern=r"^model:"),
                CallbackQueryHandler(provider_callback, pattern=r"^prov:"),
                CallbackQueryHandler(length_callback,   pattern=r"^length:"),
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

    story_conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("story",  story_command),
            CommandHandler("random", random_command),
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
                CallbackQueryHandler(action_handler,       pattern=r"^action:"),
                CallbackQueryHandler(rating_callback,      pattern=r"^rate:"),
                CallbackQueryHandler(history_nav_callback, pattern=r"^hist:"),
                CallbackQueryHandler(length_callback,      pattern=r"^length:"),
            ],
            VIEWING_HISTORY: [
                CommandHandler("story",   story_command),
                CallbackQueryHandler(history_nav_callback, pattern=r"^hist:"),
                CallbackQueryHandler(rating_callback,      pattern=r"^rate:"),
            ],
        },
        fallbacks=[
            CommandHandler("cancel",   cancel),
            CommandHandler("settings", settings_command),
            CommandHandler("start",    start),
        ],
        allow_reentry=True,
    )

    history_conv = ConversationHandler(
        entry_points=[CommandHandler("history", history_command)],
        states={
            VIEWING_HISTORY: [
                CommandHandler("story", story_command),
                CallbackQueryHandler(history_nav_callback, pattern=r"^hist:"),
                CallbackQueryHandler(rating_callback,      pattern=r"^rate:"),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start",  start),
        ],
        allow_reentry=True,
    )

    app.add_handler(settings_conv)
    app.add_handler(story_conv)
    app.add_handler(history_conv)
    app.add_handler(CommandHandler("random", random_command))
    app.add_handler(CommandHandler("stats",  stats_command))
    app.add_handler(CommandHandler("help",   help_command))
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    logger.info("Khmer Storytelling Bot v4 started! (Random + Stats + Continue)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
