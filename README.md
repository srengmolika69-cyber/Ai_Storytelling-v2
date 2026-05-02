# 🎭 AI Khmer Storytelling Telegram Bot v2

Bot Telegram ដែលប្រើ AI **Groq + Gemini** សម្រាប់និទានរឿងខ្មែរ!

## ✨ What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| AI Provider | Gemini only | ⚡ Groq + 🤖 Gemini |
| Speed | ~3-5s | < 1s (Groq) |
| Auto-fallback | ❌ | ✅ |
| Model selection | ❌ | ✅ 7 models |
| Story genres | 8 | 10 (+Comedy, +Mystery) |
| Story history | ❌ | ✅ Last 20 stories |
| Key management | /setkey, /mykey | Unified /settings |

## 🤖 Supported AI Models

### ⚡ Groq (Free, Ultra-Fast)
- `llama-3.3-70b-versatile` — Best quality
- `llama-3.1-8b-instant` — Fastest (~0.3s)
- `mixtral-8x7b-32768` — Balanced
- `gemma2-9b-it` — Google's model via Groq

### 🤖 Gemini (Free)
- `gemini-1.5-flash` — Fast & good
- `gemini-1.5-pro` — Best quality
- `gemini-2.0-flash` — Newest

## 📖 Story Genres (10 types)

| Genre | Khmer | English |
|-------|-------|---------|
| 🏮 | រឿងនិទាន | Folk Tales |
| 👻 | រឿងខ្មោច | Ghost Stories |
| 💕 | រឿងស្នេហ៍ | Love Stories |
| ⚔️ | រឿងផ្សងព្រេង | Adventures |
| 🐘 | រឿងសត្វ | Animal Fables |
| 🌟 | រឿងព្រេង | Legends |
| 🏙️ | រឿងទំនើប | Modern Stories |
| 🌈 | រឿងកុមារ | Children's Stories |
| 😄 | រឿងកំប្លែង | Comedy *(NEW)* |
| 🔍 | រឿងអាថ៌កំបាំង | Mystery *(NEW)* |

## 🚀 Setup

### Step 1: Get API Keys

**Telegram Bot Token** — [@BotFather](https://t.me/BotFather) → `/newbot`

**Groq API Key (FREE, recommended):**
1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up free → Create API Key
3. Starts with `gsk_...`

**Gemini API Key (FREE, optional):**
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create API Key → starts with `AIzaSy...`

> 💡 You can add both! Bot will auto-fallback to the other if one fails.

### Step 2: Install

```bash
cd khmer_story_bot_v2
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in your keys
```

### Step 3: Configure .env

```env
TELEGRAM_TOKEN=1234567890:ABCdef...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...    # optional
```

### Step 4: Run

```bash
python bot.py
```

## 📱 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome + setup guide |
| `/story` | Create a new story |
| `/settings` | Manage keys, provider, model |
| `/history` | View last 20 stories |
| `/help` | Full help guide |
| `/cancel` | Cancel current action |

## 🔄 Auto-Fallback System

If your primary provider fails (quota, network, etc.), the bot **automatically** tries the other provider:

```
Groq fails → tries Gemini  ✅
Gemini fails → tries Groq  ✅
```

Story output shows which provider was actually used.

## 🌐 Deploy

### Railway.app
```bash
railway init && railway up
# Set env vars in Railway dashboard
```

### Render.com
1. Connect repo
2. Add environment variables
3. Deploy as "Web Service" with `python bot.py`

### Fly.io
```bash
fly launch
fly secrets set TELEGRAM_TOKEN=... GROQ_API_KEY=...
fly deploy
```

## 📦 Tech Stack

- `python-telegram-bot` v21 — Telegram Bot SDK
- `groq` v0.11 — Groq AI (Llama, Mixtral, Gemma)
- `google-generativeai` — Gemini models
- Python 3.10+

## 💡 Free Tier Limits

| Provider | RPM | RPD | Notes |
|----------|-----|-----|-------|
| Groq | 30 | 14,400 | Ultra-fast |
| Gemini | 15 | 1,500 | Reliable |

Use both keys together for maximum capacity!

---
_✨ Made with ❤️ for Khmer culture — v2 with Groq support_
