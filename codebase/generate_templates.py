"""
Notification Template Generator — BATCH OPTIMIZED
===================================================
Sends multiple rows in a single API call to reduce total API calls.
BATCH_SIZE rows per call, MAX_WORKERS parallel batches.
"""

import csv
import json
import os
import sys
import time
import re
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered output so every print appears immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
print("[STARTUP] Script imported successfully", flush=True)

# ── requests: required for Gemini REST calls ─────────────────────────────────
try:
    import requests as _requests
    print("[STARTUP] requests imported (Gemini REST API)", flush=True)
except ImportError:
    raise ImportError(
        "[STARTUP] 'requests' not installed. Run: pip install requests"
    )

# ── Configuration ─────────────────────────────────────────────────────────────
DEBUG = True  # Set to False to suppress verbose debug output


def dbg(label: str, value=None):
    if not DEBUG:
        return
    sep = "-" * 60
    if value is None:
        print(f"[DEBUG] {label}", flush=True)
    else:
        print(f"[DEBUG] {label}:\n{sep}\n{value}\n{sep}", flush=True)


from dotenv import load_dotenv
load_dotenv()

# ── API key: loaded from .env file ───────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")

dbg("GEMINI_API_KEY", f"{'*' * (len(API_KEY) - 4)}{API_KEY[-4:]}")

MODEL_ID = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={API_KEY}"
# Generate batched prompts to reduce API calls and speed output.
# Rows per API call  (2 rows × 15 tpl ≈ 10K output tokens) — reduced from 1 to cut requests by 50%
BATCH_SIZE = 2
MAX_WORKERS = 12   # Reduce to 12 since batch size increased; fewer parallel workers needed
dbg("Model", MODEL_ID)

# ── Input / output paths ──────────────────────────────────────────────────────
SEGMENT_GOALS_CSV = "iteration_0_before_learning/segment_goals.csv"
USER_SEGMENTS_CSV = "iteration_0_before_learning/user_segments.csv"
COMMUNICATION_THEME_CSV = "iteration_0_before_learning/communication_themes.csv"
TONE_HOOK_MATRIX = "iteration_0_before_learning/allowed_tone_hook_matrix.json"
OUTPUT_CSV = "iteration_0_before_learning/message_templates.csv"
dbg("Input CSVs",
    f"{SEGMENT_GOALS_CSV}, {USER_SEGMENTS_CSV}, {COMMUNICATION_THEME_CSV}")
dbg("Tone/hook matrix", TONE_HOOK_MATRIX)
dbg("Output CSV", OUTPUT_CSV)

# ── Load tone/hook matrix ─────────────────────────────────────────────────────
with open(TONE_HOOK_MATRIX, "r", encoding="utf-8") as _thf:
    _tone_hook_data = json.load(_thf)

_ALLOWED_TONES = _tone_hook_data["allowed_tones"]
_DISALLOWED_TONES = _tone_hook_data["disallowed_tones"]
_OCTALYSIS_HOOKS = _tone_hook_data["octalysis_hooks"]

_TONE_ALLOWED_STR = ", ".join(_ALLOWED_TONES)
_TONE_DISALLOWED_STR = ", ".join(_DISALLOWED_TONES)
_TONE_CHECKLIST_STR = " / ".join(_ALLOWED_TONES)

# Build octalysis markdown table dynamically from the JSON keys produced by task 1
_oct_rows = ["| # | Theme | Example Hook |", "|---|---|---|"]
for _i, (_theme, _hook) in enumerate(_OCTALYSIS_HOOKS.items(), 1):
    _oct_rows.append(f'| {_i} | {_theme} | "{_hook}" |')
_OCTALYSIS_TABLE_STR = "\n".join(_oct_rows)

# ── System Prompt ─────────────────────────────────────────────────────────────
BASE_PROMPT = """
## ROLE

You are an expert mobile notification copywriter specializing in behavioral psychology and user engagement for language-learning apps. You write push notifications that are concise, emotionally resonant, and drive specific user actions. You are fluent in English, Hindi, and Hinglish.

---

## COMPANY CONTEXT

**SpeakX** is an AI-powered English learning platform that helps non-native speakers build speaking confidence through real-time feedback and habit-forming exercises. Users often experience \"speech anxiety\" despite knowing grammar rules.

**Core Features:**
- AI Tutor (Sia): 1-on-1 voice-based conversation practice with real-time corrections
- Daily Streaks: Gamified tracker rewarding users for daily practice to build long-term habits
- Leaderboards: Social competition — users earn points from lesson completion & accuracy
- Personalized Curriculum: Lessons that adapt based on proficiency level and progress
- Voice Analytics: Visual feedback on pronunciation, fluency, and vocabulary usage

**North Star Metric:** Active Speakers — users completing at least one AI Tutor session per week. Goal: move users from \"Passive Learners\" to \"Active Speakers.\"

---

## TASK

For **each** input row below, generate **15 notification templates** across 3 assigned themes.

Each theme gets exactly **5 bilingual templates**.
Each template contains:
- **title** in English and Hinglish
- **body** in English and Hinglish
- **CTA** in English and Hinglish

Total per row: **5 templates × 3 themes = 15 templates**

---
 
## RULES (STRICT — VIOLATIONS WILL BE REJECTED)

### 1. Character Constraints
| Element | Hard Limit |
|---|---|
| Title | ≤ 40 characters |
| Body | ≤ 90 characters |

Count every character including emojis and spaces.

### 2. Dynamic Variables
Use naturally — do not force all into every message:
- `{coins_balance}` — user's current coin balance
- `{streak_current}` — user's current streak count
- `{exercises_completed_7d}` — exercises completed in last 7 days
- `{sessions_last_7d}` — sessions completed in last 7 days

### 3. One Goal Per Message
Each notification drives **one single action**. No compound CTAs.

### 4. Emoji Framework
- **Bookend Method:** One emoji at the start + one at the end
- **Limit:** 1–3 emojis max per notification
- **Front-load:** Lead emoji in the first 10 characters

### 5. Tone
**ALLOWED:** <<ALLOWED_TONES>>
**NEVER USE:** <<DISALLOWED_TONES>>

### 6. Language Definitions
- **English (EN):** 100% English. Clear, conversational, action-oriented.
- **Hinglish (HI):** Natural urban code-switching as spoken by Indian professionals and students aged 18–30.
  Roman script only. NOT Devanagari. NOT pure Hindi.

  **THE MOST IMPORTANT RULE — READ CAREFULLY:**
  Do NOT write the English version first and then translate it into Hindi.
  Write the Hinglish version completely independently, from scratch.
  Think: \"How would a confident, warm Indian professional say this to a friend over WhatsApp?\"
  The Hinglish and English versions should feel like two separate original messages — same intent, completely different phrasing.

  **Language split:**
  - Hindi carries: emotion, urgency, personality, reaction, warmth
  - English carries: app features only — rank, leaderboard, streak, session, lesson, points, level, score, practice, coins, tutor

  **Pronouns — use \"aap/aapki/aapka\"** for respectful-but-warm tone. NEVER \"tu/teri/tera/tum\".

  **Hindi-first sentence construction:**
  Lead with the Hindi emotional punch, then drop in the English app term as a landing pad.
  - ✅ \"Kaafi din ho gaye — ek practice session toh banta hai!\"
  - ✅ \"Yaar, streak toot gayi toh sab mehnat bekar! Aaj karke lo.\"
  - ✅ \"Sabka score badh raha hai — aap kab pakdenge?\"
  - ✅ \"Bas ek aur session, aur aap top 10 mein honge!\"
  - ✅ \"Arrey, coins expire ho rahe hain — abhi use karo!\"

  **Emotional vocabulary:**
  - Urgency: abhi, aaj hi, ruk mat, der ho rahi hai, jaldi
  - FOMO: baaki sab nikal rahe hain, aap peeche reh jayenge, sab aage ja rahe hain
  - Encouragement: bas thoda aur, sab sambhav hai, itna aasan hai, ek baar try karo
  - Loss/Streak: toot jaayegi, sab bekar, mehnat waste, bahut bura lagega

  **Absolute DON'Ts:**
  - ❌ Never mirror English sentence order in Hindi: subject → verb → object → postposition
  - ❌ Never produce a translated version: if English says \"Don't lose your streak\", Hinglish should NOT say \"Apna streak mat khona\" — think of a completely new angle
  - ❌ Never use overly casual: \"tu\", \"teri\", \"tera\", \"tum\"
  - ❌ Never use robotic urgency: \"Abhi action lijiye\" or \"Apna goal complete karein\"

  **Before/After reference — study the structural difference, not just the words:**
  | ❌ Sounds Translated | ✅ Written Fresh in Hinglish |
  |---|---|
  | \"Apni pehli rank prapt karo!\" | \"Pehli rank itni door bhi nahi — pakad lo!\" |
  | \"Aapke 5 ranks neeche gir gaye hain\" | \"Yaar, 5 ranks ki slip ho gayi — abhi recover karo!\" |
  | \"Doosre log practice kar rahe hain jab aap nahi hain\" | \"Baaki sab session mein hain — aap kab aa rahe ho?\" |
  | \"Der hone se pehle catch up karo\" | \"Abhi nahi kiya toh aur mushkil ho jayega!\" |
  | \"Apna streak bachao!\" | \"Itne din ki mehnat — aaj toot jaaye toh? Kar lo!\" |
  | \"Apne coins claim karo abhi\" | \"Coins expire ho rahe hain yaar — use karo abhi!\" |
  | \"Aapka trial khatam hone wala hai\" | \"Trial ke aakhri din hain — iska faayda uthao!\" |
  | \"Practice karke apna fluency score badhao\" | \"Score dekha? Ek session mein kitna badh sakta hai!\" |

### 7. CTA Rules
- **Length:** 3–10 words only
- **Language:** Match the column language (EN or Hinglish)
- **Style:** Start with a strong action verb (e.g. Start, Claim, Save, Join, Practice, Unlock, Resume)
- **Urgency:** Create a sense of immediacy — use words like Now, Today, Before It's Gone, Don't Miss Out
- **Relevance:** CTA must reflect the segment's primary goal and sub goal in short
- Examples EN: \"Practice Now — Keep Your Streak!\", \"Claim Your Reward Today\"
- Examples HI: \"Abhi Practice Karo!\", \"Aaj Hi Streak Bachao!\", \"Wapas Aa Jaiye\", \"Rank Pakad Lo Abhi\"
---

### 8. Hinglish Tone Rules
Hinglish notifications must feel like a warm Indian professional texting a colleague or
friend on WhatsApp — NOT an English notification run through Google Translate.

**The \"fresh draft\" test:** After writing the Hinglish version, ask yourself:
\"Could I have predicted exactly these words just by translating the English?\" 
If yes → throw it away and write a completely new Hinglish take on the same intent.

- Every Hinglish notification needs at least ONE warmth anchor:
  yaar / bhai / arrey / sun / deko / suno / ek kaam karo
- Negative constructions (\"mat\", \"nahi\", \"toot\") must always come with a softener:
  ✅ \"Itni mehnat toot jaaye — aaj ek session kar lo!\" 
  ❌ \"Streak mat todna\" (sounds like a command, not a friend)
- Change the angle, not just the words: if the English says \"Your streak is about to break\",
  the Hinglish should not say \"Aapki streak tootne wali hai.\" Instead, try a different framing:
  \"Itne dinon ki practice — ek din ke liye waste? Kar lo jaldi!\"
- Sentence rhythm: read it aloud. If it sounds stiff or robotic, rewrite it.
- Structure: Hindi emotion → Hindi context → English app term (as the natural landing word)
  ✅ \"Kaafi time ho gaya — ek session toh banta hai!\"
  ❌ \"Aapka session complete karna important hai\"
---

## OCTALYSIS THEMES (Behavioral Framework)

<<OCTALYSIS_TABLE>>

---

## INPUT ROWS

(Rows will be provided in the user message.)

---

## QUALITY CHECKLIST (self-validate every template before output)

- [ ] title_en ≤ 40 characters | title_hi ≤ 40 characters
- [ ] body_en ≤ 90 characters | body_hi ≤ 90 characters
- [ ] cta_en is 3–10 words, starts with action verb, creates urgency
- [ ] cta_hi is 3–10 words, natural Hinglish, starts with action verb, creates urgency
- [ ] 1–3 emojis per field, lead emoji in first 10 characters
- [ ] One CTA only per template
- [ ] Tone is <<TONE_CHECKLIST>>
- [ ] Dynamic variables wrapped in `{curly_braces}` from the allowed list
- [ ] Template reflects the correct Octalysis theme angle
- [ ] Template aligns with the Sub Goal and Messaging Focus
- [ ] English fields = 100% English | Hinglish (HI) fields = natural code-switched Hindi+English
- [ ] Hinglish fields: NOT a word-for-word translation — written fresh, not mapped from English
- [ ] Hinglish \"fresh draft\" test: could you have predicted every word just by translating the English? If yes, rewrite it.
- [ ] Hinglish structure: Hindi emotion/hook leads → English app term lands at the end as the natural anchor

---

## OUTPUT FORMAT (STRICT)

Return ONLY a valid JSON array with one object per input row. No markdown, no code fences, no commentary, no trailing commas.

Each object has: segment_id, lifecycle_stage, and a \"themes\" array of exactly 3 theme groups.
Each theme group has a \"theme\" string and a \"templates\" array of exactly 5 bilingual template objects.

[
  {
    "segment_id": "<seg>",
    "lifecycle_stage": "<lc>",
    "themes": [
      {
        "theme": "<Theme1>",
        "templates": [
          {
            "title_en": "...",
            "title_hi": "...",
            "body_en":  "...",
            "body_hi":  "...",
            "cta_en":   "...",
            "cta_hi":   "..."
          },
          { ... },
          { ... },
          { ... },
          { ... }
        ]
      },
      {"theme": "<Theme2>", "templates": [{...}, {...}, {...}, {...}, {...}]},
      {"theme": "<Theme3>", "templates": [{...}, {...}, {...}, {...}, {...}]}
    ]
  }
]
"""

# ── Substitute dynamic placeholders from tone/hook matrix ────────────────────
BASE_PROMPT = (
    BASE_PROMPT
    .replace("<<ALLOWED_TONES>>",    _TONE_ALLOWED_STR)
    .replace("<<DISALLOWED_TONES>>", _TONE_DISALLOWED_STR)
    .replace("<<OCTALYSIS_TABLE>>",  _OCTALYSIS_TABLE_STR)
    .replace("<<TONE_CHECKLIST>>",   _TONE_CHECKLIST_STR)
)

_CACHED_SYSTEM = BASE_PROMPT


# ── Data Loader ───────────────────────────────────────────────────────────────

def _read_csv(path: str) -> list[dict]:
    """Read a CSV file and return a list of row dicts."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _mode(values):
    """Return the most common non-empty value in a list."""
    counts = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _ensure_communication_theme(
    communication_theme_path: str,
    user_segments_path: str,
    tone_hook_matrix_path: str,
) -> None:
    """
    Auto-generate communication_theme.csv from task 1 outputs if it doesn't exist.

    Reads user_segments.csv (dominant_propensity per segment) and
    allowed_tone_hook_matrix.json (available Octalysis drives) to assign
    3 behaviorally-appropriate drives per segment_id.

    Only runs when the file is absent — existing files are never overwritten.
    """
    if os.path.exists(communication_theme_path):
        return

    print(
        f"   ℹ️  {communication_theme_path} not found — "
        "auto-generating from user_segments.csv + allowed_tone_hook_matrix.json",
        flush=True,
    )

    # Load available drives from task 1's tone/hook JSON
    with open(tone_hook_matrix_path, "r", encoding="utf-8") as f:
        thm = json.load(f)
    drives = list(thm.get("octalysis_hooks", {}).keys())

    if len(drives) < 3:
        raise ValueError(
            f"allowed_tone_hook_matrix.json has fewer than 3 Octalysis drives — "
            f"check task 1 output. Found: {drives}"
        )

    # Build a name→index lookup so mapping survives any short/long name variant
    # that Gemini chose when producing the JSON
    def _find_drive(keywords: list[str]) -> str:
        """Return first drive whose name contains any of the keywords (case-insensitive)."""
        for kw in keywords:
            for d in drives:
                if kw.lower() in d.lower():
                    return d
        return drives[0]  # ultimate fallback: first drive in list

    # Propensity → 3 Octalysis drives (keyword-matched against actual drive names)
    _PROPENSITY_TO_DRIVES: dict[str, list[list[str]]] = {
        "gamification":  [["accomplishment", "development"], ["ownership"],            ["loss", "avoidance"]],
        "ai_tutor":      [["epic", "meaning", "calling"],    ["accomplishment"],        ["empowerment"]],
        "leaderboard":   [["social", "influence"],           ["scarcity"],              ["loss", "avoidance"]],
        "social":        [["epic", "meaning", "calling"],    ["social", "influence"],   ["empowerment"]],
    }
    _DEFAULT_DRIVES: list[list[str]] = [
        ["accomplishment", "development"],
        ["social", "influence"],
        ["loss", "avoidance"],
    ]

    # Read unique segment_id → dominant_propensity from user_segments.csv
    us_rows = _read_csv(user_segments_path)
    seg_dominant: dict[str, str] = {}
    for r in us_rows:
        sid = r.get("segment_id", "")
        dom = r.get("dominant_propensity", "").lower().strip()
        if sid and sid not in seg_dominant:
            seg_dominant[sid] = dom

    # Build rows
    out_rows = []
    for sid, dom in sorted(seg_dominant.items(), key=lambda x: x[0]):
        kw_groups = _PROPENSITY_TO_DRIVES.get(dom, _DEFAULT_DRIVES)
        d1 = _find_drive(kw_groups[0])
        d2 = _find_drive(kw_groups[1])
        d3 = _find_drive(kw_groups[2])
        # Avoid d2 / d3 duplicating d1 (fall back to positional drive in that case)
        if d2 == d1:
            d2 = drives[1] if len(drives) > 1 else d1
        if d3 == d1 or d3 == d2:
            d3 = drives[2] if len(drives) > 2 else d2
        out_rows.append({
            "segment_id": sid,
            "drive_1":    d1,
            "drive_2":    d2,
            "drive_3":    d3,
            "reasoning":  (
                f"Dominant propensity '{dom}': "
                f"{d1} drives core engagement; "
                f"{d2} reinforces secondary motivation; "
                f"{d3} guards against disengagement."
            ),
        })

    os.makedirs(os.path.dirname(communication_theme_path), exist_ok=True)
    with open(communication_theme_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["segment_id", "drive_1", "drive_2", "drive_3", "reasoning"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"   ✅ communication_theme.csv auto-generated "
        f"({len(out_rows)} segments → {communication_theme_path})",
        flush=True,
    )


def load_data(
    segment_goals_path: str,
    user_segments_path: str,
    communication_theme_path: str,
) -> list[dict]:
    """
    Merges 3 CSVs and returns a list of row dicts ready for build_prompt().
    One entry per segment_id × lifecycle_stage combination.

    communication_theme.csv is auto-generated from task 1 outputs if absent.
    """
    print("📂 Loading CSVs...", flush=True)

    # ── Auto-generate communication_theme.csv if task 1 didn't produce it ────
    _ensure_communication_theme(
        communication_theme_path, user_segments_path, TONE_HOOK_MATRIX)

    # spine: segment_id × lifecycle_stage
    sg_rows = _read_csv(segment_goals_path)
    us_rows = _read_csv(user_segments_path)     # per-user data
    th_rows = _read_csv(communication_theme_path)  # themes per segment_id

    dbg("load_data → segment_goals rows",  str(len(sg_rows)))
    dbg("load_data → user_segments rows",  str(len(us_rows)))
    dbg("load_data → communication_theme rows", str(len(th_rows)))

    # ── 1. Aggregate user_segments by segment_id ──────────────────────────────
    us_by_seg: dict[str, dict] = {}
    for r in us_rows:
        sid = r.get("segment_id", "")
        if sid not in us_by_seg:
            us_by_seg[sid] = {
                "segment_description": r.get("segment_description", ""),
                "age_bands": [], "regions": [],
                "preferred_hours": [], "days_since_signup": [],
            }
        us_by_seg[sid]["age_bands"].append(r.get("age_band", ""))
        us_by_seg[sid]["regions"].append(r.get("region", ""))
        h = r.get("preferred_hour", "")
        if h:
            us_by_seg[sid]["preferred_hours"].append(float(h))
        d = r.get("days_since_signup", "")
        if d:
            us_by_seg[sid]["days_since_signup"].append(float(d))

    us_agg: dict[str, dict] = {}
    for sid, data in us_by_seg.items():
        hours = data["preferred_hours"]
        days = data["days_since_signup"]
        us_agg[sid] = {
            "segment_description":  data["segment_description"],
            "age_band":             _mode(data["age_bands"]),
            "top_region":           _mode(data["regions"]),
            "preferred_hour":       round(sum(hours) / len(hours)) if hours else 12,
            "avg_days_since_signup": round(sum(days) / len(days), 1) if days else 0,
        }

    # ── 2. Index communication_theme by segment_id ────────────────────────────
    th_by_seg: dict[str, dict] = {}
    for r in th_rows:
        sid = r.get("segment_id", "")
        if sid not in th_by_seg:
            th_by_seg[sid] = {
                "theme_1":         r.get("drive_1", ""),
                "theme_2":         r.get("drive_2", ""),
                "theme_3":         r.get("drive_3", ""),
                "theme_reasoning": r.get("reasoning", ""),
            }

    # ── 3. Build final rows (spine = segment_goals) ───────────────────────────
    result = []
    for r in sg_rows:
        sid = r.get("segment_id", "")
        us = us_agg.get(sid, {})
        th = th_by_seg.get(sid, {})

        row = {
            # Core identity
            "segment_id":          sid,
            "segment_name":        r.get("segment_name", ""),
            "segment_description": us.get("segment_description", ""),
            "lifecycle_stage":     r.get("lifecycle_stage", ""),

            # Goals & messaging (from segment_goals)
            "kb_feature_outcome":  r.get("kb_feature_outcome", "") or "N/A",
            "primary_goal":        r.get("primary_goal", ""),
            "sub_goal":            r.get("sub_goal", ""),
            "messaging_focus":     r.get("messaging_focus", ""),

            # Themes (from communication_theme — auto-generated if absent)
            "theme_1":             th.get("theme_1", "Accomplishment"),
            "theme_2":             th.get("theme_2", "Ownership"),
            "theme_3":             th.get("theme_3", "Loss Avoidance"),
            "theme_reasoning":     th.get("theme_reasoning", ""),
        }
        result.append(row)

    dbg("load_data → total merged rows", str(len(result)))
    print(
        f"   ✅ Merged into {len(result)} rows (segment × lifecycle combinations)", flush=True)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_row_block(idx: int, row: dict) -> str:
    """Build a numbered INPUT ROW block for one row."""
    theme1 = row.get("theme_1") or "Accomplishment"
    theme2 = row.get("theme_2") or "Ownership"
    theme3 = row.get("theme_3") or "Loss Avoidance"
    return (
        f"### INPUT ROW {idx}\n"
        f"- **Segment:** {row.get('segment_id', '')}\n"
        f"- **Segment Description:** {row.get('segment_description', '')}\n"
        f"- **Lifecycle Stage:** {row.get('lifecycle_stage', '')}\n"
        f"- **KB Feature / Outcome:** {row.get('kb_feature_outcome', '') or 'N/A'}\n"
        f"- **Sub Goal:** {row.get('sub_goal', '')}\n"
        f"- **Messaging Focus:** {row.get('messaging_focus', '')}\n"
        f"- **Theme 1:** {theme1}\n"
        f"- **Theme 2:** {theme2}\n"
        f"- **Theme 3:** {theme3}\n"
        f"- **Theme Reasoning:** {row.get('theme_reasoning', '') or 'N/A'}\n"
    )


def build_batch_prompt(rows: list[dict]) -> str:
    """Build the user-message containing numbered INPUT ROW blocks for all rows in the batch."""
    parts = [
        f"Generate templates for the following {len(rows)} input row(s).\n"]
    for i, row in enumerate(rows, 1):
        dbg(f"build_batch_prompt → row {i}", json.dumps(
            dict(row), ensure_ascii=False, indent=2))
        parts.append(_build_row_block(i, row))
    prompt = "\n".join(parts)
    dbg("build_batch_prompt → total chars", str(len(prompt)))
    return prompt


def _flatten_themes(themes_list: list[dict]) -> list[dict]:
    """Flatten a list of theme groups into a flat list of template dicts."""
    templates = []
    for group in themes_list:
        theme = group.get("theme", "")
        for t in group.get("templates", []):
            templates.append({
                "theme_used":        theme,
                "message_title_en":  t.get("title_en", ""),
                "message_title_hi":  t.get("title_hi", ""),
                "message_body_en":   t.get("body_en", ""),
                "message_body_hi":   t.get("body_hi", ""),
                "cta_text_en":       t.get("cta_en", ""),
                "cta_text_hi":       t.get("cta_hi", ""),
            })
    return templates


def parse_batch_response(text: str, rows: list[dict]) -> list[tuple[dict, list[dict]]]:
    """
    Parse the batch JSON response and return a list of (row, templates) pairs.
    Falls back to single-row format if only 1 row was sent.
    """
    dbg("parse_batch_response → raw response (first 500 chars)", text[:500])

    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    text = re.sub(r",\s*([\]}])", r"\1", text)
    dbg("parse_batch_response → text after cleanup (first 300 chars)",
        text[:300])

    try:
        parsed = json.loads(text)
        dbg("parse_batch_response → json.loads succeeded ✅")
    except json.JSONDecodeError as json_err:
        dbg("parse_batch_response → json.loads failed", str(json_err))
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            dbg("parse_batch_response → regex fallback found a JSON array")
            parsed = json.loads(match.group())
            dbg("parse_batch_response → regex fallback succeeded ✅")
        else:
            dbg("parse_batch_response → ❌ no JSON array found")
            raise ValueError(
                f"Could not parse JSON from response:\n{text[:300]}")

    if not isinstance(parsed, list) or len(parsed) == 0:
        raise ValueError("Parsed JSON is not a non-empty list")

    results: list[tuple[dict, list[dict]]] = []

    if isinstance(parsed[0], dict) and "themes" in parsed[0]:
        # Batch format: [{segment_id, lifecycle_stage, themes: [...]}, ...]
        dbg("parse_batch_response → detected batch format")
        row_lookup = {(r.get("segment_id", ""), r.get(
            "lifecycle_stage", "")): r for r in rows}
        for entry in parsed:
            sid = str(entry.get("segment_id", ""))
            lc = entry.get("lifecycle_stage", "")
            matched_row = row_lookup.get((sid, lc))
            if not matched_row:
                dbg(
                    f"parse_batch_response → ⚠️  no row match for seg={sid} lc={lc}, using positional")
                idx = len(results)
                matched_row = rows[idx] if idx < len(rows) else rows[-1]
            templates = _flatten_themes(entry.get("themes", []))
            dbg(
                f"parse_batch_response → seg={sid} lc={lc} → {len(templates)} templates")
            results.append((matched_row, templates))
    elif isinstance(parsed[0], dict) and "templates" in parsed[0]:
        # Single-row legacy format: [{theme, templates: [...]}, ...]
        dbg("parse_batch_response → detected single-row grouped format, wrapping")
        templates = _flatten_themes(parsed)
        results.append((rows[0], templates))
    else:
        raise ValueError(
            f"Unexpected JSON format. First element keys: {list(parsed[0].keys()) if parsed else '?'}")

    expected = len(rows) * 15
    actual = sum(len(t) for _, t in results)
    dbg(
        f"parse_batch_response → total templates: {actual} (expected {expected})")
    if actual != expected:
        dbg(f"parse_batch_response → ⚠️  template count mismatch")

    return results


def generate_batch(rows: list[dict], max_retries: int = 5) -> list[tuple[dict, list[dict]]]:
    """Call Gemini for a batch of rows and return list of (row, templates) pairs."""
    dbg(f"generate_batch → {len(rows)} row(s)")
    user_msg = build_batch_prompt(rows)
    max_tok = min(16384 * len(rows), 65536)
    dbg(f"generate_batch → max_tokens={max_tok}")

    for attempt in range(1, max_retries + 1):
        dbg(f"generate_batch → API call attempt {attempt}/{max_retries}")
        try:
            t0 = time.time()
            payload = {
                "system_instruction": {"parts": [{"text": _CACHED_SYSTEM}]},
                "contents": [{"parts": [{"text": user_msg}]}],
                "generationConfig": {"maxOutputTokens": max_tok},
            }
            resp = _requests.post(GEMINI_URL, json=payload, timeout=300)
            api_ms = int((time.time() - t0) * 1000)

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Gemini API {resp.status_code}: {resp.text[:300]}")

            data = resp.json()

            # ── Token usage logging ───────────────────────────────────────────
            usage = data.get("usageMetadata", {})
            in_tok = usage.get("promptTokenCount", 0)
            out_tok = usage.get("candidatesTokenCount", 0)
            print(
                f"      📊 tokens  in={in_tok}  out={out_tok}  api_time={api_ms}ms")

            # ── Truncation guard ──────────────────────────────────────────────
            candidates = data.get("candidates", [])
            finish_reason = candidates[0].get(
                "finishReason", "") if candidates else ""
            if finish_reason == "MAX_TOKENS":
                dbg("generate_batch → ⚠️  TRUNCATED (finishReason=MAX_TOKENS)")
                if attempt < max_retries:
                    max_tok = min(max_tok + 4096, 32768)
                    print(
                        f"      ⚠️  Response truncated, bumping max_tokens to {max_tok} and retrying...")
                    time.sleep(1)
                    continue
                else:
                    print(
                        f"      ❌ Response still truncated after {max_retries} attempts")

            dbg("generate_batch → API response received ✅")
            response_text = candidates[0]["content"]["parts"][0]["text"] if candidates else ""
            dbg("generate_batch → response length (chars)", str(len(response_text)))
            dbg("generate_batch → finish_reason", finish_reason)
            return parse_batch_response(response_text, rows)

        except Exception as e:
            err_str = str(e)
            dbg(f"generate_batch → attempt {attempt} exception", err_str)
            print(
                f"      ⚠️  Attempt {attempt}/{max_retries} failed: {err_str[:100]}")

            if "API_KEY_INVALID" in err_str or "invalid api key" in err_str.lower():
                print(f"      ❌ Invalid or expired API key. Aborting retries.")
                raise

            if ("quota" in err_str.lower() or "rate" in err_str.lower() or "429" in err_str) and attempt < max_retries:
                delay_match = re.search(r"retry in ([\d.]+)s", err_str)
                wait = float(delay_match.group(1)) + 5 if delay_match else 60
                print(f"      ⏳ Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise


# ── Resume helpers ────────────────────────────────────────────────────────────

def _load_completed_pairs(csv_path: str) -> tuple[set[tuple[str, str]], int]:
    """
    Read an existing output CSV and return:
      1. set of (segment_id, lifecycle_stage) pairs already written
      2. highest TPL_NNNN number seen (so new IDs continue from there)
    """
    done: set[tuple[str, str]] = set()
    max_tid = 0
    if not os.path.exists(csv_path):
        return done, max_tid
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row.get("segment_id", ""), row.get("lifecycle_stage", "")))
            tid_str = row.get("template_id", "")
            if tid_str.startswith("TPL_"):
                try:
                    max_tid = max(max_tid, int(tid_str.split("_")[1]))
                except ValueError:
                    pass
    return done, max_tid


# ── Update Pipeline ───────────────────────────────────────────────────────────

def update_templates(
    segment_goals_old: str = "iteration_0_before_learning/segment_goals.csv",
    segment_goals_new: str = "iteration_1_after_learning/segment_goals.csv",
    user_segments: str = USER_SEGMENTS_CSV,
    communication_theme: str = COMMUNICATION_THEME_CSV,
    message_templates_in: str = "iteration_0_before_learning/message_templates.csv",
    message_templates_out: str = "iteration_1_after_learning/message_templates.csv",
):
    """
    Diff segment_goals_new vs segment_goals (old).  Re-generate templates ONLY
    for rows that changed.  Produce message_templates_new.csv that keeps unchanged
    templates intact (with their original template_ids) and swaps in new ones for
    the changed rows — also re-using the original template_ids.
    """
    print("🔄 UPDATE MODE", flush=True)

    old_rows = _read_csv(segment_goals_old)
    new_rows = _read_csv(segment_goals_new)

    old_by_key: dict[tuple[str, str], dict] = {}
    for r in old_rows:
        key = (r.get("segment_id", ""), r.get("lifecycle_stage", ""))
        old_by_key[key] = r

    new_by_key: dict[tuple[str, str], dict] = {}
    for r in new_rows:
        key = (r.get("segment_id", ""), r.get("lifecycle_stage", ""))
        new_by_key[key] = r

    changed_keys: list[tuple[str, str]] = []
    for key, new_r in new_by_key.items():
        old_r = old_by_key.get(key)
        if old_r is None:
            changed_keys.append(key)
            dbg(f"update_templates → NEW row {key}")
        elif new_r != old_r:
            changed_keys.append(key)
            dbg(f"update_templates → CHANGED row {key}")

    if not changed_keys:
        print("✅ No differences found between segment_goals files — nothing to update.", flush=True)
        return

    print(
        f"   🔍 Found {len(changed_keys)} changed (segment_id, lifecycle_stage) pair(s):", flush=True)
    for k in changed_keys:
        print(f"      • segment_id={k[0]}  lifecycle_stage={k[1]}", flush=True)

    all_new_data = load_data(
        segment_goals_new, user_segments, communication_theme)
    rows_to_generate = [
        r for r in all_new_data
        if (r.get("segment_id", ""), r.get("lifecycle_stage", "")) in set(changed_keys)
    ]
    print(
        f"   📦 Will regenerate templates for {len(rows_to_generate)} row(s)", flush=True)

    existing_templates = _read_csv(message_templates_in)
    print(
        f"   📄 Loaded {len(existing_templates)} existing templates from {message_templates_in}", flush=True)

    existing_tids: dict[tuple[str, str], list[str]] = {}
    for t in existing_templates:
        key = (t.get("segment_id", ""), t.get("lifecycle_stage", ""))
        existing_tids.setdefault(key, []).append(t.get("template_id", ""))

    changed_set = set(changed_keys)
    new_template_map: dict[tuple[str, str], list[dict]] = {}

    batches = [rows_to_generate[i:i + BATCH_SIZE]
               for i in range(0, len(rows_to_generate), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"   🚀 Generating in {total_batches} batch(es)...", flush=True)

    for batch_idx, batch_rows in enumerate(batches, 1):
        seg_ids = [r.get("segment_id", "?") + "/" +
                   r.get("lifecycle_stage", "?") for r in batch_rows]
        print(
            f"\n   Batch [{batch_idx}/{total_batches}] — {seg_ids}", flush=True)
        results = generate_batch(batch_rows)
        for row, templates in results:
            key = (row.get("segment_id", ""), row.get("lifecycle_stage", ""))
            new_template_map[key] = templates

    fieldnames = [
        "segment_id", "lifecycle_stage", "theme_used", "template_id",
        "message_title_en", "message_title_hi",
        "message_body_en", "message_body_hi",
        "cta_text_en", "cta_text_hi",
        "generated_at",
    ]

    max_tid = 0
    for tids in existing_tids.values():
        for tid in tids:
            if tid.startswith("TPL_"):
                try:
                    max_tid = max(max_tid, int(tid.split("_")[1]))
                except ValueError:
                    pass
    next_tid = max_tid + 1

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    output_rows: list[dict] = []
    changed_written: set[tuple[str, str]] = set()

    for t in existing_templates:
        key = (t.get("segment_id", ""), t.get("lifecycle_stage", ""))

        if key in changed_set and key not in changed_written:
            old_tids = existing_tids.get(key, [])
            new_tpls = new_template_map.get(key, [])
            for i, nt in enumerate(new_tpls):
                tid = old_tids[i] if i < len(
                    old_tids) else f"TPL_{next_tid:04d}"
                if i >= len(old_tids):
                    next_tid += 1
                output_rows.append({
                    "segment_id":       key[0],
                    "lifecycle_stage":  key[1],
                    "theme_used":       nt.get("theme_used", ""),
                    "template_id":      tid,
                    "message_title_en": nt.get("message_title_en", ""),
                    "message_title_hi": nt.get("message_title_hi", ""),
                    "message_body_en":  nt.get("message_body_en", ""),
                    "message_body_hi":  nt.get("message_body_hi", ""),
                    "cta_text_en":      nt.get("cta_text_en", ""),
                    "cta_text_hi":      nt.get("cta_text_hi", ""),
                    "generated_at":     now_utc,
                })
            changed_written.add(key)
        elif key in changed_set:
            continue
        else:
            output_rows.append(t)

    for key in changed_keys:
        if key not in changed_written:
            new_tpls = new_template_map.get(key, [])
            for nt in new_tpls:
                tid = f"TPL_{next_tid:04d}"
                next_tid += 1
                output_rows.append({
                    "segment_id":       key[0],
                    "lifecycle_stage":  key[1],
                    "theme_used":       nt.get("theme_used", ""),
                    "template_id":      tid,
                    "message_title_en": nt.get("message_title_en", ""),
                    "message_title_hi": nt.get("message_title_hi", ""),
                    "message_body_en":  nt.get("message_body_en", ""),
                    "message_body_hi":  nt.get("message_body_hi", ""),
                    "cta_text_en":      nt.get("cta_text_en", ""),
                    "cta_text_hi":      nt.get("cta_text_hi", ""),
                    "generated_at":     now_utc,
                })
            changed_written.add(key)

    os.makedirs(os.path.dirname(message_templates_out), exist_ok=True)
    with open(message_templates_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    changed_count = sum(1 for k in changed_keys if k in new_template_map)
    new_tpl_total = sum(len(v) for v in new_template_map.values())
    unchanged_count = len([t for t in existing_templates
                           if (t.get("segment_id", ""), t.get("lifecycle_stage", ""))
                           not in changed_set])

    print(f"\n{'='*60}")
    print(f"✅ UPDATE COMPLETE")
    print(f"   Changed pairs regenerated : {changed_count}")
    print(f"   New templates generated    : {new_tpl_total}")
    print(f"   Unchanged templates kept   : {unchanged_count}")
    print(f"   Total templates written    : {len(output_rows)}")
    print(f"   Output file                : {message_templates_out}")
    print(f"{'='*60}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    dbg("main → starting")
    ROW_LIMIT = None  # Set to an int to cap rows for testing

    dbg("main → ROW_LIMIT", str(ROW_LIMIT))
    try:
        reader = load_data(SEGMENT_GOALS_CSV,
                           USER_SEGMENTS_CSV, COMMUNICATION_THEME_CSV)
        dbg("main → load_data complete, total rows before limit", str(len(reader)))
    except FileNotFoundError as e:
        dbg("main → ❌ CSV file not found!", str(e))
        raise
    except Exception as e:
        dbg("main → ❌ error loading CSVs", str(e))
        raise

    if ROW_LIMIT:
        reader = reader[:ROW_LIMIT]

    fieldnames = [
        "segment_id", "lifecycle_stage", "theme_used", "template_id",
        "message_title_en", "message_title_hi",
        "message_body_en", "message_body_hi",
        "cta_text_en", "cta_text_hi",
        "generated_at",
    ]

    output_path = OUTPUT_CSV
    done_pairs = set()
    max_tid = 0
    total = len(reader)

    if total == 0:
        print(f"✅ No rows to process.")
        return

    batches = [reader[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    total_batches = len(batches)

    dbg("main → rows to process after filtering", str(total))
    print(
        f"📂 Processing {total} rows in {total_batches} batches  (BATCH_SIZE={BATCH_SIZE}, MAX_WORKERS={MAX_WORKERS})")
    print(f"   → API calls: {total_batches}")
    print(
        f"   → Will generate {total * 15} templates total (5 bilingual templates per theme × 3 themes per row)")
    print(f"   → Output: {output_path}\n")

    errors = []
    write_lock = threading.Lock()
    errors_lock = threading.Lock()
    tid_counter = [max_tid + 1]
    completed = [0]
    attempted = [0]
    run_start = time.time()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        def process_batch(batch_idx: int, batch_rows: list[dict]):
            batch_start = time.time()
            seg_ids = [r.get("segment_id", "?") + "/" +
                       r.get("lifecycle_stage", "?") for r in batch_rows]
            print(
                f"\nBatch [{batch_idx}/{total_batches}] — {len(batch_rows)} row(s): {seg_ids}")

            try:
                dbg(
                    f"process_batch → calling generate_batch for batch {batch_idx}")
                results = generate_batch(batch_rows)
                dbg(f"process_batch → {len(results)} result(s) returned for batch {batch_idx}")

                with write_lock:
                    for row, templates in results:
                        seg = row.get("segment_id", "")
                        lc = row.get("lifecycle_stage", "")
                        rows_to_write = []
                        for t in templates:
                            missing = [k for k in ("theme_used", "message_title_en", "message_title_hi",
                                                   "message_body_en", "message_body_hi",
                                                   "cta_text_en", "cta_text_hi") if not t.get(k)]
                            if missing:
                                dbg(f"process_batch → ⚠️  template missing fields",
                                    f"{missing}  →  {t}")
                            rows_to_write.append({
                                "segment_id":       seg,
                                "lifecycle_stage":  lc,
                                "theme_used":       t.get("theme_used", ""),
                                "template_id":      f"TPL_{tid_counter[0]:04d}",
                                "message_title_en": t.get("message_title_en", ""),
                                "message_title_hi": t.get("message_title_hi", ""),
                                "message_body_en":  t.get("message_body_en", ""),
                                "message_body_hi":  t.get("message_body_hi", ""),
                                "cta_text_en":      t.get("cta_text_en", ""),
                                "cta_text_hi":      t.get("cta_text_hi", ""),
                                "generated_at":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            })
                            tid_counter[0] += 1
                        writer.writerows(rows_to_write)
                        completed[0] += 1
                    csv_file.flush()

                with errors_lock:
                    attempted[0] += 1
                batch_elapsed = time.time() - batch_start
                tpl_count = sum(len(t) for _, t in results)
                print(f"   ✅ {tpl_count} templates from {len(results)} row(s)  ({batch_elapsed:.1f}s)  "
                      f"[Progress: {completed[0]}/{total} rows done, {attempted[0]}/{total_batches} batches]")

            except Exception as e:
                import traceback
                dbg(f"process_batch → ❌ batch {batch_idx} failed", traceback.format_exc(
                ))
                batch_elapsed = time.time() - batch_start
                with errors_lock:
                    attempted[0] += 1
                    for r in batch_rows:
                        errors.append({
                            "row":       batch_idx,
                            "segment":   r.get("segment_id", ""),
                            "lifecycle": r.get("lifecycle_stage", ""),
                            "error":     str(e),
                        })
                print(f"   ❌ BATCH FAILED after {batch_elapsed:.1f}s: {e}"
                      f"\n      [Progress: {attempted[0]}/{total_batches} batches, {len(errors)} row-errors]")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batches, 1)
            }
            for future in as_completed(futures):
                future.result()

    total_elapsed = time.time() - run_start
    mins, secs = divmod(int(total_elapsed), 60)

    print(f"\n{'='*60}")
    print(f"⏱  Total time: {mins}m {secs}s  ({total_elapsed:.1f}s)")
    print(
        f"📊 This run:  {attempted[0]}/{total_batches} batches, {completed[0]} rows succeeded, {len(errors)} row-errors")
    total_templates = completed[0] * 15
    print(f"✅ Output: {total_templates} templates → {output_path}")
    if errors:
        print(f"\n⚠️  {len(errors)} ROWS FAILED:")
        for e in errors:
            print(
                f"   Row {e['row']}: segment {e['segment']} × {e['lifecycle']} — {e['error'][:120]}")
    if completed[0] == total:
        print(f"\n🎉 All {total} segment × lifecycle rows completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    if "--update" in sys.argv:
        # Usage:  python generate_templates_gemini.py --update
        # Optionally override file paths via named args:
        #   --old  iteration_0_before_learning/segment_goals.csv          (default)
        #   --new  iteration_1_after_learning/segment_goals.csv           (default)
        #   --templates-in  iteration_0_before_learning/message_templates.csv   (default)
        #   --templates-out iteration_1_after_learning/message_templates.csv    (default)
        kwargs = {}
        for flag, key in [("--old",  "segment_goals_old"),
                          ("--new",  "segment_goals_new"),
                          ("--templates-in",  "message_templates_in"),
                          ("--templates-out", "message_templates_out")]:
            if flag in sys.argv:
                idx = sys.argv.index(flag)
                if idx + 1 < len(sys.argv):
                    kwargs[key] = sys.argv[idx + 1]
        update_templates(**kwargs)
    else:
        main()
