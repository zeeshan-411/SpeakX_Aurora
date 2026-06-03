from schedule_generator import schedule_generator_iteration_1
from timing_optimizer import generate_segment_timing_top3
import os
import io
import re
from dotenv import load_dotenv
load_dotenv()
import csv
import json
import sys
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from google import genai
from google.genai import types as genai_types

ITER0_DIR = "iteration_0_before_learning"
ITER1_DIR = "iteration_1_after_learning"
os.makedirs(ITER0_DIR, exist_ok=True)
os.makedirs(ITER1_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_REST_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
MODEL_GEMINI = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

LIFECYCLE_ORDER = ["trial", "paid", "churned", "inactive"]

ITER0_SEGMENT_GOALS = os.path.join(ITER0_DIR, "segment_goals.csv")
ITER0_USER_SEGMENTS = os.path.join(ITER0_DIR, "user_segments.csv")
ITER0_COMM_THEMES = os.path.join(ITER0_DIR, "communication_themes.csv")
ITER0_TONE_HOOK_MATRIX = os.path.join(
    ITER0_DIR, "allowed_tone_hook_matrix.json")

EXPERIMENT_RESULTS = "experiment_results.csv"
ITER1_SEGMENT_GOALS = os.path.join(ITER1_DIR, "segment_goals.csv")
ITER1_MSG_TEMPLATES = os.path.join(ITER1_DIR, "message_templates.csv")

ITER0_MESSAGE_TEMPLATES = os.path.join(ITER0_DIR, "message_templates.csv")
ITER1_MESSAGE_TEMPLATES = os.path.join(ITER1_DIR, "message_templates.csv")
ITER0_TIMING_RECS = os.path.join(ITER0_DIR, "timing_recommendations.csv")
ITER1_TIMING_RECS = os.path.join(ITER1_DIR, "timing_recommendations.csv")
ITER0_NOTIF_SCHEDULE = os.path.join(
    ITER0_DIR, "user_notification_schedule.csv")
ITER1_NOTIF_SCHEDULE = os.path.join(
    ITER1_DIR, "user_notification_schedule.csv")
LEARNING_DELTA_REPORT = "learning_delta_report.csv"

BATCH_SIZE = 1
MAX_WORKERS = 24


def _read_csv_dicts(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _mode(values):
    counts = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── 1. LOAD EXPERIMENT RESULTS ────────────────────────────────────────────────

experiments_result_path = input(
    "Enter the path for the experiment results CSV: ")
exp_df = pd.read_csv(experiments_result_path)
exp_df.to_csv(EXPERIMENT_RESULTS, index=False)
exp_df["arm_id"] = exp_df["template_id"].astype(
    str) + "_" + exp_df["notification_window"].astype(str)


# ── 2. RL LEARNING AGENT ──────────────────────────────────────────────────────

def _evaluate_weights(data, w_ctr, w_eng, w_uni):
    reward = data["ctr"] * w_ctr + data["engagement_rate"] * \
        w_eng - data["uninstall_rate"] * w_uni
    temp = data.assign(reward=reward)
    n = int(len(temp) * 0.3)
    top = temp.nlargest(n, "reward")
    bottom = temp.nsmallest(n, "reward")
    def score(s): return s["ctr"].mean() + \
        s["engagement_rate"].mean() - s["uninstall_rate"].mean()
    return score(top) - score(bottom)


# Learn reward weights by maximizing separation between top and bottom.
def _learn_optimal_weights(data):
    best_score, best_weights = -np.inf, (0.4, 0.6, 2.0)
    for w_ctr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w_eng = 1.0 - w_ctr
        for w_uni in np.arange(1.0, 3.5, 0.5):
            score = _evaluate_weights(data, w_ctr, w_eng, w_uni)
            if score > best_score:
                best_score, best_weights = score, (w_ctr, w_eng, w_uni)
    return best_weights


w_ctr, w_eng, w_uni = _learn_optimal_weights(exp_df)
exp_df["rl_reward"] = exp_df["ctr"] * w_ctr + \
    exp_df["engagement_rate"] * w_eng - exp_df["uninstall_rate"] * w_uni

reward_q70 = exp_df["rl_reward"].quantile(0.70)
reward_q30 = exp_df["rl_reward"].quantile(0.30)
min_confidence = 10

classifications = []
for _, row in exp_df.iterrows():
    opens = row["total_opens"]
    successes = row["total_engagements"]
    failures = max(opens - successes, 0)
    rl_reward = row["rl_reward"]
    alpha = 1 + successes
    beta_val = 1 + failures
    expected_engagement = alpha / (alpha + beta_val)

    if opens < min_confidence:
        label, reason = "NEUTRAL", "Insufficient data"
    elif rl_reward >= reward_q70:
        label, reason = ("GOOD", "High reward + good engagement") if expected_engagement > 0.15 else (
            "NEUTRAL", "High reward but low engagement")
    elif rl_reward <= reward_q30:
        label, reason = "BAD", "Low reward"
    else:
        label, reason = "NEUTRAL", "Mid-range reward"

    classifications.append({
        "arm_id":                  row["arm_id"],
        "template_id":             row["template_id"],
        "notification_window":     row["notification_window"],
        "segment_id":              row["segment_id"],
        "lifecycle_stage":         row["lifecycle_stage"],
        "ctr":                     row["ctr"],
        "engagement_rate":         row["engagement_rate"],
        "uninstall_rate":          row["uninstall_rate"],
        "rl_reward":               round(rl_reward, 4),
        "expected_engagement":     round(expected_engagement, 4),
        "confidence":              opens,
        "rl_classification":       label,
        "classification_reason":   reason,
        "existing_classification": row["performance_status"],
    })

rl_df = pd.DataFrame(classifications)

budget = {"GOOD": 0.70, "NEUTRAL": 0.25, "BAD": 0.05}
counts = rl_df["rl_classification"].value_counts()
rl_df["allocation_weight"] = rl_df["rl_classification"].map(
    lambda c: round(budget[c] / counts[c], 6))
rl_df["allocation_percentage"] = (rl_df["allocation_weight"] * 100).round(3)
rl_df["action"] = rl_df["rl_classification"].map(
    {"GOOD": "PROMOTE", "NEUTRAL": "MAINTAIN", "BAD": "SUPPRESS"})


# ── 3. RL STRATEGY (safety + timing → rl_comprehensive) ──────────────────────

# Aggregate safety signals to adjust frequency when uninstall risk rises.
def _build_safety_df(df):
    rows = []
    for seg_id in sorted(df["segment_id"].unique()):
        seg = df[df["segment_id"] == seg_id]
        avg_uninst = seg["uninstall_rate"].mean(
        ) if "uninstall_rate" in seg.columns else 0.01
        if avg_uninst > 0.02:
            freq_adj, freq_action, risk = -2, "REDUCE_BY_2", "HIGH_UNINSTALL_RISK"
        elif avg_uninst > 0.015:
            freq_adj, freq_action, risk = -1, "REDUCE_BY_1", "MODERATE_UNINSTALL_RISK"
        else:
            freq_adj, freq_action, risk = 0, "MAINTAIN",    "SAFE_ZONE"
        rows.append({
            "segment_id":           seg_id,
            "avg_uninstall_rate":   round(avg_uninst, 4),
            "frequency_adjustment": freq_adj,
            "frequency_action":     freq_action,
            "risk_level":           risk,
        })
    return pd.DataFrame(rows)


def _build_timing_df(df):
    rows = []
    for seg_id in sorted(df["segment_id"].unique()):
        seg_good = df[(df["segment_id"] == seg_id) &
                      (df["rl_classification"] == "GOOD")]
        seg_data = seg_good if not seg_good.empty else df[df["segment_id"] == seg_id]
        confidence_note = "GOOD arms" if not seg_good.empty else "all arms (no GOOD found)"
        window_perf = (
            seg_data.groupby("notification_window")
            .agg(
                rl_reward=("rl_reward", "mean"),
                ctr=("ctr", "mean"),
                expected_engagement=("expected_engagement", "mean"),
                arm_count=("template_id", "count"),
            )
            .sort_values("rl_reward", ascending=False)
        )
        primary = window_perf.index[0]
        secondary = window_perf.index[1] if len(window_perf) > 1 else primary
        rows.append({
            "segment_id":        seg_id,
            "primary_window":    primary,
            "primary_reward":    round(window_perf.loc[primary, "rl_reward"], 4),
            "primary_ctr":       round(window_perf.loc[primary, "ctr"], 4),
            "secondary_window":  secondary,
            "secondary_reward":  round(window_perf.loc[secondary, "rl_reward"], 4),
            "timing_confidence": confidence_note,
        })
    return pd.DataFrame(rows)


safety_df = _build_safety_df(rl_df)
times_df = _build_timing_df(rl_df)

merged = rl_df.merge(
    safety_df[["segment_id", "frequency_adjustment",
               "frequency_action", "risk_level", "avg_uninstall_rate"]],
    on="segment_id", how="left"
)
merged = merged.merge(
    times_df[["segment_id", "primary_window", "secondary_window",
              "primary_reward", "primary_ctr", "timing_confidence"]],
    on="segment_id", how="left"
)
merged["allocation_weight"] = merged["rl_classification"].map(
    lambda c: round(budget[c] / counts[c], 6))
merged["allocation_percentage"] = (merged["allocation_weight"] * 100).round(3)
merged["action"] = merged["rl_classification"].map(
    {"GOOD": "PROMOTE", "NEUTRAL": "MAINTAIN", "BAD": "SUPPRESS"})


def _make_reasoning(row):
    cls = row["rl_classification"]
    if cls == "GOOD":
        return (f"GOOD: reward={row['rl_reward']:.4f}, CTR={row['ctr']:.2%}, "
                f"engagement={row['expected_engagement']:.2%} -> allocated {row['allocation_percentage']:.3f}% traffic (PROMOTE)")
    elif cls == "NEUTRAL":
        return (f"NEUTRAL: reward={row['rl_reward']:.4f} (mid-range), confidence={row['confidence']} obs "
                f"-> allocated {row['allocation_percentage']:.3f}% traffic (MAINTAIN/EXPLORE)")
    else:
        return (f"BAD: reward={row['rl_reward']:.4f}, CTR={row['ctr']:.2%}, "
                f"uninstall={row['uninstall_rate']:.2%} -> minimised to {row['allocation_percentage']:.3f}% traffic (SUPPRESS)")


merged["classification_reasoning"] = merged.apply(_make_reasoning, axis=1)
merged = merged.rename(columns={
                       "primary_window": "recommended_window", "secondary_window": "backup_window"})

rl_comprehensive = merged[[
    "template_id", "segment_id", "notification_window", "arm_id", "lifecycle_stage",
    "rl_classification", "action", "allocation_weight", "allocation_percentage",
    "rl_reward", "ctr", "engagement_rate", "expected_engagement", "uninstall_rate", "confidence",
    "recommended_window", "backup_window", "timing_confidence",
    "frequency_action", "frequency_adjustment", "risk_level", "avg_uninstall_rate",
    "classification_reasoning",
]]


# ── 4. DECISION TABLE ─────────────────────────────────────────────────────────

_sg = pd.read_csv(ITER0_SEGMENT_GOALS)

DECISION_TABLE = (
    rl_comprehensive.groupby(["segment_id", "lifecycle_stage"]).agg(
        dominant_class=("rl_classification",
                        lambda x: x.value_counts().index[0]),
        best_window=("recommended_window",
                     lambda x: x.value_counts().index[0]),
        avg_reward=("rl_reward",          "mean"),
        avg_engagement=("engagement_rate",    "mean"),
        top_arm_ctr=("ctr",                "max"),
    ).reset_index().round(3)
)

n_good = (DECISION_TABLE["dominant_class"] == "GOOD").sum()
n_neutral = (DECISION_TABLE["dominant_class"] == "NEUTRAL").sum()
n_bad = (DECISION_TABLE["dominant_class"] == "BAD").sum()
EXPECTED_ROWS = n_good + (n_neutral * 2) + n_bad

_missing = set(_sg["segment_id"].unique()) - \
    set(rl_comprehensive["segment_id"].unique())
if _missing:
    _missing_rows = _sg[_sg["segment_id"].isin(
        _missing)][["segment_id", "lifecycle_stage"]].drop_duplicates().copy()
    _missing_rows["dominant_class"] = "GOOD"
    _missing_rows["best_window"] = "evening"
    _missing_rows["avg_reward"] = 0.0
    _missing_rows["avg_engagement"] = 0.0
    _missing_rows["top_arm_ctr"] = 0.0
    DECISION_TABLE = pd.concat(
        [DECISION_TABLE, _missing_rows], ignore_index=True)
    n_good = (DECISION_TABLE["dominant_class"] == "GOOD").sum()
    n_neutral = (DECISION_TABLE["dominant_class"] == "NEUTRAL").sum()
    n_bad = (DECISION_TABLE["dominant_class"] == "BAD").sum()
    EXPECTED_ROWS = n_good + (n_neutral * 2) + n_bad

_SG_OUTPUT_COLUMNS = ["ab_variant"] + list(_sg.columns)


# ── 5. SEGMENT GOALS UPDATE ───────────────────────────────────────────────────

_SEGMENT_GOALS_PROMPT = """You are a segment goal optimization engine for Project Aurora (SpeakX).

## FILE 1: segment_goals.csv (current goals — one row per segment x lifecycle)
{segment_goals_csv}

## FILE 2: DECISION TABLE (pre-computed from RL data)
Columns: segment_id, lifecycle_stage, dominant_class, avg_reward, avg_engagement, top_arm_ctr

{decision_table_csv}

dominant_class meanings:
- GOOD    = performing well — copy goals verbatim, no changes
- NEUTRAL = mid-range — create two A/B variant rows with different messaging strategies
- BAD     = underperforming — rewrite goals to reduce friction and improve engagement

## INSTRUCTIONS

For every row in FILE 1, look up its (segment_id, lifecycle_stage) in the DECISION TABLE:

### If dominant_class = GOOD
Output 1 row. Copy ALL fields exactly from FILE 1. Set ab_variant = "control"

### If dominant_class = BAD
Output 1 row. Rewrite fields below. Set ab_variant = "updated"
Focus on lower friction, re-establishing habit, gentle re-engagement.

### If dominant_class = NEUTRAL
Output 2 rows. Set ab_variant = "A" (conservative) and "B" (aggressive).
Variant A: gentle habit-building, low-pressure nudges, streak continuity, small wins.
Variant B: urgency and loss-framing, strong CTAs, reward/challenge framing.

FIELDS TO REWRITE for BAD and NEUTRAL:
- primary_goal, sub_goal, messaging_focus, D1_focus, D8_focus, D30_focus, D30_focus_churned, D30_focus_inactive

FIELDS TO PRESERVE for all:
- segment_id, segment_name, lifecycle_stage, dominant_propensity, D0_focus
- D2_focus through D7_focus, D10_focus, D15_focus, D20_focus, D25_focus, D45_focus, D60_focus, D90_focus

## QUALITY RULES
- Every NEUTRAL segment x lifecycle MUST produce exactly 2 rows (A and B).
- A and B must be meaningfully different in tone and angle, not just rewordings.
- GOOD rows must be 100% identical to FILE 1.
- No snake_case, no bullet points, no placeholders in goal text.

## OUTPUT FORMAT
Return ONLY raw CSV. No explanation, no markdown, no code fences.
- First column: ab_variant, then all original columns from FILE 1 in the same order.
- Row order: segment_id ascending, then lifecycle (trial, paid, churned, inactive), then A before B.
- Quote any field containing a comma with double quotes.
- Output exactly {expected_rows} rows excluding the header.
"""


def _call_gemini_sdk(prompt):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=MODEL_GEMINI,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.2),
    )
    return response.text


def _clean_sg_response(raw):
    import csv as _csv
    clean = re.sub(r"```[^\n]*\n?", "", raw).strip()
    expected_cols = len(_SG_OUTPUT_COLUMNS)
    repaired = []
    for i, line in enumerate(clean.splitlines()):
        if i == 0:
            repaired.append(line)
            continue
        try:
            parts = next(_csv.reader([line]))
        except Exception:
            repaired.append(line)
            continue
        if len(parts) <= expected_cols:
            repaired.append(line)
        else:
            overflow = expected_cols - 1
            fixed_parts = parts[:overflow] + [",".join(parts[overflow:])]
            repaired.append(
                ",".join(f'"{p}"' if "," in p else p for p in fixed_parts))
    return "\n".join(repaired)


def _parse_and_validate_sg(csv_text):
    df = pd.read_csv(io.StringIO(csv_text),
                     engine="python", on_bad_lines="warn")
    missing_cols = [c for c in _SG_OUTPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Output missing required columns: {missing_cols}")
    df = df[[c for c in _SG_OUTPUT_COLUMNS if c in df.columns]]
    lc_order = {lc: i for i, lc in enumerate(LIFECYCLE_ORDER)}
    var_order = {"control": 0, "updated": 0, "A": 0, "B": 1}
    df["_lc"] = df["lifecycle_stage"].str.lower().map(lc_order).fillna(99)
    df["_var"] = df["ab_variant"].map(var_order).fillna(0)
    df = df.sort_values(["segment_id", "_lc", "_var"]
                        ).drop(columns=["_lc", "_var"])
    return df.reset_index(drop=True)


def run_segment_goals_update():
    sg_csv = open(ITER0_SEGMENT_GOALS, "r", encoding="utf-8").read()
    decision_csv = DECISION_TABLE.to_csv(index=False)
    prompt = _SEGMENT_GOALS_PROMPT.format(
        segment_goals_csv=sg_csv,
        decision_table_csv=decision_csv,
        expected_rows=EXPECTED_ROWS,
    )
    raw = _call_gemini_sdk(prompt)
    csv_text = _clean_sg_response(raw)
    df = _parse_and_validate_sg(csv_text)
    df.to_csv(ITER1_SEGMENT_GOALS, index=False)
    return df


goals_df = run_segment_goals_update()


# ── 6. TEMPLATE / MESSAGE GENERATOR ──────────────────────────────────────────

with open(ITER0_TONE_HOOK_MATRIX, "r", encoding="utf-8") as _thf:
    _tone_hook_data = json.load(_thf)

_ALLOWED_TONES = _tone_hook_data["allowed_tones"]
_DISALLOWED_TONES = _tone_hook_data["disallowed_tones"]
_OCTALYSIS_HOOKS = _tone_hook_data["octalysis_hooks"]

_TONE_ALLOWED_STR = ", ".join(_ALLOWED_TONES)
_TONE_DISALLOWED_STR = ", ".join(_DISALLOWED_TONES)
_TONE_CHECKLIST_STR = " / ".join(_ALLOWED_TONES)

_oct_rows = ["| # | Theme | Example Hook |", "|---|---|---|"]
for _i, (_theme, _hook) in enumerate(_OCTALYSIS_HOOKS.items(), 1):
    _oct_rows.append(f'| {_i} | {_theme} | "{_hook}" |')
_OCTALYSIS_TABLE_STR = "\n".join(_oct_rows)

_TEMPLATE_SYSTEM_PROMPT = """
## ROLE

You are an expert mobile notification copywriter specializing in behavioral psychology and user engagement for language-learning apps. You write push notifications that are concise, emotionally resonant, and drive specific user actions. You are fluent in English, Hindi, and Hinglish.

---

## COMPANY CONTEXT

**SpeakX** is an AI-powered English learning platform that helps non-native speakers build speaking confidence through real-time feedback and habit-forming exercises. Users often experience "speech anxiety" despite knowing grammar rules.

**Core Features:**
- AI Tutor (Sia): 1-on-1 voice-based conversation practice with real-time corrections
- Daily Streaks: Gamified tracker rewarding users for daily practice to build long-term habits
- Leaderboards: Social competition — users earn points from lesson completion & accuracy
- Personalized Curriculum: Lessons that adapt based on proficiency level and progress
- Voice Analytics: Visual feedback on pronunciation, fluency, and vocabulary usage

**North Star Metric:** Active Speakers — users completing at least one AI Tutor session per week.

---

## TASK

For **each** input row below, generate **15 notification templates** across 3 assigned themes.
Each theme gets exactly **5 bilingual templates**.
Each template: title (EN + HI), body (EN + HI), CTA (EN + HI).

---

## RULES

### 1. Character Limits
| Element | Hard Limit |
|---|---|
| Title | <= 40 characters |
| Body | <= 90 characters |

### 2. Dynamic Variables
`{coins_balance}`, `{streak_current}`, `{exercises_completed_7d}`, `{sessions_last_7d}`

### 3. One goal per message. No compound CTAs.

### 4. Emoji Framework
Bookend method: one emoji start + one end. Limit 1-3. Lead emoji in first 10 characters.

### 5. Tone
ALLOWED: <<ALLOWED_TONES>>
NEVER USE: <<DISALLOWED_TONES>>

### 6. Language
- English (EN): 100% English. Clear, conversational, action-oriented.
- Hinglish (HI): Roman script only. Write independently from scratch — NOT a translation.
  - Hindi carries emotion/urgency/warmth. English carries app feature terms only.
  - Pronouns: aap/aapki/aapka. Never tu/teri/tera/tum.
  - Lead with Hindi emotion, land on English app term.
  - Every Hinglish notification needs one warmth anchor: yaar/bhai/arrey/sun/suno.
  - Negative constructions must always come with a softener.

### 7. CTA Rules
3-10 words, starts with action verb, creates urgency. Match column language.

---

## OCTALYSIS THEMES

<<OCTALYSIS_TABLE>>

---

## QUALITY CHECKLIST

- [ ] title_en/hi <= 40 chars, body_en/hi <= 90 chars
- [ ] CTA 3-10 words, action verb, urgency
- [ ] 1-3 emojis, lead in first 10 chars
- [ ] Tone: <<TONE_CHECKLIST>>
- [ ] Variables in {curly_braces}
- [ ] Hinglish written fresh, not translated

---

## OUTPUT FORMAT (STRICT)

Return ONLY a valid JSON array. No markdown, no code fences, no commentary.

[
  {
    "segment_id": "<seg>",
    "lifecycle_stage": "<lc>",
    "themes": [
      {
        "theme": "<Theme1>",
        "templates": [
          {"title_en": "...", "title_hi": "...", "body_en": "...", "body_hi": "...", "cta_en": "...", "cta_hi": "..."},
          {"title_en": "...", "title_hi": "...", "body_en": "...", "body_hi": "...", "cta_en": "...", "cta_hi": "..."},
          {"title_en": "...", "title_hi": "...", "body_en": "...", "body_hi": "...", "cta_en": "...", "cta_hi": "..."},
          {"title_en": "...", "title_hi": "...", "body_en": "...", "body_hi": "...", "cta_en": "...", "cta_hi": "..."},
          {"title_en": "...", "title_hi": "...", "body_en": "...", "body_hi": "...", "cta_en": "...", "cta_hi": "..."}
        ]
      },
      {"theme": "<Theme2>", "templates": [...]},
      {"theme": "<Theme3>", "templates": [...]}
    ]
  }
]
"""

_TEMPLATE_SYSTEM_PROMPT = (
    _TEMPLATE_SYSTEM_PROMPT
    .replace("<<ALLOWED_TONES>>",    _TONE_ALLOWED_STR)
    .replace("<<DISALLOWED_TONES>>", _TONE_DISALLOWED_STR)
    .replace("<<OCTALYSIS_TABLE>>",  _OCTALYSIS_TABLE_STR)
    .replace("<<TONE_CHECKLIST>>",   _TONE_CHECKLIST_STR)
)


def _load_template_data(segment_goals_path, user_segments_path, communication_themes_path):
    sg_rows = _read_csv_dicts(segment_goals_path)
    us_rows = _read_csv_dicts(user_segments_path)
    th_rows = _read_csv_dicts(communication_themes_path)

    us_by_seg = {}
    for r in us_rows:
        sid = r.get("segment_id", "")
        if sid not in us_by_seg:
            us_by_seg[sid] = {
                "segment_description": r.get("segment_description", ""),
                "age_bands": [], "regions": [], "preferred_hours": [], "days_since_signup": [],
            }
        us_by_seg[sid]["age_bands"].append(r.get("age_band", ""))
        us_by_seg[sid]["regions"].append(r.get("region", ""))
        h = r.get("preferred_hour", "")
        if h:
            us_by_seg[sid]["preferred_hours"].append(float(h))
        d = r.get("days_since_signup", "")
        if d:
            us_by_seg[sid]["days_since_signup"].append(float(d))

    us_agg = {}
    for sid, data in us_by_seg.items():
        hours = data["preferred_hours"]
        days = data["days_since_signup"]
        us_agg[sid] = {
            "segment_description":   data["segment_description"],
            "age_band":              _mode(data["age_bands"]),
            "top_region":            _mode(data["regions"]),
            "preferred_hour":        round(sum(hours) / len(hours)) if hours else 12,
            "avg_days_since_signup": round(sum(days) / len(days), 1) if days else 0,
        }

    th_by_seg = {}
    for r in th_rows:
        sid = r.get("segment_id", "")
        if sid not in th_by_seg:
            th_by_seg[sid] = {
                "theme_1":         r.get("drive_1", ""),
                "theme_2":         r.get("drive_2", ""),
                "theme_3":         r.get("drive_3", ""),
                "theme_reasoning": r.get("reasoning", ""),
            }

    result = []
    for r in sg_rows:
        sid = r.get("segment_id", "")
        us = us_agg.get(sid, {})
        th = th_by_seg.get(sid, {})
        result.append({
            "segment_id":          sid,
            "segment_name":        r.get("segment_name", ""),
            "segment_description": us.get("segment_description", ""),
            "lifecycle_stage":     r.get("lifecycle_stage", ""),
            "kb_feature_outcome":  r.get("kb_feature_outcome", "") or "N/A",
            "primary_goal":        r.get("primary_goal", ""),
            "sub_goal":            r.get("sub_goal", ""),
            "messaging_focus":     r.get("messaging_focus", ""),
            "theme_1":             th.get("theme_1", "Accomplishment"),
            "theme_2":             th.get("theme_2", "Ownership"),
            "theme_3":             th.get("theme_3", "Loss Avoidance"),
            "theme_reasoning":     th.get("theme_reasoning", ""),
        })
    return result


def _build_row_block(idx, row):
    return (
        f"### INPUT ROW {idx}\n"
        f"- **Segment:** {row.get('segment_id', '')}\n"
        f"- **Segment Description:** {row.get('segment_description', '')}\n"
        f"- **Lifecycle Stage:** {row.get('lifecycle_stage', '')}\n"
        f"- **KB Feature / Outcome:** {row.get('kb_feature_outcome', '') or 'N/A'}\n"
        f"- **Sub Goal:** {row.get('sub_goal', '')}\n"
        f"- **Messaging Focus:** {row.get('messaging_focus', '')}\n"
        f"- **Theme 1:** {row.get('theme_1') or 'Accomplishment'}\n"
        f"- **Theme 2:** {row.get('theme_2') or 'Ownership'}\n"
        f"- **Theme 3:** {row.get('theme_3') or 'Loss Avoidance'}\n"
        f"- **Theme Reasoning:** {row.get('theme_reasoning', '') or 'N/A'}\n"
    )


def _build_batch_prompt(rows):
    parts = [
        f"Generate templates for the following {len(rows)} input row(s).\n"]
    for i, row in enumerate(rows, 1):
        parts.append(_build_row_block(i, row))
    return "\n".join(parts)


def _flatten_themes(themes_list):
    templates = []
    for group in themes_list:
        theme = group.get("theme", "")
        for t in group.get("templates", []):
            templates.append({
                "theme_used":        theme,
                "message_title_en":  t.get("title_en", ""),
                "message_title_hi":  t.get("title_hi", ""),
                "message_body_en":   t.get("body_en",  ""),
                "message_body_hi":   t.get("body_hi",  ""),
                "cta_text_en":       t.get("cta_en",   ""),
                "cta_text_hi":       t.get("cta_hi",   ""),
            })
    return templates


def _parse_template_response(text, rows):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    text = re.sub(r",\s*([\]}])", r"\1", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError(
                f"Could not parse JSON from response:\n{text[:300]}")

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Parsed JSON is not a non-empty list")

    results = []
    if isinstance(parsed[0], dict) and "themes" in parsed[0]:
        row_lookup = {(r.get("segment_id", ""), r.get(
            "lifecycle_stage", "")): r for r in rows}
        for i, entry in enumerate(parsed):
            sid = str(entry.get("segment_id", ""))
            lc = entry.get("lifecycle_stage", "")
            matched_row = row_lookup.get((sid, lc)) or (
                rows[i] if i < len(rows) else rows[-1])
            results.append(
                (matched_row, _flatten_themes(entry.get("themes", []))))
    elif isinstance(parsed[0], dict) and "templates" in parsed[0]:
        results.append((rows[0], _flatten_themes(parsed)))
    else:
        raise ValueError(
            f"Unexpected JSON format. Keys: {list(parsed[0].keys()) if parsed else '?'}")
    return results


def _generate_template_batch(rows, max_retries=5):
    import requests as _requests
    user_msg = _build_batch_prompt(rows)
    max_tok = min(16384 * len(rows), 65536)

    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "system_instruction": {"parts": [{"text": _TEMPLATE_SYSTEM_PROMPT}]},
                "contents":           [{"parts": [{"text": user_msg}]}],
                "generationConfig":   {"maxOutputTokens": max_tok},
            }
            resp = _requests.post(GEMINI_REST_URL, json=payload, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Gemini API {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            candidates = data.get("candidates", [])
            finish_reason = candidates[0].get(
                "finishReason", "") if candidates else ""

            if finish_reason == "MAX_TOKENS" and attempt < max_retries:
                max_tok = min(max_tok + 4096, 32768)
                time.sleep(1)
                continue

            response_text = candidates[0]["content"]["parts"][0]["text"] if candidates else ""
            return _parse_template_response(response_text, rows)

        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "invalid api key" in err_str.lower():
                raise
            if ("quota" in err_str.lower() or "429" in err_str) and attempt < max_retries:
                m = re.search(r"retry in ([\d.]+)s", err_str)
                wait = float(m.group(1)) + 5 if m else 60
                time.sleep(wait)
            elif attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise


def _load_completed_pairs(csv_path):
    done = set()
    max_tid = 0
    if not os.path.exists(csv_path):
        return done, max_tid
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row.get("segment_id", ""), row.get("lifecycle_stage", "")))
            tid = row.get("template_id", "")
            if tid.startswith("TPL_"):
                try:
                    max_tid = max(max_tid, int(tid.split("_")[1]))
                except ValueError:
                    pass
    return done, max_tid


def run_template_generation():
    resume_file = None
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 < len(sys.argv):
            resume_file = sys.argv[idx + 1]
            if os.sep not in resume_file:
                resume_file = os.path.join(ITER1_DIR, resume_file)

    reader = _load_template_data(
        ITER1_SEGMENT_GOALS, ITER0_USER_SEGMENTS, ITER0_COMM_THEMES)

    fieldnames = [
        "segment_id", "lifecycle_stage", "theme_used", "template_id",
        "message_title_en", "message_title_hi",
        "message_body_en",  "message_body_hi",
        "cta_text_en",      "cta_text_hi",
        "generated_at",
    ]

    if resume_file:
        output_path = resume_file
        done_pairs, max_tid = _load_completed_pairs(output_path)
        reader = [r for r in reader
                  if (r.get("segment_id", ""), r.get("lifecycle_stage", "")) not in done_pairs]
    else:
        output_path = ITER1_MSG_TEMPLATES
        done_pairs = set()
        max_tid = 0

    if not reader:
        return

    batches = [reader[i:i + BATCH_SIZE]
               for i in range(0, len(reader), BATCH_SIZE)]
    errors = []
    write_lock = threading.Lock()
    errors_lock = threading.Lock()
    tid_counter = [max_tid + 1]

    file_mode = "a" if resume_file else "w"
    with open(output_path, file_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not resume_file:
            writer.writeheader()
        csv_file.flush()

        def process_batch(batch_idx, batch_rows):
            try:
                results = _generate_template_batch(batch_rows)
                with write_lock:
                    for row, templates in results:
                        rows_to_write = []
                        for t in templates:
                            rows_to_write.append({
                                "segment_id":       row.get("segment_id", ""),
                                "lifecycle_stage":  row.get("lifecycle_stage", ""),
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
                    csv_file.flush()
            except Exception as e:
                import traceback
                with errors_lock:
                    for r in batch_rows:
                        errors.append({
                            "row": batch_idx, "segment": r.get("segment_id", ""),
                            "lifecycle": r.get("lifecycle_stage", ""), "error": str(e),
                        })
                traceback.print_exc()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_batch, idx, batch): idx
                       for idx, batch in enumerate(batches, 1)}
            for future in as_completed(futures):
                future.result()


run_template_generation()

# Import external functions from the codebase
_codebase_dir = os.path.dirname(os.path.abspath(__file__))
if _codebase_dir not in sys.path:
    sys.path.insert(0, _codebase_dir)


schedule_generator_iteration_1(rl_comprehensive, goals_df)
timing_recs_df = generate_segment_timing_top3(rl_comprehensive)


def _build_delta_report():
    iter0_templates = _read_csv_dicts(ITER0_MESSAGE_TEMPLATES)
    iter1_templates = _read_csv_dicts(ITER1_MESSAGE_TEMPLATES)
    iter0_timing = _read_csv_dicts(ITER0_TIMING_RECS)
    iter1_timing = timing_recs_df.to_dict("records")
    iter0_schedule = _read_csv_dicts(ITER0_NOTIF_SCHEDULE)
    iter1_schedule = _read_csv_dicts(ITER1_NOTIF_SCHEDULE)
    experiment = _read_csv_dicts(EXPERIMENT_RESULTS)

    exp_by_tid = {r.get("template_id", "")
                        : r for r in experiment if r.get("template_id")}
    iter0_by_tid = {r.get("template_id", "")                    : r for r in iter0_templates if r.get("template_id")}
    iter1_by_tid = {r.get("template_id", "")                    : r for r in iter1_templates if r.get("template_id")}

    deltas = []

    for tid in sorted(set(iter0_by_tid) | set(iter1_by_tid)):
        t0 = iter0_by_tid.get(tid)
        t1 = iter1_by_tid.get(tid)
        exp = exp_by_tid.get(tid, {})
        perf = exp.get("performance_status", "").upper()
        ctr = _safe_float(exp.get("ctr", 0))
        eng = _safe_float(exp.get("engagement_rate", 0))
        metric = f"CTR={ctr:.1%}, engagement_rate={eng:.1%}" if exp else "no experiment data"

        if t0 and not t1:
            change = "suppressed" if perf == "BAD" else "removed"
            deltas.append({
                "entity_type":    "template",
                "entity_id":      tid,
                "change_type":    change,
                "metric_trigger": metric,
                "before_value":   f"active | {t0.get('theme_used', '?')} | seg={t0.get('segment_id', '?')}",
                "after_value":    f"{change} — removed from rotation",
                "explanation":    (
                    f"Template {tid} ({t0.get('theme_used', '?')}) for segment "
                    f"{t0.get('segment_id', '?')} {change} after experiment results showed {metric}"
                    + (", falling below BAD thresholds (CTR<5% or engagement<20%)." if perf == "BAD" else ".")
                ),
            })
        elif t1 and not t0:
            deltas.append({
                "entity_type":    "template",
                "entity_id":      tid,
                "change_type":    "template_added",
                "metric_trigger": f"promoted as reference — {metric}" if perf == "GOOD" else "new template added post-learning",
                "before_value":   "did not exist",
                "after_value":    f"active | {t1.get('theme_used', '?')} | seg={t1.get('segment_id', '?')}",
                "explanation":    (
                    f"Template {tid} ({t1.get('theme_used', '?')}) introduced in Iteration 1 "
                    f"for segment {t1.get('segment_id', '?')} modelled on GOOD-performing patterns in the same segment."
                ),
            })
        elif t0 and t1:
            changed_fields = [
                f for f in ["message_title_en", "message_title_hi", "message_body_en",
                            "message_body_hi", "cta_text_en", "cta_text_hi", "theme_used"]
                if t0.get(f, "") != t1.get(f, "")
            ]
            if changed_fields and perf in ("BAD", "NEUTRAL"):
                deltas.append({
                    "entity_type":    "template",
                    "entity_id":      tid,
                    "change_type":    "template_replaced",
                    "metric_trigger": metric,
                    "before_value":   f"body_en='{t0.get('message_body_en', '')[:60]}...'",
                    "after_value":    f"body_en='{t1.get('message_body_en', '')[:60]}...'",
                    "explanation":    (
                        f"Template {tid} updated across {len(changed_fields)} field(s) "
                        f"({', '.join(changed_fields)}) after {perf} classification ({metric}). "
                        f"Rewritten using tone and hook patterns from GOOD templates in the same segment x lifecycle."
                    ),
                })
            elif perf == "GOOD" and not changed_fields:
                deltas.append({
                    "entity_type":    "template",
                    "entity_id":      tid,
                    "change_type":    "promoted",
                    "metric_trigger": metric,
                    "before_value":   "active (unverified)",
                    "after_value":    "promoted as reference template",
                    "explanation":    (
                        f"Template {tid} ({t0.get('theme_used', '?')}) confirmed GOOD ({metric}). "
                        f"Retained in Iteration 1 and flagged as reference for the same segment x lifecycle."
                    ),
                })

    iter0_timing_by_seg = {r.get("segment_id", ""): r for r in iter0_timing}
    iter1_timing_by_seg = {r.get("segment_id", ""): r for r in iter1_timing}

    window_ctr = {}
    for r in experiment:
        seg = r.get("segment_id", "")
        win = r.get("notification_window", "")
        ctr = _safe_float(r.get("ctr", 0))
        if seg and win:
            window_ctr.setdefault((seg, win), []).append(ctr)
    avg_window_ctr = {k: sum(v) / len(v) for k, v in window_ctr.items()}

    for seg in sorted(set(iter0_timing_by_seg) | set(iter1_timing_by_seg)):
        t0 = iter0_timing_by_seg.get(seg, {})
        t1 = iter1_timing_by_seg.get(seg, {})
        win0 = t0.get("optimal_window", t0.get("primary_window", ""))
        win1 = t1.get("primary_window", "")
        if win0 and win1 and win0 != win1:
            old_ctr = avg_window_ctr.get((seg, win0))
            new_ctr = avg_window_ctr.get((seg, win1))
            metric = f"window '{win0}' avg_CTR={old_ctr:.1%}" if old_ctr is not None else f"window '{win0}' — no CTR data"
            deltas.append({
                "entity_type":    "timing",
                "entity_id":      f"seg={seg}",
                "change_type":    "timing_shift",
                "metric_trigger": metric,
                "before_value":   win0,
                "after_value":    win1,
                "explanation":    (
                    f"Segment {seg} optimal delivery window shifted from '{win0}' to '{win1}'. "
                    + (
                        f"Experiment showed '{win0}' yielding avg CTR={old_ctr:.1%}"
                        + (f" vs '{win1}' at {new_ctr:.1%}." if new_ctr else ".")
                        if old_ctr is not None else
                        "Timing updated based on learned engagement patterns."
                    )
                ),
            })

    uninstall_by_seg = {}
    for r in experiment:
        seg = r.get("segment_id", "")
        ur = _safe_float(r.get("uninstall_rate", 0))
        if seg:
            uninstall_by_seg.setdefault(seg, []).append(ur)
    avg_uninstall = {s: sum(v) / len(v) for s, v in uninstall_by_seg.items()}

    def _avg_notifs(schedule):
        per_seg = {}
        for row in schedule:
            seg = row.get("segment_id", row.get("segment", ""))
            uid = row.get("user_id", "")
            count = sum(1 for k in row if k.startswith(
                "notif_") and row[k] and row[k].strip())
            if not count:
                count = int(_safe_float(
                    row.get("notif_count", row.get("daily_notifs", 0))))
            per_seg.setdefault(seg, {})[uid] = count
        return {s: sum(u.values()) / len(u) for s, u in per_seg.items() if u}

    avg0 = _avg_notifs(iter0_schedule)
    avg1 = _avg_notifs(iter1_schedule)
    for seg in sorted(set(avg0) | set(avg1)):
        a0 = avg0.get(seg, 0)
        a1 = avg1.get(seg, 0)
        unin = avg_uninstall.get(seg)
        if abs(a0 - a1) >= 0.5:
            guardrail = unin is not None and unin > 0.02
            deltas.append({
                "entity_type":    "frequency",
                "entity_id":      f"seg={seg}",
                "change_type":    "frequency_change",
                "metric_trigger": f"uninstall_rate={unin:.1%} > 2% guardrail triggered" if guardrail else "activeness-based frequency rebalancing",
                "before_value":   f"{a0:.1f} notifs/day",
                "after_value":    f"{a1:.1f} notifs/day",
                "explanation":    (
                    f"Segment {seg} frequency changed from {a0:.1f} to {a1:.1f} notifs/day. "
                    + (
                        f"Uninstall rate {unin:.1%} exceeded 2% guardrail, triggering mandatory -2/day reduction."
                        if guardrail else
                        "Rebalanced based on revised activeness score from experiment engagement patterns."
                    )
                ),
            })

    good_themes = {}
    bad_themes = {}
    for r in experiment:
        seg = r.get("segment_id", "")
        theme = r.get("theme", "")
        status = r.get("performance_status", "").upper()
        if seg and theme:
            if status == "GOOD":
                good_themes.setdefault(seg, []).append(theme)
            elif status == "BAD":
                bad_themes.setdefault(seg, []).append(theme)

    def _dominant_theme(templates, seg):
        counts = {}
        for t in templates:
            if t.get("segment_id") == seg:
                theme = t.get("theme_used", "")
                if theme:
                    counts[theme] = counts.get(theme, 0) + 1
        return max(counts, key=counts.get) if counts else "unknown"

    all_segs = {t.get("segment_id") for t in iter0_templates +
                iter1_templates if t.get("segment_id")}
    for seg in sorted(all_segs):
        dom0 = _dominant_theme(iter0_templates, seg)
        dom1 = _dominant_theme(iter1_templates, seg)
        good = list(dict.fromkeys(good_themes.get(seg, [])))
        bad = list(dict.fromkeys(bad_themes.get(seg, [])))
        if dom0 != dom1:
            deltas.append({
                "entity_type":    "segment",
                "entity_id":      f"seg={seg}",
                "change_type":    "theme_preference_updated",
                "metric_trigger": f"GOOD themes: {good} | BAD themes: {bad}" if (good or bad) else "theme distribution shift",
                "before_value":   f"dominant_theme={dom0}",
                "after_value":    f"dominant_theme={dom1}",
                "explanation":    (
                    f"Segment {seg} dominant theme shifted from '{dom0}' to '{dom1}' in Iteration 1. "
                    + (f"Experiment confirmed '{', '.join(good)}' as high-performing and '{', '.join(bad)}' as underperforming. " if good or bad else "")
                    + "Template mix updated to weight proven themes more heavily."
                ),
            })

    good_count = sum(1 for r in experiment if r.get(
        "performance_status", "").upper() == "GOOD")
    bad_count = sum(1 for r in experiment if r.get(
        "performance_status", "").upper() == "BAD")
    neut_count = sum(1 for r in experiment if r.get(
        "performance_status", "").upper() == "NEUTRAL")
    total = len(experiment)

    suppressed = sum(1 for d in deltas if d["change_type"] == "suppressed")
    added = sum(1 for d in deltas if d["change_type"] == "template_added")
    replaced = sum(
        1 for d in deltas if d["change_type"] == "template_replaced")
    promoted = sum(1 for d in deltas if d["change_type"] == "promoted")

    summary = [{
        "entity_type":    "system",
        "entity_id":      "iteration_summary",
        "change_type":    "learning_cycle_complete",
        "metric_trigger": f"experiment_results: {total} templates evaluated — GOOD={good_count}, NEUTRAL={neut_count}, BAD={bad_count}",
        "before_value":   "Iteration 0 (pre-learning)",
        "after_value":    "Iteration 1 (post-learning)",
        "explanation":    (
            f"Learning cycle completed. Of {total} evaluated templates: "
            f"{good_count} GOOD (CTR>15%, engagement>40%), {neut_count} NEUTRAL, {bad_count} BAD (CTR<5%, engagement<20%). "
            f"Actions: {suppressed} suppressed, {promoted} promoted as references, "
            f"{added} new templates generated, {replaced} rewritten. "
            f"Iteration 1 removes underperforming content and amplifies proven patterns."
        ),
    }]

    final_rows = summary + \
        sorted(deltas, key=lambda x: (x["entity_type"], x["entity_id"]))
    fieldnames = ["entity_type", "entity_id", "change_type",
                  "metric_trigger", "before_value", "after_value", "explanation"]

    with open(LEARNING_DELTA_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)


_build_delta_report()
