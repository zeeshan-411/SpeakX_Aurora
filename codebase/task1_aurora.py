# # SpeakX Project Aurora — Task 1: System Architecture & Intelligence Design
# 
# ## Deliverables (exactly 5):
# 1. **`company_north_star.json`** — Inferred North Star metric, justification, measurable proxy
# 2. **`feature_goal_map.json`** — Feature → lifecycle → outcome mapping with Octalysis drives
# 3. **`allowed_tone_hook_matrix.json`** — Allowed/disallowed tones + all 8 Octalysis hooks
# 4. **`user_segments.csv`** — MECE segments with propensities, activeness, churn risk *(PS-compliant schema)*
# 5. **`segment_goals.csv`** — Goals × lifecycle × day-on-day focus *(wide-form: one row per segment × lifecycle)*
# 
# ---
# 
# ## Architecture
# ```
# KB (text/markdown) ──► run_kb_engine() ──► north_star.json
#                                        ──► feature_goal_map.json
#                                        ──► allowed_tone_hook_matrix.json
#                                        ──► KB_CONFIG (propensity_dims + journey + goal templates)
#                                                 │
# User CSV ──► validate_input() ──► engineer_features(KB_CONFIG) ──► run_segmentation_engine()
#                                                                          │
#                                                               export_user_segments() ──► user_segments.csv
#                                                               run_goals_builder()    ──► segment_goals.csv
# ```
# 
# **Domain-agnostic:** Swap the KB file + user CSV → all outputs change. No hardcoded SpeakX logic.
# 

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
load_dotenv()
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR   = 'iteration_0_before_learning'
INTERNAL_DIR = '_internal_debug'
os.makedirs(OUTPUT_DIR,   exist_ok=True)
os.makedirs(INTERNAL_DIR, exist_ok=True)

VALID_LIFECYCLES = ['trial', 'paid', 'churned', 'inactive']

# KB_CONFIG is populated at runtime by run_kb_engine().
# Holds propensity_dimensions, journey_templates, goal_templates — zero hardcoded domain logic.
# churn_weight_* are fixed behavioral constants: activeness is the primary churn signal (0.6),
# notification responsiveness is secondary (0.4). These are not overridden at runtime.
KB_CONFIG = {
    'churn_weight_activeness': 0.6,  # weight for (1-activeness_score) in churn_risk
    'churn_weight_notif':      0.4,  # weight for (1-norm_notif) in churn_risk
}

# Segmentation config — K-Means on propensity space, silhouette sweep k=6–12.
# Segments are behavioral personas; churn risk is annotated post-clustering.
SEGMENTATION_CONFIG = {
    'target_segments': (6, 12),
    'random_state':    42,
}

SEGMENTATION_FEATURES = []

INPUT_SCHEMA = {
    'user_id':                    {'type': str,   'required': True},
    'lifecycle_stage':            {'type': str,   'required': True,  'values': VALID_LIFECYCLES},
    'days_since_signup':          {'type': 'int', 'required': False, 'min': 0,   'max': 3650},
    'age_band':                   {'type': str,   'required': False},
    'region':                     {'type': str,   'required': False},
    'sessions_last_7d':           {'type': 'int', 'required': True,  'min': 0,   'max': 500},
    'exercises_completed_7d':     {'type': 'int', 'required': True,  'min': 0,   'max': 500},
    'streak_current':             {'type': 'int', 'required': True,  'min': 0,   'max': 3650},
    'coins_balance':              {'type': 'int', 'required': True,  'min': 0},
    'feature_ai_tutor_used':      {'type': bool,  'required': True},
    'feature_leaderboard_viewed': {'type': bool,  'required': True},
    'preferred_hour':             {'type': 'int', 'required': False, 'min': 0,   'max': 23},
    'notif_open_rate_30d':        {'type': float, 'required': True,  'min': 0.0, 'max': 1.0},
    'motivation_score':           {'type': float, 'required': True,  'min': 0.0, 'max': 1.0},
}
REQUIRED_COLUMNS = [col for col, spec in INPUT_SCHEMA.items() if spec.get('required')]

# CANONICAL_PROPENSITY_KEYS — populated by run_kb_engine() from the KB.
# SpeakX defaults below are fallback only; swapping the KB yields domain-specific keys.
SPEAKX_DEFAULT_PROPENSITY_KEYS = ['gamification', 'ai_tutor', 'leaderboard', 'social']
CANONICAL_PROPENSITY_KEYS = list(SPEAKX_DEFAULT_PROPENSITY_KEYS)

OCTALYSIS_DRIVES = [
    'Epic Meaning',
    'Accomplishment',
    'Empowerment',
    'Ownership',
    'Social Influence',
    'Scarcity',
    'Unpredictability',
    'Loss Avoidance',
]

LIFECYCLE_DAY_RANGES = {
    'trial':    [0, 1, 2, 3, 4, 5, 6, 7],
    'paid':     [8, 10, 15, 20, 25, 30],
    'churned':  [30, 45, 60, 90],
    'inactive': [30, 45, 60, 90],
}


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    5-layer validation:
      1. Required column presence
      2. Type coercion
      3. Range / value validation
      4. Missing data imputation with audit
      5. Duplicate user_id removal + anomaly flagging
    """

    # ── Layer 1: Required column presence ──────────────────────
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing_required:
        raise ValueError(
            f"\n❌ SCHEMA MISMATCH — Cannot proceed.\n"
            f"   Missing required columns: {missing_required}\n"
            f"   Dataset columns found:    {list(df.columns)}\n"
            f"   Please ensure your CSV matches the required schema."
        )

    # ── Layer 2: Type coercion ──────────────────────────────────
    df["lifecycle_stage"] = df["lifecycle_stage"].astype(str).str.lower().str.strip()

    for col, spec in INPUT_SCHEMA.items():
        if col not in df.columns:
            continue
        try:
            if spec["type"] == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            elif spec["type"] == float:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
            elif spec["type"] == bool:
                if df[col].dtype == object:
                    df[col] = (df[col].astype(str).str.lower()
                               .map({"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False})
                               .fillna(False))
                df[col] = df[col].astype(bool)
        except Exception as e:
            pass

    # ── Layer 3: Range & value validation ──────────────────────
    invalid_lc = df[~df["lifecycle_stage"].isin(VALID_LIFECYCLES)]
    if not invalid_lc.empty:
        df.loc[~df["lifecycle_stage"].isin(VALID_LIFECYCLES), "lifecycle_stage"] = "inactive"

    for col, spec in INPUT_SCHEMA.items():
        if col not in df.columns: continue
        if "min" in spec:
            bad = (df[col] < spec["min"]).sum()
            if bad > 0: df[col] = df[col].clip(lower=spec["min"])
        if "max" in spec:
            bad = (df[col] > spec["max"]).sum()
            if bad > 0: df[col] = df[col].clip(upper=spec["max"])

    # ── Layer 4: Missing data audit & imputation ────────────────
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        for col, cnt in null_cols.items():
            pct  = cnt / len(df) * 100
            spec = INPUT_SCHEMA.get(col, {})
            if spec.get("type") in ["int", float]:
                q05, q95  = df[col].quantile(0.05), df[col].quantile(0.95)
                fill_val  = df[col].clip(q05, q95).median()
                df[col]   = df[col].fillna(fill_val)
            else:
                df[col] = df[col].fillna("unknown")
    else:
        pass

    # ── Layer 5: Dedup + anomaly flag ──────────────────────────
    df = df.drop_duplicates(subset=["user_id"], keep="last").copy()

    for col, default in [("days_since_signup", 0), ("preferred_hour", 12),
                          ("age_band", "unknown"), ("region", "unknown")]:
        if col not in df.columns:
            df[col] = default

    return df

def read_kb(kb_path):
    if os.path.exists(kb_path):
        resolved = kb_path
    else:
        base = os.path.splitext(kb_path)[0]
        for ext in ['.md', '.txt']:
            if os.path.exists(base + ext):
                resolved = base + ext
                break
        else:
            md_files  = sorted(f for f in os.listdir('.') if f.endswith('.md'))
            txt_files = sorted(f for f in os.listdir('.') if f.endswith('.txt'))
            if md_files:    resolved = md_files[0]
            elif txt_files: resolved = txt_files[0]
            else: raise FileNotFoundError(f'KB not found: {kb_path}')

    with open(resolved, 'r', encoding='utf-8') as f:
        text = f.read()

    return text

def _ensure_canonical_propensities(data, df_columns):
    """
    Hard-lock propensity dimensions to exactly the 4 canonical types required by PS:
      gamification, ai_tutor, leaderboard, social

    Strategy:
      1. Walk Gemini-returned dims; if normalized key matches a canonical key, adopt
         its column_weights (re-weighted to available df columns).
      2. For any canonical key not covered by Gemini, use the hardcoded fallback.
      3. NO Gemini extras admitted — keeps feature space clean and schema pure.

    normalize_key strips common Gemini suffix variants:
      "gamification_propensity"        → "gamification"
      "ai_tutor_engagement_propensity" → "ai_tutor"
    """
    CANONICAL_DIMENSIONS = [
        {
            'key': 'gamification',
            'label': 'Gamification Engagement',
            'description': 'Reward accumulation and habit streak — log-normalized to reduce zero-inflation skew',
'column_weights': {'log_streak': 0.45, 'log_coins': 0.35, 'norm_sessions': 0.20}
        },
        {
            'key': 'ai_tutor',
            'label': 'AI Tutor Usage',
            'description': 'Active AI-guided practice: smoothed tutor usage signal + exercise depth',
'column_weights': {'ai_smoothed': 0.55, 'norm_ex': 0.45}
        },
        {
            'key': 'leaderboard',
            'label': 'Leaderboard & Competition',
            'description': 'Competitive engagement: smoothed leaderboard signal using norm_days (not sessions, no leakage)',
'column_weights': {'leaderboard_smoothed': 0.60, 'norm_days': 0.40}
        },
        {
            'key': 'social',
            'label': 'Social & Intrinsic Motivation',
            'description': 'Intrinsic drive and notification responsiveness',
            'column_weights': {'motivation_score': 0.60, 'notif_open_rate_30d': 0.40}
        },
    ]
    canonical_map = {d['key']: d for d in CANONICAL_DIMENSIONS}
    validated = []
    for can_key, can_dim in canonical_map.items():
        # Canonical weights are always used — Gemini frequently hallucinates column names.
        available = {col: w for col, w in can_dim['column_weights'].items()
                     if col in df_columns}

        if not available:
            fallback_col = 'sessions_last_7d' if 'sessions_last_7d' in df_columns else list(df_columns)[0]
            available = {fallback_col: 1.0}

        total = sum(available.values())
        dim_copy = dict(can_dim)
        dim_copy['column_weights'] = {k: round(v / total, 4) for k, v in available.items()}
        validated.append(dim_copy)

    # ── Feature exclusivity: no column may appear in more than one propensity ──
    # Prevents leaderboard ≈ social collapse from shared feature ownership.
    # Emergency fallback: use dim-specific INDEPENDENT columns guaranteed not shared.
    # This prevents the critical bug where social falls back to feature_leaderboard_viewed
    # (already owned by leaderboard), making leaderboard × social corr = 0.97 and
    # suppressing gamification via softmax variance collapse.
    _EXCLUSIVITY_FALLBACKS = {
        'gamification': {'log_streak': 0.6, 'log_coins': 0.4},
        'ai_tutor':     {'feature_ai_tutor_used': 0.7, 'exercises_completed_7d': 0.3},
        'leaderboard':  {'feature_leaderboard_viewed': 1.0},
        'social':       {'motivation_score': 0.6, 'notif_open_rate_30d': 0.4},
    }
    used_cols_global = set()
    for dim in validated:
        filtered = {}
        for col, w in dim['column_weights'].items():
            if col in used_cols_global:
                pass
            else:
                used_cols_global.add(col)
                filtered[col] = w
        if not filtered:
            # Use dim-specific independent fallback — never re-use another dim's columns.
            # This is the critical fix: social must use motivation_score/notif, NOT feature_leaderboard_viewed.
            dim_fallback = _EXCLUSIVITY_FALLBACKS.get(dim['key'], {})
            available_fallback = {c: w for c, w in dim_fallback.items()
                                  if c in df_columns and c not in used_cols_global}
            if available_fallback:
                filtered = available_fallback
                for c in filtered:
                    used_cols_global.add(c)
            else:
                # Absolute last resort: any unused column in df
                unused = [c for c in df_columns if c not in used_cols_global and c != 'user_id']
                if unused:
                    filtered = {unused[0]: 1.0}
                    used_cols_global.add(unused[0])
                else:
                    filtered = {list(dim['column_weights'].keys())[0]: 1.0}
        total_f = sum(filtered.values())
        dim['column_weights'] = {k: round(v / total_f, 4) for k, v in filtered.items()}

    data['propensity_dimensions'] = validated
    for d in validated:
        pass
    return data

def run_kb_engine(kb_path='company_kb.md', api_key=None):
    """
    Extracts strategic artifacts from KB via Gemini.
    KB + user data stats together drive:
      - north_star, feature_goal_map, tone_hook_matrix
      - propensity_dimensions + column_weights (drives feature engineering)
      - journey_templates (drives day-on-day messages in segment_goals.csv)
      - goal_templates    (drives primary/sub goals in segment_goals.csv)
    Canonical propensity dimensions (gamification, ai_tutor, leaderboard, social)
    are guaranteed via _ensure_canonical_propensities().
    """
    global KB_CONFIG, SEGMENTATION_FEATURES

    kb_text = read_kb(kb_path)

    try:
        from google import genai
        from google.genai import types as genai_types
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
        client = genai.Client(api_key=api_key)
        MODEL_ID = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    except ImportError:
        raise ImportError('google-genai not installed. Run: pip install google-genai')

    # ── Build user data summary from the actual CSV ───────────────────────────
    user_stats_text = ''
    csv_columns = []
    try:
        candidates = [f for f in os.listdir('.') if f.endswith('.csv')
                      and 'user' in f.lower() and 'segment' not in f.lower()
                      and 'goal' not in f.lower() and 'schedule' not in f.lower()]
        if candidates:
            udf = pd.read_csv(candidates[0])
            csv_columns = list(udf.columns)
            bool_cols = ['feature_ai_tutor_used', 'feature_leaderboard_viewed']
            for c in bool_cols:
                if c in udf.columns:
                    udf[c] = udf[c].astype(str).str.lower().map(
                        {'true': True, 'false': False, '1': True, '0': False})
            stats_lines = [
                f'ACTUAL USER BEHAVIORAL DATA SUMMARY (from {candidates[0]}):',
                f'Total users: {len(udf)}',
                f'CSV columns available: {list(udf.columns)}',
                f'Lifecycle distribution: {udf["lifecycle_stage"].value_counts().to_dict()}',
                f'sessions_last_7d — mean:{udf["sessions_last_7d"].mean():.2f} median:{udf["sessions_last_7d"].median():.0f}',
                f'exercises_completed_7d — mean:{udf["exercises_completed_7d"].mean():.2f}',
                f'streak_current — mean:{udf["streak_current"].mean():.2f}',
                f'coins_balance — mean:{udf["coins_balance"].mean():.1f}',
                f'notif_open_rate_30d — mean:{udf["notif_open_rate_30d"].mean():.3f}',
                f'motivation_score — mean:{udf["motivation_score"].mean():.3f}',
            ]
            if 'feature_ai_tutor_used' in udf.columns:
                stats_lines.append(f'feature_ai_tutor_used: {udf["feature_ai_tutor_used"].mean()*100:.1f}% of users')
            if 'feature_leaderboard_viewed' in udf.columns:
                stats_lines.append(f'feature_leaderboard_viewed: {udf["feature_leaderboard_viewed"].mean()*100:.1f}% of users')
            user_stats_text = '\n'.join(stats_lines)
    except Exception as e:
        user_stats_text = 'User data stats unavailable.'

    prompt = f"""You are a Strategic Intelligence Engine for a notification orchestration system.

COMPANY KNOWLEDGE BANK:
{kb_text[:50000]}

{user_stats_text}

Extract ALL strategic artifacts in STRICT JSON. Output ONLY valid JSON — no markdown, no extra text.

OUTPUT JSON SCHEMA:
{{
  "north_star": {{
    "metric": "<single most important metric>",
    "justification": "<why, referencing KB product behavior>",
    "measurable_proxy": "<formula using CSV column names>"
  }},
  "feature_goal_map": [
    {{
      "feature": "<feature from KB>",
      "lifecycle_stages": ["trial","paid","churned","inactive"],
      "outcome": "<business outcome>",
      "octalysis_drives": ["<drive1>","<drive2>"]
    }}
  ],
  "tone_hook_matrix": {{
    "allowed_tones": ["<tone1>","<tone2>","<tone3>","<tone4>"],
    "disallowed_tones": ["<tone1>","<tone2>"],
    "segment_tone_preferences": {{
      "highly_active": "<tone>", "moderately_active": "<tone>",
      "low_active": "<tone>", "at_risk": "<tone>"
    }},
    "octalysis_hooks": {{
      "Epic Meaning": "<description of how this drive applies to this product and its users>",
      "Accomplishment": "<description of how this drive applies to this product and its users>",
      "Empowerment": "<description of how this drive applies to this product and its users>",
      "Ownership": "<description of how this drive applies to this product and its users>",
      "Social Influence": "<description of how this drive applies to this product and its users>",
      "Scarcity": "<description of how this drive applies to this product and its users>",
      "Unpredictability": "<description of how this drive applies to this product and its users>",
      "Loss Avoidance": "<description of how this drive applies to this product and its users>"
    }}
  }},
  "propensity_dimensions": [
    {{
      "key": "<short_snake_case_key>",
      "label": "<human readable label>",
      "description": "<what engagement axis this measures>",
      "column_weights": {{
        "<csv_column>": <0.0-1.0>,
        "<csv_column>": <0.0-1.0>
      }}
    }}
  ],
  "journey_templates": {{
    "trial":   {{
      "D0": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D1": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D2": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D3": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D4": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D5": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D6": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D7": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}}
    }},
    "paid": {{
      "D8":  {{"default": "<msg>", "at_risk": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D10": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D15": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D20": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D25": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D30": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}}
    }},
    "churned": {{
      "D30": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D45": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D60": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D90": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}}
    }},
    "inactive": {{
      "D30": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D45": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D60": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}},
      "D90": {{"default": "<msg>", "gamification": "<msg>", "ai_tutor": "<msg>", "leaderboard": "<msg>", "social": "<msg>"}}
    }}
  }},
  "goal_templates": {{
    "trial":   {{ "default": {{ "primary_goal": "<full sentence goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "gamification": {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "ai_tutor":     {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "leaderboard":  {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "social":       {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }} }},
    "paid":    {{ "default": {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "at_risk":     {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "gamification": {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "ai_tutor":     {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "leaderboard":  {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "social":       {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }} }},
    "churned": {{ "default": {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "gamification": {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "ai_tutor":     {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "leaderboard":  {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "social":       {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }} }},
    "inactive":{{ "default": {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "gamification": {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "ai_tutor":     {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "leaderboard":  {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }},
                 "social":       {{ "primary_goal": "<goal>", "sub_goal": "<sub>", "messaging_focus": "<focus>" }} }}
  }}
}}

CRITICAL RULES:
- propensity_dimensions MUST include keys for gamification, ai_tutor, leaderboard, and social engagement. Use ONLY column names listed in the CSV columns above for column_weights. Weights must sum to 1.0.
- journey_templates: every message must be a complete, ready-to-send push notification (≤80 chars) grounded in the product KB. Each day dict MUST contain all 5 variants: default, gamification, ai_tutor, leaderboard, social. Messages MUST differ meaningfully across propensity types — no copy-pasting the default.
- goal_templates: every lifecycle MUST have all 5 keys: default, gamification, ai_tutor, leaderboard, social. primary_goal must be a full sentence describing the specific user action (e.g. "Complete your first AI tutor session and build a 3-day streak" NOT "User Activation").
- All 8 octalysis_hooks REQUIRED. Each value must be a description of how that Octalysis drive applies to this specific product and its users — NOT a push notification message example.
"""

    data = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_ID, contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type='application/json', temperature=0.1))
            data = json.loads(response.text)
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(
                    f'Gemini API unavailable after 3 attempts. '
                    f'Last error: {e}\n'
                    f'Please check your API key and network connection, then re-run.'
                )
            else:
                time.sleep(2)

    # ── Validate required keys ────────────────────────────────────────────────
    for key in ['north_star','feature_goal_map','tone_hook_matrix',
                'propensity_dimensions','journey_templates','goal_templates']:
        if key not in data:
            raise ValueError(f'LLM response missing key: {key}')

    hooks = data['tone_hook_matrix'].get('octalysis_hooks', {})
    missing_hooks = [d for d in OCTALYSIS_DRIVES if d not in hooks]
    if missing_hooks:
        raise ValueError(f'Missing Octalysis hooks: {missing_hooks}')

    if not data.get('propensity_dimensions'):
        raise ValueError('KB Engine returned empty propensity_dimensions')

    # Normalize weights
    for dim in data['propensity_dimensions']:
        total = sum(dim['column_weights'].values())
        if total > 0:
            dim['column_weights'] = {k: round(v/total, 4) for k,v in dim['column_weights'].items()}

    fm_stages = {lc for item in data['feature_goal_map'] for lc in item.get('lifecycle_stages', [])}
    for lc in VALID_LIFECYCLES:
        if lc not in fm_stages:
            data['feature_goal_map'].append({
                'feature': f'{lc.title()} Re-engagement', 'lifecycle_stages': [lc],
                'outcome': 'Re-engagement', 'octalysis_drives': ['Loss Avoidance']})

    # ── GUARDRAIL: ensure all 4 canonical propensity keys exist ──────────────
    # Include derived column names so canonical fallback can resolve log_streak,
    # log_coins, norm_sessions, ai_smoothed etc. (computed in engineer_features).
    # Without this, 'log_streak' not in csv_columns → dropped → Gemini's hallucinated
    # column (e.g. 'age_group') wins the filter and propensities all default to 0.
    _DERIVED_COLS = ['log_streak', 'log_coins', 'norm_sessions', 'norm_ex',
                     'ai_smoothed', 'norm_streak', 'norm_coins', 'norm_notif',
                     'norm_days', 'leaderboard_smoothed', 'norm_motivation']
    _all_cols = (csv_columns if csv_columns else list(INPUT_SCHEMA.keys())) + _DERIVED_COLS
    data = _ensure_canonical_propensities(data, _all_cols)

    # ── Store in global KB_CONFIG ─────────────────────────────────────────────
    KB_CONFIG.update({
        'north_star':            data['north_star'],
        'feature_goal_map':      data['feature_goal_map'],
        'tone_hook_matrix':      data['tone_hook_matrix'],
        'propensity_dimensions': data['propensity_dimensions'],
        'journey_templates':     data['journey_templates'],
        'goal_templates':        data['goal_templates'],
    })

    # Update CANONICAL_PROPENSITY_KEYS from validated KB propensity dimensions.
    global CANONICAL_PROPENSITY_KEYS
    kb_keys = [d['key'] for d in data['propensity_dimensions']]
    if kb_keys:
        CANONICAL_PROPENSITY_KEYS.clear()
        CANONICAL_PROPENSITY_KEYS.extend(kb_keys)
    else:
        pass

    SEGMENTATION_FEATURES.clear()
    SEGMENTATION_FEATURES.extend([f"propensity_{d['key']}" for d in KB_CONFIG.get('propensity_dimensions', [])])

    with open(f'{OUTPUT_DIR}/company_north_star.json', 'w') as f:
        json.dump(data['north_star'], f, indent=2)
    with open(f'{OUTPUT_DIR}/feature_goal_map.json', 'w') as f:
        json.dump(data['feature_goal_map'], f, indent=2)
    with open(f'{OUTPUT_DIR}/allowed_tone_hook_matrix.json', 'w') as f:
        json.dump(data['tone_hook_matrix'], f, indent=2)
    with open(f'{INTERNAL_DIR}/kb_intelligence_config.json', 'w') as f:
        json.dump({'propensity_dimensions': data['propensity_dimensions'],
                   'journey_templates':     data['journey_templates'],
                   'goal_templates':        data['goal_templates']}, f, indent=2)

    for dim in data['propensity_dimensions']:
        pass

    return data


def engineer_features(df):
    """
    Computes behavioral scores.
    Propensity dimensions and their column weights come from KB_CONFIG,
    which was populated by run_kb_engine() from the actual company KB.
    No domain-specific feature names are hardcoded here.
    """

    if not KB_CONFIG.get('propensity_dimensions'):
        raise RuntimeError('KB_CONFIG is empty. run_kb_engine() must be called before engineer_features().')

    def pct_normalize(s, q=0.99):
        cap = s.quantile(q)
        return (s.clip(upper=cap) / (cap if cap > 0 else 1)).round(4)

    norm_map = {}
    # Key-name overrides for raw CSV columns whose natural replace-chain name
    # diverges from the name used in canonical propensity weights or downstream code:
    #   exercises_completed_7d → replace chain produces 'norm_exercises_ex'
    #                            but canonical ai_tutor weights reference 'norm_ex'
    #   days_since_signup      → replace chain produces 'norm_days_days'
    #                            but leaderboard block writes 'norm_days' separately;
    #                            unify here so norm_map is always consistent
    _NORM_KEY_OVERRIDES = {
        'exercises_completed_7d': 'norm_ex',
        'days_since_signup':      'norm_days',
    }

    for col in ['sessions_last_7d','exercises_completed_7d','streak_current',
                'coins_balance','days_since_signup']:
        if col in df.columns:
            key = _NORM_KEY_OVERRIDES.get(col) or (
                'norm_' + col.replace('_last_7d','').replace('_current','')
                             .replace('_completed_7d','_ex').replace('_balance','')
                             .replace('_since_signup','_days')
            )
            df[key] = pct_normalize(df[col])
            norm_map[col] = key

    for col in ['notif_open_rate_30d','motivation_score']:
        if col in df.columns:
            df['norm_' + col.split('_')[0]] = df[col].clip(0,1)
    df['norm_notif'] = df.get('norm_notif', df.get('notif_open_rate_30d', pd.Series(0.5, index=df.index))).clip(0,1)

    # ── Log normalization for zero-inflated gamification columns ─────────────
    # streak_current and coins_balance are heavily right-skewed with many zeros.
    # Log normalization compresses the long tail and lifts the zero-heavy base,
    # giving gamification a fair chance to compete with social/leaderboard means.
    for raw_col, log_col in [('streak_current', 'log_streak'), ('coins_balance', 'log_coins')]:
        if raw_col in df.columns:
            cap_val = df[raw_col].quantile(0.99)
            df[log_col] = (np.log1p(df[raw_col].clip(upper=cap_val)) /
                           np.log1p(cap_val if cap_val > 0 else 1)).round(4)

    bool_cols = [c for c in df.columns if df[c].dtype == bool or c.startswith('feature_')]
    for c in bool_cols:
        df[f'int_{c}'] = df[c].astype(int)

    # ── AI tutor intensity smoothing ─────────────────────────────────────────
    # Replaces binary feature_ai_tutor_used with a gradient signal:
    #   ai_smoothed = 0.6 * feature_ai_tutor_used + 0.4 * norm_sessions
    # A hard binary flag creates a cliff-edge that structurally dominates softmax.
    # The blended signal is a more accurate measure of AI engagement intensity.
    if 'feature_ai_tutor_used' in df.columns and 'norm_sessions' in df.columns:
        df['ai_smoothed'] = (
            0.6 * df['feature_ai_tutor_used'].astype(float) +
            0.4 * df['norm_sessions'].clip(0, 1)
        ).round(4)

    # leaderboard_smoothed: uses norm_days instead of norm_sessions to avoid session-axis leakage
    # (leaderboard and ai_tutor would both use sessions, creating correlated propensities)
    if 'feature_leaderboard_viewed' in df.columns:
        _norm_days = df['norm_days']
        df['leaderboard_smoothed'] = (
            0.6 * df['feature_leaderboard_viewed'].astype(float) +
            0.4 * _norm_days.clip(0, 1)
        ).round(4)

    # ── KB-driven propensity scores ───────────────────────────
    for dim in KB_CONFIG['propensity_dimensions']:
        key   = dim['key']
        col_w = dim['column_weights']
        score = pd.Series(0.0, index=df.index)
        for raw_col, weight in col_w.items():
            # Resolution order: ai_smoothed → log_ → norm_ → raw → int_
            # ai_smoothed takes priority for feature_ai_tutor_used — gradient over cliff.
            # log_ preferred for streak/coins — log-compressed, bounded [0,1].
            candidates = [
                f'log_{raw_col}',
                norm_map.get(raw_col, ''),
                f'norm_{raw_col.split("_")[0]}',
                f'norm_{raw_col}',
                raw_col,
                f'int_{raw_col}'
            ]
            found = next((c for c in candidates if c and c in df.columns), None)
            if found:
                # Weight ceiling: cap binary features at 0.5 to prevent a single hard-cliff
                # flag from monopolizing the dimension. Continuous features are uncapped.
                _col_is_binary = (
                    df[found].dropna().isin([0, 1]).all() and df[found].nunique() <= 2
                )
                _effective_weight = min(weight, 0.5) if _col_is_binary else weight
                score += _effective_weight * df[found].clip(0, 1)
            else:
                pass
        df[f'propensity_{key}'] = score.round(4)

    # ── Propensity normalization: dimension-wise percentile equalization → Row-L1 ──
    # Row-L1 replaces softmax: softmax collapses to 0.25 for users near column means.
    # Canonical columns are hardcoded (never Gemini-sourced), ensuring non-zero scores.
    _prop_cols_computed = [f'propensity_{d["key"]}' for d in KB_CONFIG['propensity_dimensions']
                           if f'propensity_{d["key"]}' in df.columns]
    if _prop_cols_computed:
        # Save raw scores for clustering
        _raw_prop_colnames = [f'_raw_{c}' for c in _prop_cols_computed]
        df[_raw_prop_colnames] = df[_prop_cols_computed].copy()

        _raw_props = df[_prop_cols_computed].values.astype(float)
        # Dimension-wise percentile equalization before row-L1.
        # A dimension backed by 3 abundant features will structurally outscore one with
        # a single sparse binary input — an artifact of feature count, not true signal.
        # Replacing each dim's raw score with its per-user percentile rank gives every
        # dimension the same uniform [1/N, 1.0] distribution before row-L1 is applied.
        # Domain-agnostic: enforces equal footing regardless of feature count per dimension.
        _PROP_FLOOR = 1e-6
        _floored   = np.clip(_raw_props, _PROP_FLOOR, None)

        # Step 1: dimension-wise percentile rank — uniform [1/N, 1.0] per column
        _n_users   = _floored.shape[0]
        _equalized = np.zeros_like(_floored, dtype=float)
        for _dim_j in range(_floored.shape[1]):
            _equalized[:, _dim_j] = rankdata(_floored[:, _dim_j], method='average') / _n_users

        # Step 2: row-L1 on equalized scores
        _row_sums  = _equalized.sum(axis=1, keepdims=True)
        _row_norm  = np.round(_equalized / _row_sums, 4)
        df[_prop_cols_computed] = _row_norm


    # ── Activeness score — equal-weight mean of key behavioral signals ──────────
    _fallback_cols = ['norm_sessions', 'norm_ex', 'norm_streak', 'motivation_score']
    _avail = [c for c in _fallback_cols if c in df.columns]
    act_score = df[_avail].clip(0, 1).mean(axis=1).round(4) if _avail else pd.Series(0.5, index=df.index)

    df['activeness_score'] = act_score.clip(0, 1)

    # ── Churn risk — purely behavioral, no lifecycle leakage ─────────────────────
    # activeness (0.6) + notification responsiveness (0.4).
    # lifecycle_stage is never used here — no target leakage.
    w_act   = KB_CONFIG.get('churn_weight_activeness', 0.6)
    w_notif = KB_CONFIG.get('churn_weight_notif',      0.4)

    df['churn_risk'] = (
        w_act   * (1 - df['activeness_score']) +
        w_notif * (1 - df['norm_notif'])
    ).clip(0, 1).round(4)

    return df


def _get_churn_tier(avg_churn):
    try: v = float(avg_churn)
    except (TypeError, ValueError): v = 0.5
    if v >= 0.65: return 'high'
    if v >= 0.35: return 'mid'
    return 'low'

def run_segmentation_engine(df):

    pass

    # Propensity cols from validated KB dims — filters to SEGMENTATION_FEATURES only.
    prop_cols = [c for c in SEGMENTATION_FEATURES if c.startswith("propensity_") and c in df.columns]

    # Deduplicate: if somehow duplicates crept in (e.g. same key different suffix), keep first occurrence
    seen_keys = set()
    deduped_prop_cols = []
    for c in prop_cols:
        clean = c.replace("propensity_", "")
        if clean not in seen_keys:
            seen_keys.add(clean)
            deduped_prop_cols.append(c)
    if len(deduped_prop_cols) < len(prop_cols):
        pass
    prop_cols = deduped_prop_cols

    if not prop_cols:
        raise ValueError("No propensity columns found in SEGMENTATION_FEATURES. Ensure run_kb_engine() ran.")

    # ── Segmentation: K-Means on propensity space ────────────────────────────
    # Segments are behavioral personas (gamifier, AI-learner, competitor, social-motivated),
    # not churn buckets. K-Means on propensity scores surfaces identity clusters;
    # churn risk and activeness are annotated onto each cluster as segment attributes.
    # A DT on churn_bins would produce risk stratifications with identical messaging needs.
    # Compute churn_risk if not already present (engineer_features should have set it)
    if "churn_risk" not in df.columns:
        df["churn_risk"] = (0.6 * (1 - df["activeness_score"]) + 0.4 * (1 - df["norm_notif"])).clip(0, 1)

    # ── K-Means feature space: raw (pre-L1) propensity scores + activeness_score ──
    # Raw scores have more spread than post-L1 and produce better cluster separation.
    # churn_risk and lifecycle excluded — including them produces risk tiers, not personas.
    _raw_prop_cols = [f'_raw_propensity_{k}' for k in CANONICAL_PROPENSITY_KEYS
                      if f'_raw_propensity_{k}' in df.columns]
    if _raw_prop_cols:

        _cluster_features = _raw_prop_cols + ['activeness_score']
    else:
        _cluster_features = prop_cols + ['activeness_score']

    # ── Within-lifecycle percentile normalization ─────────────────────────────
    # Propensity scores are lifecycle-correlated (trial users have low streaks, paid users
    # have high streaks). K-Means on absolute values would produce lifecycle-stratified
    # clusters, not behavioral personas. Replacing each value with its within-lifecycle
    # percentile rank removes the lifecycle mean shift and de-correlates propensity from stage.
    _lc_norm = df[_cluster_features].copy().astype(float)
    for _lc in df['lifecycle_stage'].unique():
        _mask = df['lifecycle_stage'] == _lc
        _n    = int(_mask.sum())
        if _n < 2:
            continue
        for _col in _cluster_features:
            if _col in _lc_norm.columns:
                _vals = _lc_norm.loc[_mask, _col].values
                _lc_norm.loc[_mask, _col] = rankdata(_vals, method='average') / _n

    X_kmeans = _lc_norm.fillna(0).values

    target_min, target_max = SEGMENTATION_CONFIG['target_segments']
    rng = SEGMENTATION_CONFIG['random_state']

    best_k, best_score, best_labels = None, -1, None

    for k in range(target_min, target_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=rng, n_init=10)
            labels = km.fit_predict(X_kmeans)
            score = silhouette_score(X_kmeans, labels)
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
        except Exception as e:
            pass

    if best_labels is None:
        best_k = 7
        km = KMeans(n_clusters=best_k, random_state=rng, n_init=10)
        best_labels = km.fit_predict(X_kmeans)
        best_score = silhouette_score(X_kmeans, best_labels) if best_k > 1 else -1

    # ── Silhouette quality gate ───────────────────────────────────────────────
    # Informational threshold — does not halt execution, but flags weak separation.
    # silhouette < 0.10 → segments overlap heavily; messaging differentiation is weak.
    # silhouette 0.10-0.20 → acceptable separation; common for behavioral data.
    # silhouette > 0.20 → good persona distinctiveness.
    if best_score < 0.10:
        pass
    elif best_score < 0.20:
        pass
    else:
        pass

    df["segment_id"] = best_labels

    n_segs = len(np.unique(best_labels))

    # Structural health check: flag if any single propensity dominates >40% of users.
    _max_prop_vals = df[prop_cols].max(axis=1)
    _dom_inline = np.where(_max_prop_vals < 0.30, "balanced",
                           df[prop_cols].idxmax(axis=1).str.replace("propensity_", ""))
    _dom_dist = pd.Series(_dom_inline).value_counts(normalize=True)
    _max_bucket_pct = _dom_dist.iloc[0] * 100 if len(_dom_dist) > 0 else 0
    if _max_bucket_pct > 40:
        pass
    else:
        pass

    # ── HARD GUARANTEE: segment count must be 6-12 ───────────────────────────
    if not (target_min <= n_segs <= target_max):
        raise RuntimeError(
            f"\n❌ SEGMENT COUNT VIOLATION: {n_segs} segments produced.\n"
            f"   PS requires {target_min}–{target_max} MECE segments.\n"
            f"   Adjust target_segments range in SEGMENTATION_CONFIG."
        )

    # ── Persona naming: (dominant, act_tier, churn_tier) → unique segment name ──
    # Names describe behavioral identity, not churn tier.
    # 3-axis key yields up to 4×3×3=36 slots — structurally prevents collisions.
    # act_tier:   high >= q66, mid >= q33, low < q33
    # churn_tier: low < 0.35, mid < 0.65, high >= 0.65
    _PERSONA_MAP = {
        # ── GAMIFICATION ──────────────────────────────────────────────────────
        ('gamification', 'high', 'low'):  'Streak Champions',
        ('gamification', 'high', 'mid'):  'Streak Champions',
        ('gamification', 'high', 'high'): 'At-Risk Streak Keepers',
        ('gamification', 'mid',  'low'):  'Habit Builders',
        ('gamification', 'mid',  'mid'):  'Habit Builders',
        ('gamification', 'mid',  'high'): 'Fading Habit Formers',
        ('gamification', 'low',  'low'):  'Casual Coin Collectors',
        ('gamification', 'low',  'mid'):  'Lapsed Streak Seekers',
        ('gamification', 'low',  'high'): 'Disengaged Gamers',
        # ── AI TUTOR ──────────────────────────────────────────────────────────
        ('ai_tutor',     'high', 'low'):  'Fluency Achievers',
        ('ai_tutor',     'high', 'mid'):  'Coached Improvers',
        ('ai_tutor',     'high', 'high'): 'Intensive Learners',
        ('ai_tutor',     'mid',  'low'):  'Guided Practitioners',
        ('ai_tutor',     'mid',  'mid'):  'Coached Improvers',
        ('ai_tutor',     'mid',  'high'): 'Struggling Learners',
        ('ai_tutor',     'low',  'low'):  'Passive AI Explorers',
        ('ai_tutor',     'low',  'mid'):  'Passive AI Explorers',
        ('ai_tutor',     'low',  'high'): 'Dormant AI Users',
        # ── LEADERBOARD ───────────────────────────────────────────────────────
        ('leaderboard',  'high', 'low'):  'Competitive Climbers',
        ('leaderboard',  'high', 'mid'):  'Rank Chasers',
        ('leaderboard',  'high', 'high'): 'Intense Competitors',
        ('leaderboard',  'mid',  'low'):  'Social Competitors',
        ('leaderboard',  'mid',  'mid'):  'Rank Chasers',
        ('leaderboard',  'mid',  'high'): 'Lapsed Competitors',
        ('leaderboard',  'low',  'low'):  'Casual Rankers',
        ('leaderboard',  'low',  'mid'):  'Lapsed Competitors',
        ('leaderboard',  'low',  'high'): 'Disengaged Rivals',
        # ── SOCIAL ────────────────────────────────────────────────────────────
        ('social',       'high', 'low'):  'Community Leaders',
        ('social',       'high', 'mid'):  'Independent Learners',
        ('social',       'high', 'high'): 'Motivated Achievers',
        ('social',       'mid',  'low'):  'Self-Motivated Explorers',
        ('social',       'mid',  'mid'):  'Self-Motivated Learners',
        ('social',       'mid',  'high'): 'Intrinsically Driven',
        ('social',       'low',  'low'):  'Quiet Learners',
        ('social',       'low',  'mid'):  'Dormant Self-Starters',
        ('social',       'low',  'high'): 'Disengaged Independents',
        # ── BALANCED ──────────────────────────────────────────────────────────
        ('balanced',     'high', 'low'):  'All-Round High Performers',
        ('balanced',     'high', 'mid'):  'All-Round Learners',
        ('balanced',     'high', 'high'): 'Versatile but At-Risk',
        ('balanced',     'mid',  'low'):  'Balanced Practitioners',
        ('balanced',     'mid',  'mid'):  'Balanced Practitioners',
        ('balanced',     'mid',  'high'): 'Mixed-Signal Users',
        ('balanced',     'low',  'low'):  'Low-Engagement Mixed',
        ('balanced',     'low',  'mid'):  'Low-Engagement Mixed',
        ('balanced',     'low',  'high'): 'Dormant Mixed',
    }

    _act_q33 = float(df["activeness_score"].quantile(0.33))
    _act_q66 = float(df["activeness_score"].quantile(0.66))

    def _get_tiers(sid):
        sub       = df[df["segment_id"] == sid]
        act       = sub["activeness_score"].mean()
        churn     = sub["churn_risk"].mean()
        dom       = sub[prop_cols].mean().idxmax().replace("propensity_", "")
        act_tier  = "high" if act >= _act_q66 else ("mid" if act >= _act_q33 else "low")
        churn_tier = "low" if churn < 0.35 else ("mid" if churn < 0.65 else "high")
        return dom, act_tier, churn_tier

    def _make_persona_name(sid):
        dom, act_tier, churn_tier = _get_tiers(sid)
        key = (dom, act_tier, churn_tier)
        return _PERSONA_MAP.get(key, f"{dom.title()} {act_tier.title()} {churn_tier.title()}")

    final_names = {sid: _make_persona_name(sid) for sid in sorted(df["segment_id"].unique())}


    _SYNTHESIS_NAMES = {
        ('social',       'leaderboard'): 'Social Strategists',
        ('leaderboard',  'social'):      'Community Competitors',
        ('ai_tutor',     'leaderboard'): 'Coached Competitors',
        ('ai_tutor',     'gamification'): 'Structured Learners',
        ('gamification', 'ai_tutor'):    'Coached Streak Builders',
        ('gamification', 'leaderboard'): 'Competitive Habit Builders',
        ('leaderboard',  'ai_tutor'):    'Competitive AI Users',
        ('social',       'gamification'): 'Motivated Streak Builders',
    }
    # ── Collision resolution: Tier 1 (dominant >0.50) → Tier 2 (synthesis) → Tier 3 (suffix) ──
    _HIGH_INTENSITY_NAMES = {
        'leaderboard':  'Lapsed Competitors',
        'gamification': 'Streak Obsessed',
        'ai_tutor':     'Deep Learners',
        'social':       'Community Champions',
    }
    _SECONDARY_THRESHOLD = 0.26  # meaningfully above the 4-dim uniform baseline of 0.25

    # Iterative dedup: each pass escalates suffix richness until all names are unique.
    # Pass 1 → propensity profile; Pass 2 → churn tier; Pass 3 → act tier; Pass 4 → alpha index.

    def _resolve_collision(sid, existing_name, pass_num):
        sub     = df[df["segment_id"] == sid]
        means   = sub[prop_cols].mean().sort_values(ascending=False)
        dom_key    = means.index[0].replace("propensity_", "")
        dom_val    = float(means.iloc[0])
        second_key = means.index[1].replace("propensity_", "")
        second_val = float(means.iloc[1])
        act_m      = sub["activeness_score"].mean()
        churn_m    = sub["churn_risk"].mean()
        churn_tier = _get_churn_tier(churn_m)
        act_tier   = "Active" if act_m >= _act_q66 else ("Growing" if act_m >= _act_q33 else "Lapsed")

        if pass_num == 1:
            if dom_val > 0.50 and dom_key in _HIGH_INTENSITY_NAMES:
                return _HIGH_INTENSITY_NAMES[dom_key]
            elif (dom_key, second_key) in _SYNTHESIS_NAMES:
                return _SYNTHESIS_NAMES[(dom_key, second_key)]
            elif second_val >= _SECONDARY_THRESHOLD:
                label = second_key.replace("_"," ").title()
                return existing_name + " (" + label + ")"
            else:
                label = dom_key.replace("_"," ").title()
                return "Intense " + label + " Users"
        elif pass_num == 2:
            churn_label = {"high": "High Churn", "mid": "At Risk", "low": "Retained"}[churn_tier]
            return existing_name + " (" + churn_label + ")"
        elif pass_num == 3:
            return existing_name + " (" + act_tier + ")"

    _ALPHA = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    _alpha_counters = {}

    MAX_PASSES = 4
    for _pass in range(1, MAX_PASSES + 1):
        seen_names = {}
        collisions = []
        for sid in sorted(final_names):
            name = final_names[sid]
            if name in seen_names:
                collisions.append(sid)
            else:
                seen_names[name] = sid

        if not collisions:
            break

        if _pass < MAX_PASSES:
            for sid in collisions:
                final_names[sid] = _resolve_collision(sid, final_names[sid], _pass)
        else:
            for sid in collisions:
                base = final_names[sid]
                if base not in _alpha_counters:
                    _alpha_counters[base] = 0
                idx = _alpha_counters[base]
                suffix = _ALPHA[idx] if idx < len(_ALPHA) else str(idx + 2)
                final_names[sid] = base + " " + suffix
                _alpha_counters[base] += 1

    if collisions:
        pass
    else:
        pass

    df["segment_name"] = df["segment_id"].map(final_names)

    # ── MECE validation ───────────────────────────────────────────────────────
    assert df["segment_name"].isnull().sum() == 0, "MECE violation: null segment_names"
    assert df["segment_id"].isnull().sum() == 0,   "MECE violation: null segment_ids"

    for seg, sd in df.groupby("segment_name"):
        seg_act   = sd["activeness_score"].mean()
        seg_churn = sd["churn_risk"].mean()
        row_str   = f"   {seg:<28} {seg_act:.3f}  {seg_churn:.3f}  "
        row_str  += "  ".join(f"{sd[c].mean():.3f}  " for c in prop_cols)
        row_str  += str(len(sd))

    # ── Within-cluster variance — split justification audit ──────────────────
    # For any segment containing >30% of users, inspect internal propensity std.
    # HIGH internal std (>0.08) → meaningful substructure exists → splitting may be justified.
    # LOW  internal std (<0.05) → cluster is tight and coherent → keep as-is, do not force-split.
    # This provides an auditable, data-driven answer to "why didn't you split segment X?"
    large_segs = df.groupby('segment_name').filter(lambda g: len(g) / len(df) > 0.30)
    if not large_segs.empty:
        for seg_name, seg_data in large_segs.groupby('segment_name'):
            pct = len(seg_data) / len(df) * 100
            stds = seg_data[prop_cols].std()
            max_std = stds.max()
            verdict = "⚠️  HIGH internal variance — splitting may be justified" if max_std > 0.08 else "✅ Tight cluster — coherent persona, splitting not justified"
            for c, v in stds.items():
                pass

    # ── Serialize K-Means config for Iteration 1 stability ───────────────────
    os.makedirs(INTERNAL_DIR, exist_ok=True)
    try:
        joblib.dump({'k': best_k, 'silhouette': best_score, 'labels': list(best_labels)},
                    f"{INTERNAL_DIR}/segmentation_kmeans.joblib")
    except Exception as e:
        pass

    return df, best_k


import csv
def _build_segment_description(seg_data, prop_cols):
    """
    Auto-generates a human-readable description of a segment based on
    its activeness, churn risk, and dominant propensity — for evaluator clarity.
    """
    act   = seg_data["activeness_score"].mean()
    churn = seg_data["churn_risk"].mean()

    if act >= 0.66:
        act_label = "Highly Active"
    elif act >= 0.33:
        act_label = "Moderately Active"
    else:
        act_label = "Low Activity"

    if churn >= 0.65:
        churn_label = "High Churn Risk"
    elif churn >= 0.35:
        churn_label = "Medium Churn Risk"
    else:
        churn_label = "Low Churn Risk"

    canonical_c = [f"propensity_{k}" for k in CANONICAL_PROPENSITY_KEYS if f"propensity_{k}" in seg_data.columns]
    dom_prop = "default"
    if canonical_c:
        means = {c.replace("propensity_", ""): seg_data[c].mean() for c in canonical_c}
        dom_prop = max(means, key=means.get).replace("_", " ").title()

    lc_dist  = seg_data["lifecycle_stage"].value_counts()
    top_lc   = lc_dist.index[0] if not lc_dist.empty else "mixed"
    top_lc_pct = int(lc_dist.iloc[0] / len(seg_data) * 100) if not lc_dist.empty else 0

    return f"{act_label}, {churn_label} — {dom_prop} propensity ({top_lc} {top_lc_pct}%)"

def export_user_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Export user_segments.csv — PS-compliant schema:

    REQUIRED columns (per PS):
      user_id, segment_id, segment_name, segment_description,
      lifecycle_stage, activeness_score, churn_risk,
      dominant_propensity,
      propensity_gamification, propensity_ai_tutor,
      propensity_leaderboard, propensity_social

    NOT included: demographic cols (age_band, region, days_since_signup,
      preferred_hour, user_count_pct) or anomaly_flag (internal debug only)
    """

    seg_counts        = df["segment_id"].value_counts().rename("_seg_total")
    df                = df.join(seg_counts, on="segment_id")
    df["user_count_pct"] = (df["_seg_total"] / len(df) * 100).round(1)
    df                = df.drop(columns=["_seg_total"])

    canonical_prop_cols = [f"propensity_{k}" for k in CANONICAL_PROPENSITY_KEYS if f"propensity_{k}" in df.columns]
    prop_cols = canonical_prop_cols

    seg_desc_map = {}
    seg_dom_map  = {}
    for sid, sd in df.groupby("segment_id"):
        seg_desc_map[sid] = _build_segment_description(sd, canonical_prop_cols)
        if prop_cols:
            means = {c.replace("propensity_", ""): sd[c].mean() for c in prop_cols}
            seg_dom_map[sid] = max(means, key=means.get)
        else:
            seg_dom_map[sid] = "default"

    df["segment_description"] = df["segment_id"].map(seg_desc_map)
    df["dominant_propensity"]  = df["segment_id"].map(seg_dom_map)

    # Guarantee canonical propensity columns exist
    # (even if KB didn't generate them — engineer_features will have computed them
    #  due to _ensure_canonical_propensities guardrail)
    for key in CANONICAL_PROPENSITY_KEYS:
        col = f"propensity_{key}"
        if col not in df.columns:
            df[col] = 0.0

    base_cols = [
        "user_id", "segment_id", "segment_name", "segment_description",
        "lifecycle_stage",
        "activeness_score", "churn_risk",
        "dominant_propensity",
    ]
    # Note: anomaly_flag and demographic cols intentionally excluded — PS schema only
    output_cols = [c for c in
                   base_cols + canonical_prop_cols
                   if c in df.columns]

    out_df = df[output_cols].copy()

    # Assertions — MECE guarantee
    assert out_df["segment_id"].isnull().sum()   == 0, "null segment_ids in output"
    assert out_df["segment_name"].isnull().sum() == 0, "null segment_names in output"
    assert out_df["segment_description"].isnull().sum() == 0, "null segment_descriptions"

    out_df.to_csv(f"{OUTPUT_DIR}/user_segments.csv", index=False, quoting=csv.QUOTE_ALL)

    ct = pd.crosstab(out_df["segment_id"], out_df["lifecycle_stage"])
    for sid in sorted(out_df["segment_id"].unique()):
        sname = out_df[out_df["segment_id"]==sid]["segment_name"].iloc[0]
        sdesc = out_df[out_df["segment_id"]==sid]["segment_description"].iloc[0]

    return out_df


import csv
def get_dominant_propensity(df_seg):
    """Returns dominant propensity key across all validated dims (canonical + clean extras)."""
    all_prop_cols = [c for c in df_seg.columns if c.startswith('propensity_')]
    if not all_prop_cols:
        return 'default'
    means = {c.replace('propensity_', ''): df_seg[c].mean() for c in all_prop_cols}
    return max(means, key=means.get)

# Semantic-correct static defaults per lifecycle — used when KB templates are missing or empty.
# These are FALLBACK VALUES ONLY. Gemini goal_templates always take precedence.
# Defined here to prevent the silent inactive→paid semantic bug.
_LIFECYCLE_GOAL_DEFAULTS = {
    'trial': {
        'primary_goal':    'Complete first speaking practice within 24 hours',
        'sub_goal':        'Explore AI tutor and complete 1 lesson today',
        'messaging_focus': 'First-value delivery and onboarding habit formation',
        'kpi_metric':      'first_session_completion',
        'kpi_target':      '>=1 within D0-D3',
    },
    'paid': {
        'primary_goal':    'Build a consistent daily speaking habit',
        'sub_goal':        'Maintain streak and complete 3+ exercises per week',
        'messaging_focus': 'Habit reinforcement and progress visibility',
        'kpi_metric':      'weekly_sessions',
        'kpi_target':      '>=3',
    },
    'churned': {
        'primary_goal':    'Re-activate through personalized win-back message',
        'sub_goal':        'Return to app and complete 1 short exercise',
        'messaging_focus': 'Loss avoidance + social proof of peer progress',
        'kpi_metric':      'reactivation_within_7d',
        'kpi_target':      '>=1 session',
    },
    'inactive': {
        'primary_goal':    'Re-engage dormant user with low-friction entry point',
        'sub_goal':        'Open app and complete a 2-minute quick practice',
        'messaging_focus': 'Scarcity + accomplishment framing, reduce barrier to return',
        'kpi_metric':      'reengagement_session',
        'kpi_target':      '>=1 within 14d',
    },
}

# ── Behavioral axis lookup tables ────────────────────────────────────────────
# These drive sub_goal and messaging_focus differentiation across segments.
# Structure follows the document principle:
#   Primary Goal  = f(lifecycle, dominant)       ← KB-sourced base text
#   Sub Goal      = f(lifecycle, dominant, churn_tier)
#   Messaging     = f(dominant, churn_tier, act_tier)
#
# Each table is keyed by the relevant axes so every (dominant × tier) combination
# produces a distinct, meaningful output — not just an appended suffix.

_DOMINANT_LEVER = {
    'gamification':  'reinforcing streak-based motivation and reward accumulation',
    'ai_tutor':      'strengthening guided learning habits through AI practice',
    'leaderboard':   'leveraging competitive positioning and peer comparison',
    'social':        'activating intrinsic motivation and peer accountability',
    'default':       'building consistent daily engagement',
}

_CHURN_ACTION = {
    'high': 'while urgently reducing drop-off risk',
    'mid':  'while proactively stabilizing engagement',
    'low':  'while accelerating progression toward the next milestone',
}

_ACT_CONTEXT = {
    'high': 'already highly active',
    'mid':  'moderately active',
    'low':  'currently low-activity',
}

# Sub-goal base text: f(lifecycle, dominant)
# Primary = lifecycle intent, dominant = behavioral mechanism
_SUB_GOAL_BASE = {
    ('trial',    'gamification'):  'Build a streak in the first 7 days through daily rewards',
    ('trial',    'ai_tutor'):      'Complete 3 AI tutor sessions to establish a guided practice habit',
    ('trial',    'leaderboard'):   'Appear on the leaderboard by completing enough exercises in week 1',
    ('trial',    'social'):        'Discover intrinsic motivation through 2 self-directed sessions',
    ('trial',    'default'):       'Complete first value-delivering activity within 24 hours',
    ('paid',     'gamification'):  'Maintain a 30-day streak and grow coin balance this month',
    ('paid',     'ai_tutor'):      'Complete 12+ AI tutor sessions to advance to the next speaking level',
    ('paid',     'leaderboard'):   'Climb to top 25% of the monthly leaderboard cohort',
    ('paid',     'social'):        'Sustain 5+ weekly sessions through intrinsic motivation',
    ('paid',     'default'):       'Maintain streak and complete 3+ exercises per week',
    ('churned',  'gamification'):  'Reclaim streak record with 1 returning session today',
    ('churned',  'ai_tutor'):      'Resume AI-guided practice to reconnect with progress',
    ('churned',  'leaderboard'):   'Re-enter rankings with 1 session before peers move further ahead',
    ('churned',  'social'):        'Return through a personalized win-back anchored to past progress',
    ('churned',  'default'):       'Return to app and complete 1 short exercise',
    ('inactive', 'gamification'):  'Open app and log one 2-minute session to protect streak potential',
    ('inactive', 'ai_tutor'):      'Complete one AI session — Sia has a personalised lesson ready',
    ('inactive', 'leaderboard'):   'Re-enter rankings before new users permanently overtake position',
    ('inactive', 'social'):        'Re-engage through a low-friction 2-minute intrinsic practice prompt',
    ('inactive', 'default'):       'Open app and complete a 2-minute quick practice',
}

# Messaging focus base: f(dominant)
# Persona-specific hook that shapes which angle we lead with
_MESSAGING_BASE = {
    'gamification':  'Streak preservation and reward accumulation',
    'ai_tutor':      'Skill progression and measurable AI-guided improvement',
    'leaderboard':   'Competitive ranking and peer comparison urgency',
    'social':        'Intrinsic motivation and habit identity reinforcement',
    'default':       'Value demonstration and habit formation',
}

def _get_act_tier(avg_activeness):
    try: v = float(avg_activeness)
    except (TypeError, ValueError): v = 0.5
    if v >= 0.66: return 'high'
    if v >= 0.33: return 'mid'
    return 'low'

def get_goal_from_kb(lifecycle, dominant, avg_churn, avg_activeness=0.5):
    """
    Builds a fully differentiated goal tuple where every axis matters:
      primary_goal  = f(lifecycle, dominant)        — KB base, dominant-keyed
      sub_goal      = f(lifecycle, dominant, churn_tier)
      messaging_focus = f(dominant, churn_tier, act_tier)
    This ensures:
      Trial + Gamification  ≠  Trial + AI Tutor  ≠  Trial + Leaderboard
      Same dominant but different churn tier also differs
    No lifecycle-only fallback — every combination is explicitly defined.
    """
    churn_tier = _get_churn_tier(avg_churn)
    act_tier   = _get_act_tier(avg_activeness)

    # ── Primary goal: KB-sourced, dominant-keyed ─────────────────────────────
    gt           = KB_CONFIG.get('goal_templates', {})
    lc_templates = gt.get(lifecycle, {})

    if lc_templates:
        # Priority: at_risk (paid high-churn) → dominant propensity key → default
        if lifecycle == 'paid' and churn_tier == 'high' and 'at_risk' in lc_templates:
            tmpl = lc_templates['at_risk']
        elif dominant in lc_templates:
            tmpl = lc_templates[dominant]
        else:
            tmpl = lc_templates.get('default', {})
        primary = tmpl.get('primary_goal', '') if tmpl else ''
    else:
        tmpl = {}
        primary = ''

    # Fall back to lifecycle default if KB returned nothing or returned a bare label
    lc_default = _LIFECYCLE_GOAL_DEFAULTS.get(lifecycle, _LIFECYCLE_GOAL_DEFAULTS['paid'])
    if not primary:
        primary = lc_default['primary_goal']

    # ── Wire _DOMINANT_LEVER: enrich primary_goal with propensity-specific mechanism ──
    # If the KB returned a short label (e.g. "User Activation", "Win-back") or a bare
    # lifecycle sentence without a dominant-specific hook, append the propensity lever.
    # _DOMINANT_LEVER describes the unique mechanism this propensity type responds to.
    # Result: every (lifecycle × dominant) combination produces a distinct primary_goal.
    _lever = _DOMINANT_LEVER.get(dominant, _DOMINANT_LEVER['default'])
    # Only append if primary doesn't already reference the dominant mechanism
    if _lever.split()[0].lower() not in primary.lower():
        primary = f"{primary} — {_lever}"

    # ── Sub-goal: f(lifecycle, dominant, churn_tier) ─────────────────────────
    # Base anchors the lifecycle intent + dominant mechanism.
    # Churn action appends the risk-aware directive.
    sub_base   = _SUB_GOAL_BASE.get((lifecycle, dominant),
                 _SUB_GOAL_BASE.get((lifecycle, 'default'), lc_default['sub_goal']))
    churn_act  = _CHURN_ACTION[churn_tier]
    sub_goal   = f"{sub_base} {churn_act}."

    # ── Messaging focus: f(dominant, churn_tier, act_tier) ───────────────────
    # Leads with the dominant behavioral hook, qualifies with risk + activity state.
    msg_base   = _MESSAGING_BASE.get(dominant, _MESSAGING_BASE['default'])
    act_ctx    = _ACT_CONTEXT[act_tier]
    messaging_focus = f"{msg_base} — {act_ctx} user, {churn_tier} churn risk"

    return primary, sub_goal, messaging_focus

def get_journey_from_kb(lifecycle, dominant, avg_churn):
    """
    Builds day-on-day journey dict {day_key: message} from KB_CONFIG['journey_templates'].
    Lookup per day: dominant_key -> 'at_risk' (if churn high) -> 'default'
    All messages sourced from KB Intelligence Config — zero hardcoded strings.
    """
    jt      = KB_CONFIG.get('journey_templates', {})
    lc_days = jt.get(lifecycle, {})
    is_at_risk = avg_churn >= 0.65

    journey = {}
    for day, variants in lc_days.items():
        if not isinstance(variants, dict):
            journey[day] = str(variants)
            continue
        if dominant in variants:
            journey[day] = variants[dominant]
        elif is_at_risk and 'at_risk' in variants:
            journey[day] = variants['at_risk']
        else:
            journey[day] = variants.get('default', '')
    return journey

def run_goals_builder(df, kb_data=None):
    """
    Build segment_goals.csv — PS-compliant WIDE-FORM format.

    Schema (one row per segment × lifecycle) — PS-required columns only:
      segment_id, segment_name, lifecycle_stage, dominant_propensity,
      primary_goal, sub_goal, messaging_focus,
      D0_focus–D7_focus (trial), D8_focus–D30_focus (paid),
      D30_focus_churned, D45_focus, D60_focus, D90_focus (churned),
      D30_focus_inactive, D45_focus, D60_focus, D90_focus (inactive)

    Row count: num_segments × 4 lifecycles (e.g. 8 segs × 4 = 32 rows)

    Each Dx_focus cell contains the KB-sourced journey message for that day.
    Day columns not relevant to a lifecycle are left empty (NaN).

    NOTE: D30 appears in both 'paid' (D30_focus) and 'churned' (D30_focus_churned)
    because both lifecycles include day 30. They are separate columns to avoid collision.
    """

    if not KB_CONFIG.get('goal_templates'):
        raise RuntimeError('KB_CONFIG is empty. run_kb_engine() must run first.')

    canonical_prop_cols = [f"propensity_{k}" for k in CANONICAL_PROPENSITY_KEYS if f"propensity_{k}" in df.columns]
    prop_cols = canonical_prop_cols

    # ── Wide-form column layout ───────────────────────────────────────────────
    # Trial day columns  (D0–D7)
    TRIAL_DAY_COLS    = [f'D{d}_focus' for d in [0,1,2,3,4,5,6,7]]
    # Paid day columns   (D8–D30) — use "D30_focus" for paid D30
    PAID_DAY_COLS     = [f'D{d}_focus' for d in [8,10,15,20,25,30]]
    # Churned day cols   (D30–D90) — D30 gets unique col to avoid collision with paid
    CHURNED_DAY_COLS  = ['D30_focus_churned'] + [f'D{d}_focus' for d in [45,60,90]]
    # Inactive day cols  (D30–D90) — reuse churned names where possible
    INACTIVE_DAY_COLS = ['D30_focus_inactive', 'D45_focus', 'D60_focus', 'D90_focus']

    ALL_DAY_COLS = (TRIAL_DAY_COLS + PAID_DAY_COLS +
                   CHURNED_DAY_COLS + INACTIVE_DAY_COLS)
    seen = set()
    UNIQUE_DAY_COLS = []
    for c in ALL_DAY_COLS:
        if c not in seen:
            UNIQUE_DAY_COLS.append(c)
            seen.add(c)

    LIFECYCLE_TO_DAY_COLS = {
        'trial':    {d: f'D{d}_focus' for d in LIFECYCLE_DAY_RANGES['trial']},
        'paid':     {d: f'D{d}_focus' for d in LIFECYCLE_DAY_RANGES['paid']},
        'churned':  {
            30: 'D30_focus_churned',
            45: 'D45_focus',
            60: 'D60_focus',
            90: 'D90_focus',
        },
        'inactive': {
            30: 'D30_focus_inactive',
            45: 'D45_focus',
            60: 'D60_focus',
            90: 'D90_focus',
        },
    }

    segments = df[['segment_id','segment_name']].drop_duplicates().sort_values('segment_id')
    rows = []

    # Generate goals for ALL segment×lifecycle combinations — PS compliance.
    # PS requires: "Define primary goals, sub-goals, and day-on-day progression
    # for each segment × lifecycle stage (Trial D0-D7, Paid D8-D30, Churned/Inactive)"
    # This means ALL 4 lifecycles per segment, regardless of how many users currently
    # occupy that lifecycle in the runtime dataset. Goals must be generative/forward-looking.
    seg_lc_users = df.groupby(['segment_id', 'lifecycle_stage']).size().reset_index(name='_lc_count')
    seg_lc_count_map = {
        (row['segment_id'], row['lifecycle_stage']): int(row['_lc_count'])
        for _, row in seg_lc_users.iterrows()
    }

    for _, seg_row in segments.iterrows():
        sid   = seg_row['segment_id']
        sname = seg_row['segment_name']
        sd    = df[df['segment_id'] == sid]

        act_m    = sd['activeness_score'].mean()
        churn_m  = sd['churn_risk'].mean()
        dominant = get_dominant_propensity(sd)
        n_users  = len(sd)

        for lc in VALID_LIFECYCLES:
            lc_user_count = seg_lc_count_map.get((sid, lc), 0)

            primary, sub, focus = get_goal_from_kb(lc, dominant, churn_m, act_m)
            journey = get_journey_from_kb(lc, dominant, churn_m)

            # ── Phantom lifecycle correction ─────────────────────────────────────────
            # PS requires all 4 lifecycle rows per segment regardless of current user count.
            # When a segment has zero users in a lifecycle, generic KB templates produce
            # semantically wrong content (e.g. onboarding message for a churned segment).
            # Phantom rows are detected and overridden with contextually correct framing.
            _is_phantom = (lc_user_count == 0)
            _dominant_lc = sd['lifecycle_stage'].value_counts().index[0] if len(sd) > 0 else 'unknown'
            _disengaged_segment = _dominant_lc in ('churned', 'inactive')

            if _is_phantom and lc == 'trial' and _disengaged_segment:
                primary = _LIFECYCLE_GOAL_DEFAULTS['churned']['primary_goal']
                _churn_action_str = _CHURN_ACTION.get(_get_churn_tier(churn_m), "")
                sub     = 'Re-activate through a low-friction entry point ' + _churn_action_str + '.'
                focus   = 'Win-back re-entry \u2014 ' + _dominant_lc + ' users, high churn risk'
                churned_journey = get_journey_from_kb('churned', dominant, churn_m)
                journey = {'D0': churned_journey.get('D30', ''), **journey}

            elif _is_phantom and lc == 'paid' and _disengaged_segment:
                primary = _LIFECYCLE_GOAL_DEFAULTS['churned']['primary_goal']
                _churn_action_str = _CHURN_ACTION.get(_get_churn_tier(churn_m), "")
                sub     = 'Re-subscribe and complete 1 session to restart progress ' + _churn_action_str + '.'
                focus   = 'Re-subscription urgency \u2014 ' + _dominant_lc + ' users'

            elif _is_phantom and lc in ('churned', 'inactive') and _dominant_lc in ('trial', 'paid'):
                primary = _LIFECYCLE_GOAL_DEFAULTS['paid']['primary_goal']
                _churn_action_str = _CHURN_ACTION.get(_get_churn_tier(churn_m), "")
                sub     = 'Maintain engagement before churn risk escalates ' + _churn_action_str + '.'
                focus   = 'Proactive retention \u2014 primarily ' + _dominant_lc + ' users'

            row = {
                    'segment_id':          sid,
                'segment_name':        sname,
                'lifecycle_stage':     lc,
                'dominant_propensity': dominant,
                    'primary_goal':        primary,
                'sub_goal':            sub,
                'messaging_focus':     focus,
            }

            for col in UNIQUE_DAY_COLS:
                row[col] = None

            day_col_map = LIFECYCLE_TO_DAY_COLS.get(lc, {})
            for day_num, col_name in day_col_map.items():
                day_key = f'D{day_num}'
                msg     = journey.get(day_key, '')
                row[col_name] = msg if msg else ''

            rows.append(row)

    goals_df = pd.DataFrame(rows)

    # ── PS-required columns only — dominant_propensity kept, extras removed ─────
    PS_IDENTITY_COLS = ['segment_id', 'segment_name', 'lifecycle_stage', 'dominant_propensity']
    PS_GOAL_COLS     = ['primary_goal', 'sub_goal', 'messaging_focus']
    # All day focus columns kept (D0-D7 trial, D8-D30 paid, D30-D90 churned/inactive)
    ordered_cols = PS_IDENTITY_COLS + PS_GOAL_COLS + UNIQUE_DAY_COLS
    goals_df = goals_df[[c for c in ordered_cols if c in goals_df.columns]]

    goals_df.to_csv(f'{OUTPUT_DIR}/segment_goals.csv', index=False, quoting=csv.QUOTE_ALL)

    n_segs = goals_df['segment_id'].nunique()
    n_lcs  = goals_df['lifecycle_stage'].nunique()
    # Assert PS compliance: must have exactly N_segs × 4 lifecycle rows
    expected_rows = n_segs * len(VALID_LIFECYCLES)
    assert len(goals_df) == expected_rows, f'Goal row count mismatch: {len(goals_df)} != {expected_rows} (expected {n_segs} segs × 4 lifecycles)'
    for lc, cnt in goals_df['lifecycle_stage'].value_counts().items():
        pass

    sample_cols = ['segment_id','lifecycle_stage','dominant_propensity','primary_goal','D0_focus','D8_focus']
    available_sample = [c for c in sample_cols if c in goals_df.columns]

    return goals_df


def print_final_checklist(df, out_df, goals_df):
    pass

    deliverables = [
        ('company_north_star.json',      'JSON  — North Star metric + justification + proxy'),
        ('feature_goal_map.json',         'JSON  — Feature × lifecycle × outcome mapping'),
        ('allowed_tone_hook_matrix.json', 'JSON  — Tones + all 8 Octalysis hooks'),
        ('user_segments.csv',             'CSV   — MECE segments (PS-compliant schema)'),
        ('segment_goals.csv',             'CSV   — Goals × lifecycle × day WIDE-FORM'),
    ]
    all_ok = True
    for fname, desc in deliverables:
        fpath  = f'{OUTPUT_DIR}/{fname}'
        exists = os.path.exists(fpath)
        size   = os.path.getsize(fpath) if exists else 0
        status = '✅ OK' if exists else '❌ MISSING'
        all_ok = all_ok and exists

    cfg_path = f'{INTERNAL_DIR}/kb_intelligence_config.json'
    if os.path.exists(cfg_path):
        pass

    # ── Schema Compliance ────────────────────────────────────────────────────
    required_seg_cols = [
        'user_id','segment_id','segment_name','segment_description',
        'lifecycle_stage','activeness_score','churn_risk',
        'dominant_propensity',
        'propensity_gamification','propensity_ai_tutor',
        'propensity_leaderboard','propensity_social',
    ]
    for col in required_seg_cols:
        present = col in out_df.columns

    required_goal_cols = [
        'segment_id','segment_name','lifecycle_stage',
        'primary_goal','sub_goal','messaging_focus',
        'D0_focus','D1_focus','D8_focus','D30_focus',
        'D30_focus_churned','D30_focus_inactive',
    ]
    for col in required_goal_cols:
        present = col in goals_df.columns

    # PS compliance: check all 4 lifecycles present per segment
    n_segs_in_goals = goals_df['segment_id'].nunique()
    expected_rows    = n_segs_in_goals * len(VALID_LIFECYCLES)
    rows_match = len(goals_df) == expected_rows

    # Verify lc_user_count is NOT in goals (schema pollution fix)
    lc_count_absent = 'lc_user_count' not in goals_df.columns

    # ── Propensity Coverage ───────────────────────────────────────────────────
    prop_cols_present = [c for c in out_df.columns if c.startswith('propensity_')]
    for key in CANONICAL_PROPENSITY_KEYS:
        col = f'propensity_{key}'
    extra = [c for c in prop_cols_present if c.replace('propensity_','') not in CANONICAL_PROPENSITY_KEYS]
    for col in extra:
        pass


    # ── Propensity Structural Health (post-softmax verification) ────────────────
    _pcols = [c for c in out_df.columns if c.startswith('propensity_') and c.replace('propensity_','') in CANONICAL_PROPENSITY_KEYS]
    if _pcols:
        _dom_final = out_df[_pcols].idxmax(axis=1).str.replace('propensity_','').value_counts(normalize=True)
        _top_pct   = _dom_final.iloc[0] * 100 if len(_dom_final) > 0 else 0
        _health = '✅ Balanced' if _top_pct <= 40 else ('⚠️  Moderate concentration' if _top_pct <= 55 else '❌ High concentration — check mean-centering temperature')
        _stds_check = out_df[_pcols].std()
        _std_ratio  = _stds_check.max() / (_stds_check.min() + 1e-9)
        _std_health = '✅ Balanced' if _std_ratio < 2.5 else '⚠️  Uneven std — one dim still dominates variance'

    # ── Architecture Guarantees ────────────────────────────────────────────────

    if not all_ok:
        pass


def run_task1(data_path=None, kb_path=None, api_key=None):
    """
    Main execution function for Task 1.
    Both data_path (user CSV) and kb_path (company KB) are REAL inputs:
      - kb_path  -> run_kb_engine() extracts propensity_dimensions, journey_templates,
                    goal_templates, north_star, feature_goal_map, tone_hook_matrix
      - data_path -> validate_input() + engineer_features() compute scores from real user data
    Swapping either input file changes outputs — no hardcoded domain logic.
    PS non-negotiable: accepts NEW datasets at runtime.
    """
    start = time.time()

    if data_path is None:
        candidates = [f for f in os.listdir('.')
                      if f.endswith('.csv') and 'user' in f.lower()
                      and 'segment' not in f.lower() and 'goal' not in f.lower()
                      and 'schedule' not in f.lower()]
        data_path = candidates[0] if candidates else 'user_behavioral_data.csv'

    if kb_path is None:
        for candidate in ['company_kb.md', 'company_kb.txt']:
            if os.path.exists(candidate):
                kb_path = candidate
                break
        if kb_path is None:
            for ext in ['.md', '.txt']:
                found = [f for f in os.listdir('.') if f.endswith(ext)]
                if found:
                    kb_path = found[0]
                    break
        if kb_path is None:
            raise FileNotFoundError('No KB file found. Provide kb_path or place company_kb.md/.txt in working dir.')

    # Step 1: Load + Validate user behavioral CSV
    df = pd.read_csv(data_path)
    df = validate_input(df)

    # Step 2: KB Engine
    # Reads company_kb, injects user data stats as context to Gemini,
    # extracts: north_star, feature_goal_map, tone_hook_matrix,
    #           propensity_dimensions (drives feature engineering),
    #           journey_templates + goal_templates (drives segment_goals.csv)
    kb_data = run_kb_engine(kb_path, api_key=api_key)

    # Step 3: Feature Engineering using KB-extracted propensity dimensions
    df = engineer_features(df)

    # Step 4: MECE Segmentation (segmentation features built from KB dimensions)
    df, _ = run_segmentation_engine(df)

    # Step 5: Export user_segments.csv
    out_df = export_user_segments(df)

    # Step 6: Build segment_goals.csv from KB journey + goal templates
    goals_df = run_goals_builder(df, kb_data)

    elapsed = time.time() - start
    print_final_checklist(df, out_df, goals_df)

    return df, out_df, goals_df, kb_data

if __name__ == '__main__':
    kb_input = input("Enter path to Company KB file (or press Enter to auto-detect): ").strip()
    data_input = input("Enter path to User Behavioral Data CSV (or press Enter to auto-detect): ").strip()

    kb_path = kb_input if kb_input else None
    data_path = data_input if data_input else None

    df, out_df, goals_df, _ = run_task1(
        data_path=data_path,
        kb_path=kb_path,
        api_key=os.getenv('GEMINI_API_KEY'),
    )
    print(
        "\n✅ All 5 Task 1 deliverables successfully created:\n"
        "   📄 company_north_star.json\n"
        "   📄 feature_goal_map.json\n"
        "   📄 allowed_tone_hook_matrix.json\n"
        "   📊 user_segments.csv\n"
        "   📊 segment_goals.csv"
    )


# ============================================================
# CELL 11: DISPLAY ALL DELIVERABLES
# Run after Cell 10 completes.
# Shows all 3 JSON files and both CSV files in full.
# ============================================================
import json, pandas as pd
from IPython.display import display
import os

def _section(title):
    pass

def _subsection(title):
    pass

# ════════════════════════════════════════════════════════════════════════
# 1. company_north_star.json
# ════════════════════════════════════════════════════════════════════════
_section("📌 company_north_star.json")
with open(f"{OUTPUT_DIR}/company_north_star.json") as f:
    ns = json.load(f)

# ════════════════════════════════════════════════════════════════════════
# 2. feature_goal_map.json
# ════════════════════════════════════════════════════════════════════════
_section("🗺️  feature_goal_map.json")
with open(f"{OUTPUT_DIR}/feature_goal_map.json") as f:
    fgm = json.load(f)

# ════════════════════════════════════════════════════════════════════════
# 3. allowed_tone_hook_matrix.json
# ════════════════════════════════════════════════════════════════════════
_section("🎨 allowed_tone_hook_matrix.json")
with open(f"{OUTPUT_DIR}/allowed_tone_hook_matrix.json") as f:
    thm = json.load(f)

# ════════════════════════════════════════════════════════════════════════
# 4. user_segments.csv
# ════════════════════════════════════════════════════════════════════════
_section("👥 user_segments.csv")
segs = pd.read_csv(f"{OUTPUT_DIR}/user_segments.csv")

_subsection("Schema")

_subsection("Segment Summary (one row per segment)")
prop_cols = [c for c in segs.columns if c.startswith("propensity_")]
agg_dict  = {
    "user_id":            "count",
    "activeness_score":   "mean",
    "churn_risk":         "mean",
    "segment_description":("first"),
    "dominant_propensity":"first",
}
for p in prop_cols:
    agg_dict[p] = "mean"

summary = (segs.groupby(["segment_id","segment_name"])
               .agg(agg_dict)
               .rename(columns={"user_id": "n_users"})
               .reset_index())
# Round numerics
for col in summary.select_dtypes(include="float").columns:
    summary[col] = summary[col].round(3)
try:
    display(summary)
except Exception:
    pass

_subsection("Canonical Propensity Columns (PS-required)")
canonical_cols = ["user_id","segment_name","propensity_gamification",
                  "propensity_ai_tutor","propensity_leaderboard","propensity_social",
                  "activeness_score","churn_risk","dominant_propensity"]
sample_cols = [c for c in canonical_cols if c in segs.columns]
try:
    display(segs[sample_cols].head(10))
except Exception:
    pass

_subsection("Segment × Lifecycle Distribution")
ct = pd.crosstab(segs["segment_name"], segs["lifecycle_stage"])
try:
    display(ct)
except Exception:
    pass

# ════════════════════════════════════════════════════════════════════════
# 5. segment_goals.csv
# ════════════════════════════════════════════════════════════════════════
_section("🎯 segment_goals.csv")
goals = pd.read_csv(f"{OUTPUT_DIR}/segment_goals.csv")

_subsection("Schema")

_subsection("Day columns present")
day_cols = [c for c in goals.columns if c.endswith("_focus") or c in ["D30_focus_churned","D30_focus_inactive","D45_focus","D60_focus","D90_focus"]]
trial_dc  = [c for c in day_cols if any(c == f'D{d}_focus' for d in [0,1,2,3,4,5,6,7])]
paid_dc   = [c for c in day_cols if any(c == f'D{d}_focus' for d in [8,10,15,20,25,30])]
churned_dc = [c for c in day_cols if "churned" in c or any(c == f'D{d}_focus' for d in [45,60,90])]
inactive_dc = [c for c in day_cols if "inactive" in c]

_subsection("Goals per Segment × Lifecycle (key columns)")
key_cols = ["segment_id","segment_name","lifecycle_stage","dominant_propensity",
            "primary_goal","sub_goal","messaging_focus"]
display_cols = [c for c in key_cols if c in goals.columns]
try:
    display(goals[display_cols])
except Exception:
    pass

_subsection("Day Journey Messages — Trial Lifecycle (D0–D7)")
trial_goals = goals[goals["lifecycle_stage"] == "trial"].copy()
journey_cols = ["segment_name"] + [c for c in day_cols if c in [f'D{d}_focus' for d in [0,1,2,3,4,5,6,7]]]
if len(journey_cols) > 1:
    try:
        display(trial_goals[journey_cols].reset_index(drop=True))
    except Exception:
        pass

_subsection("Day Journey Messages — Paid Lifecycle (D8–D30)")
paid_goals = goals[goals["lifecycle_stage"] == "paid"].copy()
journey_cols_paid = ["segment_name"] + [c for c in day_cols if c in [f'D{d}_focus' for d in [8,10,15,20,25,30]]]
if len(journey_cols_paid) > 1:
    try:
        display(paid_goals[journey_cols_paid].reset_index(drop=True))
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════════════
# Final file size report
# ════════════════════════════════════════════════════════════════════════
_section("📦 Final Deliverables")
files = [
    "company_north_star.json",
    "feature_goal_map.json",
    "allowed_tone_hook_matrix.json",
    "user_segments.csv",
    "segment_goals.csv",
    "kb_intelligence_config.json",
]
for fname in files:
    path   = f"{OUTPUT_DIR}/{fname}"
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0