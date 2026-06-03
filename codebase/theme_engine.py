"""
Theme Engine — Project Aurora
==============================
Reads segment_goals.csv from iteration_0_before_learning/
Uses Gemini to assign top-3 Octalysis drives per segment.
Outputs communication_themes.csv to iteration_0_before_learning/
"""

import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
ITER0_DIR = "iteration_0_before_learning"

SEGMENT_GOALS_CSV = os.path.join(ITER0_DIR, "segment_goals.csv")
OUTPUT_CSV = os.path.join(ITER0_DIR, "communication_themes.csv")


def run_theme_engine():
    genai.configure(api_key=API_KEY)

    segment_goals = pd.read_csv(SEGMENT_GOALS_CSV)
    print(f"[Theme Engine] Loaded segment_goals: {segment_goals.shape}")

    # Build per-segment profiles to send into Gemini for reasoning analysis.
    segment_profiles = []
    for segment_id, group in segment_goals.groupby('segment_id'):
        profile = {
            "segment_id": int(segment_id) if not isinstance(segment_id, str) else segment_id,
            "segment_name": group['segment_name'].iloc[0],
            "segment_data": group.to_dict(orient='records')
        }
        segment_profiles.append(profile)

    # Prompt Gemini to choose top drives using full lifecycle context.
    prompt = f"""
You are an expert in behavioral design, EdTech retention, and the Octalysis Framework.
I have a dataset of user segments. Each segment profile contains all temporal data (D0 to D90 focus), propensity scores, and a specific goal hierarchy.

Analyze EACH segment profile provided in the JSON array below. For each unique 'segment_id', synthesize all the provided data and select exactly the 3 most effective Octalysis Core Drives from the allowed list to motivate them globally across their lifecycle.

Allowed Octalysis Core Drives:
- Epic Meaning
- Accomplishment
- Empowerment
- Ownership
- Social Influence
- Scarcity
- Unpredictability
- Loss Avoidance

When making your decision, heavily weigh:
1. The specific propensity scores and dominant propensity for each segment.
2. The hierarchical relationship between the 'primary_goal' (which shifts based on their lifecycle period) and their 'sub_goal' (which is unique to their behavioral segment).
3. The progression of their messaging focuses from D0 through D90.

Here are the segment profiles:
{json.dumps(segment_profiles, indent=2)}

Return a JSON array of objects with this strict structure. Return exactly ONE object per unique 'segment_id':
[
  {{
    "segment_id": "string or int",
    "top_3_drives": ["drive1", "drive2", "drive3"],
    "reasoning": "A 1-sentence explanation synthesizing why these 3 drives perfectly align this segment's specific subgoals with the broader lifecycle primary goals."
  }}
]
"""

    model = genai.GenerativeModel('gemini-2.5-flash')

    print("[Theme Engine] Sending segment profiles to Gemini API...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )

    # Parse JSON response and split top drives into columns cleanly.
    mapped_drives = json.loads(response.text)
    drives_df = pd.DataFrame(mapped_drives)
    print(f"[Theme Engine] Received {len(drives_df)} unique segment mappings from the API.")

    drives_df[['drive_1', 'drive_2', 'drive_3']] = pd.DataFrame(
        drives_df['top_3_drives'].tolist(), index=drives_df.index
    )
    drives_df = drives_df.drop(columns=['top_3_drives'])

    # Ensure segment_id types match
    drives_df['segment_id'] = drives_df['segment_id'].astype(str)

    # Persist deduplicated segment themes for downstream template and scheduling steps.
    output_df = drives_df[['segment_id', 'drive_1', 'drive_2', 'drive_3', 'reasoning']].drop_duplicates(subset=['segment_id'])
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Theme Engine] Saved communication_themes.csv → {OUTPUT_CSV} ({len(output_df)} segments)")

    return output_df


if __name__ == "__main__":
    run_theme_engine()
