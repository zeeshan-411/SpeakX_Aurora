import os
import re
import pandas as pd
import json
import anthropic
import random
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _find_behavioral_csv():
    candidates = [f for f in os.listdir('.') if f.endswith('.csv')
                  and 'user' in f.lower() and 'segment' not in f.lower()
                  and 'goal' not in f.lower() and 'schedule' not in f.lower()
                  and 'template' not in f.lower() and 'theme' not in f.lower()
                  and 'timing' not in f.lower() and 'experiment' not in f.lower()
                  and 'delta' not in f.lower() and 'notification' not in f.lower()
                  and 'rl_' not in f.lower() and 'learning' not in f.lower()]
    return candidates[0] if candidates else "user_behavioral_data.csv"


def _repair_truncated_json(raw):
    """
    Repair JSON that was truncated mid-stream by the LLM hitting max_tokens.

    Strategy:
      1. Strip markdown fences and whitespace.
      2. Try parsing as-is — if it works, return immediately.
      3. Otherwise, progressively strip characters from the end and try to close
         any open braces/brackets so the outer object is valid.
      4. As a last resort, regex-extract all "key": {drive: score} pairs
         and reconstruct the object manually.
    """
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # Attempt 1: parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip trailing garbage and close open braces
    # Find the last complete key-value pair (ends with a number before truncation)
    # Then close all open braces/brackets.
    text = raw
    # Remove any trailing partial string (unterminated quote)
    # Walk backwards to find last complete value
    last_good = len(text)
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth >= 0:
                last_good = i + 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
            if bracket_depth >= 0:
                last_good = i + 1

    # Try trimming to last_good and closing
    trimmed = text[:last_good]
    # Remove any trailing comma
    trimmed = re.sub(r',\s*$', '', trimmed)
    # Close any open structures
    closers = ''
    open_braces = trimmed.count('{') - trimmed.count('}')
    open_brackets = trimmed.count('[') - trimmed.count(']')
    closers = '}' * max(open_braces, 0) + ']' * max(open_brackets, 0)
    attempt = trimmed + closers

    try:
        return json.loads(attempt)
    except json.JSONDecodeError:
        pass

    # Attempt 3: regex extraction — find all "key": { ... } blocks
    # Pattern: "segment|lifecycle": { "Drive": score, ... }
    pattern = r'"([^"]+)"\s*:\s*\{([^}]*)\}'
    matches = re.findall(pattern, raw)
    if matches:
        result = {}
        for key, inner in matches:
            drives = {}
            # Parse "DriveName": 0.8 pairs
            drive_pattern = r'"([^"]+)"\s*:\s*([\d.]+)'
            for drive_name, score_str in re.findall(drive_pattern, inner):
                try:
                    drives[drive_name] = float(score_str)
                except ValueError:
                    drives[drive_name] = 0.5
            if drives:
                result[key] = drives
        if result:
            return result

    raise json.JSONDecodeError(
        f"Could not repair truncated JSON. Raw starts with: {raw[:200]}...",
        raw, 0
    )


def engagement_state(activeness):
    if activeness > 0.7:
        return "high"
    elif activeness > 0.4:
        return "medium"
    else:
        return "low"


def activeness_multiplier(state):
    if state == "low":
        return {
            "Loss Avoidance":1.5,
            "Scarcity":1.4,
            "Social Influence":1.2,
            "Empowerment":1.1,
            "Ownership":1.0,
            "Epic Meaning":1.0,
            "Unpredictability":1.0,
            "Accomplishment":0.7
        }
    elif state == "high":
        return {
            "Accomplishment":1.5,
            "Ownership":1.3,
            "Empowerment":1.2,
            "Social Influence":1.1,
            "Epic Meaning":1.1,
            "Unpredictability":1.0,
            "Scarcity":0.8,
            "Loss Avoidance":0.7
        }
    return {
        "Loss Avoidance":1.0,
        "Scarcity":1.0,
        "Social Influence":1.0,
        "Empowerment":1.0,
        "Ownership":1.0,
        "Epic Meaning":1.0,
        "Unpredictability":1.0,
        "Accomplishment":1.0
    }


whatsapp_sent = {}

zone_channel_map = {
    1:"push",
    2:"push",
    3:"in_app",
    4:"in_app",
    5:"push",
    6:"push"
}

def get_channel(user_id, zone, lifecycle):
    zone = int(zone)
    if lifecycle == "churned":
        if not whatsapp_sent.get(user_id, False):
            whatsapp_sent[user_id] = True
            return "whatsapp"
        if zone in [1,2,3,4]:
            return "push"
        return "sms"
    if lifecycle == "inactive":
        if not whatsapp_sent.get(user_id, False):
            whatsapp_sent[user_id] = True
            return "whatsapp"
        if zone in [3,4,5]:
            return "sms"
        return "push"
    if lifecycle == "trial":
        if zone == 5 and not whatsapp_sent.get(user_id, False):
            whatsapp_sent[user_id] = True
            return "whatsapp"
        if zone in [1,3]:
            return "push"
        if zone in [2,4]:
            return "in_app"
        return "push"
    return zone_channel_map.get(zone,"push")


segments = pd.read_csv("iteration_0_before_learning/user_segments.csv")
goal = pd.read_csv("iteration_0_before_learning/segment_goals.csv")
timings = pd.read_csv("iteration_0_before_learning/timing_recommendations.csv")
themes = pd.read_csv("iteration_0_before_learning/communication_themes.csv")
templates = pd.read_csv("iteration_0_before_learning/message_templates.csv")
user_behavior = pd.read_csv(_find_behavioral_csv())


df = segments[
    ["user_id","segment_id","lifecycle_stage","activeness_score"]
].copy()

df = df.merge(
    themes[["segment_id","drive_1","drive_2","drive_3"]],
    on="segment_id"
)

df = df.merge(
    goal[["segment_id","lifecycle_stage","primary_goal","sub_goal"]],
    on=["segment_id","lifecycle_stage"]
)


tasks = []

for (segment,lifecycle), group in df.groupby(["segment_id","lifecycle_stage"]):
    sample = group.iloc[0]
    tasks.append({
        "segment":segment,
        "lifecycle":lifecycle,
        "text":f"""
GOAL:
{sample['primary_goal']}

SUBGOAL:
{sample['sub_goal']}
""",
        "drives":[
            sample["drive_1"],
            sample["drive_2"],
            sample["drive_3"]
        ]
    })


# Ask LLM to score drives for each segment and lifecycle.
prompt = """
Score Octalysis drive effectiveness for each task.

Return ONLY JSON.

FORMAT:

{
"segment|lifecycle": {
"DriveName": score
}
}

Scores must be between 0 and 1.

TASKS:
"""

for t in tasks:
    prompt += f"""

SEGMENT: {t['segment']}
LIFECYCLE: {t['lifecycle']}

TEXT:
{t['text']}

DRIVES:
{t['drives'][0]}
{t['drives'][1]}
{t['drives'][2]}
"""


response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=8192,
    temperature=0,
    messages=[{"role":"user","content":[{"type":"text","text":prompt}]}]
)

raw = response.content[0].text.strip()
raw = raw.replace("```json","").replace("```","")
scores = _repair_truncated_json(raw)


for i in range(1,4):
    df[f"score_final_drive_{i}"] = 0.0

# Distribute templates across top zones using drive scores and counts.
for index,row in df.iterrows():
    key = f"{row['segment_id']}|{row['lifecycle_stage']}"
    segment_scores = scores.get(key,{})
    drives=[row["drive_1"],row["drive_2"],row["drive_3"]]
    state = engagement_state(row["activeness_score"])
    multiplier = activeness_multiplier(state)
    values=[]
    for drive in drives:
        val = segment_scores.get(drive,0)*multiplier.get(drive,1)
        values.append(val)
    total=sum(values) or 1
    for i in range(3):
        df.at[index,f"score_final_drive_{i+1}"]=values[i]/total


def get_notification_count(a):
    if a<0.2: return 3
    elif a<0.4: return 4
    elif a<0.55: return 5
    elif a<=0.7: return 6
    elif a<=0.8: return 7
    elif a<=0.9: return 8
    return 9

df["notification_count"]=df["activeness_score"].apply(get_notification_count)


for index,row in df.iterrows():
    notif=int(row["notification_count"])
    scores_list=[
        row["score_final_drive_1"]*notif,
        row["score_final_drive_2"]*notif,
        row["score_final_drive_3"]*notif
    ]
    floors=[int(x) for x in scores_list]
    remainder=notif-sum(floors)
    fractions=[scores_list[i]-floors[i] for i in range(3)]
    order=sorted(range(3), key=lambda i:fractions[i], reverse=True)
    for i in range(remainder):
        floors[order[i]]+=1
    df.at[index,"score_final_drive_1"]=floors[0]
    df.at[index,"score_final_drive_2"]=floors[1]
    df.at[index,"score_final_drive_3"]=floors[2]


df=df.merge(
    timings[["user_id","top_zone_1","top_zone_2","top_zone_3"]],
    on="user_id",
    how="left"
)


template_lookup={}

for _,row in templates.iterrows():
    key=(row["segment_id"],row["lifecycle_stage"],row["theme_used"])
    template_lookup.setdefault(key,[]).append(str(row["template_id"]))

df["template_drive_1"]=""
df["template_drive_2"]=""
df["template_drive_3"]=""

for index,row in df.iterrows():
    drives=[row["drive_1"],row["drive_2"],row["drive_3"]]
    for i,drive in enumerate(drives,1):
        key=(row["segment_id"],row["lifecycle_stage"],drive)
        df.at[index,f"template_drive_{i}"]=",".join(
            template_lookup.get(key,[])
        )


def collect_templates(row):
    selected=[]
    drives=[
        ("template_drive_1",int(row["score_final_drive_1"])),
        ("template_drive_2",int(row["score_final_drive_2"])),
        ("template_drive_3",int(row["score_final_drive_3"]))
    ]
    for col,count in drives:
        if count<=0:
            continue
        template_list=[t for t in str(row[col]).split(",") if t]
        if not template_list:
            continue
        take=min(count,len(template_list))
        selected.extend(random.sample(template_list,take))
    random.shuffle(selected)
    return selected


df["templates_top_zone_1"]=""
df["templates_top_zone_2"]=""
df["templates_top_zone_3"]=""

for index,row in df.iterrows():
    all_templates=collect_templates(row)
    if not all_templates:
        continue
    df.at[index,"templates_top_zone_1"]=",".join(all_templates[:1])
    df.at[index,"templates_top_zone_2"]=",".join(all_templates[1:2])
    df.at[index,"templates_top_zone_3"]=",".join(all_templates[2:])


zone_time_map={
    1:(6,9),
    2:(9,12),
    3:(12,15),
    4:(15,18),
    5:(18,21),
    6:(21,24)
}


notifications=[]

for _,row in df.iterrows():
    zones=[
        ("top_zone_1","templates_top_zone_1"),
        ("top_zone_2","templates_top_zone_2"),
        ("top_zone_3","templates_top_zone_3")
    ]
    for zone_col,template_col in zones:
        zone=row[zone_col]
        if pd.isna(zone):
            continue
        template_list=[t for t in str(row[template_col]).split(",") if t]
        if not template_list:
            continue
        start,end=zone_time_map[int(zone)]
        interval=(end-start)/(len(template_list)+1)
        for i,t in enumerate(template_list):
            time_value=start+interval*(i+1)
            hour=int(time_value)
            minute=int((time_value-hour)*60)
            channel=get_channel(row["user_id"],zone,row["lifecycle_stage"])
            notifications.append({
                "user_id":row["user_id"],
                "time":f"{hour:02d}:{minute:02d}",
                "template_id":t,
                "channel":channel
            })


def enrich_notifications(output):
    message_cols = [
        "template_id",
        "message_title_en",
        "message_title_hi",
        "message_body_en",
        "message_body_hi"
    ]
    template_text = templates[message_cols].copy()
    template_text["template_id"] = template_text["template_id"].astype(str)
    output["template_id"] = output["template_id"].astype(str)
    output = output.merge(
        template_text,
        on="template_id",
        how="left"
    )
    behavior_cols = [
        "user_id",
        "coins_balance",
        "streak_current",
        "exercises_completed_7d",
        "sessions_last_7d"
    ]
    output = output.merge(
        user_behavior[behavior_cols],
        on="user_id",
        how="left"
    )
    placeholders = [
        "coins_balance",
        "streak_current",
        "exercises_completed_7d",
        "sessions_last_7d"
    ]
    message_fields = [
        "message_title_en",
        "message_title_hi",
        "message_body_en",
        "message_body_hi"
    ]
    def replace_placeholders(row):
        for col in message_fields:
            text = str(row[col])
            for ph in placeholders:
                text = text.replace(
                    f"{{{ph}}}",
                    str(row[ph])
                )
            row[col] = text
        return row
    output = output.apply(replace_placeholders, axis=1)
    output = output.drop(columns=placeholders)
    output = output.rename(columns={
        "message_title_en":"template_message_title_en",
        "message_title_hi":"template_message_title_hi",
        "message_body_en":"template_message_body_en",
        "message_body_hi":"template_message_body_hi"
    })
    return output


output=pd.DataFrame(notifications)
output["time_dt"]=pd.to_datetime(output["time"],format="%H:%M")
output=output.sort_values(["user_id","time_dt"]).drop(columns=["time_dt"])
output = enrich_notifications(output)
output.to_csv("iteration_0_before_learning/user_notification_schedule.csv",index=False)


def schedule_generator_iteration_1(rl_strategy, goals_df):

    global whatsapp_sent

    whatsapp_sent = {}

    segments = pd.read_csv("iteration_1_after_learning/user_segments.csv")
    goal = goals_df.copy()
    timings = pd.read_csv("iteration_1_after_learning/timing_recommendations.csv")
    themes = pd.read_csv("iteration_0_before_learning/communication_themes.csv")
    templates_rl = pd.read_csv("iteration_1_after_learning/message_templates.csv")
    user_behavior_rl = pd.read_csv(_find_behavioral_csv())

    df_rl = segments[
        ["user_id","segment_id","lifecycle_stage","activeness_score"]
    ].copy()

    df_rl = df_rl.merge(
        themes[["segment_id","drive_1","drive_2","drive_3"]],
        on="segment_id"
    )

    df_rl = df_rl.merge(
        goal[["segment_id","lifecycle_stage","primary_goal","sub_goal"]],
        on=["segment_id","lifecycle_stage"]
    )

    df_rl = df_rl.merge(
        timings[["user_id","top_zone_1","top_zone_2","top_zone_3"]],
        on="user_id",
        how="left"
    )

    tasks = []

    for (segment,lifecycle), group in df_rl.groupby(["segment_id","lifecycle_stage"]):
        sample = group.iloc[0]
        tasks.append({
            "segment":segment,
            "lifecycle":lifecycle,
            "text":f"""
GOAL:
{sample['primary_goal']}

SUBGOAL:
{sample['sub_goal']}
""",
            "drives":[
                sample["drive_1"],
                sample["drive_2"],
                sample["drive_3"]
            ]
        })

    prompt = """
Score Octalysis drive effectiveness for each task.

Return ONLY JSON.

FORMAT:

{
"segment|lifecycle": {
"DriveName": score
}
}

Scores must be between 0 and 1.

TASKS:
"""

    for t in tasks:
        prompt += f"""

SEGMENT: {t['segment']}
LIFECYCLE: {t['lifecycle']}

TEXT:
{t['text']}

DRIVES:
{t['drives'][0]}
{t['drives'][1]}
{t['drives'][2]}
"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=8192,
        temperature=0,
        messages=[{
            "role":"user",
            "content":[{"type":"text","text":prompt}]
        }]
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json","").replace("```","")
    drive_scores = _repair_truncated_json(raw)

    for i in range(1,4):
        df_rl[f"score_final_drive_{i}"] = 0.0

    for index,row in df_rl.iterrows():
        key = f"{row['segment_id']}|{row['lifecycle_stage']}"
        segment_scores = drive_scores.get(key,{})
        drives=[row["drive_1"],row["drive_2"],row["drive_3"]]
        state = engagement_state(row["activeness_score"])
        multiplier = activeness_multiplier(state)
        values=[]
        for drive in drives:
            val = segment_scores.get(drive,0)*multiplier.get(drive,1)
            values.append(val)
        total=sum(values) or 1
        for i in range(3):
            df_rl.at[index,f"score_final_drive_{i+1}"]=values[i]/total

    template_scores = {}

    for _, r in rl_strategy.iterrows():
        score = 1
        if r["rl_classification"] == "GOOD":
            score += 1
        elif r["rl_classification"] == "BAD":
            score -= 1
        template_scores[str(r["template_id"])] = score

    segment_uninstall_rate = (
        rl_strategy
        .groupby("segment_id")["uninstall_rate"]
        .mean()
    )

    high_uninstall_segments = set(
        segment_uninstall_rate[segment_uninstall_rate > 2].index
    )

    df_rl["notification_count"] = df_rl["activeness_score"].apply(
        get_notification_count
    )

    for index,row in df_rl.iterrows():
        notif = row["notification_count"]
        if row["segment_id"] in high_uninstall_segments:
            notif = max(1, notif - 2)
        df_rl.at[index,"notification_count"] = notif

    for index,row in df_rl.iterrows():
        notif = int(row["notification_count"])
        scores_list = [
            row["score_final_drive_1"] * notif,
            row["score_final_drive_2"] * notif,
            row["score_final_drive_3"] * notif
        ]
        floors = [int(x) for x in scores_list]
        remainder = notif - sum(floors)
        fractions = [scores_list[i] - floors[i] for i in range(3)]
        order = sorted(range(3), key=lambda i: fractions[i], reverse=True)
        for i in range(remainder):
            floors[order[i]] += 1
        df_rl.at[index,"score_final_drive_1"] = floors[0]
        df_rl.at[index,"score_final_drive_2"] = floors[1]
        df_rl.at[index,"score_final_drive_3"] = floors[2]

    template_lookup = {}

    for _,row in templates_rl.iterrows():
        key=(row["segment_id"],row["lifecycle_stage"],row["theme_used"])
        template_lookup.setdefault(key,[]).append(str(row["template_id"]))

    df_rl["template_drive_1"]=""
    df_rl["template_drive_2"]=""
    df_rl["template_drive_3"]=""

    for index,row in df_rl.iterrows():
        drives=[row["drive_1"],row["drive_2"],row["drive_3"]]
        for i,drive in enumerate(drives,1):
            key=(row["segment_id"],row["lifecycle_stage"],drive)
            df_rl.at[index,f"template_drive_{i}"]=",".join(
                template_lookup.get(key,[])
            )

    def collect_templates_rl(row):
        selected=[]
        drives=[
            ("template_drive_1", int(row["score_final_drive_1"])),
            ("template_drive_2", int(row["score_final_drive_2"])),
            ("template_drive_3", int(row["score_final_drive_3"]))
        ]
        for col,count in drives:
            if count <= 0:
                continue
            template_list=[t for t in str(row[col]).split(",") if t]
            if not template_list:
                continue
            scored=[]
            for t in template_list:
                score=template_scores.get(str(t),1)
                scored.append((t,score))
            scored.sort(key=lambda x:x[1],reverse=True)
            take=min(count,len(scored))
            selected.extend([t[0] for t in scored[:take]])
        random.shuffle(selected)
        return selected

    df_rl["templates_top_zone_1"]=""
    df_rl["templates_top_zone_2"]=""
    df_rl["templates_top_zone_3"]=""

    for index,row in df_rl.iterrows():
        templates_selected=collect_templates_rl(row)
        if not templates_selected:
            continue
        df_rl.at[index,"templates_top_zone_1"]=",".join(templates_selected[:1])
        df_rl.at[index,"templates_top_zone_2"]=",".join(templates_selected[1:2])
        df_rl.at[index,"templates_top_zone_3"]=",".join(templates_selected[2:])

    notifications=[]

    for _,row in df_rl.iterrows():
        zones=[
            ("top_zone_1","templates_top_zone_1"),
            ("top_zone_2","templates_top_zone_2"),
            ("top_zone_3","templates_top_zone_3")
        ]
        for zone_col,template_col in zones:
            zone=row[zone_col]
            if pd.isna(zone):
                continue
            template_list=[t for t in str(row[template_col]).split(",") if t]
            if not template_list:
                continue
            start,end=zone_time_map[int(zone)]
            interval=(end-start)/(len(template_list)+1)
            for i,t in enumerate(template_list):
                time_value=start+interval*(i+1)
                hour=int(time_value)
                minute=int((time_value-hour)*60)
                channel=get_channel(row["user_id"],zone,row["lifecycle_stage"])
                notifications.append({
                    "user_id":row["user_id"],
                    "time":f"{hour:02d}:{minute:02d}",
                    "template_id":t,
                    "channel":channel
                })

    output=pd.DataFrame(notifications)
    output["time_dt"]=pd.to_datetime(output["time"],format="%H:%M")
    output=output.sort_values(["user_id","time_dt"]).drop(columns=["time_dt"])

    def enrich_notifications_rl(output):
        message_cols = [
            "template_id",
            "message_title_en",
            "message_title_hi",
            "message_body_en",
            "message_body_hi"
        ]
        template_text = templates_rl[message_cols].copy()
        template_text["template_id"] = template_text["template_id"].astype(str)
        output["template_id"] = output["template_id"].astype(str)
        output = output.merge(template_text, on="template_id", how="left")
        behavior_cols = [
            "user_id",
            "coins_balance",
            "streak_current",
            "exercises_completed_7d",
            "sessions_last_7d"
        ]
        output = output.merge(user_behavior_rl[behavior_cols], on="user_id", how="left")
        placeholders = [
            "coins_balance",
            "streak_current",
            "exercises_completed_7d",
            "sessions_last_7d"
        ]
        message_fields = [
            "message_title_en",
            "message_title_hi",
            "message_body_en",
            "message_body_hi"
        ]
        def replace_placeholders(row):
            for col in message_fields:
                text = str(row[col])
                for ph in placeholders:
                    text = text.replace(f"{{{ph}}}", str(row[ph]))
                row[col] = text
            return row
        output = output.apply(replace_placeholders, axis=1)
        output = output.drop(columns=placeholders)
        output = output.rename(columns={
            "message_title_en":"template_message_title_en",
            "message_title_hi":"template_message_title_hi",
            "message_body_en":"template_message_body_en",
            "message_body_hi":"template_message_body_hi"
        })
        return output

    output = enrich_notifications_rl(output)
    output.to_csv("iteration_1_after_learning/user_notification_schedule.csv",index=False)
