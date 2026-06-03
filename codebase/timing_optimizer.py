"""
Timing Optimizer — Project Aurora
==================================
Inputs  : user_behavioral_data.csv, iteration_0_before_learning/user_segments.csv
Outputs : iteration_0_before_learning/timing_recommendations.csv
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

ITER0_DIR = "iteration_0_before_learning"
ITER1_DIR = "iteration_1_after_learning"

WINDOW_TO_ZONE = {
    "early_morning": 1, "mid_morning": 2, "afternoon": 3,
    "late_afternoon": 4, "evening": 5, "night": 6,
}
ALL_ZONES = [1, 2, 3, 4, 5, 6]


def _find_behavioral_csv():
    candidates = [f for f in os.listdir('.') if f.endswith('.csv')
                  and 'user' in f.lower() and 'segment' not in f.lower()
                  and 'goal' not in f.lower() and 'schedule' not in f.lower()
                  and 'template' not in f.lower() and 'theme' not in f.lower()
                  and 'timing' not in f.lower() and 'experiment' not in f.lower()
                  and 'delta' not in f.lower() and 'notification' not in f.lower()]
    return candidates[0] if candidates else "user_behavioral_data.csv"


def run_timing_optimizer(behavioral_path=None, segments_path=None, output_dir=None):
    if output_dir is None:
        output_dir = ITER0_DIR
    if behavioral_path is None:
        behavioral_path = _find_behavioral_csv()
    if segments_path is None:
        segments_path = os.path.join(ITER0_DIR, "user_segments.csv")

    OUT_TIMING_PATH = os.path.join(output_dir, "timing_recommendations.csv")

    print("=" * 60)
    print("  TIMING OPTIMIZER — PROJECT AURORA")
    print("=" * 60)

    beh = pd.read_csv(behavioral_path)
    seg = pd.read_csv(segments_path)

    seg_cols = ["user_id", "segment_id", "segment_name",
                "activeness_score", "churn_risk",
                "propensity_gamification", "propensity_ai_tutor",
                "propensity_leaderboard", "propensity_social"]
    available_seg_cols = [c for c in seg_cols if c in seg.columns]
    seg_slim = seg[available_seg_cols].drop_duplicates("user_id")

    df = beh.merge(seg_slim, on="user_id", how="left")
    print(f"\n[OK] Merged — {df.shape}")

    # Map preferred hour into coarse time zones for classification modeling.
    def map_to_zone(hour):
        if 6 <= hour <= 8: return 1
        elif 9 <= hour <= 11: return 2
        elif 12 <= hour <= 14: return 3
        elif 15 <= hour <= 17: return 4
        elif 18 <= hour <= 20: return 5
        elif 21 <= hour <= 23: return 6
        elif hour == 5: return 1
        else: return np.nan

    df["time_zone"] = df["preferred_hour"].apply(map_to_zone)
    df.dropna(subset=["time_zone"], inplace=True)
    df["time_zone"] = df["time_zone"].astype(int)

    le_lifecycle = LabelEncoder()
    le_age = LabelEncoder()
    le_region = LabelEncoder()

    df["lifecycle_enc"] = le_lifecycle.fit_transform(df["lifecycle_stage"].astype(str))
    df["age_enc"] = le_age.fit_transform(df["age_band"].astype(str))
    df["region_enc"] = le_region.fit_transform(df["region"].astype(str))
    df["ai_tutor_enc"] = df["feature_ai_tutor_used"].map({True:1,False:0,"True":1,"False":0}).fillna(0).astype(int)
    df["leaderboard_enc"] = df["feature_leaderboard_viewed"].map({True:1,False:0,"True":1,"False":0}).fillna(0).astype(int)
    df["progress_enc"] = df["feature_progress_checked"].map({True:1,False:0,"True":1,"False":0}).fillna(0).astype(int) if "feature_progress_checked" in df.columns else 0

    FEATURES = ["days_since_signup","sessions_last_7d","exercises_completed_7d",
                "streak_current","coins_balance","notif_open_rate_30d","motivation_score"]
    for f in ["activeness_score","churn_risk","propensity_gamification","propensity_ai_tutor",
              "propensity_leaderboard","propensity_social","segment_id"]:
        if f in df.columns: FEATURES.append(f)
    FEATURES += ["lifecycle_enc","age_enc","region_enc","ai_tutor_enc","leaderboard_enc","progress_enc"]
    FEATURES = [f for f in FEATURES if f in df.columns]

    df_model = df[FEATURES + ["time_zone","user_id"]].dropna()
    X = df_model[FEATURES].values
    y = df_model["time_zone"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train several classifiers and pick the one with best accuracy.
    models = {
        "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42))]),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))]),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=42))]),
    }

    results = {}
    fitted_models = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cv = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean()
        results[name] = {"accuracy": acc, "f1_macro": f1, "cv_accuracy": cv}
        fitted_models[name] = pipe
        print(f"  {name:<23} {acc:.4f}  {f1:.4f}  {cv:.4f}")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = fitted_models[best_name]
    print(f"\n[BEST] {best_name}")

    classes = best_model.classes_
    proba = best_model.predict_proba(df_model[FEATURES].values)
    proba_df = pd.DataFrame(proba, columns=[f"prob_zone_{c}" for c in classes])
    proba_df["user_id"] = df_model["user_id"].values
    if "segment_id" in df_model.columns:
        proba_df["segment_id"] = df_model["segment_id"].values

    prob_cols = [f"prob_zone_{c}" for c in classes]

    def top3_zones(row):
        scores = {c: row[f"prob_zone_{c}"] for c in classes}
        sz = sorted(scores, key=scores.get, reverse=True)[:3]
        return pd.Series({
            "top_zone_1": int(sz[0]), "top_zone_1_prob": round(scores[sz[0]], 4),
            "top_zone_2": int(sz[1]), "top_zone_2_prob": round(scores[sz[1]], 4),
            "top_zone_3": int(sz[2]), "top_zone_3_prob": round(scores[sz[2]], 4),
        })

    user_top3 = proba_df.apply(top3_zones, axis=1)
    user_top3["user_id"] = proba_df["user_id"].values
    user_top3 = user_top3[["user_id","top_zone_1","top_zone_1_prob","top_zone_2","top_zone_2_prob","top_zone_3","top_zone_3_prob"]]

    os.makedirs(output_dir, exist_ok=True)
    user_top3.to_csv(OUT_TIMING_PATH, index=False)
    print(f"\n[OK] → {OUT_TIMING_PATH} ({user_top3.shape[0]} rows)")
    return user_top3


def generate_segment_timing_top3(rl_comprehensive_df, output_path=None):
    """
    Called by t3 after RL strategy is built.
    Maps notification_window → time zone for each segment using rl_reward-weighted scores.
    Outputs timing_recommendations.csv to iteration_1_after_learning/
    """
    if output_path is None:
        output_path = os.path.join(ITER1_DIR, "timing_recommendations.csv")

    df = rl_comprehensive_df.copy()
    # Strip time-range suffix like "early_morning (06:00 - 08:59)" → "early_morning"
    df["_window_clean"] = df["notification_window"].astype(str).str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    df["time_zone"] = df["_window_clean"].map(WINDOW_TO_ZONE)
    df = df.dropna(subset=["time_zone"])
    df["time_zone"] = df["time_zone"].astype(int)

    df["score"] = np.clip(df["rl_reward"], 0, None) * df["allocation_weight"]

    agg = df.groupby(["segment_id", "time_zone"])["score"].sum().reset_index().rename(columns={"score": "zone_score"})

    # Read segment names from iter0
    seg_names_dict = {}
    seg_path = os.path.join(ITER0_DIR, "user_segments.csv")
    if os.path.exists(seg_path):
        _seg = pd.read_csv(seg_path)
        seg_names_dict = dict(zip(_seg["segment_id"].astype(str), _seg["segment_name"]))

    rows = []
    for seg_id in sorted(df["segment_id"].unique()):
        seg_agg = agg[agg["segment_id"] == seg_id].set_index("time_zone")["zone_score"]
        zone_scores = {z: seg_agg.get(z, 0.0) for z in ALL_ZONES}
        total = sum(zone_scores.values())
        zone_probs = {z: v / total for z, v in zone_scores.items()} if total > 0 else {z: 1/6 for z in ALL_ZONES}
        top3 = sorted(zone_probs, key=zone_probs.get, reverse=True)[:3]
        rows.append({
            "segment_id": seg_id,
            "segment_name": seg_names_dict.get(str(seg_id), f"Segment_{seg_id}"),
            "row_count": int(df[df["segment_id"] == seg_id].shape[0]),
            "primary_window": WINDOW_TO_ZONE and [k for k, v in WINDOW_TO_ZONE.items() if v == top3[0]][0] if top3 else "",
            "top1_zone": top3[0], "top1_probability": round(zone_probs[top3[0]], 4),
            "top2_zone": top3[1], "top2_probability": round(zone_probs[top3[1]], 4),
            "top3_zone": top3[2], "top3_probability": round(zone_probs[top3[2]], 4),
        })

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[Timing Iter1] → {output_path} ({len(result)} segments)")
    return result


if __name__ == "__main__":
    run_timing_optimizer()
