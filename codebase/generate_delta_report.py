#!/usr/bin/env python3

import csv
import os
import argparse

ITER0_DIR = "iteration_0_before_learning"
ITER1_DIR = "iteration_1_after_learning"

DEFAULTS = {
    "iter0_templates": os.path.join(ITER0_DIR, "message_templates.csv"),
    "iter1_templates": os.path.join(ITER1_DIR, "message_templates.csv"),
    "iter0_schedule":  os.path.join(ITER0_DIR, "user_notification_schedule.csv"),
    "iter1_schedule":  os.path.join(ITER1_DIR, "user_notification_schedule.csv"),
    "iter0_timing":    os.path.join(ITER0_DIR, "timing_recommendations.csv"),
    "iter1_timing":    os.path.join(ITER1_DIR, "timing_recommendations.csv"),
    "experiment":      "experiment_results.csv",
    "output":          "learning_delta_report.csv",
}

FIELDNAMES = [
    "entity_type",
    "entity_id",
    "change_type",
    "metric_trigger",
    "before_value",
    "after_value",
    "explanation",
]


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Compare iteration outputs and produce human readable delta explanations summary.
def detect_template_changes(iter0, iter1, experiment):
    deltas = []
    exp_by_tid = {r.get("template_id", "")
                        : r for r in experiment if r.get("template_id")}
    iter0_by_tid = {r.get("template_id", "")                    : r for r in iter0 if r.get("template_id")}
    iter1_by_tid = {r.get("template_id", "")                    : r for r in iter1 if r.get("template_id")}

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
                    f"for segment {t1.get('segment_id', '?')} to replace suppressed templates, "
                    f"modelled on patterns from GOOD-performing templates in the same segment."
                ),
            })

        elif t0 and t1:
            changed_fields = [
                f for f in ["message_title_en", "message_title_hi",
                            "message_body_en", "message_body_hi",
                            "cta_text_en", "cta_text_hi", "theme_used"]
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
                        f"Retained unchanged in Iteration 1 and flagged as a reference pattern "
                        f"for generating new templates in the same segment x lifecycle."
                    ),
                })

    return deltas


def detect_timing_changes(iter0_timing, iter1_timing, experiment):
    deltas = []
    window_ctr = {}
    for r in experiment:
        seg = r.get("segment_id", "")
        win = r.get("notification_window", "")
        ctr = _safe_float(r.get("ctr", 0))
        if seg and win:
            window_ctr.setdefault((seg, win), []).append(ctr)
    avg_window_ctr = {k: sum(v) / len(v) for k, v in window_ctr.items()}

    iter0_by_seg = {r.get("segment_id", "")                    : r for r in iter0_timing if r.get("segment_id")}
    iter1_by_seg = {r.get("segment_id", "")                    : r for r in iter1_timing if r.get("segment_id")}

    for seg in sorted(set(iter0_by_seg) | set(iter1_by_seg)):
        t0 = iter0_by_seg.get(seg, {})
        t1 = iter1_by_seg.get(seg, {})
        win0 = t0.get("optimal_window", t0.get("primary_window", ""))
        win1 = t1.get("optimal_window", t1.get("primary_window", ""))

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
                        "Timing updated based on learned engagement patterns for this segment."
                    )
                ),
            })

    return deltas


def detect_frequency_changes(iter0_schedule, iter1_schedule, experiment):
    deltas = []
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
                    f"Segment {seg} daily notification frequency changed from {a0:.1f} to {a1:.1f} notifs/day. "
                    + (
                        f"Uninstall rate of {unin:.1%} exceeded the 2% guardrail, "
                        f"triggering a mandatory -2/day reduction regardless of activeness score."
                        if guardrail else
                        f"Frequency rebalanced based on revised activeness score from experiment engagement patterns."
                    )
                ),
            })

    return deltas


def detect_segment_changes(iter0, iter1, experiment):
    deltas = []
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

    all_segs = {t.get("segment_id")
                for t in iter0 + iter1 if t.get("segment_id")}
    for seg in sorted(all_segs):
        dom0 = _dominant_theme(iter0, seg)
        dom1 = _dominant_theme(iter1, seg)
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
                    + (f"Experiment confirmed '{', '.join(good)}' as high-performing and "
                       f"'{', '.join(bad)}' as underperforming. "
                       if good or bad else "")
                    + "Template mix updated to weight proven themes more heavily."
                ),
            })

    return deltas


def _summary_row(all_deltas, experiment):
    good = sum(1 for r in experiment if r.get(
        "performance_status", "").upper() == "GOOD")
    bad = sum(1 for r in experiment if r.get(
        "performance_status", "").upper() == "BAD")
    neut = sum(1 for r in experiment if r.get(
        "performance_status", "").upper() == "NEUTRAL")
    total = len(experiment)

    suppressed = sum(1 for d in all_deltas if d["change_type"] == "suppressed")
    added = sum(1 for d in all_deltas if d["change_type"] == "template_added")
    replaced = sum(
        1 for d in all_deltas if d["change_type"] == "template_replaced")
    promoted = sum(1 for d in all_deltas if d["change_type"] == "promoted")

    return [{
        "entity_type":    "system",
        "entity_id":      "iteration_summary",
        "change_type":    "learning_cycle_complete",
        "metric_trigger": f"experiment_results: {total} templates evaluated — GOOD={good}, NEUTRAL={neut}, BAD={bad}",
        "before_value":   "Iteration 0 (pre-learning)",
        "after_value":    "Iteration 1 (post-learning)",
        "explanation":    (
            f"Learning cycle completed. Of {total} evaluated templates: "
            f"{good} GOOD (CTR>15%, engagement>40%), {neut} NEUTRAL, {bad} BAD (CTR<5%, engagement<20%). "
            f"Actions: {suppressed} suppressed, {promoted} promoted as references, "
            f"{added} new templates generated, {replaced} rewritten. "
            f"Iteration 1 removes underperforming content and amplifies proven patterns."
        ),
    }]


def generate_delta_report(
    iter0_templates_path, iter1_templates_path,
    iter0_schedule_path,  iter1_schedule_path,
    iter0_timing_path,    iter1_timing_path,
    experiment_path,      output_path,
):
    iter0_templates = _read_csv(iter0_templates_path)
    iter1_templates = _read_csv(iter1_templates_path)
    iter0_schedule = _read_csv(iter0_schedule_path)
    iter1_schedule = _read_csv(iter1_schedule_path)
    iter0_timing = _read_csv(iter0_timing_path)
    iter1_timing = _read_csv(iter1_timing_path)
    experiment = _read_csv(experiment_path)

    template_deltas = detect_template_changes(
        iter0_templates, iter1_templates, experiment)
    timing_deltas = detect_timing_changes(
        iter0_timing, iter1_timing, experiment)
    frequency_deltas = detect_frequency_changes(
        iter0_schedule, iter1_schedule, experiment)
    segment_deltas = detect_segment_changes(
        iter0_templates, iter1_templates, experiment)

    all_deltas = template_deltas + timing_deltas + frequency_deltas + segment_deltas
    final_rows = _summary_row(all_deltas, experiment) + sorted(
        all_deltas, key=lambda x: (x["entity_type"], x["entity_id"]))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(
        output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(final_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter0-templates",
                        default=DEFAULTS["iter0_templates"])
    parser.add_argument("--iter1-templates",
                        default=DEFAULTS["iter1_templates"])
    parser.add_argument("--iter0-schedule",
                        default=DEFAULTS["iter0_schedule"])
    parser.add_argument("--iter1-schedule",
                        default=DEFAULTS["iter1_schedule"])
    parser.add_argument("--iter0-timing",    default=DEFAULTS["iter0_timing"])
    parser.add_argument("--iter1-timing",    default=DEFAULTS["iter1_timing"])
    parser.add_argument("--experiment",      default=DEFAULTS["experiment"])
    parser.add_argument("--output",          default=DEFAULTS["output"])
    args = parser.parse_args()

    generate_delta_report(
        iter0_templates_path=args.iter0_templates,
        iter1_templates_path=args.iter1_templates,
        iter0_schedule_path=args.iter0_schedule,
        iter1_schedule_path=args.iter1_schedule,
        iter0_timing_path=args.iter0_timing,
        iter1_timing_path=args.iter1_timing,
        experiment_path=args.experiment,
        output_path=args.output,
    )
