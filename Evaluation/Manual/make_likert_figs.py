#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRIC_ALIASES = {
    "Temporal consistency": ["temporal consistency", "temporal_consistency", "temporal"],
    "Texture": ["texture", "textures"],
    "Motion smoothness": ["motion smoothness", "motion_smoothness", "motion", "smoothness"],
    "Artifacting": ["artifacting", "artifacts", "artifact"],
}
METRIC_ORDER = ["Temporal consistency", "Texture", "Motion smoothness", "Artifacting"]
MODEL_CANDIDATES = ["model", "pipeline", "method"]
VIDEO_CANDIDATES = ["video", "clip", "id", "name", "filename"]
FLAG_CANDIDATES  = ["top", "best", "worst", "bucket", "group", "ranking"]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def find_col(df: pd.DataFrame, candidates):
    for c in df.columns:
        for cand in candidates:
            if cand in c:
                return c
    return None

def infer_model_from_text(text: str) -> str:
    t = str(text).lower()
    if "diffueraser" in t: return "DiffuEraser"
    if "wan 1.3" in t or "wan1.3" in t: return "WAN 1.3B"
    if "wan 14" in t or "wan14" in t or "wan_14" in t: return "WAN 14B"
    if "stable diffusion" in t or "stablediffusion" in t or re.search(r"\bsd\b", t): return "Stable Diffusion"
    if "baseline" in t or "orig" in t or "original" in t: return "Baseline"
    return "Unknown"

def locate_metric_columns(df: pd.DataFrame):
    metric_cols = {}
    for canonical, aliases in METRIC_ALIASES.items():
        found = None
        for col in df.columns:
            for a in aliases:
                if a in col:
                    found = col
                    break
            if found: break
        if found: metric_cols[canonical] = found
    return metric_cols

def detect_top_worst_flag(df: pd.DataFrame):
    flag_col = find_col(df, FLAG_CANDIDATES)
    if flag_col is None or not (df[flag_col].dtype == object or pd.api.types.is_string_dtype(df[flag_col])): 
        return None, False
    def label_bucket(x):
        s = str(x).lower()
        if "top" in s or "best" in s: return "top"
        if "worst" in s or "bottom" in s: return "worst"
        return None
    bucket = df[flag_col].apply(label_bucket)
    if bucket.notna().any():
        df["__bucket__"] = bucket
        return "__bucket__", True
    return None, False

# ---------- Plotters ----------
def plot_heatmap_delta(data: pd.DataFrame, out_path: str):
    if data.empty: return
    data = data.copy()
    data["delta"] = data["top3_mean"] - data["worst3_mean"]
    pivot = data.pivot(index="model", columns="metric", values="delta")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(pivot.values)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    ax.set_title("Top3 − Worst3 (Δ Likert) by Model × Metric")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _autolabel(ax, bars, decimals=2, frac_offset=0.04, fontsize=10):
    """
    Write values on top of bars with spacing that scales with the y-range.
    frac_offset is a fraction of the current y-range, so labels won't collide
    even when bars are near the top.
    """
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    dy = frac_offset * yrange
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2.0,
            h + dy,
            f"{h:.{decimals}f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=fontsize,
            clip_on=False,     # allow label to render slightly outside axes if needed
        )

def plot_grouped_bar_for_metric(data: pd.DataFrame, metric_name: str, out_path: str):
    """Shows numbers on bars, no error bars, with extra headroom for labels."""
    sub = data[data["metric"] == metric_name].copy()
    if sub.empty: 
        return

    x = np.arange(len(sub))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.8))  # a bit taller/wider for spacing

    # Bars (no yerr)
    bars_top   = ax.bar(x - width/2, sub["top3_mean"].values,   width, label="Top 3 mean")
    bars_worst = ax.bar(x + width/2, sub["worst3_mean"].values, width, label="Worst 3 mean")

    # --- Dynamic headroom so labels never clash with the plot frame ---
    ymax_val = float(max(sub["top3_mean"].max(), sub["worst3_mean"].max()))
    # Add absolute headroom of ~0.6 Likert points or 12% of bar height, whichever is larger
    headroom = max(0.6, 0.12 * ymax_val)
    # Cap at 5.5 to keep axes sensible for Likert; raise top if your values can exceed 5
    upper = min(5.5, ymax_val + headroom)
    ax.set_ylim(0, upper)

    # Labels on bars (offset scales with y-range so it adapts to your data)
    _autolabel(ax, bars_top,   decimals=2, frac_offset=0.04, fontsize=10)
    _autolabel(ax, bars_worst, decimals=2, frac_offset=0.04, fontsize=10)

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Likert mean (1–5)")
    ax.set_title(f"{metric_name}: Top 3 vs Worst 3 by Model")
    ax.legend()

    # A little extra padding for the title/labels so nothing crowds the frame
    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.18)  # more room for title & rotated xticks

    fig.savefig(out_path, dpi=200)
    # Optional vector export:
    # fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)

def plot_lollipop_delta_by_model(data: pd.DataFrame, out_path: str):
    if data.empty: return
    delta = (data.groupby("model")[["top3_mean","worst3_mean"]]
             .mean().assign(delta=lambda d: d["top3_mean"] - d["worst3_mean"])
             .reset_index())
    delta = delta.sort_values("delta")
    y = np.arange(len(delta))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hlines(y, 0, delta["delta"].values)
    pts = ax.plot(delta["delta"].values, y, "o")
    for i, v in enumerate(delta["delta"].values):
        ax.text(v, y[i], f" {v:.2f}", va="center", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(delta["model"].tolist())
    ax.set_xlabel("Δ Likert (Top3 − Worst3) averaged over metrics")
    ax.set_title("Overall improvement from Worst 3 to Top 3 (per model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_radar_per_model(data: pd.DataFrame, model_name: str, out_path: str):
    sub = data[data["model"] == model_name].copy()
    if sub.empty: return
    order = [m for m in METRIC_ORDER if m in data["metric"].unique()]
    ordered = sub.set_index("metric").loc[order].reset_index()
    values_top = ordered["top3_mean"].tolist()
    values_worst = ordered["worst3_mean"].tolist()
    labels = ordered["metric"].tolist()
    values_top += values_top[:1]
    values_worst += values_worst[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2*np.pi, len(labels), endpoint=False))
    ax.set_xticklabels(labels)
    ax.set_ylim(1, 5)
    ax.plot(angles, values_top, linewidth=2, label="Top 3 mean")
    ax.fill(angles, values_top, alpha=0.1)
    ax.plot(angles, values_worst, linewidth=2, label="Worst 3 mean")
    ax.fill(angles, values_worst, alpha=0.1)
    ax.set_title(f"{model_name}: Top 3 vs Worst 3 (Radar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------- Core ----------
def main(input_csv: str, out_dir: str, top_k: int = 3):
    os.makedirs(out_dir, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    df = df_raw.copy()
    df.columns = [norm(c) for c in df.columns]

    metric_cols = locate_metric_columns(df)
    if not metric_cols:
        raise ValueError("No metric columns detected; ensure your CSV has the four metrics (case-insensitive).")
    value_cols = [metric_cols[m] for m in metric_cols.keys()]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    model_col = find_col(df, MODEL_CANDIDATES)
    video_col = find_col(df, VIDEO_CANDIDATES)
    if model_col is None:
        if video_col is not None:
            df["__model__"] = df[video_col].apply(infer_model_from_text)
        else:
            df["__model__"] = "Model"
        model_col = "__model__"
    if video_col is None:
        df["__video__"] = np.arange(len(df))
        video_col = "__video__"

    flag_col, has_flag = detect_top_worst_flag(df)
    df["__overall__"] = df[value_cols].mean(axis=1)

    agg = {col: "mean" for col in value_cols}
    agg["__overall__"] = "mean"
    per_video = df.groupby([model_col, video_col], as_index=False).agg(agg)

    summary_rows = []
    for m in sorted(per_video[model_col].unique()):
        if has_flag:
            dsub = df[df[model_col] == m]
            top_rows = dsub[dsub[flag_col] == "top"].groupby(video_col)[value_cols].mean().reset_index().head(top_k)
            worst_rows = dsub[dsub[flag_col] == "worst"].groupby(video_col)[value_cols].mean().reset_index().head(top_k)
            top, worst = top_rows, worst_rows
        else:
            g = per_video[per_video[model_col] == m].sort_values("__overall__", ascending=False)
            top = g.head(top_k)[[video_col] + value_cols].copy()
            worst = g.tail(top_k)[[video_col] + value_cols].copy()

        for metric_name, col in metric_cols.items():
            top_vals = top[col].dropna()
            worst_vals = worst[col].dropna()
            if len(top_vals) == 0:
                top_mean = top_std = np.nan
            else:
                top_mean = float(top_vals.mean())
                top_std = float(top_vals.std(ddof=1) if len(top_vals) > 1 else 0.0)
            if len(worst_vals) == 0:
                worst_mean = worst_std = np.nan
            else:
                worst_mean = float(worst_vals.mean())
                worst_std = float(worst_vals.std(ddof=1) if len(worst_vals) > 1 else 0.0)

            summary_rows.append({
                "model": m if m else "Unknown",
                "metric": metric_name,
                "top3_mean": round(top_mean, 3) if not math.isnan(top_mean) else np.nan,
                "top3_std": round(top_std, 3) if not math.isnan(top_std) else np.nan,
                "worst3_mean": round(worst_mean, 3) if not math.isnan(worst_mean) else np.nan,
                "worst3_std": round(worst_std, 3) if not math.isnan(worst_std) else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "model_likert_summary_computed.csv")
    summary.to_csv(summary_path, index=False)

    # Figures
    plot_heatmap_delta(summary, os.path.join(out_dir, "heatmap_delta_model_metric.png"))
    for metric in METRIC_ORDER:
        if metric in summary["metric"].unique():
            plot_grouped_bar_for_metric(summary, metric, os.path.join(out_dir, f"bar_{metric.replace(' ', '_').lower()}.png"))
    plot_lollipop_delta_by_model(summary, os.path.join(out_dir, "lollipop_overall_delta.png"))
    for model_name in sorted(summary["model"].unique()):
        plot_radar_per_model(summary, model_name, os.path.join(out_dir, f"radar_{str(model_name).replace(' ', '_').lower()}.png"))

    print(f"Done. Summary: {summary_path}\nFigures saved in: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="Video Output Likert Scale (Responses) - Form responses 1.csv")
    parser.add_argument("--out", default="likert_figures_real")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    main(args.csv, args.out, args.topk)
