import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt  
import matplotlib.cm as cm       
import plotly.graph_objects as go
import os
import numpy as np
from scipy.optimize import curve_fit
import datetime
import io
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import re

from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Border, Side

import plotly.express as px  # for the Auto Tune scatter

# Define 4PL function
def four_param_logistic(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

# Define inverse function to calculate Ct
def inverse_four_pl(threshold, a, b, c, d):
    try:
        return c * ((a - d) / (threshold - d) - 1)**(1 / b)
    except:
        return None


# ---- HomeBrew Function ----
# HomeBrew background subtraction (configurable)
# HomeBrew background subtraction with selectable fit region
def spr_qpcr_background_correction(
    test_signal,
    std_win: int = 7,            # rolling window length (cycles)
    prefit_start: int = 2,       # 0-based start for scanning & baseline fit
    ratio: float = 1.5,          # STD-ratio detector threshold
    fit_end_pad: int = 6,        # used when fit_region="prefit": fit end = i + pad (can be negative if you allow)
    start_point_pad: int = 4,    # reported start = i + pad (can be negative if you allow)
    detector: str = "std",       # "std" or "slope"
    slope_min: float = 50.0,     # slope threshold (y-units per cycle) when detector="slope"
    fit_region: str = "prefit",  # "prefit", "window", or "window_back"
    back_offset: int = 0         # used when fit_region="window_back": slide the window back by K cycles
):
    """
    Returns (baseline_corrected_signal, start_point_index).

    Fit region choices (inclusive indices):
      - "prefit":      [prefit_start .. clamp(i + fit_end_pad)]
      - "window":      [i .. i + std_win - 1]
      - "window_back": [i - (std_win + K) .. i - K]   (K = back_offset)

    Notes:
      - All indices are 0-based; cycle number = index + 1.
      - start_point_index is 0-based (or -1 if not found).
    """
    x = np.asarray(test_signal, dtype=float)
    n = len(x)
    if n < 5:
        return x - np.mean(x[:max(1, n//3)]), -1

    # sanitize inputs
    std_win         = max(3, int(std_win))
    prefit_start    = max(0, int(prefit_start))
    ratio           = float(ratio)
    fit_end_pad     = int(fit_end_pad)
    start_point_pad = int(start_point_pad)
    detector        = (detector or "std").lower()
    back_offset     = max(0, int(back_offset))

    # reference noise (like original S[1])
    ref_start = 1 if 1 + std_win <= n else 0
    S_ref = np.std(x[ref_start: ref_start + std_win]) if ref_start + std_win <= n else np.std(x[:min(std_win, n)])

    A = np.arange(n)
    last_i = n - std_win

    for i in range(prefit_start, last_i + 1):
        win = slice(i, i + std_win)

        # ---- detector ----
        if detector == "slope":
            m = np.polyfit(A[win], x[win], 1)[0]
            triggered = abs(m) > float(slope_min)
        else:
            S_i = np.std(x[win])
            triggered = (S_ref > 0) and ((S_i / (S_ref + 1e-12)) > ratio)

        if not triggered:
            continue

        # ---- choose baseline fit region ----
        if fit_region == "window":
            fit_start = i
            fit_end   = i + std_win - 1

        elif fit_region == "window_back":
            # Example: detect at cycle 20 (i=19), win=7, K=2 -> [11..18] cycles
            fit_start = i - (std_win + back_offset)
            fit_end   = i - back_offset

        else:  # "prefit"
            fit_start = prefit_start
            fit_end   = i + fit_end_pad

        # clamp & ensure at least 2 points for line fit
        fit_start = max(0, min(fit_start, n - 2))
        fit_end   = max(fit_start + 1, min(fit_end, n - 1))

        # linear baseline over chosen region
        if fit_end - fit_start >= 1:
            p = np.polyfit(A[fit_start:fit_end + 1], x[fit_start:fit_end + 1], 1)
            f = np.polyval(p, A)
            E = x - f
        else:
            base = np.mean(x[prefit_start:i]) if i > prefit_start else np.mean(x[:min(5, n)])
            E = x - base

        # reported start point
        start_point = i + start_point_pad
        start_point = max(0, min(start_point, n - 1))
        return E, start_point

    # fallback
    return x - np.mean(x[:min(5, n)]), -1



def calculate_ct(x, y, threshold, startpoint = 10, use_4pl=True, return_std=False):
    x = np.array(x)
    y = np.array(y)
    
    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return (None, None) if return_std else None

    if use_4pl:
        try:
            post_cycle_10 = x >= startpoint
            x_fit = x[post_cycle_10]
            y_fit = y[post_cycle_10]

            if len(x_fit) >= 5:
                popt, pcov = curve_fit(four_param_logistic, x_fit, y_fit, maxfev=10000)
                ct = inverse_four_pl(threshold, *popt)

                if ct is not None and x_fit[0] <= ct <= x_fit[-1]:
                    if return_std:
                        # Estimate gradient numerically
                        eps = 1e-8
                        grads = np.zeros(4)
                        for i in range(4):
                            p_hi = np.array(popt)
                            p_lo = np.array(popt)
                            p_hi[i] += eps
                            p_lo[i] -= eps
                            ct_hi = inverse_four_pl(threshold, *p_hi)
                            ct_lo = inverse_four_pl(threshold, *p_lo)
                            grads[i] = (ct_hi - ct_lo) / (2 * eps)

                        ct_variance = np.dot(grads.T, np.dot(pcov, grads))
                        ct_std = np.sqrt(ct_variance) if ct_variance >= 0 else np.nan
                        return float(ct), float(ct_std)
                    else:
                        return float(ct)
        except:
            pass

    # Linear interpolation fallback
    # above = y > threshold
    # if not np.any(above):
    #     return (None, None) if return_std else None

    # Linear interpolation fallback (respect startpoint)
    post = x >= startpoint
    x_use = x[post]
    y_use = y[post]
    if len(x_use) < 2:
        return (None, None) if return_std else None

    above = y_use > threshold
    if not np.any(above):
        return (None, None) if return_std else None

    idx = np.argmax(above)

    # if threshold is already exceeded at the first post-startpoint sample, don't return x[0]
    if idx == 0:
        return (None, None) if return_std else None

    x1, x2 = x_use[idx - 1], x_use[idx]
    y1, y2 = y_use[idx - 1], y_use[idx]
    ct = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    return (float(ct), None) if return_std else float(ct)


    idx = np.argmax(above)
    if idx == 0:
        ct = float(x[0])
    else:
        x1, x2 = x[idx - 1], x[idx]
        y1, y2 = y[idx - 1], y[idx]
        ct = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)

    return (ct, None) if return_std else float(ct)


# --- Pair builder reused for scoring ---
def _pairs_for_group(wells_in_group: set, replicate_mode: str):
    pair_set = set()
    if replicate_mode == "Custom (paired)":
        for a, b in st.session_state.get("replicate_pairs", []):
            if a in wells_in_group and b in wells_in_group:
                pair_set.add(tuple(sorted((a, b))))
    else:
        for w in wells_in_group:
            for p in replicate_partners(w):
                if p in wells_in_group:
                    pair_set.add(tuple(sorted((w, p))))
    return sorted(pair_set)

# --- Preprocess a single trace exactly like your plotting path ---
def _preprocess_trace(
    df, well, channel_name,
    rox_df,
    enable_deconvolution, deconv_target_channel, df_corr, alpha_value,
    norm_mode: str,
    firstcycle_cycle: int,
    autofluor_dict: dict,
):
    """
    Returns a processed y series for a single well:
      - optional deconvolution (e.g., Cy5 += α * Cy5.5)
      - normalization mode:
         * 'ROX channel'         -> y = y / ROX
         * 'First-cycle formula' -> y = (y - F1) / (F1 - autofluor[channel])
         * 'None'                -> unchanged
    """
    y = df[well].copy()

    # Deconvolution, if requested
    if enable_deconvolution and (channel_name == deconv_target_channel) and (df_corr is not None) and (well in df_corr.columns):
        y = y + alpha_value * df_corr[well]

    # Normalization
    if norm_mode == "ROX channel" and (rox_df is not None) and (well in rox_df.columns) and (channel_name.upper() != "ROX"):
        rs = rox_df[well]
        if np.all(rs > 0):
            y = y / rs

    elif norm_mode == "First-cycle formula":
        # choose which cycle is F1 (1-based in UI)
        idx = max(0, min(len(y) - 1, int(firstcycle_cycle) - 1))
        F1 = float(y.iloc[idx])
        auto = float(autofluor_dict.get(channel_name, 0.0))
        denom = F1 - auto
        # protect against zero/negative scaling
        if denom <= 0:
            denom = 1e-9
        y = (y - F1) / denom

    return y

# --- Apply a baseline config + compute Ct for all selected wells -> replicate STDs + score ---
# def score_config(
#     dfs_by_channel: dict,         # {channel -> df}
#     rox_df,
#     df_corr,
#     enable_deconvolution: bool,
#     deconv_target_channel: str,
#     alpha_value: float,
#     normalize_to_rox: bool,
#     selected_channels: list,
#     per_channel_thresholds: dict,
#     groups: dict,                 # st.session_state["groups"]
#     replicate_mode: str,
#     config: dict,                 # see examples in run_autotune()
#     # NEW:
#     qpos_channel: str = None,
#     qpos_well: str = None,
#     target_ct: float = 23.0,
# ):
def score_config(
    dfs_by_channel: dict, rox_df, df_corr,
    enable_deconvolution, deconv_target_channel, alpha_value,
    norm_mode: str, firstcycle_cycle: int, autofluor_dict: dict,
    selected_channels, per_channel_thresholds,
    groups, replicate_mode, config,
    qpos_channel: str = None, qpos_well: str = None, target_ct: float = 23.0,
):
    """
    Evaluate one baseline config:
    - If qpos_channel & qpos_well are provided AND exist, compute a per-config threshold for qpos_channel
      so that Ct(qPOS) == target_ct. Use that threshold for ALL wells of qpos_channel during scoring.
    - Return median/p75 replicate STD and coverage, plus stats.
    """

    # --- Compute a per-config threshold for qPOS channel (if available) ---
    qpos_threshold = None
    if qpos_channel and (qpos_channel in dfs_by_channel):
        df_q = dfs_by_channel[qpos_channel]
        if (df_q is not None) and (qpos_well in df_q.columns):
            # preprocess + baseline just like other wells, but only for qPOS
            # y_q = _preprocess_trace(df_q, qpos_well, qpos_channel, rox_df,
            #                         enable_deconvolution, deconv_target_channel, df_corr,
            #                         alpha_value, normalize_to_rox)
            y_q = _preprocess_trace(
                df_q, qpos_well, qpos_channel, rox_df,
                enable_deconvolution, deconv_target_channel, df_corr, alpha_value,
                norm_mode, firstcycle_cycle, autofluor_dict
            )

            # apply the candidate baseline method 'config'
            if config["method"] == "avgN":
                bs = config["baseline_start"]
                bc = config["baseline_cycles"]
                baseline = y_q.iloc[bs-1: bs-1+bc].mean()
                yq_corr = y_q - baseline
            else:
                yq_corr, _ = spr_qpcr_background_correction(
                    np.array(y_q),
                    std_win=config["det_win"],
                    prefit_start=config["prefit_start"] - 1,
                    ratio=config.get("lift_ratio", 1.5),
                    fit_end_pad=config.get("fit_end_pad", 6),
                    start_point_pad=config.get("start_point_pad", max(0, config.get("det_win", 7)//2)),
                    detector=config.get("detector", "std"),
                    slope_min=config.get("slope_min", 50.0),
                    fit_region=config.get("fit_region", "prefit"),
                    back_offset=config.get("back_offset", 0),
                )

            x_q = df_q["Cycle"].values
            # This is the ONLY place we "auto tune" the per-config threshold:
            qpos_threshold = _qpos_threshold_from_trace(x_q, np.array(yq_corr), target_ct=target_ct, startpoint=10)

    # --- Build Ct table for this config (uses the qPOS threshold override if present) ---
    ct_rows = []
    for channel_name in selected_channels:
        df = dfs_by_channel.get(channel_name)
        if df is None:
            continue

        # decide channel threshold for this config:
        channel_thr = per_channel_thresholds.get(channel_name, 0.13)
        if (qpos_threshold is not None) and (channel_name == qpos_channel):
            channel_thr = qpos_threshold

        for group, info in groups.items():
            for well in info["wells"]:
                if well not in df.columns:
                    continue

                # y = _preprocess_trace(df, well, channel_name, rox_df,
                #                       enable_deconvolution, deconv_target_channel, df_corr,
                #                       alpha_value, normalize_to_rox)
                y = _preprocess_trace(
                    df, well, channel_name, rox_df,
                    enable_deconvolution, deconv_target_channel, df_corr, alpha_value,
                    norm_mode, firstcycle_cycle, autofluor_dict
                )

                # baseline modes
                if config["method"] == "avgN":
                    bs = config["baseline_start"]
                    bc = config["baseline_cycles"]
                    baseline = y.iloc[bs-1: bs-1+bc].mean()
                    y_corr = y - baseline
                else:
                    y_corr, _ = spr_qpcr_background_correction(
                        np.array(y),
                        std_win=config["det_win"],
                        prefit_start=config["prefit_start"] - 1,
                        ratio=config.get("lift_ratio", 1.5),
                        fit_end_pad=config.get("fit_end_pad", 6),
                        start_point_pad=config.get("start_point_pad", max(0, config.get("det_win", 7)//2)),
                        detector=config.get("detector", "std"),
                        slope_min=config.get("slope_min", 50.0),
                        fit_region=config.get("fit_region", "prefit"),
                        back_offset=config.get("back_offset", 0),
                    )

                x = dfs_by_channel[channel_name]["Cycle"].values
                thr = channel_thr

                # If this is exactly the qPOS well & channel, we force Ct = target_ct for scoring.
                if (qpos_threshold is not None) and (channel_name == qpos_channel) and (well == qpos_well):
                    ct_val = float(target_ct)
                else:
                    ct_val, _ = calculate_ct(x, np.array(y_corr), threshold=thr, startpoint=10, use_4pl=True, return_std=True)

                ct_rows.append({
                    "Group": group, "Well": well, "Channel": channel_name,
                    "Ct_num": float(ct_val) if ct_val is not None else np.nan
                })

    if not ct_rows:
        return {"score": (float("inf"), float("inf"), 0.0), "coverage": 0.0, "stats": None, "qpos_threshold": qpos_threshold}

    ct_df_num = pd.DataFrame(ct_rows)

    # Build replicate STDs
    rep_stds = []
    total_pairs = 0
    for group, info in groups.items():
        wells_in_group = set(info["wells"])
        pair_list = _pairs_for_group(wells_in_group, replicate_mode)
        if not pair_list:
            continue
        for ch in ct_df_num["Channel"].unique():
            sub = ct_df_num[(ct_df_num["Group"] == group) & (ct_df_num["Channel"] == ch)]
            ct_map = {row["Well"]: row["Ct_num"] for _, row in sub.iterrows()}
            for a, b in pair_list:
                total_pairs += 1
                v1, v2 = ct_map.get(a, np.nan), ct_map.get(b, np.nan)
                if not (np.isnan(v1) or np.isnan(v2)):
                    rep_stds.append(float(np.std([v1, v2], ddof=1)))

    if total_pairs == 0 or len(rep_stds) == 0:
        return {"score": (float("inf"), float("inf"), 0.0), "coverage": 0.0, "stats": None, "qpos_threshold": qpos_threshold}

    rep_stds = np.array(rep_stds, dtype=float)
    median_std = float(np.nanmedian(rep_stds))
    p75_std    = float(np.nanpercentile(rep_stds, 75))
    coverage   = float(rep_stds.size) / float(total_pairs)

    return {
        "score": (median_std, p75_std, -coverage),
        "coverage": coverage,
        "stats": {"median": median_std, "p75": p75_std, "count": rep_stds.size, "total_pairs": total_pairs},
        "qpos_threshold": qpos_threshold,
    }



# def run_autotune(
#     dfs_by_channel, rox_df, df_corr,
#     enable_deconvolution, deconv_target_channel, alpha_value, normalize_to_rox,
#     selected_channels, per_channel_thresholds,
#     groups, replicate_mode, log_y: bool,
#     # NEW:
#     qpos_channel: str, qpos_well: str, target_ct: float = 23.0,
# ):
def run_autotune(
    dfs_by_channel, rox_df, df_corr,
    enable_deconvolution, deconv_target_channel, alpha_value,
    norm_mode, firstcycle_cycle, autofluor_dict,
    selected_channels, per_channel_thresholds,
    groups, replicate_mode, log_y: bool,
    qpos_channel: str, qpos_well: str, target_ct: float = 23.0,
):

    # ---- Search space (same as before; trimmed for readability) ----
    grid = []

    # Average-of-N cycles
    for bs in [3, 4, 5]:
        for bc in [8, 10, 12]:
            grid.append({"method": "avgN", "baseline_start": bs, "baseline_cycles": bc})

    # Homebrew: STD detector
    for win in [5, 7, 9]:
        for ratio in [1.3, 1.5, 1.8]:
            for pre in [2, 3, 4]:
                grid += [
                    {"method": "homebrew", "detector": "std", "det_win": win, "lift_ratio": ratio,
                     "prefit_start": pre, "fit_region": "prefit", "fit_end_pad": 6, "back_offset": 0,
                     "start_point_pad": win//2},
                    {"method": "homebrew", "detector": "std", "det_win": win, "lift_ratio": ratio,
                     "prefit_start": pre, "fit_region": "window", "fit_end_pad": 0, "back_offset": 0,
                     "start_point_pad": win//2},
                    {"method": "homebrew", "detector": "std", "det_win": win, "lift_ratio": ratio,
                     "prefit_start": pre, "fit_region": "window_back", "fit_end_pad": 0, "back_offset": 2,
                     "start_point_pad": win//2},
                ]

    # Homebrew: slope detector
    slope_cands = ([0.01, 0.02, 0.05] if log_y else [30.0, 60.0, 90.0])
    for win in [5, 7, 9]:
        for sm in slope_cands:
            for pre in [2, 3, 4]:
                grid += [
                    {"method": "homebrew", "detector": "slope", "det_win": win, "slope_min": sm,
                     "prefit_start": pre, "fit_region": "prefit", "fit_end_pad": 6, "back_offset": 0,
                     "start_point_pad": win//2},
                    {"method": "homebrew", "detector": "slope", "det_win": win, "slope_min": sm,
                     "prefit_start": pre, "fit_region": "window", "fit_end_pad": 0, "back_offset": 0,
                     "start_point_pad": win//2},
                    {"method": "homebrew", "detector": "slope", "det_win": win, "slope_min": sm,
                     "prefit_start": pre, "fit_region": "window_back", "fit_end_pad": 0, "back_offset": 2,
                     "start_point_pad": win//2},
                ]

    rows = []
    for cfg in grid:
        # res = score_config(
        #     dfs_by_channel, rox_df, df_corr,
        #     enable_deconvolution, deconv_target_channel, alpha_value, normalize_to_rox,
        #     selected_channels, per_channel_thresholds,
        #     groups, replicate_mode, cfg,
        #     # NEW:
        #     qpos_channel=qpos_channel, qpos_well=qpos_well, target_ct=target_ct
        # )
        res = score_config(
            dfs_by_channel, rox_df, df_corr,
            enable_deconvolution, deconv_target_channel, alpha_value,
            norm_mode, firstcycle_cycle, autofluor_dict,
            selected_channels, per_channel_thresholds,
            groups, replicate_mode, cfg,
            qpos_channel=qpos_channel, qpos_well=qpos_well, target_ct=target_ct
        )
        med, p75, neg_cov = res["score"]
        rows.append({
            "method": cfg["method"],
            "detector": cfg.get("detector", "avgN"),
            "config": cfg,
            "median_STD": med,
            "p75_STD": p75,
            "coverage": -neg_cov,
            "qPOS_threshold": res.get("qpos_threshold", None),
        })

    df_res = pd.DataFrame(rows).sort_values(
        ["median_STD", "p75_STD", "coverage"], ascending=[True, True, False]
    ).reset_index(drop=True)

    best_cfg = df_res.iloc[0]["config"] if not df_res.empty else None
    best_thr = df_res.iloc[0]["qPOS_threshold"] if not df_res.empty else None
    return df_res, best_cfg, best_thr

# def _qpos_threshold_from_trace(x, y_corr, target_ct=23.0, startpoint=10):
#     """
#     Return the fluorescence threshold that makes Ct(target_ct) for the given corrected trace.
#     Try 4PL fit -> y(target_ct); fallback to linear interpolation on raw (x,y_corr).
#     """
#     x = np.asarray(x, dtype=float)
#     y_corr = np.asarray(y_corr, dtype=float)
#     try:
#         mask = x >= startpoint
#         if np.count_nonzero(mask) < 5:
#             # too few points to fit a 4PL: interpolate
#             return float(np.interp(target_ct, x, y_corr))
#         popt, _ = curve_fit(four_param_logistic, x[mask], y_corr[mask], maxfev=10000)
#         y_at_target = four_param_logistic(float(target_ct), *popt)
#         return float(y_at_target)
#     except Exception:
#         return float(np.interp(target_ct, x, y_corr))
def _qpos_threshold_from_trace(x, y_corr, target_ct=23.0, startpoint=10):
    x = np.asarray(x, dtype=float)
    y_corr = np.asarray(y_corr, dtype=float)
    try:
        mask = x >= startpoint
        if np.count_nonzero(mask) < 5:
            thr = float(np.interp(target_ct, x, y_corr))
        else:
            popt, _ = curve_fit(four_param_logistic, x[mask], y_corr[mask], maxfev=10000)
            thr = float(four_param_logistic(float(target_ct), *popt))
    except Exception:
        thr = float(np.interp(target_ct, x, y_corr))

    # --- guard rail: keep threshold above early baseline -------------------
    pre = x < startpoint
    if np.any(pre):
        early_max = float(np.nanmax(y_corr[pre]))
        thr = max(thr, early_max + 1e-9)   # tiny margin  # <--
    return thr


# === Here we GO! ===




# timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
version = "v1.2.1"

st.set_page_config(layout="wide")
st.title("qPCR Viewer - Supports Bio-Rad")
st.markdown(f"**Version:** {version}")
# st.markdown(f"**Last updated:** {timestamp}")
st.write("Contact JIACHONG CHU for questions.")

# Choose plate type
plate_type = st.radio("Select Plate Type", ["96-well", "384-well"])
if plate_type == "96-well":
    rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cols = list(range(1, 13))
else:
    rows = [chr(i) for i in range(ord("A"), ord("P") + 1)]
    cols = list(range(1, 25))

well_names = [f"{r}{c}" for r in rows for c in cols]

# Choose platform
platform = st.radio("Select qPCR Platform", ["Bio-Rad"], index=0)

# Upload files
uploaded_files = []
uploaded_files = st.file_uploader("Upload Bio-Rad CSVs (1 per channel)", type=["csv"], accept_multiple_files=True)

# Group state
if "groups" not in st.session_state:
    st.session_state["groups"] = {}

# ==== choose groups ====

# ---------- Replicates controls ----------
st.subheader("Replicates (optional)")
use_replicates = st.toggle(
    "Enable Replicate Selection",
    value=False,
    help="Auto-select paired wells when you click a well."
)
replicate_mode = st.selectbox(
    "Replicate Pattern",
    [
        "Left-Right (paired across halves)",
        "Top-Down (paired across halves)",
        "Neighbors (horizontal pair)",
        "Neighbors (vertical pair)",
        "Custom (paired)"
    ],
    disabled=not use_replicates,
    help="LR: A1↔A7 (96) / A1↔A13 (384); TD: A1↔E1 (96) / A1↔I1 (384); Neighbors: A1↔A2 or A1↔B1; Custom: define your own pairs."
)

# ---------- Helpers ----------
def _safe_key(s: str) -> str:
    k = re.sub(r'[^A-Za-z0-9_]+', '_', s).strip('_')
    return k or "Group_1"

nrows, ncols = len(rows), len(cols)
row_to_idx = {r: i for i, r in enumerate(rows)}
idx_to_row = {i: r for i, r in enumerate(rows)}
col_min, col_max = min(cols), max(cols)
well_names = [f"{r}{c}" for r in rows for c in cols]

# Initialize global storage for custom replicate pairs
if "replicate_pairs" not in st.session_state:
    st.session_state["replicate_pairs"] = []          # list of (a,b)
if "replicate_map" not in st.session_state:
    st.session_state["replicate_map"] = {}            # both directions

def _lr_pair(well: str):
    """Left-Right pair across plate halves on the same row (e.g., A1↔A7 for 96)."""
    r = well[0]
    c = int(well[1:])
    half = ncols // 2
    partner_c = c + half if c <= half else c - half
    return [f"{r}{partner_c}"] if 1 <= partner_c <= ncols else []

def _td_pair(well: str):
    """Top-Down pair across plate halves on the same column (e.g., A1↔E1 for 96)."""
    r = well[0]
    c = int(well[1:])
    ri = row_to_idx[r]
    half = nrows // 2
    partner_ri = ri + half if ri < half else ri - half
    return [f"{idx_to_row[partner_ri]}{c}"] if 0 <= partner_ri < nrows else []

def _neighbors_h_pair(well: str):
    """Neighbors horizontal PAIR: (1↔2), (3↔4), ... within the same row."""
    r = well[0]
    c = int(well[1:])
    partner_c = c + 1 if (c % 2 == 1) else c - 1
    return [f"{r}{partner_c}"] if 1 <= partner_c <= ncols else []

def _neighbors_v_pair(well: str):
    """Neighbors vertical PAIR: (A↔B), (C↔D), ... within the same column."""
    r = well[0]
    c = int(well[1:])
    ri = row_to_idx[r]
    partner_ri = ri + 1 if (ri % 2 == 0) else ri - 1
    return [f"{idx_to_row[partner_ri]}{c}"] if 0 <= partner_ri < nrows else []

def _custom_pair_partner(well: str):
    """Return custom partner if defined."""
    return [st.session_state["replicate_map"][well]] if well in st.session_state["replicate_map"] else []

def replicate_partners(well: str):
    """Return partner(s) based on replicate mode."""
    if not use_replicates:
        return []
    if replicate_mode.startswith("Left-Right"):
        return _lr_pair(well)
    if replicate_mode.startswith("Top-Down"):
        return _td_pair(well)
    if replicate_mode.startswith("Neighbors (horizontal"):
        return _neighbors_h_pair(well)
    if replicate_mode.startswith("Neighbors (vertical"):
        return _neighbors_v_pair(well)
    # Custom (paired)
    return _custom_pair_partner(well)

# ---------- Custom replicate pair builder ----------
# if use_replicates and replicate_mode == "Custom (paired)":
#     st.markdown("**Define custom replicate pairs**")
#     # Only allow wells that aren't already in a pair to be chosen as 'first'
#     already_paired = set(st.session_state["replicate_map"].keys())
#     available_first = ["— choose —"] + [w for w in well_names if w not in already_paired]
#     first = st.selectbox("1) Select first well", available_first, key="custom_rep_first")

#     # Partner list excludes the first choice and wells already paired
#     if first != "— choose —":
#         available_partner = ["— choose —"] + [w for w in well_names if (w != first and w not in already_paired)]
#     else:
#         available_partner = ["— choose —"]
#     second = st.selectbox("2) Select its replicate", available_partner, key="custom_rep_second")

#     colA, colB = st.columns([1,1])
#     def _add_pair(a, b):
#         if a == "— choose —" or b == "— choose —":
#             st.warning("Pick both wells before adding.")
#             return
#         if a == b:
#             st.warning("A replicate cannot be paired with itself.")
#             return
#         if a in st.session_state["replicate_map"] or b in st.session_state["replicate_map"]:
#             st.warning("One or both wells are already paired. Remove the old pair first.")
#             return
#         # store tuple (ordered for consistency)
#         tup = (a, b) if a < b else (b, a)
#         st.session_state["replicate_pairs"].append(tup)
#         # and store both directions for quick lookup
#         st.session_state["replicate_map"][a] = b
#         st.session_state["replicate_map"][b] = a
#         st.success(f"Added pair: {tup[0]} ↔ {tup[1]}")
#         # reset selectors for next pair
#         st.session_state["custom_rep_first"] = "— choose —"
#         st.session_state["custom_rep_second"] = "— choose —"
#         st.rerun()

# ---------- Custom replicate pair builder (click-on-plate) ----------

if use_replicates and replicate_mode == "Custom (paired)":
    st.markdown("**Define custom replicate pairs by clicking directly on the plate**")

    # Ensure state exists
    if "replicate_pairs" not in st.session_state:
        st.session_state["replicate_pairs"] = []
    if "replicate_map" not in st.session_state:
        st.session_state["replicate_map"] = {}
    if "pair_buffer" not in st.session_state:
        st.session_state["pair_buffer"] = None          # holds first click
    if "pair_on_plate" not in st.session_state:
        st.session_state["pair_on_plate"] = True        # allow disabling if needed

    ui_left, ui_right = st.columns([4, 1])

    with ui_right:
        st.toggle("Pair on plate map", key="pair_on_plate", value=st.session_state["pair_on_plate"])
        if st.button("Clear current pick"):
            st.session_state["pair_buffer"] = None
            st.rerun()

        if st.session_state["replicate_pairs"]:
            display_pairs = [f"{a} ↔ {b}" for (a, b) in st.session_state["replicate_pairs"]]
            to_remove = st.selectbox("Remove a saved pair", ["— none —"] + display_pairs, key="custom_rep_remove2")
            if st.button("Remove Selected Pair"):
                if to_remove != "— none —":
                    a, b = [x.strip() for x in to_remove.split("↔")]
                    ordered = (a, b) if a < b else (b, a)
                    try:
                        st.session_state["replicate_pairs"].remove(ordered)
                    except ValueError:
                        pass
                    st.session_state["replicate_map"].pop(a, None)
                    st.session_state["replicate_map"].pop(b, None)
                    st.success(f"Removed pair: {a} ↔ {b}")
                    st.rerun()

        if st.button("Clear ALL pairs"):
            st.session_state["replicate_pairs"].clear()
            st.session_state["replicate_map"].clear()
            st.session_state["pair_buffer"] = None
            st.rerun()

        st.caption("Legend:")
        st.write("• Click one well, then another to create a pair.")
        st.write("• ✓ = already paired")
        st.write("• ● = currently selected (first click)")

    with ui_left:
        # clickable plate (separate from selection checkboxes)
        for r in rows:
            row_cols = st.columns(len(cols))
            for c, col in zip(cols, row_cols):
                w = f"{r}{c}"
                already = (w in st.session_state["replicate_map"])
                picked  = (st.session_state["pair_buffer"] == w)
                label_prefix = "✓ " if already else ("● " if picked else "")
                clicked = col.button(
                    f"{label_prefix}{w}",
                    key=f"pairbtn_{w}",
                    use_container_width=True,
                    disabled=(not st.session_state["pair_on_plate"]) or (already and not picked)
                )
                if clicked:
                    buf = st.session_state["pair_buffer"]

                    # first click (arm) or unselect if same cell
                    if buf is None or buf == w:
                        st.session_state["pair_buffer"] = None if buf == w else w
                        st.rerun()
                    else:
                        a, b = sorted((buf, w))
                        if a == b:
                            st.session_state["pair_buffer"] = None
                            st.warning("A replicate cannot be paired with itself.")
                            st.rerun()

                        if (a in st.session_state["replicate_map"]) or (b in st.session_state["replicate_map"]):
                            st.warning("One or both wells are already paired. Remove the old pair first.")
                        else:
                            st.session_state["replicate_pairs"].append((a, b))
                            st.session_state["replicate_map"][a] = b
                            st.session_state["replicate_map"][b] = a
                            st.session_state["pair_buffer"] = None
                            st.success(f"Added pair: {a} ↔ {b}")
                            st.rerun()

    # summary
    if st.session_state["replicate_pairs"]:
        st.caption("Current replicate pairs (used for STD later):")
        st.write(", ".join([f"{a}↔{b}" for (a, b) in st.session_state["replicate_pairs"]]))
    else:
        st.caption("No pairs added yet.")


# ---------- Group assignment ----------
st.subheader("Step 1: Assign Wells to a Group")
group_name = st.text_input("Group Name", "Group 1")
safe_group_key = _safe_key(group_name)

# --- Initialize per-well checkbox state once per group key ---
for w in well_names:
    k = f"{safe_group_key}_{w}"
    if k not in st.session_state:
        st.session_state[k] = False


preset_colors = {
    "Red": "#FF0000", "Green": "#28A745", "Blue": "#007BFF", "Orange": "#FD7E14",
    "Purple": "#6F42C1", "Brown": "#8B4513", "Black": "#000000", "Gray": "#6C757D", "Custom HEX": None
}
selected_color_name = st.selectbox("Select Group Color", list(preset_colors.keys()))
group_color = st.color_picker("Pick a Custom Color", "#FF0000") if selected_color_name == "Custom HEX" else preset_colors[selected_color_name]

if "groups" not in st.session_state:
    st.session_state["groups"] = {}
if "__suppress_rep_cb__" not in st.session_state:
    st.session_state["__suppress_rep_cb__"] = False

# # ---------- Quick select ----------
st.write("Quick Select:")
col1, col2 = st.columns(2)
selected_row = col1.selectbox("Select Entire Row", ["None"] + rows)
selected_col = col2.selectbox("Select Entire Column", ["None"] + [str(c) for c in cols])
select_all = st.checkbox("Select All Wells")

# ---------- Replicate callback ----------
def _on_checkbox_change(well: str):
    """Mirror the clicked well's state to its replicate partner(s) per current mode."""
    if st.session_state["__suppress_rep_cb__"] or not use_replicates:
        return
    key = f"{safe_group_key}_{well}"
    state = bool(st.session_state.get(key, False))
    st.session_state["__suppress_rep_cb__"] = True
    try:
        for partner in replicate_partners(well):
            pkey = f"{safe_group_key}_{partner}"
            if pkey in st.session_state:  # guard to avoid Streamlit exceptions
                st.session_state[pkey] = state
    finally:
        st.session_state["__suppress_rep_cb__"] = False
        
# --- Grey-out mask for replicate modes ---
greyed_wells = set()
active_wells = set(well_names)

if use_replicates:
    half_cols = ncols // 2
    half_rows = nrows // 2

    if replicate_mode.startswith("Left-Right"):
        greyed_wells = {f"{r}{c}" for r in rows for c in cols if c > half_cols}
        active_wells = {f"{r}{c}" for r in rows for c in cols if c <= half_cols}

    elif replicate_mode.startswith("Top-Down"):
        greyed_wells = {f"{rows[i]}{c}" for i in range(half_rows, nrows) for c in cols}
        active_wells = {f"{rows[i]}{c}" for i in range(0, half_rows) for c in cols}

    elif replicate_mode.startswith("Neighbors (horizontal"):
        # grey EVEN columns (2,4,6,...) ; click ODD (1,3,5,...)
        greyed_wells = {f"{r}{c}" for r in rows for c in cols if (c % 2) == 0}
        active_wells = {f"{r}{c}" for r in rows for c in cols if (c % 2) == 1}

    elif replicate_mode.startswith("Neighbors (vertical"):
        # grey EVEN rows (B,D,F,...) ; click ODD rows (A,C,E,...)
        greyed_wells = {f"{rows[i]}{c}" for i in range(1, nrows, 2) for c in cols}
        active_wells = {f"{rows[i]}{c}" for i in range(0, nrows, 2) for c in cols}

    else:
        greyed_wells = set()
        active_wells = set(well_names)




# --- Bulk apply selection to session_state (and mirror to replicates) ---
def _apply_bulk_selection(target_wells, state=True):
    st.session_state["__suppress_rep_cb__"] = True
    changed = False
    try:
        for w in target_wells:
            k = f"{safe_group_key}_{w}"
            if k not in st.session_state:
                st.session_state[k] = False
            if st.session_state[k] != state:
                st.session_state[k] = state
                changed = True
            if use_replicates:
                for p in replicate_partners(w):
                    pk = f"{safe_group_key}_{p}"
                    if pk not in st.session_state:
                        st.session_state[pk] = False
                    if st.session_state[pk] != state:
                        st.session_state[pk] = state
                        changed = True
    finally:
        st.session_state["__suppress_rep_cb__"] = False
    if changed:
        st.rerun()

# --- Apply "Select All" to the clickable side (replicate mirroring fills the grey side) ---
if select_all:
    _apply_bulk_selection(active_wells, True)

# --- Apply quick row/column selection (restricted to clickable side) ---
bulk_targets = set()
if selected_row != "None":
    bulk_targets |= {f"{selected_row}{c}" for c in cols}
if selected_col != "None":
    bulk_targets |= {f"{r}{selected_col}" for r in rows}
bulk_targets &= active_wells
if bulk_targets:
    _apply_bulk_selection(bulk_targets, True)

# (Optional) Clear helpers
col_clear1, col_clear2 = st.columns(2)
with col_clear1:
    if st.button("Clear Active Half"):
        _apply_bulk_selection(active_wells, False)
with col_clear2:
    if st.button("Clear All Wells"):
        _apply_bulk_selection(set(well_names), False)

    
# ---------- Manual well selection grid ----------

st.write("Select Wells (click checkboxes):")
for r in rows:
    cols_container = st.columns(len(cols))
    for c, col in zip(cols, cols_container):
        well = f"{r}{c}"
        key = f"{safe_group_key}_{well}"
        disabled_cell = use_replicates and (well in greyed_wells)
        if key not in st.session_state:
            st.session_state[key] = False
        col.checkbox(
            well,
            key=key,
            disabled=disabled_cell,
            on_change=_on_checkbox_change,
            args=(well,)
        )
        

# ---------- Build selection ----------
selected_wells = [
    f"{r}{c}"
    for r in rows for c in cols
    if st.session_state.get(f"{safe_group_key}_{r}{c}", False)
]
selected_wells = sorted(set(selected_wells), key=lambda x: (x[0], int(x[1:])))

# ---------- Add / show / delete groups ----------
if st.button("Add Group"):
    if group_name and selected_wells:
        st.session_state["groups"][group_name] = {"color": group_color, "wells": selected_wells}
        st.success(f"Added {group_name}: {len(selected_wells)} wells")

st.subheader("Current Groups")
for group, info in st.session_state["groups"].items():
    st.markdown(f"**{group}** ({info['color']}): {', '.join(info['wells'])}")

st.subheader("Delete a Group")
if st.session_state["groups"]:
    group_to_delete = st.selectbox("Select Group to Delete", list(st.session_state["groups"].keys()))
    if st.button("Delete Group"):
        st.session_state["groups"].pop(group_to_delete, None)
        st.success(f"Deleted group: {group_to_delete}")
else:
    st.info("No groups available to delete.")




# ==== plot ====
channel_name_map = {
    "FAM": "FAM",
    "HEX": "HEX",
    "Cy5": "Cy5",
    "Cy5.5": "Cy5-5",
    "ROX": "ROX",
    "SYBR": "SYBR"
}

channel_styles = [
    {"dash": "solid",     "symbol": None},
    {"dash": "dash",      "symbol": None},
    {"dash": "solid",     "symbol": "triangle-up"},
    {"dash": "solid",     "symbol": "square"},
    {"dash": "solid",     "symbol": "x"}
]


st.sidebar.subheader("Plot Color Scheme")
color_mode = st.sidebar.radio("Color Mode", ["Solid", "Gradient", "Colormap"])

colormap_name = None
if color_mode == "Colormap":
    available_colormaps = sorted(m for m in plt.colormaps() if not m.endswith("_r"))  # Optional: filter out reversed
    colormap_name = st.sidebar.selectbox(
        "Select a Colormap", ["jet", "viridis", "plasma", "cividis", "cool", "hot", "spring", "summer", "winter"]
    )
    

channel_options = ["FAM", "HEX", "Cy5", "Cy5.5", "ROX", "SYBR"]
default_channels = ["FAM"]

# ---- Safe defaults (prevent NameError when deconvolution is off) ----
deconv_target_channel = "Cy5"      # ignored if enable_deconvolution is False
deconv_correction_channel = "Cy5.5"
alpha_value = 0.7


st.sidebar.subheader("Deconvolution Settings (Bio-Rad only)")
enable_deconvolution = st.sidebar.checkbox("Enable Deconvolution for Bio-Rad")
if enable_deconvolution:
    deconv_target_channel = st.sidebar.selectbox("Channel to Deconvolve", channel_options, index=2)   # e.g. Cy5
    deconv_correction_channel = st.sidebar.selectbox("Correction Channel", channel_options, index=3)   # e.g. Cy5.5
    alpha_value = st.sidebar.number_input("Alpha Multiplier (α)", min_value=-10.0, max_value=10.0, value=0.07, step=0.01)

    
selected_channels = st.sidebar.multiselect("Select Channels to Plot", channel_options, default=default_channels)


# normalize_to_rox = st.sidebar.checkbox("Normalize fluorescence to ROX channel")

st.sidebar.subheader("Normalization")
norm_mode = st.sidebar.radio(
    "Normalization mode",
    ["None", "ROX channel", "First-cycle formula"],
    index=0,
    help="Pick one normalization. ROX divides by ROX trace; First-cycle uses (Fn - F1)/(F1 - autofluorescence)."
)

# First-cycle options
if "autofluor_dict" not in st.session_state:
    st.session_state["autofluor_dict"] = {"FAM": 2800.0, "HEX": 0.0, "Cy5": 0.0, "Cy5.5": 0.0, "ROX": 0.0, "SYBR": 0.0}
if "firstcycle_cycle" not in st.session_state:
    st.session_state["firstcycle_cycle"] = 1

if norm_mode == "First-cycle formula":
    st.sidebar.markdown("**First-cycle parameters**")
    st.session_state["firstcycle_cycle"] = st.sidebar.number_input(
        "Cycle to treat as F1", min_value=1, max_value=5, value=int(st.session_state["firstcycle_cycle"]), step=1
    )
    st.sidebar.markdown("**Autofluorescence (per channel)**")
    for ch in channel_options:
        # show controls only for channels you're plotting or loaded; keeping it simple: show for all
        default_val = float(st.session_state["autofluor_dict"].get(ch, 0.0))
        new_val = st.sidebar.number_input(f"{ch} autofluor", min_value=0.0, value=default_val, step=100.0)
        st.session_state["autofluor_dict"][ch] = float(new_val)

# For convenience in later code
firstcycle_cycle = int(st.session_state["firstcycle_cycle"])
autofluor_dict   = dict(st.session_state["autofluor_dict"])




# --- Always-available thresholds container (so Auto Tune has something to use) ---
DEFAULT_THR = 0.13
if "per_channel_thresholds" not in st.session_state:
    st.session_state["per_channel_thresholds"] = {ch: DEFAULT_THR for ch in channel_options}

per_channel_thresholds = st.session_state["per_channel_thresholds"]

# --- Build dfs_by_channel + rox_df (+ df_corr if deconvolution) for Auto Tune & plotting ---
dfs_by_channel = {}
rox_df = None
df_corr = None  # for deconvolution (Cy5.5 -> Cy5), if enabled

if uploaded_files:
    # Load each channel CSV once
    for ch in channel_options:  # you can switch to selected_channels if you prefer
        match_key = channel_name_map.get(ch, ch)
        f = next((f for f in uploaded_files if match_key.lower() in f.name.lower()), None)
        if f is not None:
            df_tmp = pd.read_csv(f, comment='#')
            df_tmp.columns = df_tmp.columns.str.strip()
            df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains("Unnamed")]
            dfs_by_channel[ch] = df_tmp

    # ROX normalization table (if requested)
    # if normalize_to_rox:
    #     # 1) Prefer the ROX dataframe you already loaded by channel
    #     if "ROX" in dfs_by_channel:
    #         rox_df = dfs_by_channel["ROX"]
    #     else:
    #         # 2) Fallback: looser filename match (handles "PassiveRef", "Reference", etc.)
    #         rox_file = next(
    #             (f for f in uploaded_files
    #              if re.search(r'(?:^|[^a-z])rox(?:[^a-z]|$)|passive[_\s-]*ref|reference', f.name, flags=re.I)),
    #             None
    #         )
    #         if rox_file:
    #             try:
    #                 tmp = pd.read_csv(rox_file, comment="#")
    #                 tmp.columns = tmp.columns.str.strip()
    #                 rox_df = tmp.loc[:, ~tmp.columns.str.contains("Unnamed")]
    #             except pd.errors.EmptyDataError:
    #                 rox_df = None
    # ROX normalization table (only if requested)
    rox_df = None
    if norm_mode == "ROX channel":
        if "ROX" in dfs_by_channel:
            rox_df = dfs_by_channel["ROX"]
        else:
            rox_file = next(
                (f for f in uploaded_files
                 if re.search(r'(?:^|[^a-z])rox(?:[^a-z]|$)|passive[_\s-]*ref|reference', f.name, flags=re.I)),
                None
            )
            if rox_file:
                try:
                    tmp = pd.read_csv(rox_file, comment="#")
                    tmp.columns = tmp.columns.str.strip()
                    rox_df = tmp.loc[:, ~tmp.columns.str.contains("Unnamed")]
                except pd.errors.EmptyDataError:
                    rox_df = None


    # Deconvolution correction table (if enabled)
    if enable_deconvolution:
        corr_key = channel_name_map.get(deconv_correction_channel, deconv_correction_channel)
        corr_file = next((f for f in uploaded_files if corr_key.lower() in f.name.lower()), None)
        if corr_file:
            try:
                df_corr = pd.read_csv(corr_file, comment="#")
                df_corr.columns = df_corr.columns.str.strip()
                df_corr = df_corr.loc[:, ~df_corr.columns.str.contains("Unnamed")]
                if df_corr.drop(columns='Cycle', errors='ignore').sum().sum() == 0:
                    df_corr = None  # all-zero guard
            except pd.errors.EmptyDataError:
                df_corr = None
                
# ---- ONE source of truth for log_y (put this BEFORE Auto Tune) ----
if "log_y" not in st.session_state:
    st.session_state["log_y"] = True  # default

log_y = st.sidebar.toggle(
    "Use Semilog Y-axis (log scale)",
    value=st.session_state["log_y"],
    key="log_y"   # keep it in session_state
)





# ---------- Auto Tune (qPOS anchored) ----------
st.sidebar.subheader("Auto Tune (qPOS anchored)")
enable_autotune = st.sidebar.checkbox("Enable Auto Tune", value=False)

if enable_autotune:
    if not uploaded_files:
        st.sidebar.info("Upload CSVs first.")
    elif not st.session_state.get("groups"):
        st.sidebar.info("Create at least one Group with wells.")
    else:
        # Only channels that actually loaded
        loaded_channels = [ch for ch in selected_channels if ch in dfs_by_channel]
        if not loaded_channels:
            st.sidebar.info("Select at least one loaded channel.")
        else:
            qpos_channel = st.sidebar.selectbox("qPOS Channel", loaded_channels)
            qpos_well    = st.sidebar.selectbox("qPOS Well", well_names)
            target_ct    = st.sidebar.number_input("Target Ct (qPOS)", 5.0, 45.0, 23.0, 0.1)

            # >>> SHOW WARNING HERE & USE A SINGLE BUTTON <<<
            if norm_mode == "ROX channel" and rox_df is None:
                st.sidebar.error(
                    "ROX normalization is selected, but no ROX CSV is loaded. "
                    "Rename the file to include 'ROX' (or 'PassiveRef'/'Reference') or switch normalization mode."
                )
            
            run_btn = st.sidebar.button("Run Auto Tune")
            if run_btn:
                df_res, best_cfg, best_thr = run_autotune(
                    dfs_by_channel, rox_df, df_corr,
                    enable_deconvolution, deconv_target_channel, alpha_value,
                    norm_mode, firstcycle_cycle, autofluor_dict,
                    selected_channels, per_channel_thresholds,
                    st.session_state["groups"], replicate_mode, st.session_state["log_y"],
                    qpos_channel=qpos_channel, qpos_well=qpos_well, target_ct=target_ct
                )
            
                scale_label = (
                    "ROX-normalized" if norm_mode == "ROX channel"
                    else "First-cycle normalized" if norm_mode == "First-cycle formula"
                    else "raw RFU"
                )
                df_res["scale"] = scale_label
                # --- NEW: add per-channel Ct thresholds to df_res ---
                thr_cols = {}
                for ch in selected_channels:
                    colname = f"thr_{ch}"
                    if ch == qpos_channel:
                        # per-config threshold solved by Auto Tune
                        thr_cols[colname] = df_res["qPOS_threshold"].astype(float)
                    else:
                        # fixed manual threshold (same for all configs)
                        thr_val = float(per_channel_thresholds.get(ch, DEFAULT_THR))
                        thr_cols[colname] = [thr_val] * len(df_res)
                
                thr_df = pd.DataFrame(thr_cols)
                df_res = pd.concat([df_res, thr_df], axis=1)
                df_res["target_Ct"] = float(target_ct)

            # norm_active = bool(normalize_to_rox and (rox_df is not None))
            # if normalize_to_rox and not norm_active:
            #     st.sidebar.error(
            #         "ROX normalization is ON, but no ROX CSV is loaded. "
            #         "Rename the file to include 'ROX' (or 'PassiveRef'/'Reference') or turn off normalization."
            #     )
                
            # # button
            # run_btn = st.sidebar.button("Run Auto Tune")

            # if run_btn:
            #     df_res, best_cfg, best_thr = run_autotune(
            #         dfs_by_channel, rox_df, df_corr,
            #         enable_deconvolution, deconv_target_channel, alpha_value, normalize_to_rox,
            #         selected_channels, per_channel_thresholds,
            #         st.session_state["groups"], replicate_mode, st.session_state["log_y"],
            #         qpos_channel=qpos_channel, qpos_well=qpos_well, target_ct=target_ct
            #     )


            #     # df_res["scale"] = "ROX-normalized" if norm_active else "raw RFU"
            #     scale_label = (
            #         "ROX-normalized" if norm_mode == "ROX channel"
            #         else "First-cycle normalized" if norm_mode == "First-cycle formula"
            #         else "raw RFU"
            #     )
            #     df_res["scale"] = scale_label


                # ---- Persist a version for the Explorer ----
                df_plot = df_res.copy()
                df_plot["rank"] = np.arange(1, len(df_plot) + 1)
                
                # Expand the 'config' dict into columns the Explorer expects (cfg_*)
                cfg_cols = pd.json_normalize(df_plot["config"]).add_prefix("cfg_")
                df_plot = pd.concat([df_plot.drop(columns=["config"]), cfg_cols], axis=1)
                
                # Keep it for the Explorer + other consumers
                st.session_state["autotune_df"] = df_plot
                st.session_state["autotune_best_config"] = df_plot.iloc[0].to_dict()

            
                if df_res is not None and not df_res.empty:
                    # add 1-based rank
                    df_res = df_res.copy()
                    df_res["rank"] = np.arange(1, len(df_res) + 1)
            
                    # ---- catalog: rank → (config, qpos_threshold, metrics) ----
                    catalog = []
                    for _, row in df_res.iterrows():
                        catalog.append({
                            "rank": int(row["rank"]),
                            "config": row["config"],
                            "qpos_threshold": row.get("qPOS_threshold", None),
                            "median_STD": float(row["median_STD"]),
                            "p75_STD": float(row["p75_STD"]),
                            "coverage": float(row["coverage"]),
                        })
                    st.session_state["autotune_catalog"] = catalog
                    st.session_state["autotune_selected_rank"] = 1  # default to best
                    st.session_state["autotune_qpos_channel"] = qpos_channel
                    st.session_state["autotune_qpos_threshold"] = df_res.iloc[0]["qPOS_threshold"]
                    st.session_state["autotune_target_ct"]      = float(target_ct)
            
                    st.subheader("Auto Tune Results (ranked)")
                    cols_common = ["rank","method","detector","median_STD","p75_STD",
                       "coverage","qPOS_threshold","target_Ct","scale"]
                    cols_thr = [c for c in df_res.columns if c.startswith("thr_")]  # thr_FAM, thr_HEX, ...
                    
                    st.dataframe(df_res[cols_common + cols_thr], use_container_width=True)
                    # st.dataframe(df_res[["rank","method","detector","median_STD","p75_STD","coverage","qPOS_threshold","scale"]])
          
                    st.success(
                        f"Best config → median STD={df_res.iloc[0]['median_STD']:.3f}, "
                        f"p75={df_res.iloc[0]['p75_STD']:.3f}, coverage={df_res.iloc[0]['coverage']:.2f}"
                    )
                else:
                    st.info("Auto Tune produced no results. Check groups, thresholds, or data quality.")
# =========================
# Auto Tune Explorer (Persistent)
# =========================
st.subheader("Auto Tune Explorer (persistent)")
if "autotune_df" in st.session_state and st.session_state["autotune_df"] is not None:
    df_plot = st.session_state["autotune_df"]

    # Nice readable label of the parameters (optional)
    def _param_label(row):
        m = row.get("cfg_method")
        if m == "avgN":
            bs = row.get("cfg_baseline_start")
            bc = row.get("cfg_baseline_cycles")
            return f"avgN: start={bs}, N={bc}"
        det = row.get("cfg_detector")
        win = row.get("cfg_det_win")
        pre = row.get("cfg_prefit_start")
        fit = row.get("cfg_fit_region")
        pad = row.get("cfg_fit_end_pad")
        K   = row.get("cfg_back_offset")
        mark= row.get("cfg_start_point_pad")
        if str(det).lower() == "std":
            ratio = row.get("cfg_lift_ratio")
            tail  = f"ratio={ratio}"
        else:
            slope = row.get("cfg_slope_min")
            tail  = f"slope={slope}"
        return f"{det} | win={win} | pre={pre} | fit={fit} | pad={pad} | K={K} | mark={mark} | {tail}"

    df_plot = df_plot.copy()
    df_plot["params"] = df_plot.apply(_param_label, axis=1)

    # Scatter: median_STD vs p75_STD; color=coverage; symbol=method; facet by detector
    fig_at = px.scatter(
        df_plot,
        x="median_STD",
        y="p75_STD",
        color="coverage",
        symbol="method",           # keep top-level for legend
        facet_col="detector",      # keep top-level for facet
        hover_data=[
            "rank","coverage","qPOS_threshold","method","detector",
            "cfg_det_win","cfg_lift_ratio","cfg_slope_min","cfg_prefit_start",
            "cfg_fit_region","cfg_fit_end_pad","cfg_back_offset","cfg_start_point_pad",
            "cfg_baseline_start","cfg_baseline_cycles","params"
        ],
        title="Auto Tune configurations: replicate dispersion vs. coverage",
    )
    fig_at.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    st.plotly_chart(fig_at, use_container_width=True)

    # Always-visible table of all configs
    st.caption("All Auto Tune candidates (sorted by rank).")
    st.dataframe(
        df_plot.sort_values("rank")[
            [
                "rank","method","detector","median_STD","p75_STD","coverage","qPOS_threshold",
                "cfg_det_win","cfg_lift_ratio","cfg_slope_min","cfg_prefit_start","cfg_fit_region",
                "cfg_fit_end_pad","cfg_back_offset","cfg_start_point_pad","cfg_baseline_start","cfg_baseline_cycles"
            ]
        ],
        use_container_width=True
    )

    # Download
    st.download_button(
        "Download Auto Tune results (CSV)",
        data=df_plot.to_csv(index=False).encode("utf-8"),
        file_name="autotune_results.csv",
        mime="text/csv"
    )

    # Optional: keep or clear results
    cols = st.columns([1,1])
    with cols[0]:
        if st.button("Clear Auto Tune Results"):
            st.session_state.pop("autotune_df", None)
            st.session_state.pop("autotune_best_config", None)
            st.session_state.pop("autotune_qpos_channel", None)
            st.session_state.pop("autotune_qpos_threshold", None)
            st.session_state.pop("autotune_target_ct", None)
            st.rerun()
else:
    st.info("Run Auto Tune to populate this view.")



# ---------- Apply to Plot (manual vs by rank) ----------
st.sidebar.subheader("Apply to Plot")
catalog = st.session_state.get("autotune_catalog", None)

if catalog:
    apply_mode = st.sidebar.radio(
        "Baseline/threshold source",
        ["Manual settings", "Auto Tune (select rank)"],
        index=1 if st.session_state.get("autotune_enabled_for_plot", False) else 0,
        help="Choose manual baseline/thresholds or select an Auto Tune configuration by rank."
    )
    st.session_state["autotune_enabled_for_plot"] = (apply_mode == "Auto Tune (select rank)")

    if st.session_state["autotune_enabled_for_plot"]:
        ranks = [c["rank"] for c in catalog]
        default_idx = max(0, ranks.index(st.session_state.get("autotune_selected_rank", 1))) if ranks else 0
        picked_rank = st.sidebar.selectbox("Use Auto Tune rank", ranks, index=default_idx)

        # remember selection
        st.session_state["autotune_selected_rank"] = picked_rank

        # fetch the chosen config/threshold for plotting
        chosen = next((c for c in catalog if c["rank"] == picked_rank), None)
        st.session_state["autotune_selected_config"] = chosen["config"] if chosen else None
        st.session_state["autotune_selected_threshold"] = chosen["qpos_threshold"] if chosen else None

        # (optional) small summary
        if chosen:
            st.sidebar.caption(
                f"rank {chosen['rank']} • median={chosen['median_STD']:.3f}, "
                f"p75={chosen['p75_STD']:.3f}, cov={chosen['coverage']:.2f}"
            )
else:
    st.sidebar.radio(
        "Baseline/threshold source",
        ["Manual settings"],
        index=0, disabled=True,
        help="Run Auto Tune to enable selection by rank."
    )




# Baseline, Log Y, Threshold
st.sidebar.subheader("Step 3: Baseline Settings")
use_baseline = st.sidebar.toggle("Apply Baseline Subtraction", value=False)


baseline_method = st.sidebar.radio(
    "Baseline Method",
    ["Average of N cycles", "Homebrew Lift-off Fit"],
    index=1
)

if use_baseline and baseline_method == "Average of N cycles":
    baseline_start = st.sidebar.number_input("Baseline start cycle", min_value=3, max_value=15, value=3, step=1)
    baseline_cycles = st.sidebar.number_input("Number of cycles to average", min_value=1, max_value=20, value=10, step=1)

elif use_baseline and baseline_method == "Homebrew Lift-off Fit":
    # 1) Choose detector first
    detector_mode = st.sidebar.radio(
        "Lift-off detector",
        ["STD ratio", "Slope threshold"],
        help="Choose how lift-off is detected."
    )
    detector_flag = "std" if detector_mode == "STD ratio" else "slope"

    # 2) Show the appropriate window length input
    if detector_flag == "std":
        det_win = st.sidebar.number_input(
            "STD window length (cycles)", min_value=3, max_value=25, value=7, step=1,
            help="Rolling window used to compute S[i]."
        )
        lift_ratio = st.sidebar.number_input(
            "Sensitivity ratio (S[i]/S_ref)", min_value=0.1, max_value=5.0, value=1.5, step=0.1
        )
        slope_min = 50.0  # unused in this mode
    else:
        det_win = st.sidebar.number_input(
            "Slope window length (cycles)", min_value=3, max_value=25, value=7, step=1,
            help="Window over which the linear slope is estimated."
        )
        slope_min = st.sidebar.number_input(
            "Slope threshold (|Δy| per cycle)", min_value=0.0, max_value=1e6, value=50.0, step=1.0,
            help="Units match your Y-axis: RFU/cycle (linear) or log10(RFU)/cycle (semilog)."
        )
        lift_ratio = 1.5  # unused in this mode

    # Common knobs
    prefit_start_cycle = st.sidebar.number_input(
        "Prefit start cycle (1-based)", min_value=1, max_value=40, value=3, step=1
    )
    fit_region_label = st.sidebar.radio(
        "Baseline fit region",
        ["From prefit start", "Detection window", "Window slid back by K"]
    )
    if fit_region_label == "From prefit start":
        fit_region = "prefit"
        fit_end_pad = st.sidebar.number_input("Fit end offset (i + ...)", -30, 30, 6, 1)
        back_offset = 0
    elif fit_region_label == "Detection window":
        fit_region, fit_end_pad, back_offset = "window", 0, 0
    else:
        fit_region = "window_back"
        back_offset = st.sidebar.number_input("K (slide window back by K)", 0, 50, 2, 1)
        fit_end_pad = 0

    start_point_pad = st.sidebar.number_input(
        "Lift-off mark offset (i + ...)", min_value=-30, max_value=30, value=max(0, det_win//2), step=1
    )
    




# log_y = st.sidebar.toggle("Use Semilog Y-axis (log scale)")

threshold_enabled = st.sidebar.checkbox("Enable Threshold & Ct Calculation")

# Make sure per_channel_thresholds exists even if box is off
per_channel_thresholds = st.session_state.get("per_channel_thresholds", {ch: DEFAULT_THR for ch in channel_options})

if threshold_enabled:
    st.sidebar.markdown("**Per-Channel Thresholds:**")
    for ch in selected_channels:
        val = st.sidebar.number_input(
            f"Threshold for {ch}",
            min_value=0.0,
            value=float(per_channel_thresholds.get(ch, DEFAULT_THR)),
            step=0.01,
            key=f"threshold_{ch}"
        )
        # persist
        st.session_state["per_channel_thresholds"][ch] = val

# keep a local alias up-to-date
per_channel_thresholds = st.session_state["per_channel_thresholds"]








## ===== now let's do some business! ===

ct_results = []

# Plotting
if uploaded_files and st.sidebar.button("Plot Curves"):
    fig = go.Figure()

    # Apply Auto Tune best config/threshold?
    use_auto = st.session_state.get("autotune_enabled_for_plot", False)
    sel_cfg  = st.session_state.get("autotune_selected_config")  # may be None
    sel_thr  = st.session_state.get("autotune_selected_threshold")  # may be None
    qpos_ch  = st.session_state.get("autotune_qpos_channel")  # used for threshold override



    for i, channel_name in enumerate(selected_channels):
        chan_str = channel_name 
        match_key = channel_name_map.get(channel_name, channel_name.lower())
        df = dfs_by_channel.get(channel_name)
        if df is None:
            st.warning(f"No data loaded for channel: {channel_name}")
            continue

        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]

         # ======= DECONVOLUTION: Load correction file ONCE here =======
        df_corr = None
        if enable_deconvolution and channel_name == deconv_target_channel:
            match_key_corr = channel_name_map.get(deconv_correction_channel, deconv_correction_channel.lower())
            corr_file = next((f for f in uploaded_files if match_key_corr.lower() in f.name.lower()), None)
            if corr_file:
                try:
                    df_corr = pd.read_csv(corr_file, comment='#')
                    df_corr.columns = df_corr.columns.str.strip()
                    df_corr = df_corr.loc[:, ~df_corr.columns.str.contains("Unnamed")]
    
                    if df_corr.empty:
                        st.warning(f"Correction file '{corr_file.name}' is empty. Deconvolution skipped for channel {channel_name}.")
                        df_corr = None
                    elif (df_corr.drop(columns='Cycle', errors='ignore').sum().sum() == 0):
                        st.warning(f"Correction file '{corr_file.name}' has all-zero data. Deconvolution skipped for channel {channel_name}.")
                        df_corr = None
                except pd.errors.EmptyDataError:
                    st.warning(f"Correction file '{corr_file.name}' is empty or invalid. Deconvolution skipped for channel {channel_name}.")
                    df_corr = None
            else:
                st.warning(f"Correction channel file not found: {deconv_correction_channel}")

        for group, info in st.session_state["groups"].items():
            wells = info["wells"]
            base_color = info["color"]

            if color_mode == "Solid":
                color_list = [base_color] * len(wells)
            elif color_mode == "Gradient":
                gradient = mcolors.LinearSegmentedColormap.from_list("gradient", [
                    tuple(1 - 0.5 * (1 - c) for c in mcolors.to_rgb(base_color)), mcolors.to_rgb(base_color)
                ])
                color_list = [gradient(i / max(1, len(wells) - 1)) for i in range(len(wells))]
            elif color_mode == "Colormap" and colormap_name:
                cmap = plt.get_cmap(colormap_name)
                color_list = [mcolors.to_hex(cmap(i / max(1, len(wells) - 1))) for i in range(len(wells))]
            else:
                color_list = [base_color] * len(wells)
            
            # ======= For each well =======
            for well, color in zip(wells, color_list):
                if well in df.columns:
                    # Unified preprocessing (deconvolution + normalization)
                    y = _preprocess_trace(
                        df, well, channel_name, rox_df,
                        enable_deconvolution, deconv_target_channel, df_corr, alpha_value,
                        norm_mode, firstcycle_cycle, autofluor_dict
                    )
            
                    # --- Baseline with Auto Tune override if selected ---
                    if use_auto and sel_cfg is not None:
                        cfg = sel_cfg
                        if cfg.get("method") == "avgN":
                            bs = int(cfg["baseline_start"])
                            bc = int(cfg["baseline_cycles"])
                            baseline = y.iloc[bs - 1: bs - 1 + bc].mean()
                            y = y - baseline
                        else:
                            y, _ = spr_qpcr_background_correction(
                                np.array(y),
                                std_win=int(cfg["det_win"]),
                                prefit_start=int(cfg["prefit_start"]) - 1,
                                ratio=float(cfg.get("lift_ratio", 1.5)),
                                fit_end_pad=int(cfg.get("fit_end_pad", 6)),
                                start_point_pad=int(cfg.get("start_point_pad", max(0, int(cfg.get("det_win", 7)) // 2))),
                                detector=str(cfg.get("detector", "std")),
                                slope_min=float(cfg.get("slope_min", 50.0)),
                                fit_region=str(cfg.get("fit_region", "prefit")),
                                back_offset=int(cfg.get("back_offset", 0)),
                            )
                    else:
                        # manual baseline logic
                        if use_baseline:
                            if baseline_method == "Average of N cycles":
                                baseline = y.iloc[baseline_start - 1: baseline_start - 1 + baseline_cycles].mean()
                                y = y - baseline
                            elif baseline_method == "Homebrew Lift-off Fit":
                                y, start_point_idx = spr_qpcr_background_correction(
                                    np.array(y),
                                    std_win=int(det_win),
                                    prefit_start=int(prefit_start_cycle) - 1,
                                    ratio=float(lift_ratio),
                                    fit_end_pad=int(fit_end_pad),
                                    start_point_pad=int(start_point_pad),
                                    detector=detector_flag,
                                    slope_min=float(slope_min),
                                    fit_region=fit_region,
                                    back_offset=int(back_offset),
                                )
            
                    # --- draw the curve ---
                    x = df["Cycle"].values
                    y_plot = np.asarray(y, dtype=float)
                    if log_y:
                        y_plot = np.where(y_plot > 0, y_plot, np.nan)
            
                    style = channel_styles[i % len(channel_styles)]
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_plot,
                            mode="lines+markers" if style["symbol"] else "lines",
                            name=f"{group}: {well} ({channel_name})",
                            line=dict(color=mcolors.to_hex(color), dash=style["dash"]),
                            marker=dict(symbol=style["symbol"], size=6) if style["symbol"] else None,
                        )
                    )
            
                    if threshold_enabled:
                        channel_threshold = per_channel_thresholds.get(chan_str, 1000.0)
                        if use_auto and (qpos_ch == channel_name) and (sel_thr is not None):
                            channel_threshold = float(sel_thr)
            
                        try:
                            ct_value, ct_std = calculate_ct(
                                x, y,
                                threshold=channel_threshold,
                                startpoint=10, use_4pl=True, return_std=True
                            )
                            if ct_value is not None:
                                ct_results.append({"Group": group, "Well": well, "Channel": channel_name, "Ct": f"{ct_value:.2f}"})
                            else:
                                ct_results.append({"Group": group, "Well": well, "Channel": channel_name, "Ct": "Undetermined"})
                        except Exception:
                            ct_results.append({"Group": group, "Well": well, "Channel": channel_name, "Ct": "Undetermined"})



    if threshold_enabled:
        for ch in selected_channels:
            channel_threshold = per_channel_thresholds.get(ch, 1000.0)  # fallback default
            fig.add_hline(y=channel_threshold, line_dash="dot", line_color="gray",
                          annotation_text=f"{ch} Threshold", annotation_position="top right")
    
    fig.update_layout(
        title="Amplification Curves",
        xaxis_title="Cycle",
        yaxis_title="log₁₀(RFU)" if log_y else "RFU",
        yaxis_type="log" if log_y else "linear",
        legend=dict(font=dict(size=8),orientation = "v",x= 1.02, y = 1, xanchor ="left",yanchor = "top" ),
        width=800,          # width in pixels
        height=600          # height in pixels (6:8 ratio)
        )

    st.plotly_chart(fig, use_container_width=False)
    

    
    if ct_results:
        st.subheader("Ct Values")
        ct_df = pd.DataFrame(ct_results)
        st.dataframe(ct_df)
    


        # ----- Replicate STD block -----
        if use_replicates:
            st.subheader("Replicate Ct STD (paired)")

            # Make a numeric copy of Ct values
            ct_df_num = ct_df.copy()
            ct_df_num["Ct_num"] = pd.to_numeric(ct_df_num["Ct"], errors="coerce")

            rep_rows = []

            # Helper: get replicate pairs for a group given current mode
            def _pairs_for_group(wells_in_group: set):
                pair_set = set()
                if replicate_mode == "Custom (paired)":
                    for a, b in st.session_state.get("replicate_pairs", []):
                        if a in wells_in_group and b in wells_in_group:
                            pair_set.add(tuple(sorted((a, b))))
                else:
                    # Derive pairs from the selected replicate pattern
                    for w in wells_in_group:
                        for p in replicate_partners(w):
                            if p in wells_in_group:
                                pair_set.add(tuple(sorted((w, p))))
                return sorted(pair_set)

            # Build rows: one line per (Group, Channel, pair)
            for group, info in st.session_state["groups"].items():
                wells_in_group = set(info["wells"])
                pair_list = _pairs_for_group(wells_in_group)
                if not pair_list:
                    continue

                # Iterate channels present in Ct table
                for ch in ct_df_num["Channel"].unique():
                    sub = ct_df_num[(ct_df_num["Group"] == group) & (ct_df_num["Channel"] == ch)]
                    # Map: well -> Ct_num
                    ct_map = {row["Well"]: row["Ct_num"] for _, row in sub.iterrows()}

                    for a, b in pair_list:
                        vals = [ct_map.get(a, np.nan), ct_map.get(b, np.nan)]
                        vals = [v for v in vals if not pd.isna(v)]
                        if len(vals) >= 2:
                            mean_ct = float(np.mean(vals))
                            std_ct  = float(np.std(vals, ddof=1))  # sample STD; for 2 reps = |Δ|/√2
                            rep_rows.append({
                                "Group": group,
                                "Channel": ch,
                                "Pair": f"{a} ↔ {b}",
                                "n": len(vals),
                                "Mean Ct": round(mean_ct, 2),
                                "STD Ct": round(std_ct, 3),
                            })
                        else:
                            rep_rows.append({
                                "Group": group,
                                "Channel": ch,
                                "Pair": f"{a} ↔ {b}",
                                "n": len(vals),
                                "Mean Ct": None,
                                "STD Ct": None,
                            })

            if rep_rows:
                rep_df = pd.DataFrame(rep_rows).sort_values(["Group", "Channel", "Pair"]).reset_index(drop=True)
                st.dataframe(rep_df, use_container_width=True)

                # Optional quick download
                rep_csv = rep_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Replicate STD (CSV)",
                    data=rep_csv,
                    file_name=f"Replicate_STD_{version}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No valid replicate pairs with numeric Ct values were found for the current selection.")
        # ----- End Replicate STD block -----
        # Persist last results so they can be shown/downloaded outside the button scope
        if threshold_enabled and ct_results:
            ct_df = pd.DataFrame(ct_results)
            st.session_state["last_ct_df"] = ct_df
        else:
            st.session_state["last_ct_df"] = None
        


# =========================
# Latest Plot Results (persistent) + Excel downloads
# =========================
st.subheader("Latest Plot Results (persistent)")

ct_df  = st.session_state.get("last_ct_df")
rep_df = st.session_state.get("last_rep_df")

# Show Ct table + quick CSV
if isinstance(ct_df, pd.DataFrame) and not ct_df.empty:
    st.markdown("**Ct Values**")
    st.dataframe(ct_df, use_container_width=True)
    st.download_button(
        "Download Ct table (CSV)",
        data=ct_df.to_csv(index=False).encode("utf-8"),
        file_name="Ct_table.csv",
        mime="text/csv"
    )
else:
    st.info("No Ct table yet. Plot with thresholds enabled to populate.")

# Show Replicate STD + quick CSV
if isinstance(rep_df, pd.DataFrame) and not rep_df.empty:
    st.markdown("**Replicate Ct STD (paired)**")
    st.dataframe(rep_df, use_container_width=True)
    st.download_button(
        "Download Replicate STD (CSV)",
        data=rep_df.to_csv(index=False).encode("utf-8"),
        file_name="Replicate_STD.csv",
        mime="text/csv"
    )
else:
    st.info("No replicate STD yet. Enable replicates and plot to populate.")


# ---------- Excel: Ct plate layout with same CF + borders ----------
include_cf_plate = st.checkbox(
    "Include conditional formatting in Excel (Ct plate layout)", value=True, key="cf_plate"
)

if isinstance(ct_df, pd.DataFrame) and not ct_df.empty:
    output_plate = io.BytesIO()
    with pd.ExcelWriter(output_plate, engine="openpyxl") as writer:
        plate_rows = ["A","B","C","D","E","F","G","H"] if plate_type == "96-well" else [chr(i) for i in range(ord("A"), ord("P")+1)]
        plate_cols = list(range(1,13)) if plate_type == "96-well" else list(range(1,25))

        for channel in ct_df["Channel"].dropna().unique():
            plate_matrix = pd.DataFrame(index=plate_rows, columns=plate_cols)
            sub = ct_df[ct_df["Channel"] == channel]

            for _, row in sub.iterrows():
                m = re.match(r"([A-Z]+)(\d+)", str(row["Well"]))
                if not m:
                    continue
                r, c = m.group(1), int(m.group(2))
                if r in plate_matrix.index and c in plate_matrix.columns:
                    val = pd.to_numeric(row["Ct"], errors="coerce")
                    plate_matrix.at[r, c] = float(val) if pd.notna(val) else None

            plate_matrix.sort_index(axis=1, inplace=True)
            plate_matrix.to_excel(writer, sheet_name=str(channel), na_rep="")

    output_plate.seek(0)

    if include_cf_plate:
        wb = load_workbook(output_plate)
        for sheetname in wb.sheetnames:
            ws = wb[sheetname]
            start_row = 2
            end_row = 9 if plate_type == "96-well" else 17
            end_col = 13 if plate_type == "96-well" else 25
            cell_range = f"B{start_row}:{get_column_letter(end_col)}{end_row}"

            rule = ColorScaleRule(
                start_type='num', start_value=14, start_color='4F81BD',
                mid_type='percentile', mid_value=50, mid_color='FFFFFF',
                end_type='num', end_value=33, end_color='F8696B'
            )
            ws.conditional_formatting.add(cell_range, rule)

            thin_border = Border(
                left=Side(style='thin', color='000000'),
                right=Side(style='thin', color='000000'),
                top=Side(style='thin', color='000000'),
                bottom=Side(style='thin', color='000000')
            )
            for row_cells in ws.iter_rows(min_row=start_row, max_row=end_row, min_col=2, max_col=end_col):
                for cell in row_cells:
                    cell.border = thin_border

        final_plate = io.BytesIO()
        wb.save(final_plate)
        final_plate.seek(0)
    else:
        final_plate = output_plate

    st.download_button(
        label="Download Ct Results as XLSX (Plate Layout)",
        data=final_plate,
        file_name=f"Ct_Results_{version}_plate_layout.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ---------- Excel: Replicate STD sheet with CF on "STD Ct" ----------
include_cf_rep = st.checkbox(
    "Include conditional formatting in Excel (Replicate STD)", value=True, key="cf_rep"
)

if isinstance(rep_df, pd.DataFrame) and not rep_df.empty:
    output_rep = io.BytesIO()
    with pd.ExcelWriter(output_rep, engine="openpyxl") as writer:
        rep_out = rep_df.copy()
        for col in ["Mean Ct","STD Ct"]:
            if col in rep_out.columns:
                rep_out[col] = pd.to_numeric(rep_out[col], errors="coerce")
        rep_out.to_excel(writer, sheet_name="Replicate_STD", index=False)

    output_rep.seek(0)

    if include_cf_rep:
        wb2 = load_workbook(output_rep)
        ws2 = wb2["Replicate_STD"]

        headers = {ws2.cell(row=1, column=j).value: j for j in range(1, ws2.max_column + 1)}
        std_col_idx = headers.get("STD Ct", None)

        if std_col_idx:
            col_letter = get_column_letter(std_col_idx)
            start_row = 2
            end_row = ws2.max_row
            cell_range = f"{col_letter}{start_row}:{col_letter}{end_row}"

            try:
                max_std = float(np.nanmax(pd.to_numeric(rep_df["STD Ct"], errors="coerce")))
                if not np.isfinite(max_std):
                    max_std = 1.0
            except Exception:
                max_std = 1.0

            rule2 = ColorScaleRule(
                start_type='num', start_value=0, start_color='63BE7B',
                mid_type='percentile', mid_value=50, mid_color='FFFFFF',
                end_type='num', end_value=max_std, end_color='F8696B'
            )
            ws2.conditional_formatting.add(cell_range, rule2)

            # Apply borders on whole table
            # Thin borders on the whole table
            thin = Border(
                left=Side(style='thin', color='000000'),
                right=Side(style='thin', color='000000'),
                top=Side(style='thin', color='000000'),
                bottom=Side(style='thin', color='000000'),
            )
            for row_cells in ws2.iter_rows(min_row=1, max_row=end_row, min_col=1, max_col=ws2.max_column):
                for cell in row_cells:
                    cell.border = thin

        final_rep = io.BytesIO()
        wb2.save(final_rep)
        final_rep.seek(0)
    else:
        final_rep = output_rep

    st.download_button(
        label="Download Replicate STD as XLSX",
        data=final_rep,
        file_name=f"Replicate_STD_{version}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
