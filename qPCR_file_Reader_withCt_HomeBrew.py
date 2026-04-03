
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import io, os
import re
from typing import List
import matplotlib.lines as mlines
import glob
import matplotlib as mpl
from matplotlib.colors import PowerNorm
from scipy.stats import norm as scnorm
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from itertools import chain
import string
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Border, Side

import plotly.express as px  # for the Auto Tune scatter


# =========================================================
# 4PL / Ct helpers
# =========================================================
def four_param_logistic(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

def inverse_four_pl(threshold, a, b, c, d):
    try:
        return c * ((a - d) / (threshold - d) - 1)**(1 / b)
    except Exception:
        return None

def calculate_ct(x, y, threshold, startpoint=10, use_4pl=False, return_std=False, scale='log'):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return (None, None) if return_std else None

    order = np.argsort(x)
    x, y = x[order], y[order]

    post = x >= startpoint
    if np.count_nonzero(post) < 2:
        return (None, None) if return_std else None
    x_fit, y_fit = x[post], y[post]

    if use_4pl:
        try:
            if x_fit.size >= 5:
                popt, pcov = curve_fit(four_param_logistic, x_fit, y_fit, maxfev=10000)
                ct = inverse_four_pl(threshold, *popt)
                if (ct is not None) and (x_fit[0] <= ct <= x_fit[-1]):
                    if return_std:
                        eps = 1e-8
                        grads = np.zeros(4)
                        for i in range(4):
                            p_hi = np.array(popt); p_hi[i] += eps
                            p_lo = np.array(popt); p_lo[i] -= eps
                            ct_hi = inverse_four_pl(threshold, *p_hi)
                            ct_lo = inverse_four_pl(threshold, *p_lo)
                            grads[i] = (ct_hi - ct_lo) / (2 * eps)
                        ct_var = float(np.dot(grads.T, np.dot(pcov, grads)))
                        ct_std = np.sqrt(ct_var) if ct_var >= 0 else np.nan
                        return float(ct), float(ct_std)
                    return float(ct)
        except Exception:
            pass

    if scale == 'linear':
        above = y_fit > threshold
        if not np.any(above):
            return (None, None) if return_std else None
        idx = int(np.argmax(above))
        if idx == 0:
            ct = float(x_fit[0])
        else:
            x1, x2 = x_fit[idx-1], x_fit[idx]
            y1, y2 = y_fit[idx-1], y_fit[idx]
            if y2 == y1:
                return (None, None) if return_std else None
            ct = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
        return (float(ct), None) if return_std else float(ct)

    elif scale == 'log':
        if (threshold is None) or (not np.isfinite(threshold)) or (threshold <= 0):
            return (None, None) if return_std else None
        pos = y_fit > 0
        if np.count_nonzero(pos) < 2:
            return (None, None) if return_std else None
        xf = x_fit[pos]
        yf_log = np.log10(y_fit[pos])
        thr_log = np.log10(threshold)
        above = yf_log > thr_log
        if not np.any(above):
            return (None, None) if return_std else None
        idx = int(np.argmax(above))
        if idx == 0:
            ct = float(xf[0])
        else:
            x1, x2 = xf[idx-1], xf[idx]
            y1, y2 = yf_log[idx-1], yf_log[idx]
            if y2 == y1:
                return (None, None) if return_std else None
            ct = x1 + (thr_log - y1) * (x2 - x1) / (y2 - y1)
        return (float(ct), None) if return_std else float(ct)

    else:
        raise ValueError("scale must be 'linear' or 'log'")

def find_threshold_for_target_ct_multi(
    x,
    ybg_list,
    target_ct,
    calculate_ct_func,
    ct_tol=0.01,
    max_iter=60,
    eps=1e-12
):
    x = np.asarray(x, dtype=float)

    y_all = np.concatenate([np.asarray(y, dtype=float) for y in ybg_list])
    lo = max(np.nanmin(y_all) + eps, eps)
    hi = np.nanmax(y_all) - eps
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid combined y range for threshold search.")

    def _ct_list(thr):
        out = []
        for y in ybg_list:
            res = calculate_ct_func(x, y, thr)
            ct = res[0] if isinstance(res, (tuple, list)) else res
            out.append(ct if (ct is not None and np.isfinite(ct)) else None)
        return out

    def _ct_avg(thr):
        cts = [ct for ct in _ct_list(thr) if ct is not None]
        if len(cts) == 0:
            return None
        return float(np.mean(cts))

    def _valid(v):
        return v is not None and np.isfinite(v)

    ct_lo = _ct_avg(lo)
    ct_hi = _ct_avg(hi)
    if not _valid(ct_lo):
        lo = np.nanpercentile(y_all, 5) + eps
        ct_lo = _ct_avg(lo)
    if not _valid(ct_hi):
        hi = np.nanpercentile(y_all, 95) - eps
        ct_hi = _ct_avg(hi)

    if not (_valid(ct_lo) and _valid(ct_hi)):
        raise RuntimeError("Could not evaluate average Ct at search bounds.")

    if ct_lo > ct_hi:
        lo, hi = hi, lo
        ct_lo, ct_hi = ct_hi, ct_lo

    if target_ct < ct_lo - 1e-9 or target_ct > ct_hi + 1e-9:
        raise ValueError(
            f"Target Ct {target_ct:.2f} is outside achievable average range "
            f"[{ct_lo:.2f}, {ct_hi:.2f}]"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ct_mid = _ct_avg(mid)

        if not _valid(ct_mid):
            mid = np.nextafter(mid, hi)
            ct_mid = _ct_avg(mid)
            if not _valid(ct_mid):
                hi = mid
                continue

        err = ct_mid - target_ct
        if abs(err) <= ct_tol:
            cts_at_mid = _ct_list(mid)
            return mid, ct_mid, cts_at_mid

        if err < 0:
            lo, ct_lo = mid, ct_mid
        else:
            hi, ct_hi = mid, ct_mid

    mid = 0.5 * (lo + hi)
    return mid, _ct_avg(mid), _ct_list(mid)


# =========================================================
# Background fitting
# =========================================================
def linear_exp_fit(x, a, b, c):
    return a * x + b * (2**x) + c

def linear_fit(x, d, e):
    return d * x + e

def SPR_fitbackground(median_intercept, fam_y, rox_y, start, end, cycles, plot=False):
    window = np.arange(start, end)
    fam_y_window = fam_y[start:end]
    rox_y_window = rox_y[start:end]

    popt, pcov = curve_fit(linear_exp_fit, window, fam_y_window, p0=[1, 2, 0.1])
    a, b, c = popt
    fam_bkg_fit = a * cycles + c

    popt, pcov = curve_fit(linear_fit, window, rox_y_window, p0=[1, 0.1])
    d, e = popt
    rox_bkg_fit = d * cycles + e

    if plot:
        plt.plot(cycles, fam_y, label='fam')
        plt.plot(cycles, rox_y, label='rox')
        plt.plot(cycles, fam_bkg_fit, '--', label='fam fit')
        plt.plot(cycles, rox_bkg_fit, '--', label='rox fit')
        plt.legend()
        plt.show()

    bkg = (a - median_intercept * d) * cycles + (c - median_intercept * e)
    if np.mean(bkg) > 0:
        return (fam_y - bkg) / rox_y
    else:
        bkg = (d - a / median_intercept) * cycles + (e - c / median_intercept)
        return fam_y / (rox_y - bkg)

def spr_QSqpcr_background_dY_residue_biorad(
    alpha,
    leaking_df,
    target_df,
    passive_ref_df,
    selected_wells,
    startcycle=10,
    window_size=8,
    StepIndY=50,
    n_cycles=40
):
    residue = []

    for _, rows in enumerate(selected_wells):
        for _, well in enumerate(rows):
            target_y = target_df[well]
            passive_ref_y = passive_ref_df[well]
            leaking_y = (
                leaking_df[well]
                if well in leaking_df.columns
                else pd.Series(np.zeros(n_cycles), name=well)
            )
            target_fixed = target_y + alpha * leaking_y
            y_norm = target_fixed / passive_ref_y

            y = np.asarray(y_norm, dtype=float)
            n = len(y)
            A = np.arange(n)
            p5, p95 = np.percentile(y, [5, 95])
            DeltaY = p95 - p5
            threshold = DeltaY / StepIndY
            Sn = np.full(n, np.nan)
            Sn[startcycle] = y[startcycle + window_size] - y[startcycle]

            for i in range(startcycle + 1, n - window_size - 2):
                Sn[i]   = y[i + window_size] - y[i]
                Sn[i+1] = y[i + 1 + window_size] - y[i+1]
                Sn[i+2] = y[i + 2 + window_size] - y[i+2]
                cond1 = (Sn[i]   - Sn[i-1] > threshold)
                cond2 = (Sn[i+1] - Sn[i]   > threshold)
                cond3 = (Sn[i+2] - Sn[i+1] > threshold)
                if (cond1 & cond2 & cond3).all():
                    start = i - 1
                    end = i + window_size
                    xf = np.arange(start, end)
                    yy = y[start:end]
                    popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])
                    a, b, c = popt
                    for xx in xf:
                        residue.append(y[xx] - (a * xx + c))

    x = np.asarray(residue, dtype=float).ravel()
    mean, std = scnorm.fit(x)
    return residue, mean, std

def spr_QSqpcr_background_dY_v5(
    std,
    test_signal,
    sigma_mult=2.0,
    min_points=4,
    max_refit_iter=3,
    startcycle=6,
    window_size=6,
    StepIndY=40,
    returnbase=False
):
    y = np.asarray(test_signal, dtype=float)
    n = len(y)
    A = np.arange(n)
    p5, p95 = np.percentile(y, [5, 95])
    DeltaY = p95 - p5
    threshold = DeltaY / StepIndY
    Sn = np.full(n, np.nan)
    Sn[startcycle] = y[startcycle + window_size] - y[startcycle]

    sigma = np.full(n, float(std))

    # if y[len(y)-1] - y[0] <= 0.1:
    #     if returnbase:
    #         return y-y, -2, 0, len(y)-1, 0, np.full(n, 0), y - 0
    #     else:
    #         return y-7, -2, 0, len(y)-1, 0
    if y[len(y)-1] - y[0] <= 0.1:
            base = np.nanmean(y[startcycle:startcycle + window_size])
            if returnbase:
                return y - base, -2, 0, len(y)-1, base, np.full(n, base), y - base
            else:
                return y - base, -2, 0, len(y)-1, base
    else:
        for i in range(startcycle + 1, n - window_size - 2):
            Sn[i]   = y[i + window_size] - y[i]
            Sn[i+1] = y[i + 1 + window_size] - y[i+1]
            Sn[i+2] = y[i + 2 + window_size] - y[i+2]

            cond1 = (Sn[i]   - Sn[i-1] > threshold)
            cond2 = (Sn[i+1] - Sn[i]   > threshold)
            cond3 = (Sn[i+2] - Sn[i+1] > threshold)

            if (cond1 & cond2 & cond3).all():
                start = i - 1
                end = i + window_size

                xf = np.arange(start, end)
                yy = y[start:end].copy()
                sig = sigma[start:end]

                mask = np.isfinite(yy) & np.isfinite(sig)
                if mask.sum() < min_points:
                    popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])
                    a, b, c = popt
                else:
                    popt, pcov = curve_fit(linear_exp_fit, xf[mask], yy[mask], p0=[1, 2, 0.1])
                    a, b, c = popt

                    for _ in range(max_refit_iter):
                        res = yy - linear_exp_fit(xf, a, b, c)
                        keep = (np.abs(res) <= sigma_mult * sig) & mask
                        if keep.sum() < min_points or keep.sum() == mask.sum():
                            break
                        popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])
                        a, b, c = popt
                        mask = keep
                        yy = yy[keep]
                        xf = xf[keep]
                        sig = sig[keep]
                        mask = mask[keep]

                baseline = a * A + c
                E = (y - baseline) / baseline
                start_point = end - 1 - 2
                expcurve = linear_exp_fit(A, a, b, c)
                intercept = c
                if returnbase:
                    return E, start_point, start, end, intercept, baseline, expcurve
                else:
                    return E, start_point, start, end, intercept

        if returnbase:
            base = np.nanmean(y[startcycle:startcycle + window_size])
            return y - base, -1, startcycle, startcycle + window_size, base, np.full(n, float(base)), y - base
        else:
            return y - np.nanmean(y[startcycle:startcycle + window_size]), -1, startcycle, startcycle + window_size, np.nanmean(y[startcycle:startcycle + window_size])

def calc_ct_func(x, y, thr):
    return calculate_ct(x, y, threshold=thr, startpoint=startcycle_to_use, use_4pl=False, return_std=False)


# =========================================================
# Well/group UI helpers
# =========================================================
def _safe_key(s: str) -> str:
    k = re.sub(r'[^A-Za-z0-9_]+', '_', s).strip('_')
    return k or "Group_1"

def select_group_ui(
    group_label: str,
    rows: list[str],
    cols: list[int],
    preset_colors: dict[str, str] | None = None,
    default_color: str = "#FF0000",
    state_key: str = "group_editor"
):
    if preset_colors is None:
        preset_colors = {
            "Red": "#FF0000", "Green": "#28A745", "Blue": "#007BFF", "Orange": "#FD7E14",
            "Purple": "#6F42C1", "Brown": "#8B4513", "Black": "#000000",
            "Gray": "#6C757D", "Custom HEX": None
        }

    well_names = [f"{r}{c}" for r in rows for c in cols]
    safe_key = _safe_key(state_key)

    for w in well_names:
        k = f"{safe_key}_{w}"
        if k not in st.session_state:
            st.session_state[k] = False

    st.subheader(f"Select Wells for **{group_label}**")

    colA, colB = st.columns([1, 1])
    with colA:
        pick_name = st.selectbox("Group color (preset)", list(preset_colors.keys()), key=f"{safe_key}_colorname")
    with colB:
        if pick_name == "Custom HEX":
            group_color = st.color_picker("Pick a custom color", default_color, key=f"{safe_key}_colorpicker")
        else:
            group_color = preset_colors[pick_name] or default_color

    st.write("Quick Select:")
    qc1, qc2, qc3, qc4 = st.columns([1, 1, 1, 1])

    with qc1:
        rows_multi = st.multiselect("Rows", rows, key=f"{safe_key}_rows_multi")
    with qc2:
        cols_multi = st.multiselect("Cols", [str(c) for c in cols], key=f"{safe_key}_cols_multi")
    with qc3:
        qc_mode = st.radio("Mode", ["Select", "Deselect"], horizontal=True, key=f"{safe_key}_qc_mode")
    with qc4:
        apply_qc = st.button("Apply", key=f"{safe_key}_apply_qc", use_container_width=True)

    def _bulk_set(wells, state=True):
        changed = False
        for w in wells:
            k = f"{safe_key}_{w}"
            if k not in st.session_state:
                st.session_state[k] = False
            if st.session_state[k] != state:
                st.session_state[k] = state
                changed = True
        if changed:
            st.rerun()

    if apply_qc:
        targets = set()
        if rows_multi:
            targets |= {f"{r}{c}" for r in rows_multi for c in cols}
        if cols_multi:
            targets |= {f"{r}{int(c)}" for r in rows for c in cols_multi}
        if targets:
            _bulk_set(targets, state=(qc_mode == "Select"))
        else:
            st.info("Pick at least one row or column before applying.")

    with st.expander("Paste wells / ranges (optional)"):
        txt = st.text_input("Examples: A1,A3,B2-B6,C1-C12", key=f"{safe_key}_paste")
        if st.button("Apply pasted wells", key=f"{safe_key}_apply_paste"):
            pat = r"([A-P])(\d+)(?:-([A-P])?(\d+))?"
            targets = set()
            for token in [t.strip().upper() for t in txt.split(",") if t.strip()]:
                m = re.fullmatch(pat, token)
                if not m:
                    continue
                r1, c1, r2, c2 = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                if not r2 and not c2:
                    if r1 in rows and c1 in cols:
                        targets.add(f"{r1}{c1}")
                else:
                    r_end = r2 or r1
                    c_end = int(c2 or c1)
                    if (r1 in rows) and (r_end in rows):
                        r_lo, r_hi = sorted([rows.index(r1), rows.index(r_end)])
                        for ri in range(r_lo, r_hi + 1):
                            for c in range(min(c1, c_end), max(c1, c_end) + 1):
                                if c in cols:
                                    targets.add(f"{rows[ri]}{c}")
            if targets:
                _bulk_set(targets, state=True)
            else:
                st.warning("No valid wells parsed.")

    st.write("Click wells to add/remove:")
    for r in rows:
        row_cols = st.columns(len(cols))
        for c, col in zip(cols, row_cols):
            w = f"{r}{c}"
            key = f"{safe_key}_{w}"
            if key not in st.session_state:
                st.session_state[key] = False
            col.checkbox(w, key=key)

    selected_wells = [
        f"{r}{c}"
        for r in rows for c in cols
        if st.session_state.get(f"{safe_key}_{r}{c}", False)
    ]
    selected_wells = sorted(set(selected_wells), key=lambda x: (x[0], int(x[1:])))
    return group_label, selected_wells, group_color

def _make_plate_df(plate_format: str) -> pd.DataFrame:
    if str(plate_format).startswith("384"):
        rows = list(string.ascii_uppercase[:16])
        cols = list(range(1, 25))
    else:
        rows = list(string.ascii_uppercase[:8])
        cols = list(range(1, 13))
    return pd.DataFrame(False, index=rows, columns=cols)

def _wells_from_df(df: pd.DataFrame) -> list[str]:
    out = []
    for r in df.index:
        for c in df.columns:
            if bool(df.loc[r, c]):
                out.append(f"{r}{c}")
    return out

def _full_plate_select(df: pd.DataFrame, row_rule: str = "All rows", col_rule: str = "All cols", select: bool = True) -> pd.DataFrame:
    out = df.copy()

    def row_ok(r_label: str) -> bool:
        pos = df.index.get_loc(r_label) + 1
        if row_rule == "Odd rows only":
            return (pos % 2) == 1
        if row_rule == "Even rows only":
            return (pos % 2) == 0
        return True

    def col_ok(c_label) -> bool:
        c = int(c_label)
        if col_rule == "Odd cols only":
            return (c % 2) == 1
        if col_rule == "Even cols only":
            return (c % 2) == 0
        return True

    for r in df.index:
        if not row_ok(r):
            continue
        allowed_cols = [c for c in df.columns if col_ok(c)]
        if allowed_cols:
            out.loc[r, allowed_cols] = select
    return out

def select_wells_ui(
    plate_format: str = "384-well (16×24)",
    key_prefix: str = "wellsel",
    show_summary: bool = True,
):
    grid_key = f"{key_prefix}_grid_{'384' if plate_format.startswith('384') else '96'}"
    if grid_key not in st.session_state:
        st.session_state[grid_key] = _make_plate_df(plate_format)
    plate_df = st.session_state[grid_key]

    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
    with cc1:
        row_rule = st.selectbox("Row rule", ["All rows", "Odd rows only", "Even rows only"], key=f"{key_prefix}_rowrule")
    with cc2:
        col_rule = st.selectbox("Column rule", ["All cols", "Odd cols only", "Even cols only"], key=f"{key_prefix}_colrule")
    with cc3:
        mode = st.radio("Mode", ["Select", "Deselect"], horizontal=True, key=f"{key_prefix}_mode")
    with cc4:
        applied = st.button("Apply full-plate selection", use_container_width=True, key=f"{key_prefix}_apply")

    if applied:
        st.session_state[grid_key] = _full_plate_select(
            st.session_state[grid_key],
            row_rule=row_rule,
            col_rule=col_rule,
            select=(mode == "Select"),
        )
        st.success(f"Applied: {mode.lower()} | {row_rule} × {col_rule}")
        plate_df = st.session_state[grid_key]

    r1, r2, r3 = st.columns([1, 1, 1])
    with r1:
        row_choice = st.multiselect("Select entire rows", plate_df.index, key=f"{key_prefix}_rows")
    with r2:
        col_choice = st.multiselect("Select entire columns", plate_df.columns, key=f"{key_prefix}_cols")
    with r3:
        if st.button("Apply row/col selection", key=f"{key_prefix}_apply_rc"):
            for r in row_choice:
                plate_df.loc[r, :] = True
            for c in col_choice:
                plate_df.loc[:, c] = True
            st.session_state[grid_key] = plate_df

    column_config = {c: st.column_config.CheckboxColumn() for c in plate_df.columns}
    edited_grid = st.data_editor(
        st.session_state[grid_key],
        use_container_width=True,
        num_rows="fixed",
        hide_index=False,
        column_config=column_config,
        key=f"{key_prefix}_editor",
    )

    st.session_state[grid_key] = edited_grid
    selected_wells = _wells_from_df(edited_grid)

    if show_summary:
        with st.expander(f"Selected wells ({len(selected_wells)})", expanded=False):
            st.write(", ".join(selected_wells) if selected_wells else "None")
        st.info(f"{len(selected_wells)} wells selected.")

    return selected_wells, edited_grid

def _delete_selector_state(rows, cols, state_key="group_editor"):
    sk = _safe_key(state_key)
    for r in rows:
        for c in cols:
            k = f"{sk}_{r}{c}"
            if k in st.session_state:
                del st.session_state[k]
    st.rerun()

def _load_wells_into_editor(wells: list[str], rows, cols, state_key="group_editor"):
    sk = _safe_key(state_key)
    for r in rows:
        for c in cols:
            k = f"{sk}_{r}{c}"
            if k in st.session_state:
                del st.session_state[k]
    for w in wells:
        m = re.match(r'^([A-P])(\d{1,2})$', w, re.I)
        if not m:
            continue
        r, c = m.group(1).upper(), int(m.group(2))
        if (r in rows) and (c in cols):
            st.session_state[f"{sk}_{r}{c}"] = True
    st.rerun()


# =========================================================
# Bio-Rad deconvolution helpers
# =========================================================

DEFAULT_BIORAD_MATRIX_TEXT = """1.0000,-0.0003,-0.3720,0.0080,0.0006
0.0019,1.0000,2.8,0.0016,0.0006
0.0003,0.0175,1.0000,0.0028,0.0019
-0.0004,-0.0006,-0.0039,1.0000,-0.0547
-0.0003,-0.0005,0.0105,0.1215,1.0000
"""

DEFAULT_BIORAD_MATRIX_TEXT_ROX_PROBE = """1.0000,-0.0003,0.0000,0.0080,0.0006
0.0019,1.0000,0.0000,0.0016,0.0006
0.0000,0.0000,1.0000,0.0000,0.0000
-0.0004,-0.0006,0.0000,1.0000,-0.0547
-0.0003,-0.0005,0.0000,0.1215,1.0000
"""

def normalize_chan_key(raw_chan: str) -> str:
    s = re.sub(r'[^A-Za-z0-9]+', '', str(raw_chan)).upper()
    if s == "CY5":
        return "CY5"
    if s in ("CY55", "CY55DOT", "CY5DOT5", "CY5_5", "CY5-5"):
        return "CY55"
    return s

def get_well_columns_from_df(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if re.fullmatch(r'^[A-P](\d{1,2})$', str(c).strip(), flags=re.I)
    ]

def parse_matrix_from_text(text: str) -> np.ndarray:
    if not text or not text.strip():
        raise ValueError("Matrix text is empty.")
    rows = []
    for line in text.strip().splitlines():
        parts = [p for p in re.split(r'[\s,;\t]+', line.strip()) if p != ""]
        if parts:
            rows.append([float(x) for x in parts])
    arr = np.array(rows, dtype=float)
    if arr.shape != (5, 5):
        raise ValueError(f"Matrix must be 5×5, got {arr.shape}.")
    return arr

def load_deconv_matrix_from_file(file_obj) -> np.ndarray:
    name = file_obj.name.lower()

    try:
        if name.endswith(".csv"):
            file_obj.seek(0)
            df = pd.read_csv(file_obj, index_col=0)
        elif name.endswith(".xlsx"):
            file_obj.seek(0)
            df = pd.read_excel(file_obj, index_col=0)
        else:
            raise ValueError("Matrix file must be .csv or .xlsx")
    except Exception:
        file_obj.seek(0)
        if name.endswith(".csv"):
            df = pd.read_csv(file_obj, header=None)
        else:
            df = pd.read_excel(file_obj, header=None)

    if isinstance(df.index, pd.Index) and isinstance(df.columns, pd.Index):
        idx = [normalize_chan_key(x) for x in df.index]
        cols = [normalize_chan_key(x) for x in df.columns]
        df.index = idx
        df.columns = cols

        want_rows = ["FAM", "HEX", "ROX", "CY5", "CY55"]
        want_cols = ["FAM", "HEX", "TAMRA", "CY5", "CY55"]

        if set(want_rows).issubset(set(df.index)) and set(want_cols).issubset(set(df.columns)):
            return df.loc[want_rows, want_cols].to_numpy(dtype=float)

    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape == (5, 5):
        return df_num.to_numpy(dtype=float)

    raise ValueError("Could not parse a 5×5 matrix from the uploaded file.")

def compute_detector_backgrounds_from_blank_wells(
    channel_dfs: dict,
    bg_wells: list[str],
    c0: int = 5,
    c1: int = 20,
) -> dict[str, float]:
    """
    Compute one scalar background per detector channel from user-defined blank wells.
    This matches the calibration logic much better than inverting raw traces directly.
    """
    bg = {}
    for ch, df in channel_dfs.items():
        vals = []
        for well in bg_wells:
            if well in df.columns:
                arr = pd.to_numeric(df[well], errors="coerce").to_numpy(dtype=float)
                if len(arr) >= c1:
                    vals.append(arr[c0:c1])
        bg[ch] = float(np.nanmedian(np.concatenate(vals))) if vals else 0.0
    return bg

def deconvolve_biorad_plate_no_rox(
    channel_dfs: dict,
    B: np.ndarray,
    bg_vals: dict[str, float] | None = None,
):
    """
    Detector rows in B: [FAM, HEX, ROX, CY5, CY55]
    Source cols in B:   [FAM, HEX, TAMRA, CY5, CY55]

    IMPORTANT:
    - Uploaded ROX is treated as the TAMRA detector in this mode.
    - Matrix inverse is applied to background-subtracted detector traces.
    - Missing uploaded detector channels are assumed zero.
    - Missing wells inside a channel are assumed zero.
    """
    req = ["FAM", "HEX", "ROX", "CY5", "CY55"]

    if not channel_dfs:
        raise ValueError("No channel data uploaded.")

    ref_key = next(iter(channel_dfs.keys()))
    ref_df = channel_dfs[ref_key].copy()

    ref_well_cols = set(get_well_columns_from_df(ref_df))
    meta_cols = [c for c in ref_df.columns if c not in ref_well_cols]

    all_well_cols = set()
    for df in channel_dfs.values():
        all_well_cols.update(get_well_columns_from_df(df))

    def _well_sort_key(w):
        m = re.fullmatch(r'^([A-P])(\d{1,2})$', str(w).strip(), flags=re.I)
        if not m:
            return ("Z", 999)
        return (m.group(1).upper(), int(m.group(2)))

    well_cols = sorted(all_well_cols, key=_well_sort_key)

    B_inv = np.linalg.inv(B)

    n_rows = len(ref_df)
    Y_list = []
    missing_channels = []
    missing_wells_by_channel = {}

    for k in req:
        if k not in channel_dfs:
            missing_channels.append(k)
            Y_list.append(np.zeros((n_rows, len(well_cols)), dtype=float))
            continue

        df = channel_dfs[k].copy()

        for mc in meta_cols:
            if mc not in df.columns and mc in ref_df.columns:
                df[mc] = ref_df[mc]

        if len(df) != n_rows:
            raise ValueError(
                f"Row count mismatch in channel {k}: got {len(df)} rows, expected {n_rows}. "
                "All uploaded CSVs must have the same number of cycle rows."
            )

        present_wells = set(get_well_columns_from_df(df))
        missing_wells = [w for w in well_cols if w not in present_wells]
        if missing_wells:
            missing_wells_by_channel[k] = missing_wells
            for w in missing_wells:
                df[w] = 0.0

        df_block = df[well_cols].copy()
        bg_val = 0.0 if bg_vals is None else float(bg_vals.get(k, 0.0))

        for c in well_cols:
            df_block[c] = pd.to_numeric(df_block[c], errors="coerce").fillna(0.0) - bg_val

        Y_list.append(df_block.to_numpy(dtype=float))

    Y = np.stack(Y_list, axis=0)  # shape = (5, n_cycles, n_wells)
    X = np.einsum("ij,jkl->ikl", B_inv, Y)

    out = {}
    out_names = ["FAM", "HEX", "TAMRA", "CY5", "CY55"]
    for i, name in enumerate(out_names):
        df_out = ref_df[meta_cols].copy()
        df_out[well_cols] = X[i]
        out[name] = df_out

    return out, B_inv, missing_channels, missing_wells_by_channel

def deconvolve_biorad_plate_rox_probe(
    channel_dfs: dict,
    B: np.ndarray,
    bg_vals: dict[str, float] | None = None,
):
    """
    ROX is treated as a probe channel, not passive reference.

    Detector rows in B: [FAM, HEX, ROX, CY5, CY55]
    Source cols in B:   [FAM, HEX, ROX, CY5, CY55]

    Current intended behavior:
    - ROX to/from all other channels = 0
    - FAM/HEX/CY5/CY55 deconvolve against each other as usual
    - ROX passes through unchanged except for optional detector background subtraction
    """
    req = ["FAM", "HEX", "ROX", "CY5", "CY55"]

    if not channel_dfs:
        raise ValueError("No channel data uploaded.")

    ref_key = next(iter(channel_dfs.keys()))
    ref_df = channel_dfs[ref_key].copy()

    ref_well_cols = set(get_well_columns_from_df(ref_df))
    meta_cols = [c for c in ref_df.columns if c not in ref_well_cols]

    all_well_cols = set()
    for df in channel_dfs.values():
        all_well_cols.update(get_well_columns_from_df(df))

    def _well_sort_key(w):
        m = re.fullmatch(r'^([A-P])(\d{1,2})$', str(w).strip(), flags=re.I)
        if not m:
            return ("Z", 999)
        return (m.group(1).upper(), int(m.group(2)))

    well_cols = sorted(all_well_cols, key=_well_sort_key)

    B_inv = np.linalg.inv(B)

    n_rows = len(ref_df)
    Y_list = []
    missing_channels = []
    missing_wells_by_channel = {}

    for k in req:
        if k not in channel_dfs:
            missing_channels.append(k)
            Y_list.append(np.zeros((n_rows, len(well_cols)), dtype=float))
            continue

        df = channel_dfs[k].copy()

        for mc in meta_cols:
            if mc not in df.columns and mc in ref_df.columns:
                df[mc] = ref_df[mc]

        if len(df) != n_rows:
            raise ValueError(
                f"Row count mismatch in channel {k}: got {len(df)} rows, expected {n_rows}. "
                "All uploaded CSVs must have the same number of cycle rows."
            )

        present_wells = set(get_well_columns_from_df(df))
        missing_wells = [w for w in well_cols if w not in present_wells]
        if missing_wells:
            missing_wells_by_channel[k] = missing_wells
            for w in missing_wells:
                df[w] = 0.0

        df_block = df[well_cols].copy()
        bg_val = 0.0 if bg_vals is None else float(bg_vals.get(k, 0.0))

        for c in well_cols:
            df_block[c] = pd.to_numeric(df_block[c], errors="coerce").fillna(0.0) - bg_val

        Y_list.append(df_block.to_numpy(dtype=float))

    Y = np.stack(Y_list, axis=0)  # shape = (5, n_cycles, n_wells)
    X = np.einsum("ij,jkl->ikl", B_inv, Y)

    out = {}
    out_names = ["FAM", "HEX", "ROX", "CY5", "CY55"]
    for i, name in enumerate(out_names):
        df_out = ref_df[meta_cols].copy()
        df_out[well_cols] = X[i]
        out[name] = df_out

    return out, B_inv, missing_channels, missing_wells_by_channel
    
def make_unity_ref_df(template_df: pd.DataFrame) -> pd.DataFrame:
    df = template_df.copy()
    for c in get_well_columns_from_df(df):
        df[c] = 1.0
    return df


# =========================================================
# App header
# =========================================================
version = "v2.2.0"

st.set_page_config(layout="wide")
st.title("qPCR homebrew - Supports Bio-Rad")
st.markdown(f"**Version:** {version}")
st.write("Contact JIACHONG CHU for questions.")

if "groups" not in st.session_state:
    st.session_state["groups"] = {}

plate_type = st.radio(
    "Plate type",
    ["96-well (8×12)", "384-well (16×24)"],
    index=0,
    horizontal=True
)

if plate_type.startswith("96"):
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
else:
    rows = list(string.ascii_uppercase[:16])
    cols = list(range(1, 25))


# =========================================================
# Uploads
# =========================================================
uploaded_files = st.file_uploader(
    "Upload Bio-Rad CSVs (1 per channel)",
    type=["csv"],
    accept_multiple_files=True
)

def clean_df(df):
    return df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')

channel_dfs = {}
if uploaded_files:
    pat = re.compile(
        r"Quantification\s+Amplification\s+Results[_\s-]*([A-Za-z0-9\.\-\+]+)\.csv$",
        re.IGNORECASE
    )
    for f in uploaded_files:
        fname = f.name.strip()
        m = pat.search(fname)
        if not m:
            st.warning(f"⚠️ Could not parse channel from filename: {fname}")
            continue

        raw_chan = m.group(1)
        chan_key = normalize_chan_key(raw_chan)

        try:
            df_raw = pd.read_csv(f)
        except Exception as e:
            st.error(f"Failed to read {fname}: {e}")
            continue

        channel_dfs[chan_key] = clean_df(df_raw)

    st.info(f"Loaded channels: {', '.join(sorted(channel_dfs.keys())) or 'None'}")


# =========================================================
# Mode selector + optional matrix
# =========================================================
working_channel_dfs = channel_dfs.copy()
deconv_enabled = False
B_deconv = None
B_inv = None
missing_deconv_channels = []

signal_mode = st.radio(
    "Bio-Rad signal mode",
    [
        "Without ROX normalization (ROX detector = TAMRA)",
        "With ROX normalization (no TAMRA)",
        "ROX as probe (no normalization; no ROX decon for now)"
    ],
    horizontal=True,
    index=0
)

if channel_dfs:
    if signal_mode.startswith("Without ROX"):
        st.caption("In this mode, uploaded ROX is treated as the TAMRA detector. No ROX normalization is used.")

        deconv_enabled = st.checkbox(
            "Apply deconvolution matrix before Ct workflow",
            value=True,
            help="Matrix rows = [FAM, HEX, ROX, CY5, CY55]; columns = [FAM, HEX, TAMRA, CY5, CY55]. Missing uploaded detector channels are assumed zero."
        )

        if deconv_enabled:
            matrix_file = st.file_uploader(
                "Upload deconvolution matrix (.csv or .xlsx)",
                type=["csv", "xlsx"],
                key="deconv_matrix_upload"
            )

            matrix_text = st.text_area(
                "Or paste 5×5 matrix here",
                value=DEFAULT_BIORAD_MATRIX_TEXT,
                height=140,
                key="deconv_matrix_text"
            )
            st.subheader("Detector background for deconvolution")

            c_bg1, c_bg2, c_bg3, c_bg4, c_bg5 = st.columns(5)
            with c_bg1:
                fam_bg_manual = st.number_input("FAM bg", value=3178.979, format="%.3f")
            with c_bg2:
                hex_bg_manual = st.number_input("HEX bg", value=2025.254, format="%.3f")
            with c_bg3:
                rox_bg_manual = st.number_input("ROX bg", value=2048.373, format="%.3f")
            with c_bg4:
                cy5_bg_manual = st.number_input("CY5 bg", value=1715.456, format="%.3f")
            with c_bg5:
                cy55_bg_manual = st.number_input("CY55 bg", value=2064.234, format="%.3f")

            bg_vals_for_deconv = {
                "FAM": fam_bg_manual,
                "HEX": hex_bg_manual,
                "ROX": rox_bg_manual,
                "CY5": cy5_bg_manual,
                "CY55": cy55_bg_manual,
            }

            try:
                if matrix_file is not None:
                    B_deconv = load_deconv_matrix_from_file(matrix_file)
                else:
                    B_deconv = parse_matrix_from_text(matrix_text)

                st.write("Loaded deconvolution matrix:")
                st.dataframe(
                    pd.DataFrame(
                        B_deconv,
                        index=["FAM", "HEX", "ROX", "CY5", "CY55"],
                        columns=["FAM", "HEX", "TAMRA", "CY5", "CY55"]
                    )
                )

                working_channel_dfs, B_inv, missing_deconv_channels, missing_wells_by_channel = deconvolve_biorad_plate_no_rox(
                    channel_dfs,
                    B_deconv,
                    bg_vals=bg_vals_for_deconv,
                )

                st.success("Deconvolution applied. Downstream workflow will use deconvolved FAM / HEX / TAMRA / CY5 / CY55.")

                if missing_deconv_channels:
                    st.warning(f"Missing detector channels assumed zero during deconvolution: {missing_deconv_channels}")

                if missing_wells_by_channel:
                    msg_lines = []
                    for ch, wells in missing_wells_by_channel.items():
                        preview = ", ".join(wells[:8])
                        more = "" if len(wells) <= 8 else f" ... (+{len(wells)-8} more)"
                        msg_lines.append(f"{ch}: {preview}{more}")
                    st.warning(
                        "Some uploaded channels were missing well columns; those wells were assumed zero.\n\n"
                        + "\n".join(msg_lines)
                    )

                with st.expander("Show inverse matrix used"):
                    st.dataframe(
                        pd.DataFrame(
                            B_inv,
                            index=["FAM", "HEX", "TAMRA", "CY5", "CY55"],
                            columns=["FAM", "HEX", "ROX", "CY5", "CY55"]
                        )
                    )

            except Exception as e:
                st.error(f"Failed to apply deconvolution: {e}")
                working_channel_dfs = channel_dfs.copy()
                deconv_enabled = False

    elif signal_mode.startswith("With ROX normalization"):
        st.caption("In this mode, there is no TAMRA channel. ROX is used as passive reference for normalization.")
        working_channel_dfs = channel_dfs.copy()

    elif signal_mode.startswith("ROX as probe"):
        st.caption("In this mode, ROX is treated as a regular probe channel. ROX-to/from-other-channel deconvolution is set to 0, while FAM/HEX/CY5/CY55 still deconvolve against each other.")
    
        deconv_enabled = st.checkbox(
            "Apply deconvolution matrix before Ct workflow",
            value=True,
            help="ROX is isolated for now. Other probe channels still deconvolve against each other."
        )
    
        if deconv_enabled:
            matrix_file = st.file_uploader(
                "Upload deconvolution matrix (.csv or .xlsx)",
                type=["csv", "xlsx"],
                key="deconv_matrix_upload_rox_probe"
            )
    
            matrix_text = st.text_area(
                "Or paste 5×5 matrix here",
                value=DEFAULT_BIORAD_MATRIX_TEXT_ROX_PROBE,
                height=140,
                key="deconv_matrix_text_rox_probe"
            )
    
            st.subheader("Detector background for deconvolution")
    
            c_bg1, c_bg2, c_bg3, c_bg4, c_bg5 = st.columns(5)
            with c_bg1:
                fam_bg_manual = st.number_input("FAM bg", value=3178.979, format="%.3f", key="roxprobe_fam_bg")
            with c_bg2:
                hex_bg_manual = st.number_input("HEX bg", value=2025.254, format="%.3f", key="roxprobe_hex_bg")
            with c_bg3:
                rox_bg_manual = st.number_input("ROX bg", value=2048.373, format="%.3f", key="roxprobe_rox_bg")
            with c_bg4:
                cy5_bg_manual = st.number_input("CY5 bg", value=1715.456, format="%.3f", key="roxprobe_cy5_bg")
            with c_bg5:
                cy55_bg_manual = st.number_input("CY55 bg", value=2064.234, format="%.3f", key="roxprobe_cy55_bg")
    
            bg_vals_for_deconv = {
                "FAM": fam_bg_manual,
                "HEX": hex_bg_manual,
                "ROX": rox_bg_manual,
                "CY5": cy5_bg_manual,
                "CY55": cy55_bg_manual,
            }
    
            try:
                if matrix_file is not None:
                    B_deconv = load_deconv_matrix_from_file(matrix_file)
                else:
                    B_deconv = parse_matrix_from_text(matrix_text)
    
                st.write("Loaded deconvolution matrix:")
                st.dataframe(
                    pd.DataFrame(
                        B_deconv,
                        index=["FAM", "HEX", "ROX", "CY5", "CY55"],
                        columns=["FAM", "HEX", "ROX", "CY5", "CY55"]
                    )
                )
    
                working_channel_dfs, B_inv, missing_deconv_channels, missing_wells_by_channel = deconvolve_biorad_plate_rox_probe(
                    channel_dfs,
                    B_deconv,
                    bg_vals=bg_vals_for_deconv,
                )
    
                st.success("Deconvolution applied. ROX is treated as an independent probe channel.")
    
                if missing_deconv_channels:
                    st.warning(f"Missing detector channels assumed zero during deconvolution: {missing_deconv_channels}")
    
                if missing_wells_by_channel:
                    msg_lines = []
                    for ch, wells in missing_wells_by_channel.items():
                        preview = ", ".join(wells[:8])
                        more = "" if len(wells) <= 8 else f" ... (+{len(wells)-8} more)"
                        msg_lines.append(f"{ch}: {preview}{more}")
                    st.warning(
                        "Some uploaded channels were missing well columns; those wells were assumed zero.\n\n"
                        + "\n".join(msg_lines)
                    )
    
                with st.expander("Show inverse matrix used"):
                    st.dataframe(
                        pd.DataFrame(
                            B_inv,
                            index=["FAM", "HEX", "ROX", "CY5", "CY55"],
                            columns=["FAM", "HEX", "ROX", "CY5", "CY55"]
                        )
                    )
    
            except Exception as e:
                st.error(f"Failed to apply deconvolution: {e}")
                working_channel_dfs = channel_dfs.copy()
                deconv_enabled = False
        else:
            working_channel_dfs = channel_dfs.copy()


# =========================================================
# Step 2–4: Ct workflow (plate-wide)
# =========================================================
if plate_type.startswith("96"):
    full_rows = list("ABCDEFGH")
    full_cols = list(range(1, 13))
else:
    full_rows = list(string.ascii_uppercase[:16])
    full_cols = list(range(1, 25))

st.subheader("Deconvolution / STD estimation / Ct workflow")

if not working_channel_dfs:
    st.info("Upload Bio-Rad CSVs first.")
else:
    first_df = next(iter(working_channel_dfs.values()))
    cycles_col = "Cycle" if "Cycle" in first_df.columns else None
    if cycles_col:
        cycles = first_df[cycles_col].to_numpy(dtype=float)
    else:
        cycles = np.arange(0, 40, dtype=float)
    n_cycles = len(cycles)

    c1_ui, c2_ui, c3_ui = st.columns([2, 2, 2])
    with c1_ui:
        startcycle_to_use = st.number_input(
            "Background start cycle",
            min_value=1,
            max_value=max(5, int(cycles[-1]) - 5),
            value=10,
            step=1
        )
    with c2_ui:
        window_size_to_use = st.number_input(
            "Background window size",
            min_value=4,
            max_value=20,
            value=8,
            step=1
        )
    with c3_ui:
        StepIndY_to_use = st.number_input(
            "ΔY step index (StepIndY)",
            min_value=10,
            max_value=200,
            value=50,
            step=5
        )

    if signal_mode.startswith("Without ROX"):
        chan_opts = [c for c in sorted(working_channel_dfs.keys()) if c in {"FAM", "HEX", "TAMRA", "CY5", "CY55", "ROX"}]
        ref_choices = ["NONE"]
        default_target = "FAM" if "FAM" in chan_opts else (chan_opts[0] if chan_opts else None)
        default_ref = "NONE"
    
    elif signal_mode.startswith("With ROX normalization"):
        chan_opts = [c for c in sorted(working_channel_dfs.keys()) if c != "ROX"]
        ref_choices = ["ROX", "NONE"] if "ROX" in working_channel_dfs else ["NONE"]
        default_target = "FAM" if "FAM" in chan_opts else (chan_opts[0] if chan_opts else None)
        default_ref = "ROX" if "ROX" in ref_choices else "NONE"
    
    else:
        # ROX as probe
        chan_opts = [c for c in sorted(working_channel_dfs.keys()) if c in {"FAM", "HEX", "ROX", "CY5", "CY55"}]
        ref_choices = ["NONE"]
        default_target = "ROX" if "ROX" in chan_opts else ("FAM" if "FAM" in chan_opts else (chan_opts[0] if chan_opts else None))
        default_ref = "NONE"

    if not chan_opts:
        st.error("No usable target channels found.")
        st.stop()

    c1_ui, c2_ui = st.columns([3, 3])
    with c1_ui:
        target_chan = st.selectbox(
            "Target channel",
            chan_opts,
            index=chan_opts.index(default_target) if default_target in chan_opts else 0
        )
    with c2_ui:
        passive_ref_chan = st.selectbox(
            "Passive reference channel",
            ref_choices,
            index=ref_choices.index(default_ref) if default_ref in ref_choices else 0
        )

    if signal_mode.startswith("Without ROX") or signal_mode.startswith("ROX as probe"):
        st.caption("Recommended in this mode: passive reference = NONE.")

    use_leak_correction = st.checkbox(
        "Apply extra leak correction term (target + α·leak) before background fitting?",
        value=False
    )
        

    raw_target_df = channel_dfs.get(target_chan)
    raw_target_wells = set(get_well_columns_from_df(raw_target_df)) if raw_target_df is not None else set()
        
    leaking_chan = None
    alpha = 0.0
    if use_leak_correction:
        leak_choices = [c for c in chan_opts if c != target_chan]
        if leak_choices:
            c1_ui, c2_ui = st.columns([3, 3])
            with c1_ui:
                leaking_chan = st.selectbox(
                    "Leaking channel",
                    leak_choices,
                    index=(leak_choices.index("HEX") if "HEX" in leak_choices else 0)
                )
            with c2_ui:
                alpha = st.number_input(
                    "alpha (target + α·leak)",
                    value=-0.010,
                    step=0.001,
                    format="%.3f"
                )

    target_df = working_channel_dfs.get(target_chan)
    leaking_df = working_channel_dfs.get(leaking_chan) if (use_leak_correction and leaking_chan is not None) else None

    if passive_ref_chan == "NONE":
        passive_ref_df = make_unity_ref_df(target_df) if target_df is not None else None
    else:
        passive_ref_df = working_channel_dfs.get(passive_ref_chan)

    if target_df is None or passive_ref_df is None:
        st.error("Selected target or passive-reference channel is missing.")
        st.stop()

    rows_sorted = full_rows
    cols_sorted = full_cols
    selected_wells_plate = [[f"{r}{c}" for c in cols_sorted] for r in rows_sorted]

    def _present_wells(grid, target_df, passive_ref_df, raw_target_wells, leaking_df=None):
        out = []
        for row in grid:
            ok = []
            for w in row:
                if w not in raw_target_wells:
                    continue
                if (w in target_df.columns) and (w in passive_ref_df.columns):
                    ok.append(w)
            if ok:
                out.append(ok)
        return out

    present_wells = _present_wells(
            selected_wells_plate,
            target_df,
            passive_ref_df,
            raw_target_wells,
            leaking_df if use_leak_correction else None
        )

    if not present_wells:
        st.error("No overlapping well columns in the selected channels.")
        st.stop()

    residue, mean_bg, res_std = spr_QSqpcr_background_dY_residue_biorad(
        alpha=(alpha if use_leak_correction else 0.0),
        leaking_df=(leaking_df if use_leak_correction else pd.DataFrame(columns=[])),
        target_df=target_df,
        passive_ref_df=passive_ref_df,
        selected_wells=present_wells,
        startcycle=int(startcycle_to_use),
        window_size=int(window_size_to_use),
        StepIndY=int(StepIndY_to_use),
        n_cycles=int(n_cycles),
    )
    st.success(f"Estimated plate residual σ = {res_std:.4f} (mean {mean_bg:.4f}, n={len(residue)})")

    st.subheader("Pick qPOS/reference wells & target Ct → find threshold")

    all_well_flat = sorted(raw_target_wells, key=lambda x: (x[0], int(x[1:])))
    qpos_default = all_well_flat[:2] if len(all_well_flat) >= 2 else all_well_flat

    qpos_wells = st.multiselect(
            "qPOS / reference wells",
            all_well_flat,
            default=qpos_default
        )
    target_ct = st.number_input(
        "Target Ct (avg over qPOS)",
        min_value=0.0,
        max_value=float(cycles[-1]),
        value=23.0,
        step=0.1
    )

    threshold_Ct = None
    ct_ref_avg = None
    ct_ref_list = None

    if qpos_wells:
        ref_y_bg = []
        for well in qpos_wells:
            target_y = target_df[well]
            ref_y = passive_ref_df[well]
            leak_y = (
                leaking_df[well]
                if use_leak_correction and (leaking_df is not None) and (well in leaking_df.columns)
                else pd.Series(np.zeros(n_cycles), name=well)
            )

            y_norm = (target_y + alpha * leak_y) / ref_y
            y_bg, start_point, start_win, end_win, intercept = spr_QSqpcr_background_dY_v5(
                res_std,
                y_norm,
                sigma_mult=2.0,
                min_points=4,
                max_refit_iter=3,
                startcycle=int(startcycle_to_use),
                window_size=int(window_size_to_use),
                StepIndY=int(StepIndY_to_use),
                returnbase=False
            )
            ref_y_bg.append(y_bg)

        def _calc_ct_func(x, y, thr):
            return calculate_ct(
                x, y,
                threshold=thr,
                startpoint=int(startcycle_to_use),
                use_4pl=False,
                return_std=False
            )

        threshold_Ct, ct_ref_avg, ct_ref_list = find_threshold_for_target_ct_multi(
            x=cycles,
            ybg_list=ref_y_bg,
            target_ct=float(target_ct),
            calculate_ct_func=_calc_ct_func,
            ct_tol=0.01,
        )
        st.success(f"Chosen threshold = {threshold_Ct:.5g} → avg Ct on qPOS = {ct_ref_avg:.2f}")
        st.caption("Per-qPOS Cts: " + ", ".join(f"{ct:.2f}" if ct is not None else "None" for ct in ct_ref_list))

    st.subheader("Compute Ct for whole plate & show as plate table")
    if threshold_Ct is None:
        st.info("Pick qPOS wells and target Ct above to compute threshold.")
    else:
        Ct_plate = {}
        for r in rows_sorted:
            for c in cols_sorted:
                well = f"{r}{c}"
                if well not in raw_target_wells:
                    Ct_plate[well] = None
                    continue
                
                if (well not in target_df.columns) or (well not in passive_ref_df.columns):
                    Ct_plate[well] = None
                    continue

                target_y = target_df[well]
                ref_y = passive_ref_df[well]
                leak_y = (
                    leaking_df[well]
                    if use_leak_correction and (leaking_df is not None) and (well in leaking_df.columns)
                    else pd.Series(np.zeros(n_cycles), name=well)
                )

                y_norm = (target_y + alpha * leak_y) / ref_y
                y_bg, start_point, *_ = spr_QSqpcr_background_dY_v5(
                    res_std,
                    y_norm,
                    sigma_mult=2.0,
                    min_points=4,
                    max_refit_iter=3,
                    startcycle=int(startcycle_to_use),
                    window_size=int(window_size_to_use),
                    StepIndY=int(StepIndY_to_use),
                    returnbase=False
                )
                if np.nanmax(y_bg) < threshold_Ct:
                    Ct_plate[well] = None
                    continue
                ct_val, _ = calculate_ct(
                    cycles,
                    y_bg,
                    threshold=threshold_Ct,
                    return_std=True,
                    use_4pl=False,
                    startpoint=int(max(start_point, startcycle_to_use))
                )
                Ct_plate[well] = None if ct_val is None else float(ct_val)

        st.subheader("Preview curves at chosen threshold")

        target_cols = {c.strip() for c in target_df.columns}
        passive_cols = {c.strip() for c in passive_ref_df.columns}
        leak_cols = {c.strip() for c in (leaking_df.columns if (use_leak_correction and leaking_df is not None) else [])}

        present_well_set = (target_cols & passive_cols) & raw_target_wells
        present_wells = sorted(
            [w for w in present_well_set if re.fullmatch(r'^[A-P](\d{1,2})$', w, flags=re.I)],
            key=lambda x: (x[0].upper(), int(x[1:]))
        )

        if not present_wells:
            st.warning("No wells found in the uploaded CSVs to plot.")
        else:
            c1_plot, c2_plot, c3_plot = st.columns([1.3, 1, 1])
            with c1_plot:
                color_mode = st.radio(
                    "Coloring mode",
                    ["One color", "By row", "By column"],
                    horizontal=True
                )
            with c2_plot:
                use_semilog = st.checkbox("Semilog Y", value=True)
            with c3_plot:
                lw = st.slider("Line width", 0.5, 3.0, 1.0, 0.1)

            one_color = None
            if color_mode == "One color":
                one_color = st.color_picker("Curve color", "#1f77b4")

            def _mk_palette(n):
                cmap = mpl.cm.get_cmap("tab20", max(1, n))
                return [cmap(i % cmap.N) for i in range(n)]

            rows_all = sorted({w[0].upper() for w in present_wells})
            cols_all = sorted({int(w[1:]) for w in present_wells})

            row_colors = {}
            col_colors = {}

            if color_mode == "By row":
                pal = _mk_palette(len(rows_all))
                row_colors = {r: pal[i] for i, r in enumerate(rows_all)}
            elif color_mode == "By column":
                pal = _mk_palette(len(cols_all))
                col_colors = {c: pal[i] for i, c in enumerate(cols_all)}

            fig, ax = plt.subplots(figsize=(5.5, 3.8))

            for w in present_wells:
                row_label = w[0].upper()
                col_label = int(w[1:])

                tgt = target_df[w]
                ref = passive_ref_df[w]
                leak = (
                    leaking_df[w]
                    if (use_leak_correction and leaking_df is not None and w in leaking_df.columns)
                    else pd.Series(np.zeros(n_cycles), name=w)
                )

                y_norm = (tgt + alpha * leak) / ref
                y_bg, start_point, *_ = spr_QSqpcr_background_dY_v5(
                    res_std,
                    y_norm,
                    sigma_mult=2.0,
                    min_points=4,
                    max_refit_iter=3,
                    startcycle=int(startcycle_to_use),
                    window_size=int(window_size_to_use),
                    StepIndY=int(StepIndY_to_use),
                    returnbase=False
                )

                if color_mode == "One color":
                    color = one_color
                elif color_mode == "By row":
                    color = row_colors.get(row_label)
                else:
                    color = col_colors.get(col_label)

                y_plot = np.array(y_bg, dtype=float)
                if use_semilog:
                    y_plot = np.where(y_plot > 0, y_plot, np.nan)

                ax.plot(cycles, y_plot, color=color, linewidth=lw)

            ax.axhline(y=threshold_Ct, linestyle="--", color="black", linewidth=1.5)
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Background-corrected signal")
            ax.set_title(f"Amplification curves @ threshold = {threshold_Ct:.5g}")
            if use_semilog:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.25)

            handles = []
            if color_mode == "By row":
                for r in rows_all:
                    handles.append(mlines.Line2D([0], [0], color=row_colors[r], lw=2, label=f"Row {r}"))
            elif color_mode == "By column":
                max_show = 12
                show_cols = cols_all[:max_show]
                for c in show_cols:
                    handles.append(mlines.Line2D([0], [0], color=col_colors[c], lw=2, label=f"Col {c}"))
                if len(cols_all) > max_show:
                    st.caption(f"Legend shows first {max_show} columns; all columns are colored on the plot.")

            if handles:
                ax.legend(handles=handles, ncol=3, fontsize=9, frameon=True)

            st.pyplot(fig, clear_figure=True)

        plate_df = pd.DataFrame(index=rows_sorted, columns=[str(c) for c in cols_sorted], dtype=float)

        for r in rows_sorted:
            for c in cols_sorted:
                well = f"{r}{c}"
                plate_df.at[r, str(c)] = Ct_plate.get(well, np.nan)

        st.dataframe(
            plate_df.style.format("{:.2f}").background_gradient(axis=None),
            use_container_width=True
        )

        csv_bytes = plate_df.to_csv(index=True).encode("utf-8")
        safe_target = target_chan.replace("/", "_")
        st.download_button(
            "Download Ct plate (CSV)",
            data=csv_bytes,
            file_name=f"Ct_{safe_target}.csv",
            mime="text/csv"
        )
