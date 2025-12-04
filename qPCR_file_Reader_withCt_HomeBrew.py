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

# Define 4PL function
def four_param_logistic(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

# Define inverse function to calculate Ct
def inverse_four_pl(threshold, a, b, c, d):
    try:
        return c * ((a - d) / (threshold - d) - 1)**(1 / b)
    except:
        return None

def calculate_ct(x, y, threshold, startpoint=10, use_4pl=False, return_std=False, scale='log'):
    x = np.asarray(x); y = np.asarray(y)

    if y[len(y)-1] - y[0] <= 0.1:
        return (None, None) if return_std else (None)
    
    else:
            
        # drop NaNs
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size < 3:
            return (None, None) if return_std else None
    
        # ensure x ascending
        order = np.argsort(x)
        x, y = x[order], y[order]
    
        # restrict to x >= startpoint (hard guarantee)
        post = x >= startpoint
        if np.count_nonzero(post) < 2:
            return (None, None) if return_std else None
        x_fit, y_fit = x[post], y[post]
    
        # 4PL first (optional)
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
                                grads[i] = (ct_hi - ct_lo) / (2*eps)
                            ct_var = float(np.dot(grads.T, np.dot(pcov, grads)))
                            ct_std = np.sqrt(ct_var) if ct_var >= 0 else np.nan
                            return float(ct), float(ct_std)
                        return float(ct)
            except Exception:
                pass  # fall through to interpolation
    
        # Fallback: interpolate crossing within [x_fit[0], x_fit[-1]]
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
    x,                 # 1D array of cycles (shared by all wells)
    ybg_list,          # list/tuple of 1–4 background-corrected y arrays (same length as x)
    target_ct,         # desired average Ct (float)
    calculate_ct_func, # callable: (x, y, thr) -> ct OR (ct, ...)
    ct_tol=0.01,
    max_iter=60,
    eps=1e-12
):
    """
    Find a single fluorescence threshold such that the *average* Ct across
    multiple wells equals target_ct (within ct_tol).

    Returns
    -------
    threshold : float
        Threshold giving average Ct ≈ target_ct.
    ct_avg : float
        Average Ct at that threshold (over valid wells).
    ct_list : list[float or None]
        Per-well Ct at that threshold (None for wells that failed Ct).
    """
    import numpy as np

    x = np.asarray(x, dtype=float)

    # Combine all wells' y to set a robust search bracket
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

    # Evaluate ends; if invalid, nudge to percentiles
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

    # Ensure monotonic direction (Ct increases with threshold in typical qPCR)
    if ct_lo > ct_hi:
        lo, hi = hi, lo
        ct_lo, ct_hi = ct_hi, ct_lo

    # Feasibility check
    if target_ct < ct_lo - 1e-9 or target_ct > ct_hi + 1e-9:
        raise ValueError(
            f"Target Ct {target_ct:.2f} is outside achievable average range "
            f"[{ct_lo:.2f}, {ct_hi:.2f}]"
        )

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ct_mid = _ct_avg(mid)

        if not _valid(ct_mid):
            mid = np.nextafter(mid, hi)  # nudge upward
            ct_mid = _ct_avg(mid)
            if not _valid(ct_mid):
                hi = mid
                continue

        err = ct_mid - target_ct
        if abs(err) <= ct_tol:
            cts_at_mid = _ct_list(mid)
            return mid, ct_mid, cts_at_mid

        if err < 0:   # need later Ct -> increase threshold
            lo, ct_lo = mid, ct_mid
        else:         # need earlier Ct -> decrease threshold
            hi, ct_hi = mid, ct_mid

    # Fallback: best midpoint
    mid = 0.5 * (lo + hi)
    return mid, _ct_avg(mid), _ct_list(mid)

    
def linear_exp_fit(x,a,b,c):
    return a*x + b*(2**x) + c

def linear_fit(x,d,e):
    return d*x + e
    
def SPR_fitbackground(median_intercept,fam_y,rox_y,start,end,cycles,plot = False):
    window = np.arange(start,end)
    fam_y_window = fam_y[start:end]
    rox_y_window = rox_y[start:end]
    
    popt, pcov = curve_fit(linear_exp_fit, window, fam_y_window, p0=[1, 2, 0.1])  # p0 = initial guess
    a,b,c = popt
    fam_bkg_fit = a*cycles + c

    popt, pcov = curve_fit(linear_fit, window, rox_y_window, p0=[1, 0.1])  # p0 = initial guess
    d,e = popt
    rox_bkg_fit = d*cycles + e
    if plot:
        plt.plot(cycles,fam_y,label = f'fam')
        plt.plot(cycles,rox_y,label = f'rox')
        plt.plot(cycles,fam_bkg_fit,'--',label = f'fam* fit')
        plt.plot(cycles,rox_bkg_fit,'--',label = f'rox fit')
        plt.legend()
        plt.show()  
    bkg = (a - median_intercept*d) * cycles + (c - median_intercept*e)
    if np.mean(bkg)>0:
        return (fam_y - bkg) / rox_y
    else:
        bkg = (d - a/median_intercept) * cycles + (e - c/median_intercept)
        return fam_y / (rox_y - bkg)

def spr_QSqpcr_background_dY_residue_biorad(alpha, leaking_df, target_df, passive_ref_df, selected_wells, startcycle = 10, window_size = 8, StepIndY = 50,n_cycles = 40):
   
    residue = []

    for ii, rows in enumerate(selected_wells):
        for jj,well in enumerate(rows):
                target_y = target_df[well]
                passive_ref_y = passive_ref_df[well]
                leaking_y = (
                    leaking_df[well]
                    if well in leaking_df.columns
                    else pd.Series(np.zeros(n_cycles), name=well)
                )
                target_fixed = target_y + alpha * leaking_y
                y_norm = target_fixed/passive_ref_y
                
                test_signal = y_norm
                y = np.asarray(test_signal, dtype=float)
                n = len(y)
                A = np.arange(n)
                p5, p95 = np.percentile(y, [5, 95])
                DeltaY = p95 - p5
                threshold = DeltaY/StepIndY
                Sn = np.full(n,np.nan)
                Sn[startcycle] = y[startcycle+window_size] - y[startcycle]
                detected = False
                start_point = -1
                
                for i in range(startcycle + 1, n - window_size-2):
                    Sn[i] = y[i+window_size] - y[i]
                    Sn[i+1] = y[i+1+window_size] - y[i+1]
                    Sn[i+2] = y[i+2+window_size] - y[i+2]
                    cond1 = (Sn[i] - Sn[i-1] > threshold)
                    cond2 = (Sn[i+1] - Sn[i] > threshold)
                    cond3 = (Sn[i+2] - Sn[i+1] > threshold)
                    if (cond1 & cond2 & cond3).all():
                        # if window_size % 2 == 0:
                        #     start = i - window_size // 2
                        #     end = i + window_size // 2
                        # else:
                        #     start = i - (window_size - 1) // 2 - 1
                        #     end = i + (window_size - 1) // 2 - 1
                        
                        start = i - 1 
                        end = i + window_size
                        xf = np.arange(start, end)
                        yy = y[start:end]
                        popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])  # p0 = initial guess
                        a,b,c = popt
                        baseline = a * A + c
                        # baseline = np.polyval(p, A)
                        E = (y - baseline)/baseline
                        start_point = end-1
                        detected = True
                        for xx in xf:
                            residue.append(y[xx] -  (a * xx + c))
    x = np.asarray(residue, dtype=float).ravel()
    mean, std = scnorm.fit(x)
    return residue,mean,std

    
def spr_QSqpcr_background_dY_v5(std, test_signal, sigma_mult=2.0, min_points=4, max_refit_iter = 3, startcycle = 6, window_size = 6, StepIndY = 40, returnbase = False):
    y = np.asarray(test_signal, dtype=float)
    n = len(y)
    A = np.arange(n)
    p5, p95 = np.percentile(y, [5, 95])
    DeltaY = p95 - p5
    threshold = DeltaY/StepIndY
    # print (threshold)
    Sn = np.full(n,np.nan)
    Sn[startcycle] = y[startcycle+window_size] - y[startcycle]
    detected = False
    start_point = -1
    
    sigma = np.full(n, float(std))

    
    for i in range(startcycle + 1, n - window_size - 2):
        # print (f'\n')
        Sn[i] = y[i+window_size] - y[i]
        Sn[i+1] = y[i+1+window_size] - y[i+1]
        Sn[i+2] = y[i+2+window_size] - y[i+2]
        
        cond1 = (Sn[i] - Sn[i-1] > threshold)
        cond2 = (Sn[i+1] - Sn[i] > threshold)
        cond3 = (Sn[i+2] - Sn[i+1] > threshold)
        # print (f'cycle = {i}')
        # print (Sn[i] - Sn[i-1])
        # print (Sn[i+1] - Sn[i])
        # print (Sn[i+2] - Sn[i+1])
        if (cond1 & cond2 & cond3).all():

            start = i - 1 
            end = i + window_size
            # if window_size % 2 == 0:
            #     start = i - 1 - window_size // 2
            #     end = i + 2 + window_size // 2
            # else:
            #     start = i - 1 - (window_size - 1) // 2 - 1
            #     end = i + 2 + (window_size - 1) // 2 - 1
                
            xf = np.arange(start, end)
            yy = y[start:end].copy()
            sig = sigma[start:end]

            # Initial fit
            mask = np.isfinite(yy) & np.isfinite(sig)
            if mask.sum() < min_points:
                popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])  # p0 = initial guess
                # a, b = np.polyfit(xf, yy, 1)
                a,b,c = popt
            else:
                # a, b = np.polyfit(xf[mask], yy[mask], 1)
                popt, pcov = curve_fit(linear_exp_fit, xf[mask], yy[mask], p0=[1, 2, 0.1])  # p0 = initial guess
                a,b,c = popt
                # Iteratively drop > sigma_mult * sigma residuals and refit
                for _ in range(max_refit_iter):
                    
                    res  = yy - linear_exp_fit(xf,a,b,c)
                    # print (f'res : {res}')
                    keep = (np.abs(res) <= sigma_mult * sig) & mask
                    if keep.sum() < min_points or keep.sum() == mask.sum():
                        break
                    popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])  # p0 = initial guess
                    a,b,c = popt
                    mask = keep  # tighten for the next pass
                    yy = yy[keep]
                    xf = xf[keep]
                    sig = sig[keep]
                    mask = mask[keep]
            # plt.plot(xf,yy)
            # plt.plot(xf, linear_exp_fit(xf,a,b,c),'r--')
            # plt.plot(A,y)
            # plt.plot(A, linear_exp_fit(A,a,b,c),'r--')
            # print (f'fitted line = {a}x + {b}2^x + {c}')       
            # baseline = a * A + b
            baseline = a * A + c
            # print (f'baseline = {a}x + {c}')
            E = (y - baseline) / baseline
            start_point = end - 1 - 2
            detected = True
            expcurve = linear_exp_fit(A,a,b,c)
            intercept = c
            if returnbase:
                return E, start_point, start, end, intercept, baseline, expcurve
            else: return E, start_point, start, end, intercept

    # if break 
    if returnbase:
        return y - np.nanmean(y[startcycle:startcycle+window_size]), -1, startcycle, startcycle+window_size, np.nanmean(y[startcycle:startcycle+window_size]), np.full(n, float(np.nanmean(y[startcycle:startcycle+window_size]))), y - np.nanmean(y[startcycle:startcycle+window_size])
    else:
        return y - np.nanmean(y[startcycle:startcycle+window_size]), -1, startcycle, startcycle+window_size, np.nanmean(y[startcycle:startcycle+window_size])

def calc_ct_func(x, y, thr):
    return calculate_ct(x, y,threshold=thr,startpoint=startcycle_to_use,use_4pl=False,return_std=False)


# --- Group selector (reusable) ------------------------------------------------
def _safe_key(s: str) -> str:
    import re
    k = re.sub(r'[^A-Za-z0-9_]+', '_', s).strip('_')
    return k or "Group_1"

def select_group_ui(
    group_label: str,
    rows: list[str],
    cols: list[int],
    preset_colors: dict[str,str]|None = None,
    default_color: str = "#FF0000",
    state_key: str = "group_editor"
):
    """
    Render a self-contained group selection UI without replicate logic.
      - group name
      - color picker (preset or custom)
      - quick select (multi rows / multi cols)
      - click-to-select grid of wells
    Returns:
      (group_label, selected_wells, group_color)
    """
    import streamlit as st
    import re

    if preset_colors is None:
        preset_colors = {
            "Red": "#FF0000", "Green": "#28A745", "Blue": "#007BFF", "Orange": "#FD7E14",
            "Purple": "#6F42C1", "Brown": "#8B4513", "Black": "#000000",
            "Gray": "#6C757D", "Custom HEX": None
        }

    nrows, ncols = len(rows), len(cols)
    well_names = [f"{r}{c}" for r in rows for c in cols]
    safe_key = _safe_key(state_key)

    # Initialize per-well state once per group key
    for w in well_names:
        k = f"{safe_key}_{w}"
        if k not in st.session_state:
            st.session_state[k] = False

    st.subheader(f"Select Wells for **{group_label}**")

    # --- Color controls
    colA, colB = st.columns([1,1])
    with colA:
        pick_name = st.selectbox("Group color (preset)", list(preset_colors.keys()),
                                 key=f"{safe_key}_colorname")
    with colB:
        if pick_name == "Custom HEX":
            group_color = st.color_picker("Pick a custom color", default_color,
                                          key=f"{safe_key}_colorpicker")
        else:
            group_color = preset_colors[pick_name] or default_color

    # --- Quick select (multi) ---
    st.write("Quick Select:")
    qc1, qc2, qc3, qc4 = st.columns([1, 1, 1, 1])

    with qc1:
        rows_multi = st.multiselect(
            "Rows", rows, key=f"{safe_key}_rows_multi",
            help="Pick any number of rows (e.g., A, C, H)."
        )
    with qc2:
        cols_multi = st.multiselect(
            "Cols", [str(c) for c in cols], key=f"{safe_key}_cols_multi",
            help="Pick any number of columns (e.g., 1, 3, 5, 12)."
        )
    with qc3:
        qc_mode = st.radio("Mode", ["Select", "Deselect"],
                           horizontal=True, key=f"{safe_key}_qc_mode")
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

    # --- (Optional) paste wells/ranges ---
    with st.expander("Paste wells / ranges (optional)"):
        txt = st.text_input(
            "Examples: A1,A3,B2-B6,C1-C12",
            key=f"{safe_key}_paste"
        )
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

    # --- Grid of checkboxes
    st.write("Click wells to add/remove:")
    for r in rows:
        row_cols = st.columns(len(cols))
        for c, col in zip(cols, row_cols):
            w = f"{r}{c}"
            key = f"{safe_key}_{w}"
            if key not in st.session_state:
                st.session_state[key] = False
            col.checkbox(w, key=key)

    # Collect output
    selected_wells = [
        f"{r}{c}"
        for r in rows for c in cols
        if st.session_state.get(f"{safe_key}_{r}{c}", False)
    ]
    selected_wells = sorted(set(selected_wells), key=lambda x: (x[0], int(x[1:])))

    return group_label, selected_wells, group_color

# ---------------------------------------------------------------------------


# --- Well selection UI (drop-in module) --------------------------------------


def _make_plate_df(plate_format: str) -> pd.DataFrame:
    """Create an empty (all False) plate grid DataFrame for 96 or 384 well plates."""
    if str(plate_format).startswith("384"):
        rows = list(string.ascii_uppercase[:16])  # A–P
        cols = list(range(1, 24+1))              # 1–24
    else:
        rows = list(string.ascii_uppercase[:8])   # A–H
        cols = list(range(1, 12+1))              # 1–12
    return pd.DataFrame(False, index=rows, columns=cols)

def _wells_from_df(df: pd.DataFrame) -> list[str]:
    """Return wells (A1 style) where df cell is True."""
    out = []
    for r in df.index:
        for c in df.columns:
            if bool(df.loc[r, c]):
                out.append(f"{r}{c}")
    return out

def _full_plate_select(df: pd.DataFrame,
                       row_rule: str = "All rows",
                       col_rule: str = "All cols",
                       select: bool = True) -> pd.DataFrame:
    """Bulk select/deselect with row/col filters."""
    out = df.copy()

    def row_ok(r_label: str) -> bool:
        pos = df.index.get_loc(r_label) + 1  # A=1
        if row_rule == "Odd rows only":  return (pos % 2) == 1
        if row_rule == "Even rows only": return (pos % 2) == 0
        return True

    def col_ok(c_label) -> bool:
        c = int(c_label)
        if col_rule == "Odd cols only":  return (c % 2) == 1
        if col_rule == "Even cols only": return (c % 2) == 0
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
    """
    Render a well-selection widget set and return (selected_wells, grid_df).

    Parameters
    ----------
    plate_format : str
        "384-well (16×24)" or "96-well (8×12)"
    key_prefix : str
        Unique prefix for Streamlit keys if you place multiple selectors on one page.
    show_summary : bool
        If True, shows an expander with the selected wells.

    Returns
    -------
    selected_wells : list[str]
        Wells like ["A1", "B3", ...] where the checkbox is True.
    grid_df : pd.DataFrame (bool)
        The current plate grid state.
    """
    grid_key = f"{key_prefix}_grid_{'384' if plate_format.startswith('384') else '96'}"
    if grid_key not in st.session_state:
        st.session_state[grid_key] = _make_plate_df(plate_format)
    plate_df = st.session_state[grid_key]

    # ---- Bulk controls
    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
    with cc1:
        row_rule = st.selectbox(
            "Row rule", ["All rows", "Odd rows only", "Even rows only"],
            key=f"{key_prefix}_rowrule",
            help="Choose which rows to affect when bulk selecting."
        )
    with cc2:
        col_rule = st.selectbox(
            "Column rule", ["All cols", "Odd cols only", "Even cols only"],
            key=f"{key_prefix}_colrule",
            help="Choose which columns to affect when bulk selecting."
        )
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

    # ---- Whole row/col quick picks
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

    # ---- Grid editor
    column_config = {c: st.column_config.CheckboxColumn() for c in plate_df.columns}
    edited_grid = st.data_editor(
        st.session_state[grid_key],
        use_container_width=True,
        num_rows="fixed",
        hide_index=False,
        column_config=column_config,
        key=f"{key_prefix}_editor",
    )

    # Persist and compute selection
    st.session_state[grid_key] = edited_grid
    selected_wells = _wells_from_df(edited_grid)

    if show_summary:
        with st.expander(f"Selected wells ({len(selected_wells)})", expanded=False):
            st.write(", ".join(selected_wells) if selected_wells else "None")
        st.info(f"{len(selected_wells)} wells selected.")

    return selected_wells, edited_grid

def _delete_selector_state(rows, cols, state_key="group_editor"):
    """Delete per-well widget keys so they recreate with defaults next run."""
    sk = _safe_key(state_key)
    for r in rows:
        for c in cols:
            k = f"{sk}_{r}{c}"
            if k in st.session_state:
                del st.session_state[k]
    st.rerun()


def _load_wells_into_editor(wells: list[str], rows, cols, state_key="group_editor"):
    """Delete all per-well keys, then pre-seed only the requested ones."""
    sk = _safe_key(state_key)
    # 1) nuke all well keys
    for r in rows:
        for c in cols:
            k = f"{sk}_{r}{c}"
            if k in st.session_state:
                del st.session_state[k]
    # 2) set selected (safe to set BEFORE re-render)
    for w in wells:
        m = re.match(r'^([A-P])(\d{1,2})$', w, re.I)
        if not m:
            continue
        r, c = m.group(1).upper(), int(m.group(2))
        if (r in rows) and (c in cols):
            st.session_state[f"{sk}_{r}{c}"] = True
    st.rerun()
# -----------------------------------------------------------------------------






# === Group builder ===
version = "v2.0.0"

st.set_page_config(layout="wide")
st.title("qPCR homebrew - Supports Bio-Rad")
st.markdown(f"**Version:** {version}")
# st.markdown(f"**Last updated:** {timestamp}")
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
    rows = list("ABCDEFGH")          # A–H
    cols = list(range(1, 12 + 1))    # 1–12
else:
    rows = list(string.ascii_uppercase[:16])  # A–P
    cols = list(range(1, 24 + 1))             # 1–24

uploaded_files = []
uploaded_files = st.file_uploader("Upload Bio-Rad CSVs (1 per channel)", type=["csv"], accept_multiple_files=True)

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
        # Normalize: uppercase, drop non-alphanumerics so Cy5.5 -> CY55
        chan_key = re.sub(r'[^A-Za-z0-9]+', '', raw_chan).upper()

        try:
            df_raw = pd.read_csv(f)
        except Exception as e:
            st.error(f"Failed to read {fname}: {e}")
            continue

        channel_dfs[chan_key] = clean_df(df_raw)

    # Optional convenience vars (may be None if not uploaded)
    fam_df  = channel_dfs.get("FAM")
    hex_df  = channel_dfs.get("HEX")
    cy5_df  = channel_dfs.get("CY5")
    cy55_df = channel_dfs.get("CY55")
    rox_df  = channel_dfs.get("ROX")

    st.info(f"Loaded channels: {', '.join(sorted(channel_dfs.keys())) or 'None'}")



# st.subheader("Step 1: Assign Wells to a Group")

# group_name = st.text_input("Group Name", "Group 1")

# # Preset palette you already had:
# preset_colors = {
#     "Red": "#FF0000", "Green": "#28A745", "Blue": "#007BFF", "Orange": "#FD7E14",
#     "Purple": "#6F42C1", "Brown": "#8B4513", "Black": "#000000",
#     "Gray": "#6C757D", "Custom HEX": None
# }

# selector_key = f"group_editor_{'96' if plate_type.startswith('96') else '384'}"

# gname, sel_wells, gcolor = select_group_ui(
#     group_label=group_name,
#     rows=rows,
#     cols=cols,
#     preset_colors=preset_colors,
#     default_color="#FF0000",
#     state_key=selector_key,  # <- important
# )

# col_add, col_clear = st.columns([1,1])

# def _add_group_cb(gname, sel_wells, gcolor, rows, cols, selector_key):
#     if sel_wells:
#         st.session_state.setdefault("groups", {})
#         st.session_state["groups"][gname] = {"color": gcolor, "wells": sel_wells.copy()}
#         _delete_selector_state(rows, cols, state_key=selector_key)
#     else:
#         st.session_state["__last_msg"] = "No wells selected."

# with col_add:
#     st.button(
#         "Add Group",
#         key="btn_add_group",
#         on_click=_add_group_cb,
#         args=(gname, sel_wells, gcolor, rows, cols, selector_key),
#         use_container_width=True,
#     )

# with col_clear:
#     st.button(
#         "Clear Current Selection",
#         key="btn_clear_sel",
#         on_click=_delete_selector_state,
#         args=(rows, cols, selector_key),
#         use_container_width=True,
#     )

# if st.session_state.pop("__last_msg", None):
#     st.warning("No wells selected.")

    
        
# st.subheader("Saved groups")
# if st.session_state["groups"]:
#     for name, info in st.session_state["groups"].items():
#         wells_preview = ", ".join(info["wells"][:12])
#         c1, c2, c3 = st.columns([4, 1.2, 1.2])
#         with c1:
#             st.markdown(
#                 f"- **{name}** · color: `{info['color']}` · wells: {len(info['wells'])}"
#                 + (f" · e.g. {wells_preview}..." if wells_preview else "")
#             )
#         with c2:
#             st.button(
#                 f"Re-edit ▷",
#                 key=f"load_{name}",
#                 on_click=_load_wells_into_editor,
#                 args=(info["wells"], rows, cols, selector_key),
#             )
        
#         with c3:
#             def _del_group_cb(group_name):
#                 if "groups" in st.session_state and group_name in st.session_state["groups"]:
#                     del st.session_state["groups"][group_name]
#                 st.rerun()
#             st.button("Delete", key=f"del_{name}", on_click=_del_group_cb, args=(name,))
# else:
#     st.caption("No groups saved yet.")








# ===============================
# Step 2–4: Ct workflow (plate-wide)
# ===============================
if plate_type.startswith("96"):
    full_rows = list("ABCDEFGH")
    full_cols = list(range(1, 13))
else:
    full_rows = list(string.ascii_uppercase[:16])  # A–P
    full_cols = list(range(1, 25))                # 1–24

    
    
st.subheader("Deconvolution, STD estimation (plate-wide)")

if not channel_dfs:
    st.info("Upload Bio-Rad CSVs first.")
else:
    # ---- runtime knobs (safe defaults) ----
    cycles_col = "Cycle" if "Cycle" in next(iter(channel_dfs.values())).columns else None
    # If your CSVs have Cycle as index or first column, adapt here:
    if cycles_col:
        cycles = next(iter(channel_dfs.values()))[cycles_col].to_numpy(dtype=float)
    else:
        # fallback to 40 cycles
        cycles = np.arange(0, 40, dtype=float)
    n_cycles = len(cycles)
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        startcycle_to_use  = st.number_input("Background start cycle", min_value=1, max_value=int(cycles[-1])-5, value=10, step=1)
    with c2: 
        window_size_to_use = st.number_input("Background window size", min_value=4, max_value=20, value=8, step=1)
    with c3:
        StepIndY_to_use    = st.number_input("ΔY step index (StepIndY)", min_value=10, max_value=200, value=50, step=5)

    # ---- choose channels ----
    chan_opts = sorted(channel_dfs.keys())
    c1, c2 = st.columns([3,3])
    with c1:
        target_chan = st.selectbox("Target channel", chan_opts, index=(chan_opts.index("FAM") if "FAM" in chan_opts else 0))
    with c2:
        passive_ref_chan = st.selectbox("Passive reference channel", chan_opts, index=(chan_opts.index("ROX") if "ROX" in chan_opts else 0))

    use_deconv = st.checkbox("Apply deconvolution (leak correction)?", value=False)
    leaking_chan = None
    alpha = 0.0
    if use_deconv:
        leak_choices = [c for c in chan_opts if c != target_chan]
        c1, c2 = st.columns([3,3])
        with c1:
            leaking_chan = st.selectbox("Leaking channel", leak_choices, index=(leak_choices.index("HEX") if "HEX" in leak_choices else 0))
        with c2:
            alpha = st.number_input("alpha (target + α·leak)", value=-0.010, step=0.001, format="%.3f")

    # ---- fetch dataframes ----
    target_df = channel_dfs.get(target_chan)
    passive_ref_df = channel_dfs.get(passive_ref_chan)
    leaking_df = channel_dfs.get(leaking_chan) if use_deconv else None

    # sanity checks
    if target_df is None or passive_ref_df is None:
        st.error("Selected target or passive-reference channel is missing from uploads.")
    else:
        # ---- infer plate wells present in the CSVs ----
        rows_sorted = full_rows
        cols_sorted = full_cols
        selected_wells_plate = [[f"{r}{c}" for c in cols_sorted] for r in rows_sorted]
        
        # Subset for STD estimation (must exist in target & passive_ref; leak optional)
        def _present_wells(grid, target_df, passive_ref_df, leaking_df=None):
            out = []
            for row in grid:
                ok = []
                for w in row:
                    if (w in target_df.columns) and (w in passive_ref_df.columns):
                        if (leaking_df is not None) and (w not in leaking_df.columns):
                            # if you want to require leak too, keep this check; otherwise drop it
                            # continue
                            pass
                        ok.append(w)
                if ok:
                    out.append(ok)
            return out
        
        present_wells = _present_wells(
            selected_wells_plate, target_df, passive_ref_df,
            leaking_df if use_deconv else None
        )
        
        if not present_wells:
            st.error("No overlapping well columns in the selected channels.")
            st.stop()
        
        # Call with filtered wells
        residue, mean_bg, res_std = spr_QSqpcr_background_dY_residue_biorad(
            alpha=(alpha if use_deconv else 0.0),
            leaking_df=(leaking_df if use_deconv else pd.DataFrame(columns=[])),
            target_df=target_df,
            passive_ref_df=passive_ref_df,
            selected_wells=present_wells,   # <-- filtered
            startcycle=int(startcycle_to_use),
            window_size=int(window_size_to_use),
            StepIndY=int(StepIndY_to_use),
            n_cycles=int(n_cycles),
        )
        st.success(f"Estimated plate residual σ = {res_std:.4f} (mean {mean_bg:.4f}, n={len(residue)})")

        # ---- Step 3: pick qPOS/reference wells & target Ct -> threshold ----
        st.subheader("Pick qPOS/reference wells & target Ct → find threshold")

        all_well_flat = [w for row in selected_wells_plate for w in row]
        qpos_wells = st.multiselect("qPOS / reference wells", all_well_flat, default=all_well_flat[:2])
        target_ct = st.number_input("Target Ct (avg over qPOS)", min_value=0.0, max_value=float(cycles[-1]), value=17.0, step=0.1)

        threshold_Ct = None
        ct_ref_avg = None
        ct_ref_list = None

        if qpos_wells:
            ref_y_bg = []
            for well in qpos_wells:
                # pull channels; supply zeros for missing leak wells
                target_y = target_df[well]
                ref_y = passive_ref_df[well]
                leak_y = (leaking_df[well] if use_deconv and (leaking_df is not None) and (well in leaking_df.columns)
                          else pd.Series(np.zeros(n_cycles), name=well))
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

            # calc threshold by desired average Ct
            def _calc_ct_func(x, y, thr):
                return calculate_ct(x, y, threshold=thr, startpoint=int(startcycle_to_use), use_4pl=False, return_std=False)

            threshold_Ct, ct_ref_avg, ct_ref_list = find_threshold_for_target_ct_multi(
                x=cycles,
                ybg_list=ref_y_bg,
                target_ct=float(target_ct),
                calculate_ct_func=_calc_ct_func,
                ct_tol=0.01,
            )
            st.success(f"Chosen threshold = {threshold_Ct:.5g} → avg Ct on qPOS = {ct_ref_avg:.2f}")
            st.caption("Per-qPOS Cts: " + ", ".join(f"{ct:.2f}" if ct is not None else "None" for ct in ct_ref_list))

        # ---- Step 4: compute Ct for the whole plate at that threshold ----
        st.subheader("Compute Ct for whole plate & show as plate table")
        if threshold_Ct is None:
            st.info("Pick qPOS wells and target Ct above to compute threshold.")
        else:
            Ct_plate = {}
            for r in rows_sorted:
                for c in cols_sorted:
                    well = f"{r}{c}"
                    if (well not in target_df.columns) or (well not in passive_ref_df.columns):
                        Ct_plate[well] = None
                        continue
                    target_y = target_df[well]
                    ref_y = passive_ref_df[well]
                    leak_y = (leaking_df[well] if use_deconv and (leaking_df is not None) and (well in (leaking_df.columns if leaking_df is not None else []))
                              else pd.Series(np.zeros(n_cycles), name=well))
                    y_norm = (target_y + alpha * leak_y) / ref_y
                    y_bg, start_point, *_ = spr_QSqpcr_background_dY_v5(
                        res_std, y_norm,
                        sigma_mult=2.0, min_points=4, max_refit_iter=3,
                        startcycle=int(startcycle_to_use),
                        window_size=int(window_size_to_use),
                        StepIndY=int(StepIndY_to_use),
                        returnbase=False
                    )
                    ct_val, _ = calculate_ct(
                        cycles, y_bg, threshold=threshold_Ct,
                        return_std=True, use_4pl=False,
                        startpoint=int(max(start_point, startcycle_to_use))
                    )
                    Ct_plate[well] = None if ct_val is None else float(ct_val)



            # ---- Curve preview (before table) -------------------------------------------
            st.subheader("Preview curves at chosen threshold")
            
            if threshold_Ct is None:
                st.info("Choose qPOS wells and target Ct to determine a threshold first.")
            else:
                # Determine present wells (exist in both target & passive-ref)
                target_cols   = {c.strip() for c in target_df.columns}
                passive_cols  = {c.strip() for c in passive_ref_df.columns}
                leak_cols     = {c.strip() for c in (leaking_df.columns if (use_deconv and leaking_df is not None) else [])}
            
                present_well_set = target_cols & passive_cols
                # If you *require* leak channel to exist too, uncomment next line:
                # if use_deconv: present_well_set &= leak_cols
            
                # Keep only well-like labels A1..P24 and sort by (row, col)
                present_wells = sorted(
                    [w for w in present_well_set if re.fullmatch(r'^[A-P](\d{1,2})$', w, flags=re.I)],
                    key=lambda x: (x[0].upper(), int(x[1:]))
                )
            
                if not present_wells:
                    st.warning("No wells found in the uploaded CSVs to plot.")
                else:
                    c1, c2, c3 = st.columns([1.3, 1, 1])
                    with c1:
                        color_mode = st.radio(
                            "Coloring mode",
                            ["One color", "By row", "By column"],
                            horizontal=True,
                            help="Choose how to color the curves."
                        )
                    with c2:
                        use_semilog = st.checkbox("Semilog Y", value=True)
                    with c3:
                        lw = st.slider("Line width", 0.5, 3.0, 1.0, 0.1)
            
                    # For "One color" mode, let the user pick the color
                    one_color = None
                    if color_mode == "One color":
                        one_color = st.color_picker("Curve color", "#1f77b4")
            
                    # Build color lookup based on mode
                    def _mk_palette(n):
                        # tab20 covers up to 20 unique colors; for more, wrap
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
            
                    # Draw the plot
                    fig, ax = plt.subplots(figsize=(5.5, 3.8))
            
                    for w in present_wells:
                        row_label = w[0].upper()
                        col_label = int(w[1:])
            
                        # Normalize exactly like Ct pipeline
                        tgt  = target_df[w]
                        ref  = passive_ref_df[w]
                        leak = (leaking_df[w] if (use_deconv and (w in leak_cols)) else pd.Series(np.zeros(n_cycles), name=w))
                        y_norm = (tgt + alpha * leak) / ref
            
                        y_bg, start_point, *_ = spr_QSqpcr_background_dY_v5(
                            res_std, y_norm,
                            sigma_mult=2.0, min_points=4, max_refit_iter=3,
                            startcycle=int(startcycle_to_use),
                            window_size=int(window_size_to_use),
                            StepIndY=int(StepIndY_to_use),
                            returnbase=False
                        )
            
                        # Color choice
                        if color_mode == "One color":
                            color = one_color
                            label = None  # suppress per-well legend spam
                        elif color_mode == "By row":
                            color = row_colors.get(row_label)
                            label = None  # we’ll show a compact legend later
                        else:  # By column
                            color = col_colors.get(col_label)
                            label = None
            
                        # Semilog handling: mask nonpositive on log
                        y_plot = np.array(y_bg, dtype=float)
                        if use_semilog:
                            y_plot = np.where(y_plot > 0, y_plot, np.nan)
            
                        ax.plot(cycles, y_plot, color=color, linewidth=lw)
            
                    # Horizontal dashed line at threshold
                    ax.axhline(y=threshold_Ct, linestyle="--",color = "black", linewidth=1.5)
            
                    ax.set_xlabel("Cycle")
                    ax.set_ylabel("Background-corrected signal")
                    ax.set_title(f"Amplification curves @ threshold = {threshold_Ct:.5g}")
                    if use_semilog:
                        ax.set_yscale("log")
                    ax.grid(True, alpha=0.25)
            
                    # Compact legend for group colors (row/col), not every well
                    handles = []
                    if color_mode == "By row":
                        from matplotlib.lines import Line2D
                        for r in rows_all:
                            handles.append(Line2D([0], [0], color=row_colors[r], lw=2, label=f"Row {r}"))
                    elif color_mode == "By column":
                        from matplotlib.lines import Line2D
                        # Keep legend readable: show up to 12 cols; if more, show summarized chunks
                        max_show = 12
                        show_cols = cols_all[:max_show]
                        for c in show_cols:
                            handles.append(Line2D([0], [0], color=col_colors[c], lw=2, label=f"Col {c}"))
                        if len(cols_all) > max_show:
                            st.caption(f"Legend shows first {max_show} columns; all columns are colored on the plot.")
            
                    if handles:
                        ax.legend(handles=handles, ncol=3, fontsize=9, frameon=True)
            
                    st.pyplot(fig, clear_figure=True)
            # ---------------------------------------------------------------------------

            # build plate dataframe (rows × cols)
            plate_df = pd.DataFrame(index=rows_sorted,
                    columns=[str(c) for c in cols_sorted],
                    dtype=float)

            for r in rows_sorted:
                for c in cols_sorted:
                    well = f"{r}{c}"
                    plate_df.at[r, str(c)] = (
                        Ct_plate.get(well, np.nan)
                    )

            st.dataframe(plate_df.style.format("{:.2f}").background_gradient(axis=None), use_container_width=True)

            # download
            csv_bytes = plate_df.to_csv(index=True).encode("utf-8")
            st.download_button("Download Ct plate (CSV)", data=csv_bytes, file_name=f"Ct_{target_chan}.csv", mime="text/csv")
