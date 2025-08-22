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

# Define 4PL function
def four_param_logistic(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

# Define inverse function to calculate Ct
def inverse_four_pl(threshold, a, b, c, d):
    try:
        return c * ((a - d) / (threshold - d) - 1)**(1 / b)
    except:
        return None
# HomeBrew backgroun subtraction
def spr_qpcr_background_correction(test_signal):
    A = np.arange(len(test_signal))
    E = np.zeros_like(test_signal)
    S = np.zeros(len(test_signal))
    
    S[0] = np.std(test_signal[0:4])
    S[1] = np.std(test_signal[1:7])
    
    for i in range(2, len(test_signal) - 4):
        S[i] = np.std(test_signal[i:i+7])
        if S[i] / S[1] > 1.5:
            p = np.polyfit(A[2:i+6], test_signal[2:i+6], 1)
            f = np.polyval(p, A)
            E = test_signal - f
            start_point = i + 4
            return E, start_point

    # fallback if no lift-off found
    return test_signal - np.mean(test_signal[:5]), -1


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
    above = y > threshold
    if not np.any(above):
        return (None, None) if return_std else None

    idx = np.argmax(above)
    if idx == 0:
        ct = float(x[0])
    else:
        x1, x2 = x[idx - 1], x[idx]
        y1, y2 = y[idx - 1], y[idx]
        ct = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)

    return (ct, None) if return_std else float(ct)



# timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
version = "v1.2.0"

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

# 


# ---------- Replicates controls ----------
st.subheader("Replicates (optional)")
use_replicates = st.toggle(
    "Enable Replicate Selection",
    value=False,
    help="Auto-select additional wells when you click a well."
)
replicate_mode = st.selectbox(
    "Replicate Pattern",
    ["Left-Right (half-plate)", "Top-Down (half-plate)", "Neighbors (4-way)"],
    disabled=not use_replicates,
    help="Left-Right = columns split in half; Top-Down = rows split in half; Neighbors = up/down/left/right."
)

# ---------- Helpers ----------
def _safe_key(s: str) -> str:
    k = re.sub(r'[^A-Za-z0-9_]+', '_', s).strip('_')
    return k or "Group_1"

nrows, ncols = len(rows), len(cols)
row_to_idx = {r: i for i, r in enumerate(rows)}
idx_to_row = {i: r for i, r in enumerate(rows)}
col_min, col_max = min(cols), max(cols)

def _block_cols_for(c0: int):
    """Return 1-based column list in same half as zero-based column index c0."""
    half = ncols // 2
    return list(range(1, half + 1)) if c0 < half else list(range(half + 1, ncols + 1))

def _block_rows_for(r0: int):
    """Return row labels in same half as zero-based row index r0."""
    half = nrows // 2
    return [idx_to_row[i] for i in range(half)] if r0 < half else [idx_to_row[i] for i in range(half, nrows)]

def _neighbors4(well: str):
    r = well[0]
    c = int(well[1:])
    ri = row_to_idx[r]
    out = []
    if c - 1 >= col_min: out.append(f"{r}{c-1}")
    if c + 1 <= col_max: out.append(f"{r}{c+1}")
    if ri - 1 >= 0:      out.append(f"{idx_to_row[ri-1]}{c}")
    if ri + 1 < nrows:   out.append(f"{idx_to_row[ri+1]}{c}")
    return out

def replicate_set(well: str):
    """Return the full replicate set for a clicked well, per current mode."""
    if not use_replicates:
        return [well]
    r = well[0]
    c = int(well[1:])
    ri = row_to_idx[r]
    if replicate_mode.startswith("Left-Right"):
        cols_in_block = _block_cols_for(c - 1)
        return [f"{rr}{cc}" for rr in rows for cc in cols_in_block]         # all rows × half-plate columns
    elif replicate_mode.startswith("Top-Down"):
        rows_in_block = _block_rows_for(ri)
        return [f"{rr}{cc}" for rr in rows_in_block for cc in cols]          # half-plate rows × all columns
    else:
        return list(set(_neighbors4(well) + [well]))                         # neighbors + self

# ---------- Group assignment ----------
st.subheader("Step 1: Assign Wells to a Group")
group_name = st.text_input("Group Name", "Group 1")
safe_group_key = _safe_key(group_name)

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

# ---------- Quick select ----------
st.write("Quick Select:")
col1, col2 = st.columns(2)
selected_row = col1.selectbox("Select Entire Row", ["None"] + rows)
selected_col = col2.selectbox("Select Entire Column", ["None"] + [str(c) for c in cols])
select_all = st.checkbox("Select All Wells")

quick_selected = set()
if selected_row != "None":
    quick_selected.update([f"{selected_row}{c}" for c in cols])
if selected_col != "None":
    quick_selected.update([f"{r}{selected_col}" for r in rows])

# ---------- Replicate callback ----------
def _on_checkbox_change(well: str):
    """Mirror the clicked well's state to its replicate partners."""
    if st.session_state["__suppress_rep_cb__"] or not use_replicates:
        return
    key = f"{safe_group_key}_{well}"
    state = bool(st.session_state.get(key, False))
    st.session_state["__suppress_rep_cb__"] = True
    try:
        for partner in replicate_set(well):
            pkey = f"{safe_group_key}_{partner}"
            # Only set if this checkbox key exists in the current run (avoids Streamlit exceptions)
            if pkey in st.session_state:
                st.session_state[pkey] = state
    finally:
        st.session_state["__suppress_rep_cb__"] = False

# ---------- Manual well selection grid ----------
st.write("Select Wells (click checkboxes):")
for r in rows:
    cols_container = st.columns(len(cols))
    for c, col in zip(cols, cols_container):
        well = f"{r}{c}"
        key = f"{safe_group_key}_{well}"
        default_checked = select_all or (well in quick_selected)
        col.checkbox(
            well,
            key=key,
            value=bool(st.session_state.get(key, default_checked)),
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

# Plot settings
# st.sidebar.subheader("Plot Settings")
# color_mode = st.sidebar.radio("Color mode", ["Solid", "Gradient"])

st.sidebar.subheader("Plot Color Scheme")
color_mode = st.sidebar.radio("Color Mode", ["Solid", "Gradient", "Colormap"])

colormap_name = None
if color_mode == "Colormap":
    available_colormaps = sorted(m for m in plt.colormaps() if not m.endswith("_r"))  # Optional: filter out reversed
    colormap_name = st.sidebar.selectbox(
        "Select a Colormap", ["jet", "viridis", "plasma", "cividis", "cool", "hot", "spring", "summer", "winter"]
    )
    

channel_options = ["FAM", "HEX", "Cy5", "Cy5.5", "ROX", "SYBR"]
default_channels = ["FAM", "HEX"]

st.sidebar.subheader("Deconvolution Settings (Bio-Rad only)")
enable_deconvolution = st.sidebar.checkbox("Enable Deconvolution for Bio-Rad")
if enable_deconvolution:
    deconv_target_channel = st.sidebar.selectbox("Channel to Deconvolve", channel_options, index=2)   # e.g. Cy5
    deconv_correction_channel = st.sidebar.selectbox("Correction Channel", channel_options, index=3)   # e.g. Cy5.5
    alpha_value = st.sidebar.number_input("Alpha Multiplier (α)", min_value=-10.0, max_value=10.0, value=0.07, step=0.01)

    
selected_channels = st.sidebar.multiselect("Select Channels to Plot", channel_options, default=default_channels)


normalize_to_rox = st.sidebar.checkbox("Normalize fluorescence to ROX channel")



# Baseline, Log Y, Threshold
st.sidebar.subheader("Step 3: Baseline Settings")
use_baseline = st.sidebar.toggle("Apply Baseline Subtraction", value=False)

baseline_method = st.sidebar.radio("Baseline Method",["Average of N cycles", "Homebrew Lift-off Fit"],index=0)

if use_baseline and baseline_method == "Average of N cycles":
    # Choose where to start averaging (3–15 is a typical safe range)
    baseline_start = st.sidebar.number_input("Baseline start cycle",min_value=3,max_value=15,value=3,step=1)

    # Choose how many cycles to average
    baseline_cycles = st.sidebar.number_input("Number of cycles to average",min_value=1,max_value=20,value=10,step=1)

log_y = st.sidebar.toggle("Use Semilog Y-axis (log scale)")

threshold_enabled = st.sidebar.checkbox("Enable Threshold & Ct Calculation")
# threshold_value = st.sidebar.number_input("Set RFU Threshold", min_value=0.0, value=1000.0, step=100.0)
# Per-channel thresholds
per_channel_thresholds = {}
if threshold_enabled:
    st.sidebar.markdown("**Per-Channel Thresholds:**")
    for ch in selected_channels:
        default_thresh = 0.13  # you can set any default
        per_channel_thresholds[ch] = st.sidebar.number_input(
            f"Threshold for {ch}", min_value=0.0, value=default_thresh, step=100.0, key=f"threshold_{ch}"
        )
        

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


ct_results = []

# Plotting
if uploaded_files and st.sidebar.button("Plot Curves"):
    fig = go.Figure()

    rox_df = None
    if normalize_to_rox:
        rox_file = next((f for f in uploaded_files if "rox" in f.name.lower()), None)
        if rox_file:
            rox_df = pd.read_csv(rox_file)

    for i, channel_name in enumerate(selected_channels):
        chan_str = channel_name 
        match_key = channel_name_map.get(channel_name, channel_name.lower())
        matched_file = next((f for f in uploaded_files if match_key.lower() in f.name.lower()), None)
        if not matched_file:
            st.warning(f"No file found for channel: {channel_name}")
            continue

        df = pd.read_csv(matched_file)
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
                    y = df[well].copy()
        
                    # Apply deconvolution IF df_corr loaded and well present
                    if enable_deconvolution and channel_name == deconv_target_channel and df_corr is not None:
                        if well in df_corr.columns:
                            y_corr = df_corr[well]
                            y = y + alpha_value * y_corr
                        else:
                            st.warning(f"Correction file loaded but well {well} not found in correction file.")
        
                    # === Rest of your processing ===
                    if normalize_to_rox and rox_df is not None and well in rox_df.columns and channel_name.upper() != "ROX":
                        rox_signal = rox_df[well]
                        if np.all(rox_signal > 0):
                            y = y / rox_signal
        
                    if use_baseline:
                        if baseline_method == "Average of N cycles":
                            baseline = y.iloc[baseline_start-1 : baseline_start-1+baseline_cycles].mean()
                            y -= baseline
                        elif baseline_method == "Homebrew Lift-off Fit":
                            y, E = spr_qpcr_background_correction(np.array(y))
        
                    x = df["Cycle"].values
                    style = channel_styles[i % len(channel_styles)]
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers" if style["symbol"] else "lines",
                        name=f"{group}: {well} ({channel_name})",
                        line=dict(color=mcolors.to_hex(color), dash=style["dash"]),
                        marker=dict(symbol=style["symbol"], size=6) if style["symbol"] else None
                    ))
                

                    if threshold_enabled:
                            channel_threshold = per_channel_thresholds.get(chan_str, 1000.0)
                            try:
                                ct_value, ct_std = calculate_ct(x, y, threshold=channel_threshold, startpoint = 10, use_4pl=True, return_std=True)
                                if ct_value is not None:
                                    ct_results.append({
                                        "Group": group,
                                        "Well": well,
                                        "Channel": channel_name,
                                        "Ct": f"{ct_value:.2f}"
                                    })
                                else:
                                    ct_results.append({
                                        "Group": group,
                                        "Well": well,
                                        "Channel": channel_name,
                                        "Ct": "Undetermined"
                                    })
                            except Exception as e:
                                ct_results.append({
                                    "Group": group,
                                    "Well": well,
                                    "Channel": channel_name,
                                    "Ct": "Undetermined"
                                })


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
    
        include_conditional_formatting = st.checkbox("Include Conditional Formatting in Download", value=True)
    
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
    
        plate_rows = ["A", "B", "C", "D", "E", "F", "G", "H"] if plate_type == "96-well" else [chr(i) for i in range(ord("A"), ord("P")+1)]
        plate_cols = list(range(1, 13)) if plate_type == "96-well" else list(range(1, 25))
    
        for channel in ct_df["Channel"].unique():
            plate_matrix = pd.DataFrame(index=plate_rows, columns=plate_cols)
    
            channel_df = ct_df[ct_df["Channel"] == channel]
            for _, row in channel_df.iterrows():
                well = row["Well"]
                match = re.match(r"([A-Z]+)([0-9]+)", well)
                # if match:
                #     r, c = match.group(1), int(match.group(2))
                #     if r in plate_matrix.index and c in plate_matrix.columns:
                #         plate_matrix.at[r, c] = float(row["Ct"])
                if match:
                    r, c = match.group(1), int(match.group(2))
                    if r in plate_matrix.index and c in plate_matrix.columns:
                        ct_raw = str(row["Ct"]).strip()
                        try:
                            ct_value = float(ct_raw)
                        except (ValueError, TypeError):
                            ct_value = np.nan  # Assign NaN if Ct is not numeric
                        plate_matrix.at[r, c] = ct_value
                        
                if r in plate_matrix.index and c in plate_matrix.columns:
                    plate_matrix.at[r, c] = ct_value
    
            plate_matrix.sort_index(axis=1, inplace=True)
            plate_matrix.to_excel(writer, sheet_name=str(channel))
    
        writer.close()
        output.seek(0)
    
        if include_conditional_formatting:
            wb = load_workbook(output)
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
    
                for row in ws.iter_rows(min_row=start_row, max_row=end_row, min_col=2, max_col=end_col):
                    for cell in row:
                        cell.border = thin_border
                
            final_output = io.BytesIO()
            wb.save(final_output)
            final_output.seek(0)
        else:
            final_output = output
    
        st.download_button(
            label="Download Ct Results as XLSX (Plate Layout)",
            data=final_output,
            file_name=f"Ct_Results_{version}_plate_layout.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ------------------------
# Debug Section - Heatmap
# ------------------------
# st.sidebar.subheader("[Debug] Heatmap")
# enable_debug_heatmap = st.sidebar.checkbox("Enable Heatmap Debug Mode")
# if enable_debug_heatmap:
#     st.subheader("Debug Heatmap of Average Fluorescence")

#     debug_channel = st.sidebar.selectbox("Select Channel for Heatmap", channel_options)
#     debug_cycle_count = st.sidebar.number_input("Number of Cycles to Average", min_value=1, max_value=100, value=20)

#     heatmap_matrix = None  # initialize to check later

#     if platform == "QuantStudio (QS)" and uploaded_files:
#         df = pd.read_excel(uploaded_files[0][1]) if uploaded_files[0][1].name.endswith("xlsx") else pd.read_csv(uploaded_files[0][1])
#         df = df[df["Well Position"] != "Well Position"]
#         df.iloc[:, 5:] = df.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')
#         rfu_cols = [col for col in df.columns if col.startswith("X")]
#         debug_chan_idx = int(debug_channel) - 1

#         detected_wells = df["Well Position"].dropna().unique()
#         rows_used = sorted(set(w[0] for w in detected_wells if isinstance(w, str)))
#         cols_used = sorted(set(int(w[1:]) for w in detected_wells if isinstance(w, str) and w[1:].isdigit()))
#         heatmap_matrix = pd.DataFrame(np.nan, index=rows_used, columns=cols_used)

#         for well in detected_wells:
#             sub_df = df[df["Well Position"] == well].sort_values(by=df.columns[1])
#             if debug_chan_idx < len(rfu_cols):
#                 y = sub_df[rfu_cols[debug_chan_idx]].iloc[:debug_cycle_count]
#                 avg_val = y.mean()
#                 match = re.match(r"([A-Z]+)([0-9]+)", well)
#                 if match:
#                     r, c = match.group(1), int(match.group(2))
#                     if r in plate_matrix.index and c in plate_matrix.columns:
#                         plate_matrix.at[r, c] = float(row["Ct"])
#                 if r in heatmap_matrix.index and c in heatmap_matrix.columns:
#                     heatmap_matrix.loc[r, c] = avg_val

#     elif platform == "Bio-Rad" and uploaded_files:
#         match_key = channel_name_map.get(debug_channel, debug_channel.lower())
#         matched_file = next((f for f in uploaded_files if match_key.lower() in f.name.lower()), None)
#         if matched_file:
#             df = pd.read_csv(matched_file)
#             df.columns = df.columns.str.strip()
#             df = df.loc[:, ~df.columns.str.contains("Unnamed")]
#             detected_wells = [c for c in df.columns if isinstance(c, str) and len(c) >= 2 and c[0].isalpha() and c[1:].isdigit()]
#             rows_used = sorted(set(w[0] for w in detected_wells))
#             cols_used = sorted(set(int(w[1:]) for w in detected_wells))
#             heatmap_matrix = pd.DataFrame(np.nan, index=rows_used, columns=cols_used)

#             for well in detected_wells:
#                 y = df[well].iloc[:debug_cycle_count]
#                 avg_val = y.mean()
#                 match = re.match(r"([A-Z]+)([0-9]+)", well)
#                 if match:
#                     r, c = match.group(1), int(match.group(2))
#                     if r in plate_matrix.index and c in plate_matrix.columns:
#                         plate_matrix.at[r, c] = float(row["Ct"])


#     if heatmap_matrix is not None:
#         # Plot heatmap
#         rows_used = list(heatmap_matrix.index)
#         cols_used = list(heatmap_matrix.columns)
#         n_rows, n_cols = len(rows_used), len(cols_used)
#         cell_size = 0.6
#         fig, ax = plt.subplots(figsize=(n_cols * cell_size, n_rows * cell_size))
#         im = ax.imshow(heatmap_matrix.values.astype(float), cmap='viridis', aspect='equal')

#         ax.set_xticks(np.arange(len(cols_used)))
#         ax.set_yticks(np.arange(len(rows_used)))
#         ax.set_xticklabels(cols_used)
#         ax.set_yticklabels(rows_used)
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#         for i in range(n_rows):
#             for j in range(n_cols):
#                 value = heatmap_matrix.iloc[i, j]
#                 if not np.isnan(value):
#                     ax.text(j, i, f"{value:.1f}", ha="center", va="center",
#                             color="white" if value > np.nanmax(heatmap_matrix.values)/2 else "black",
#                             fontsize=5)

#         ax.set_title(f"Average RFU (First {debug_cycle_count} Cycles) - {debug_channel}")
#         fig.colorbar(im, ax=ax)
#         st.pyplot(fig)
#     else:
#         st.warning("No heatmap data available. Please check file format or platform selection.")
