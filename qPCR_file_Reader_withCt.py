import streamlit as st
import pandas as pd
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import os
import numpy as np
from scipy.optimize import curve_fit
import datetime

# Define 4PL function
def four_param_logistic(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

# Define inverse function to calculate Ct
def inverse_four_pl(threshold, a, b, c, d):
    try:
        return c * ((a - d) / (threshold - d) - 1)**(1 / b)
    except:
        return None



timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

st.set_page_config(layout="wide")
st.title("qPCR Viewer - Supports QuantStudio & Bio-Rad")
st.markdown(f"**Last updated:** {timestamp}")
st.write("Contact Jiachong CHU for questions.")

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
platform = st.radio("Select qPCR Platform", ["QuantStudio (QS)", "Bio-Rad"], index=1)

# Upload files
uploaded_files = []
if platform == "QuantStudio (QS)":
    uploaded_file = st.file_uploader("Upload QuantStudio xlsx or csv", type=["xlsx", "csv"])
    if uploaded_file:
        uploaded_files.append(("QS", uploaded_file))
else:
    uploaded_files = st.file_uploader("Upload Bio-Rad CSVs (1 per channel)", type=["csv"], accept_multiple_files=True)

# Group state
if "groups" not in st.session_state:
    st.session_state["groups"] = {}

# Group assignment
st.subheader("Step 1: Assign Wells to a Group")
group_name = st.text_input("Group Name", "Group 1")

preset_colors = {
    "Red": "#FF0000", "Green": "#28A745", "Blue": "#007BFF", "Orange": "#FD7E14",
    "Purple": "#6F42C1", "Brown": "#8B4513", "Black": "#000000", "Gray": "#6C757D", "Custom HEX": None
}
selected_color_name = st.selectbox("Select Group Color", list(preset_colors.keys()))
if selected_color_name == "Custom HEX":
    group_color = st.color_picker("Pick a Custom Color", "#FF0000")
else:
    group_color = preset_colors[selected_color_name]

selected_wells = []

# Quick select
st.write("Quick Select:")
col1, col2 = st.columns(2)
selected_row = col1.selectbox("Select Entire Row", ["None"] + rows)
selected_col = col2.selectbox("Select Entire Column", ["None"] + [str(c) for c in cols])
if selected_row != "None":
    selected_wells.extend([f"{selected_row}{c}" for c in cols])
if selected_col != "None":
    selected_wells.extend([f"{r}{selected_col}" for r in rows])

select_all = st.checkbox("Select All Wells")

# Manual well selection
st.write("Select Wells (click checkboxes):")
for r in rows:
    cols_container = st.columns(len(cols))
    for c, col in zip(cols, cols_container):
        well = f"{r}{c}"
        default_checked = select_all or False
        if col.checkbox(well, key=f"{group_name}_{well}", value=default_checked):
            selected_wells.append(well)

selected_wells = sorted(set(selected_wells), key=lambda x: (x[0], int(x[1:])))

if st.button("Add Group"):
    if group_name and selected_wells:
        st.session_state["groups"][group_name] = {"color": group_color, "wells": selected_wells}

# Display groups
st.subheader("Current Groups")
for group, info in st.session_state["groups"].items():
    st.markdown(f"**{group}** ({info['color']}): {', '.join(info['wells'])}")

# Delete group
st.subheader("Delete a Group")
if st.session_state["groups"]:
    group_to_delete = st.selectbox("Select Group to Delete", list(st.session_state["groups"].keys()))
    if st.button("Delete Group"):
        st.session_state["groups"].pop(group_to_delete, None)
        st.success(f"Deleted group: {group_to_delete}")
else:
    st.info("No groups available to delete.")

# Plot settings
st.subheader("Step 2: Plot Settings")
color_mode = st.radio("Color mode", ["Solid", "Gradient"])
if platform == "QuantStudio (QS)":
    channel_options = [str(i) for i in range(1, 13)]
    default_channels = ["1", "2"]
else:
    channel_options = ["FAM", "HEX", "Cy5", "Cy5.5", "ROX"]
    default_channels = ["FAM", "HEX"]

selected_channels = st.multiselect("Select Channels to Plot", channel_options, default=default_channels)

# Baseline, Log Y, Threshold
st.subheader("Step 3: Baseline Settings")
use_baseline = st.toggle("Apply Baseline Subtraction", value=False)
baseline_cycles = st.number_input("Use average RFU of first N cycles", min_value=1, max_value=20, value=10)
log_y = st.toggle("Use Semilog Y-axis (log scale)")

st.subheader("Step 4: Threshold Settings")
threshold_enabled = st.checkbox("Enable Threshold & Ct Calculation")
threshold_value = st.number_input("Set RFU Threshold", min_value=0.0, value=1000.0, step=100.0)

channel_name_map = {
    "FAM": "FAM",
    "HEX": "HEX",
    "Cy5": "Cy5",
    "Cy5.5": "Cy5-5",
    "ROX": "ROX"
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
if uploaded_files and st.button("Plot Curves"):
    fig = go.Figure()

    if platform == "QuantStudio (QS)":
        filetype = uploaded_files[0][1].name.split(".")[-1].lower()
        if filetype == "xlsx":
            df = pd.read_excel(uploaded_files[0][1])
        else:
            df = pd.read_csv(uploaded_files[0][1])

        df = df[df["Well Position"] != "Well Position"]
        df.iloc[:, 5:] = df.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')
        rfu_cols = [col for col in df.columns if col.startswith("X")]
        cycle_col = next((col for col in df.columns if "cycle" in col.lower()), "Cycle Number")

        for group, info in st.session_state["groups"].items():
            wells = info["wells"]
            base_color = info["color"]
            color_list = [base_color] * len(wells) if color_mode == "Solid" else [
                mcolors.LinearSegmentedColormap.from_list("gradient", [
                    tuple(1 - 0.5 * (1 - c) for c in mcolors.to_rgb(base_color)), mcolors.to_rgb(base_color)
                ])(i / max(1, len(wells) - 1)) for i in range(len(wells))
            ]

            for well, color in zip(wells, color_list):
                if well in df["Well Position"].values:
                    sub_df = df[df["Well Position"] == well].sort_values(by=cycle_col)
                    x = sub_df[cycle_col].values

                    for i, chan_str in enumerate(selected_channels):
                        chan_idx = int(chan_str) - 1
                        if 0 <= chan_idx < len(rfu_cols):
                            y = sub_df[rfu_cols[chan_idx]].copy()
                            if use_baseline:
                                baseline = y.iloc[:baseline_cycles].mean()
                                y -= baseline
                                
                            style = channel_styles[i % len(channel_styles)]
                            fig.add_trace(go.Scatter(
                                x=x,
                                y=y,
                                mode="lines+markers" if style["symbol"] else "lines",
                                name=f"{group}: {well} (Ch {chan_str})",
                                line=dict(color=mcolors.to_hex(color), dash=style["dash"]),
                                marker=dict(symbol=style["symbol"], size=6) if style["symbol"] else None
                            ))

                            if threshold_enabled:
                                try:
                                    # Remove NaNs
                                    valid = ~np.isnan(x) & ~np.isnan(y)
                                    x_fit = np.array(x[valid], dtype=float)
                                    y_fit = np.array(y[valid], dtype=float)
                                    if len(x_fit) >= 5:  # ensure enough points to fit
                                        popt, _ = curve_fit(four_param_logistic, x_fit, y_fit, maxfev=10000)
                                        ct = inverse_four_pl(threshold_value, *popt)
                                        if ct is not None and x_fit[0] <= ct <= x_fit[-1]:
                                            ct_results.append({
                                                "Group": group,
                                                "Well": well,
                                                "Channel": chan_str if platform == "QuantStudio (QS)" else channel_name,
                                                "Ct": f"{float(ct):.2f}"
                                            })
                                            fig.add_annotation(
                                                x=ct,
                                                y=threshold_value,
                                                text=f"Ct: {ct:.1f}",
                                                showarrow=True,
                                                arrowhead=2,
                                                font=dict(size=10),
                                                bgcolor="white"
                                            )
                                except:
                                    above = y > threshold_value
                                    if any(above):
                                        first_cross = above.idxmax()
                                        if first_cross > 0:
                                            y1, y2 = y[first_cross - 1], y[first_cross]
                                            x1, x2 = x[first_cross - 1], x[first_cross]
                                            ct = x1 + (threshold_value - y1) * (x2 - x1) / (y2 - y1)
                                        else:
                                            ct = x[first_cross]


    
    else:
        for i, channel_name in enumerate(selected_channels):
            chan_str = channel_name 
            match_key = channel_name_map.get(channel_name, channel_name.lower())
            matched_file = next((f for f in uploaded_files if match_key.lower() in f.name.lower()), None)
            if not matched_file:
                st.warning(f"No file found for channel: {channel_name}")
                continue

            df = pd.read_csv(matched_file)

            for group, info in st.session_state["groups"].items():
                wells = info["wells"]
                base_color = info["color"]
                color_list = [base_color] * len(wells) if color_mode == "Solid" else [
                    mcolors.LinearSegmentedColormap.from_list("gradient", [
                        tuple(1 - 0.5 * (1 - c) for c in mcolors.to_rgb(base_color)), mcolors.to_rgb(base_color)
                    ])(j / max(1, len(wells) - 1)) for j in range(len(wells))
                ]

                for well, color in zip(wells, color_list):
                    if well in df.columns:
                        y = df[well].copy()
                        if use_baseline:
                            baseline = y.iloc[:baseline_cycles].mean()
                            y -= baseline
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
                            try:
                                # Remove NaNs
                                valid = ~np.isnan(x) & ~np.isnan(y)
                                x_fit = np.array(x[valid], dtype=float)
                                y_fit = np.array(y[valid], dtype=float)
                                if len(x_fit) >= 5:  # ensure enough points to fit
                                    popt, _ = curve_fit(four_param_logistic, x_fit, y_fit, maxfev=10000)                                 
                                    ct = inverse_four_pl(threshold_value, *popt)

                                    if ct is not None and x_fit[0] <= ct <= x_fit[-1]:
                                        ct_results.append({
                                            "Group": group,
                                            "Well": well,
                                            "Channel": channel_name,
                                            "Ct": f"{float(ct):.2f}"
                                        })
                                        fig.add_annotation(
                                            x=ct,
                                            y=threshold_value,
                                            text=f"Ct: {ct:.1f}",
                                            showarrow=True,
                                            arrowhead=2,
                                            font=dict(size=10),
                                            bgcolor="white"
                                        )
                            except:
                                pass
                                # above = y > threshold_value
                                # if any(above):
                                #     first_cross = above.idxmax()
                                #     if first_cross > 0:
                                #         y1, y2 = y[first_cross - 1], y[first_cross]
                                #         x1, x2 = x[first_cross - 1], x[first_cross]
                                #         ct = x1 + (threshold_value - y1) * (x2 - x1) / (y2 - y1)
                                #     else:
                                #         ct = x[first_cross]

    if threshold_enabled:
        fig.add_hline(y=threshold_value, line_dash="dot", line_color="gray", annotation_text="Threshold", annotation_position="top right")

    fig.update_layout(
        title="Amplification Curves",
        xaxis_title="Cycle",
        yaxis_title="RFU (log)" if log_y else "RFU",
        yaxis_type="log" if log_y else "linear",
        legend=dict(font=dict(size=8),orientation = "v",x= 1.02, y = 1, xanchor ="left",yanchor = "top" ),
        width=800,          # width in pixels
        height=600          # height in pixels (6:8 ratio)
        )

    st.plotly_chart(fig, use_container_width=False)
    
    if ct_results:
        st.subheader("Ct Values")
        st.dataframe(pd.DataFrame(ct_results))
        st.download_button(
            label="Download Ct Results as CSV",
            data=pd.DataFrame(ct_results).to_csv(index=False),
            file_name="Ct_Results.csv",
            mime="text/csv"
        )

