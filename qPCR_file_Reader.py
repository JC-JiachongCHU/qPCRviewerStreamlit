import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

st.set_page_config(layout="wide")
st.title("qPCR Viewer - Supports QuantStudio & Bio-Rad")

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
platform = st.radio("Select qPCR Platform", ["QuantStudio (QS)", "Bio-Rad"])

# Upload files based on platform
uploaded_files = []
if platform == "QuantStudio (QS)":
    uploaded_file = st.file_uploader("Upload QuantStudio xlsx or csv", type=["xlsx", "csv"])
    if uploaded_file:
        uploaded_files.append(("QS", uploaded_file))
else:
    uploaded_files = st.file_uploader("Upload Bio-Rad CSVs (1 per channel)", type=["csv"], accept_multiple_files=True)

# Initialize session state
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

# Quick select row/column
st.write("Quick Select:")
col1, col2 = st.columns(2)
selected_row = col1.selectbox("Select Entire Row", ["None"] + rows)
selected_col = col2.selectbox("Select Entire Column", ["None"] + [str(c) for c in cols])
if selected_row != "None":
    selected_wells.extend([f"{selected_row}{c}" for c in cols])
if selected_col != "None":
    selected_wells.extend([f"{r}{selected_col}" for r in rows])

# Select all toggle
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

# Deduplicate and sort
selected_wells = sorted(set(selected_wells), key=lambda x: (x[0], int(x[1:])))

if st.button("Add Group"):
    if group_name and selected_wells:
        st.session_state["groups"][group_name] = {"color": group_color, "wells": selected_wells}

# Display groups
st.subheader("Current Groups")
for group, info in st.session_state["groups"].items():
    st.markdown(f"**{group}** ({info['color']}): {', '.join(info['wells'])}")

# Delete group section
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

# Define distinguishable line styles + markers for each channel
channel_styles = [
    {"linestyle": "-", "marker": None},
    {"linestyle": "--", "marker": None},
    {"linestyle": "-", "marker": "^"},
    {"linestyle": "-", "marker": "s"},
    {"linestyle": "-", "marker": "x"}
]

channel_name_map = {
    "FAM": "FAM",
    "HEX": "HEX",
    "Cy5": "Cy5",
    "Cy5.5": "Cy5-5",
    "ROX": "ROX"
}

# option for baseline subtraction
st.subheader("Step 3: Baseline Settings")
use_baseline = st.toggle("Apply Baseline Subtraction", value=False)
baseline_cycles = st.number_input("Use average RFU of first N cycles", min_value=1, max_value=20, value=10)
log_y = st.toggle("Use Semilog Y-axis (log scale)")

# Plotting
if uploaded_files and st.button("Plot Curves"):
    fig, ax = plt.subplots(figsize=(6, 4))

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
                    x = sub_df[cycle_col]

                    for i, chan_str in enumerate(selected_channels):
                        chan_idx = int(chan_str) - 1
                        if 0 <= chan_idx < len(rfu_cols):
                            y = sub_df[rfu_cols[chan_idx]].copy()
                            if use_baseline:
                                baseline = y.iloc[:baseline_cycles].mean()
                                y -= baseline
                            style = channel_styles[i % len(channel_styles)]
                            ax.plot(x, y, label=f"{group}: {well} (Ch {chan_str})",
                                    color=color, linestyle=style["linestyle"],
                                    marker=style["marker"], markersize=4)

    else:
        for i, channel_name in enumerate(selected_channels):
            match_key = channel_name_map.get(channel_name, channel_name.lower())
            matched_file = next((f for f in uploaded_files if match_key.lower() in f.name.lower()), None)
            if not matched_file:
                st.warning(f"No file found for channel: {channel_name}")
                continue

            df = pd.read_csv(matched_file)
            style = channel_styles[i % len(channel_styles)]

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
                        ax.plot(df["Cycle"], y, label=f"{group}: {well} ({channel_name})",
                                color=color, linestyle=style["linestyle"], marker=style["marker"], markersize=4)

    ax.set_title("Amplification Curves by Group")
    ax.set_ylabel("RFU (log scale)" if log_y else "RFU")
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("RFU")
    ax.grid(True)
    ax.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    st.pyplot(fig)
