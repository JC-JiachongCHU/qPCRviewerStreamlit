import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")
st.title("qPCR Viewer with 96-Well Plate & Group Color Modes")

# Define 96-well layout
rows = ["A","B","C","D","E","F","G","H"]
cols = list(range(1,13))
well_names = [f"{r}{c}" for r in rows for c in cols]

# Upload Excel file
uploaded_file = st.file_uploader("Upload your qPCR CSV file", type=["csv"])

# Initialize session state for storing group data
if "groups" not in st.session_state:
    st.session_state["groups"] = {}

# --- Group Assignment UI ---
st.subheader("Step 1: Assign Wells to a Group")
group_name = st.text_input("Group Name", "Group 1")
# Predefined color palette
preset_colors = {
    "Red": "#FF0000",
    "Green": "#28A745",
    "Blue": "#007BFF",
    "Orange": "#FD7E14",
    "Purple": "#6F42C1",
    "Brown": "#8B4513",
    "Black": "#000000",
    "Gray": "#6C757D",
    "Custom HEX": None
}

selected_color_name = st.selectbox("Select Group Color", list(preset_colors.keys()))

if selected_color_name == "Custom HEX":
    group_color = st.color_picker("Pick a Custom Color", "#FF0000")
else:
    group_color = preset_colors[selected_color_name]
selected_wells = []

st.write("Select Wells (click checkboxes):")
for r in rows:
    cols_container = st.columns(12)
    for c, col in zip(cols, cols_container):
        well = f"{r}{c}"
        if col.checkbox(well, key=f"{group_name}_{well}"):
            selected_wells.append(well)

if st.button("Add Group"):
    if group_name and selected_wells:
        st.session_state["groups"][group_name] = {
            "color": group_color,
            "wells": selected_wells
        }

# Show current group assignments
st.subheader("Current Groups")
for group, info in st.session_state["groups"].items():
    st.markdown(f"**{group}** ({info['color']}): {', '.join(info['wells'])}")

# --- Color Mode and Channel ---
st.subheader("Step 2: Plot Settings")
color_mode = st.radio("Color mode", ["Solid", "Gradient"])
channel_index = st.number_input("Channel Index (1 = first channel)", min_value=1, step=1, value=1) - 1

# --- Plot ---
if uploaded_file and st.button("Plot Curves"):
    df = pd.read_csv(uploaded_file)
    df = df[df["Well Position"] != "Well Position"]
    df.iloc[:, 5:] = df.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')
    rfu_cols = [col for col in df.columns if col.startswith("X")]

    fig, ax = plt.subplots(figsize=(6, 4))  # width=6 inches, height=4 inches

    for group, info in st.session_state["groups"].items():
        wells = info["wells"]
        base_color = info["color"]

        # Generate gradient if needed
        if color_mode == "Solid":
            color_list = [base_color] * len(wells)
        else:
            # Convert hex to RGB
            rgb = mcolors.to_rgb(base_color)
            
            # Brighten the base color (e.g., scale toward white)
            bright_rgb = tuple(1 - 0.5 * (1 - c) for c in rgb)  # adjust 0.5 as needed
            
            # Create gradient from lightened version to full color
            cmap = mcolors.LinearSegmentedColormap.from_list("gradient", [bright_rgb, rgb])
            color_list = [cmap(i / max(1, len(wells) - 1)) for i in range(len(wells))]

        for well, color in zip(wells, color_list):
            if well in df["Well Position"].values:
                rfu_curve = df[df["Well Position"] == well][rfu_cols[channel_index]].values.flatten()
                ax.plot(rfu_curve, label=f"{group}: {well}", color=color)

    ax.set_title("Amplification Curves by Group")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RFU")
    ax.legend(
        fontsize=6,
        bbox_to_anchor=(1.05, 1),  # Position legend to the right
        loc='upper left',
        borderaxespad=0.
    )
    ax.grid(True)
    st.pyplot(fig)