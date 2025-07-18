import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Exec Cycles Analysis", layout="wide")

# --- App Title
st.title("Exec Cycles vs Batch Size Analysis")

# --- Directory with experiments
base_experiments_path = "/media/mohammad/extension/experiments/astra-sim/upc/results"  # modify if needed

# --- List models and topologies available
models = [d for d in os.listdir(base_experiments_path) if os.path.isdir(os.path.join(base_experiments_path, d))]

if not models:
    st.error("No models found in the experiments directory.")
    st.stop()

model_name = 'GPT_40B_batch'

# Get topologies under selected model
model_path = os.path.join(base_experiments_path, model_name)
topologies = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]

if not topologies:
    st.error("No topologies found under this model.")
    st.stop()

topology_name = 'FoldedClos'

# --- Load available files
folder_path = os.path.join(base_experiments_path, model_name, topology_name)
all_files = [f for f in os.listdir(folder_path) if f.endswith("_res.csv")]

if not all_files:
    st.error("No CSV result files found in this topology folder.")
    st.stop()

# --- Extract all available strategies and batch sizes
available_strategies = sorted(set("_".join(f.replace("_res.csv", "").split("_")[:5]) for f in all_files))
available_batch_sizes = sorted(set(int(f.replace("_res.csv", "").split("_")[5]) for f in all_files))

# --- User selections
selected_strategies = st.multiselect("Select Strategies (dp_tp_sp_pp_fsdp)", available_strategies, default=available_strategies)
selected_batch_sizes = st.multiselect("Select Batch Sizes", available_batch_sizes, default=available_batch_sizes)

# --- If no selection, stop
if not selected_strategies or not selected_batch_sizes:
    st.warning("Please select at least one strategy and one batch size.")
    st.stop()

# --- Process files
records = []

for filename in all_files:
    parts = filename.replace("_res.csv", "").split("_")
    strategy = "_".join(parts[:5])
    batch_size = int(parts[5])

    if strategy not in selected_strategies or batch_size not in selected_batch_sizes:
        continue

    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)

    if df.empty or 'exec_cycles' not in df.columns:
        continue

    max_row = df.loc[df['exec_cycles'].idxmax()]

    records.append({
        "strategy": strategy,
        "batch_size": batch_size,
        "exec_cycles": max_row['exec_cycles']
    })

# --- Build dataframe
if not records:
    st.error("No matching data found for the selected combinations.")
    st.stop()

result_df = pd.DataFrame(records)

# --- Show data table
st.subheader("Resulting DataFrame")
st.dataframe(result_df.sort_values(["strategy", "batch_size"]))

# --- Plotting
st.subheader("Exec Cycles vs Batch Size Plot")

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.lineplot(data=result_df, x="batch_size", y="exec_cycles", hue="strategy", marker="o")

plt.title(f"Time (Cycles) vs Batch Size\nModel: GPT-30B | Topology: {topology_name} | #NPUs: 32")
plt.xlabel("Batch Size")
plt.ylabel("Time (Cycles)")
plt.xscale("log", base=2)
plt.legend(title="Strategy")
plt.tight_layout()

st.pyplot(plt)


# Make sure required columns are there in all files
missing_cols_files = []
breakdown_records = []

for filename in all_files:
    parts = filename.replace("_res.csv", "").split("_")
    strategy = "_".join(parts[:5])
    batch_size = int(parts[5])

    if strategy not in selected_strategies or batch_size not in selected_batch_sizes:
        continue

    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)

    required_cols = {'exec_cycles', 'exposed_comm_cycles', 'exposed_comp_cycles'}
    if not required_cols.issubset(df.columns):
        missing_cols_files.append(filename)
        continue

    max_row = df.loc[df['exec_cycles'].idxmax()]

    overlap = max_row['exec_cycles'] - max_row['exposed_comm_cycles'] - max_row['exposed_comp_cycles']

    breakdown_records.append({
        "strategy": strategy,
        "batch_size": batch_size,
        "Overlap": overlap,
        "Exposed Comm": max_row['exposed_comm_cycles'],
        "Exposed Comp": max_row['exposed_comp_cycles'],
        "exec_cycles": max_row['exec_cycles']
    })

# Check if no valid data
if not breakdown_records:
    st.error("No valid files with required columns ('exec_cycles', 'exposed_comm_cycles', 'exposed_comp_cycles') found.")
    st.stop()

breakdown_df = pd.DataFrame(breakdown_records)


# Reshape the dataframe to long form for seaborn
breakdown_long_df = breakdown_df.melt(
    id_vars=["strategy", "batch_size", "exec_cycles"],
    value_vars=["Overlap", "Exposed Comm", "Exposed Comp"],
    var_name="Component",
    value_name="Cycles"
)

# Calculate percentages
breakdown_long_df['Percentage'] = (breakdown_long_df['Cycles'] / breakdown_long_df['exec_cycles']) * 100

# Colors and labels
labels = ['Overlap', 'Exposed Comm', 'Exposed Comp']
colors = ['blue', 'lightcoral', 'lightgreen']
color_dict = dict(zip(labels, colors))

st.subheader("Exec Cycles Breakdown per Batch Size (Stacked Bar, with % labels)")

# Sort batch sizes for consistent plotting order
sorted_batches = sorted(breakdown_df['batch_size'].unique())

# Colors and labels
labels = ['Overlap', 'Exposed Comm', 'Exposed Comp']
colors = ['blue', 'lightcoral', 'lightgreen']

# Initialize bar positions
x = range(len(sorted_batches))
bar_width = 0.6

plt.figure(figsize=(14, 7))

# Prepare data: for each label, get value list in batch size order
for strategy in breakdown_df['strategy'].unique():
    st.write(f"**Strategy:** `{strategy}`")

    strategy_df = breakdown_df[breakdown_df['strategy'] == strategy].set_index('batch_size').loc[sorted_batches]
    overlaps = strategy_df['Overlap'].values
    comms = strategy_df['Exposed Comm'].values
    comps = strategy_df['Exposed Comp'].values
    totals = strategy_df['exec_cycles'].values

    # Stack bottom positions
    bottoms_comm = overlaps
    bottoms_comp = overlaps + comms

    plt.figure(figsize=(14, 7))
    
    # Plot bars
    plt.bar(x, overlaps, bar_width, label='Overlap', color='blue')
    plt.bar(x, comms, bar_width, bottom=bottoms_comm, label='Exposed Comm', color='lightcoral')
    plt.bar(x, comps, bar_width, bottom=bottoms_comp, label='Exposed Comp', color='lightgreen')

    # Add percentage labels
    for i in range(len(x)):
        total = totals[i]
        if total == 0:
            continue

        # Overlap %
        plt.text(x[i], overlaps[i]/2, f"{overlaps[i]/total*100:.1f}%", ha='center', va='center', color='white', fontsize=10)
        # Comm %
        plt.text(x[i], bottoms_comm[i] + comms[i]/2, f"{comms[i]/total*100:.1f}%", ha='center', va='center', color='black', fontsize=10)
        # Comp %
        plt.text(x[i], bottoms_comp[i] + comps[i]/2, f"{comps[i]/total*100:.1f}%", ha='center', va='center', color='black', fontsize=10)

    plt.xticks(x, sorted_batches)
    plt.xlabel("Batch Size")
    plt.ylabel("Cycles")
    plt.title(f"Exec Cycles Breakdown \nStrategy: {strategy}\nModel: GPT30B | Topology: {topology_name}")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)