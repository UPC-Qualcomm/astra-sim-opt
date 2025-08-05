
import streamlit as st
import sections.trace_picker as picker
import pandas as pd
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import helper.constants as constants

@st.cache_data
def load_batch_data(selected_model, selected_config, batch_size_dir="batch_study"):
    batch_res_file = f"../results/{selected_model}/{selected_config}/{batch_size_dir}/{selected_config}.csv"
    df = pd.read_csv(batch_res_file).sort_values(by="exec_cycles", ascending=True)
    return df

@st.cache_data
def get_unique_strategies_and_batches(df):
    strategies = df["dp_mp_sp_pp_sharded"].unique()
    batches = sorted(df['batch'].astype(int).unique())
    return strategies, batches

@st.cache_data
def filter_and_prepare_df(df, selected_strategies, selected_batch_sizes):
    df = df[df["dp_mp_sp_pp_sharded"].isin(selected_strategies) & df["batch"].isin(selected_batch_sizes)]
    df["overlap"] = df['exec_cycles'] - df['exposed_comm_cycles'] - df['exposed_comp_cycles']
    df["strategy"] = df["dp_mp_sp_pp_sharded"]
    return df

@st.cache_data
def plot_simulation_time_breakdown(df, selected_batch_sizes, selected_model, selected_config):
    labels = ['Overlap', 'Exposed Comp', 'Exposed Comm']  # Flipped order
    colors = ['lightblue', 'lightgreen', 'lightcoral']         # Flipped colors
    strategies = df['strategy'].unique()
    cols = 4
    rows = math.ceil(len(strategies) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strategy_df = df[df['strategy'] == strategy].set_index('batch').reindex(selected_batch_sizes)
        overlaps = strategy_df['overlap'].fillna(0).values
        comps = strategy_df['exposed_comp_cycles'].fillna(0).values      # Flipped
        comms = strategy_df['exposed_comm_cycles'].fillna(0).values      # Flipped
        totals = strategy_df['exec_cycles'].fillna(0).values

        x = range(len(selected_batch_sizes))
        bar_width = 0.6
        bottoms_comp = overlaps
        bottoms_comm = overlaps + comps

        ax.bar(x, overlaps, bar_width, label=labels[0], color=colors[0])
        ax.bar(x, comps, bar_width, bottom=bottoms_comp, label=labels[1], color=colors[1])   # Flipped
        ax.bar(x, comms, bar_width, bottom=bottoms_comm, label=labels[2], color=colors[2])   # Flipped

        startegy_label = strategy.split('_')
        ax.set_xticks(x)
        ax.set_xticklabels(selected_batch_sizes, fontsize=constants.XTICK_SIZE)
        ax.set_xlabel("Batch Size", fontsize=constants.LABEL_SIZE)
        ax.set_title(f"DP:{startegy_label[0]}, TP:{startegy_label[1]}, SP:{startegy_label[2]}\nPP:{startegy_label[3]}, FSDP:{startegy_label[4]}", fontsize=constants.TITLE_SIZE, pad=20)  # Add padding to title

        if idx == 0:
            ax.set_ylabel("Cycles", fontsize=constants.LABEL_SIZE)
            ax.tick_params(axis='y', labelsize=constants.YTICK_SIZE)
            ax.yaxis.get_offset_text().set_fontsize(constants.FONT_SIZE)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)
            ax.yaxis.get_offset_text().set_fontsize(constants.FONT_SIZE)
    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add one global legend in a row (horizontal)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc='upper center',
        ncol=len(labels),
        fontsize=constants.LEGEND_SIZE,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Fix: Add tight_layout to prevent overlap
    #filename = "batch_study.svg"
    #plt.savefig(filename, format="svg", transparent=True, bbox_inches='tight', 
    #            pad_inches=0.3, facecolor='white', dpi=300)

    st.pyplot(fig)

@st.cache_data
def plot_3d_simulation_time_breakdown(df, selected_batch_sizes, selected_model, selected_config):
    strategies = df['strategy'].unique()
    strategy_indices = {strategy: i for i, strategy in enumerate(strategies)}
    batch_indices = {batch: i for i, batch in enumerate(selected_batch_sizes)}

    labels = ['overlap', 'exposed_comm_cycles', 'exposed_comp_cycles']
    # Use the same colors as the 2D plot
    colors = ['blue', 'lightcoral', 'lightgreen']
    traces = []

    bar_thickness = 18  # Thicker bars for better visibility
    marker_size = 10    # Larger markers for easier hover

    for i, label in enumerate(labels):
        xs, ys, zs, texts = [], [], [], []
        x_lines, y_lines, z_lines = [], [], []

        for strategy in strategies:
            for batch in selected_batch_sizes:
                row = df[(df['strategy'] == strategy) & (df['batch'] == batch)]
                if row.empty:
                    value = 0
                else:
                    value = row[label].values[0]

                x = batch_indices[batch]
                y = strategy_indices[strategy]

                # Stack base
                base = sum([
                    row[l].values[0] if not row.empty else 0
                    for l in labels[:i]
                ])

                top = base + value

                # Tooltip for both base and top
                tooltip = f"{label.replace('_', ' ')}: {value:.0f}<br>Batch: {batch}<br>Strategy: {strategy}<br>From: {base:.0f} To: {top:.0f}"

                # Center point for marker (for hover)
                xs.append(x)
                ys.append(y)
                zs.append(top)
                texts.append(tooltip)

                # Vertical bar segment (thicker)
                x_lines += [x, x, None]
                y_lines += [y, y, None]
                z_lines += [base, top, None]

        # Add vertical bars (lines) with thicker width
        traces.append(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color=colors[i], width=bar_thickness),
            name=label.replace('_', ' ').title(),
            showlegend=True,
            text=[tooltip for tooltip in texts for _ in range(2)] + [None]*len(xs),
            hoverinfo='text'
        ))

        # Add visible markers for hover only (not legend)
        traces.append(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(size=marker_size, color=colors[i], symbol='circle', opacity=0.9),
            text=texts,
            hoverinfo='text',
            name=f"{label.replace('_', ' ').title()} Value",
            showlegend=False
        ))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='Batch Size',
                tickvals=list(batch_indices.values()),
                ticktext=[str(b) for b in selected_batch_sizes],
                backgroundcolor='rgba(240,240,240,0.8)',
                gridcolor='gray',
                showbackground=True,
                tickangle=45
            ),
            yaxis=dict(
                title='Strategy',
                tickvals=list(strategy_indices.values()),
                ticktext=list(strategy_indices.keys()),
                backgroundcolor='rgba(240,240,240,0.8)',
                gridcolor='gray',
                showbackground=True,
                tickangle=45
            ),
            zaxis=dict(
                title='Cycles',
                backgroundcolor='rgba(240,240,240,0.8)',
                gridcolor='gray',
                showbackground=True
            ),
            camera=dict(
                eye=dict(x=-2.0, y=0.0, z=1.2)  # Batch size on left, strategy on right
            )
        ),
        title=f"3D Simulation Time Breakdown<br>Model: {selected_model} | Topology: {selected_config}",
        margin=dict(l=10, r=10, b=10, t=50),
        height=700,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1
        ),
        showlegend=True
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)
st.subheader(
    "Exploration: Global Batch Size",
    help=(
        "This section allows you to explore the impact of different global batch sizes on the simulation time for the chosen model and configuration.\n"
        "- Analyze how varying batch sizes affect the performance metrics.\n"
        "- Understand the trade-offs between batch size and simulation time."
    )
)

BATCH_SIZE_DIR = "batch_study"
selected_model, selected_config = picker.get_model_and_config()
picker.set_session_peak_perf_bw(selected_config)

df = load_batch_data(selected_model, selected_config, BATCH_SIZE_DIR)
parallelism_strategies, sorted_batches = get_unique_strategies_and_batches(df)
max_select = 8
selected_strategies = st.multiselect(
    "Select Strategies (dp_tp_sp_pp_fsdp) [max 20]", 
    parallelism_strategies, 
    default=parallelism_strategies[:max_select], 
    max_selections=max_select
)

exp_opt = st.radio("Select Exploration Mode", ["Saved Data", "Generate New Data"], key="exploration_mode")

if exp_opt == "Generate New Data":
    st.warning("This feature is under development. Please check back later.")

else:
    if exp_opt == "Saved Data":
        st.info("Using pre-generated data for global batch size exploration.")
        selected_batch_sizes = st.multiselect("Select Batch Sizes", sorted_batches, default=sorted_batches)

        if not selected_strategies or not selected_batch_sizes:
            st.warning("Please select at least one strategy and one batch size.")
            st.stop()

        filtered_df = filter_and_prepare_df(df, selected_strategies, selected_batch_sizes)
        st.markdown("---")
        st.subheader("Simulation Time Breakdown per Batch Size (Stacked Bar, with % labels)")

        plot_type = st.radio(
            "Select Plot Type",
            ["2D Stacked Bar", "3D Stacked Bar"],
            key="plot_type"
        )
        if plot_type == "2D Stacked Bar":
            plot_simulation_time_breakdown(filtered_df, selected_batch_sizes, selected_model, selected_config)
        else:
            plot_3d_simulation_time_breakdown(filtered_df, selected_batch_sizes, selected_model, selected_config)

