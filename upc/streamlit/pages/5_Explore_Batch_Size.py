
import streamlit as st
import sections.trace_picker as picker
import pandas as pd
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import helper.constants as constants

@st.cache_data(show_spinner='Loading Batch Data...')
def load_batch_data(selected_model, selected_config, batch_size_dir="batch_study"):
    batch_res_file = f"../results/{selected_model}/{selected_config}/{batch_size_dir}/{selected_config}.csv"
    df = pd.read_csv(batch_res_file).sort_values(by="exec_cycles", ascending=True)
    return df

@st.cache_data(show_spinner='Getting Unique Strategies and Batches...')
def get_unique_strategies_and_batches(df):
    strategies = df["dp_mp_sp_pp_sharded"].unique()
    batches = sorted(df['batch'].astype(int).unique())
    return strategies, batches

@st.cache_data(show_spinner='Filtering and Preparing DataFrame...')
def filter_and_prepare_df(df, selected_strategies, selected_batch_sizes):
    df = df[df["dp_mp_sp_pp_sharded"].isin(selected_strategies) & df["batch"].isin(selected_batch_sizes)]
    df["overlap"] = df['exec_cycles'] - df['exposed_comm_cycles'] - df['exposed_comp_cycles']
    df["strategy"] = df["dp_mp_sp_pp_sharded"]
    return df

@st.cache_data(show_spinner='Plotting Simulation Time Breakdown...')
def plot_simulation_time_breakdown(df, selected_batch_sizes, selected_model, selected_config):
    labels = ['Overlap', 'Exposed Comp', 'Exposed Comm']  # Flipped order
    colors = ['lightblue', 'lightgreen', 'lightcoral']         # Flipped colors
    strategies = df['strategy'].unique()
    cols = 4
    rows = math.ceil(len(strategies) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 10))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strategy_df = df[df['strategy'] == strategy].set_index('batch').reindex(selected_batch_sizes)
        overlaps = strategy_df['overlap'].fillna(0).values
        comps = strategy_df['exposed_comp_cycles'].fillna(0).values      # Flipped
        comms = strategy_df['exposed_comm_cycles'].fillna(0).values      # Flipped
        totals = strategy_df['exec_cycles'].fillna(0).values
        
        # Get percentage values for labels
        comp_percents = strategy_df['exposed_comp_cycles_percent'].fillna(0).values
        comm_percents = strategy_df['exposed_comm_cycles_percent'].fillna(0).values
        overlap_percents = 100 - (comp_percents + comm_percents)

        x = range(len(selected_batch_sizes))
        bar_width = 0.6
        bottoms_comp = overlaps
        bottoms_comm = overlaps + comps

        bars1 = ax.bar(x, overlaps, bar_width, label=labels[0], color=colors[0])
        bars2 = ax.bar(x, comps, bar_width, bottom=bottoms_comp, label=labels[1], color=colors[1])   # Flipped
        bars3 = ax.bar(x, comms, bar_width, bottom=bottoms_comm, label=labels[2], color=colors[2])   # Flipped

        # Add percentage labels to each bar segment
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            # Overlap label (bottom segment)
            if overlaps[i] > 0:
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height()/2,
                       f'{overlap_percents[i]:.1f}%', ha='center', va='center', 
                       fontsize=constants.FONT_SIZE-2, fontweight='normal')
            
            # Exposed comp label (middle segment)
            if comps[i] > 0:
                ax.text(bar2.get_x() + bar2.get_width()/2, bottoms_comp[i] + bar2.get_height()/2,
                       f'{comp_percents[i]:.1f}%', ha='center', va='center', 
                       fontsize=constants.FONT_SIZE-2, fontweight='normal')
            
            # Exposed comm label (top segment)
            if comms[i] > 0:
                ax.text(bar3.get_x() + bar3.get_width()/2, bottoms_comm[i] + bar3.get_height()/2,
                       f'{comm_percents[i]:.1f}%', ha='center', va='center', 
                       fontsize=constants.FONT_SIZE-2, fontweight='normal')

        startegy_label = strategy.split('_')
        ax.set_xticks(x)
        ax.set_xticklabels(selected_batch_sizes, fontsize=constants.XTICK_SIZE)
        ax.set_xlabel("Batch Size", fontsize=constants.LABEL_SIZE)
        ax.set_title(f"DP:{startegy_label[0]}, TP:{startegy_label[1]}, SP:{startegy_label[2]}\nPP:{startegy_label[3]}, FSDP:{startegy_label[4]}", fontsize=constants.TITLE_SIZE, pad=20)  # Add padding to title

        # Show y-axis labels on the first subplot of each row
        if idx % cols == 0:
            ax.set_ylabel("Time (Cycles)    ", fontsize=constants.LABEL_SIZE)
            ax.tick_params(axis='y', labelsize=constants.YTICK_SIZE)
            ax.yaxis.get_offset_text().set_fontsize(constants.FONT_SIZE)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)
            ax.yaxis.get_offset_text().set_fontsize(constants.FONT_SIZE)
    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add one legend in a row (horizontal)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc='upper center',
        #bbox_to_anchor=(0.5, 1.),  # Move legend higher above the titles
        ncol=len(labels),
        fontsize=constants.LEGEND_SIZE,#+4,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Fix: Add tight_layout to prevent overlap
    #filename = "batch_study.svg"
    #plt.savefig(filename, format="svg", transparent=True, bbox_inches='tight', 
    #            pad_inches=0.3, facecolor='white', dpi=300)

    st.pyplot(fig)

@st.cache_data(show_spinner='Plotting 3D Simulation Time Breakdown...')
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

    # Format model name (remove underscores)
    formatted_model = selected_model.replace('_', ' ')
    
    # Format config name
    config_mapping = {
        '2D_Torus': '2D Torus',
        '3D_Torus': '3D Torus', 
        'Dragonfly': 'Dragonfly',
        'FoldedClos': 'Folded-Clos'
    }
    formatted_config = config_mapping.get(selected_config, selected_config)

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
        title=dict(
            text=f"3D Simulation Time Breakdown<br>Model: {formatted_model} | Topology: {formatted_config}",
            x=0.5,  # Center the title horizontally
            xanchor='center'
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        height=700,
        legend=dict(
            x=0.85, y=0.5,
            xanchor='left',
            yanchor='middle',
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
    "Exploration: Batch Size",
    help=(
        "This section allows you to explore the impact of different batch sizes on the simulation time for the chosen model and configuration.\n"
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

#### TODO exp_opt = st.radio("Select Exploration Mode", ["Saved Data", "Generate New Data"], key="exploration_mode")
exp_opt = "Saved Data"
if exp_opt == "Generate New Data":
    st.warning("This feature is under development. Please check back later.")

else:
    if exp_opt == "Saved Data":
        ####st.info("Using pre-generated data for batch size exploration.")
        selected_batch_sizes = st.multiselect("Select Batch Sizes", sorted_batches, default=sorted_batches)

        #TODO: Switch variables rather than hardcoded values
        st.info(f"""
            **Peak performance for Single NPU**: {989} TFLOPs, \t
            **Peak memory bandwidth**: {3350} GB/s, \n
            **Inter node Bandwidth**: {200} GB/s, \t
            **Intra node Bandwidth**: {900} GB/s, \t
            **Number of NPUs**: {32}
        """)
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
            st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666; margin-top: 1em;'>Execution time breakdown showing how different batch sizes affect overlap (blue), exposed computation (green), and exposed communication (red) for each parallelism strategy. Percentage labels indicate the proportion of each component. Higher overlap and lower exposed communication generally indicate better performance.</p>", unsafe_allow_html=True)
        else:
            plot_3d_simulation_time_breakdown(filtered_df, selected_batch_sizes, selected_model, selected_config)
            st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666; margin-top: 1em;'>Interactive 3D visualization of execution time breakdown across batch sizes and parallelism strategies. Each vertical bar segment represents different execution components stacked on top of each other. Use mouse controls to rotate and zoom for better exploration of the performance landscape.</p>", unsafe_allow_html=True)

