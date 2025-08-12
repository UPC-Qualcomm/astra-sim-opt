import os
import json
import pandas as pd
import streamlit as st
import trace_visualization as tv
import roofline_visualization as rv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import helper.constants as constants

def trace_picker():
    st.title("📊 Trace Picker")

    BASE_MODELS_DIR = _get_output_dir()

    selected_model, selected_config = _model_and_config_selection(BASE_MODELS_DIR)

    if selected_model and selected_config:
        return set_sim_input(selected_model, selected_config)
    #else:
    #    st.error(
    #        f"❌ The combination you entered does not correspond to an existing trace file: `{trace_file_name}`"
    #    )

    return -1


def _get_output_dir():
    base_model_output = os.path.abspath(os.path.join(os.getcwd(), "../output"))
    return base_model_output


def _get_configs_dir():
    configs_dir = os.path.abspath(os.path.join(os.getcwd(), "../configuration"))
    return configs_dir


def _get_config_names(model_dir):
    return [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]


def _get_model_names(base_model_dir):
    return [
        d
        for d in os.listdir(base_model_dir)
        if os.path.isdir(os.path.join(base_model_dir, d))
    ]

def model_selector(base_model_dir):
    model_names = _get_model_names(base_model_dir)
    # Create a display map: "model_name_with_underscore" -> "Model Name With Underscore"
    display_map = {name: name.replace('_', ' ') for name in model_names}
    display_names = [display_map[name] for name in model_names]
    selected_display = st.selectbox("Select a Model", display_names)
    # Reverse map to get the original model name
    reverse_map = {v: k for k, v in display_map.items()}
    return reverse_map[selected_display]

def config_selector(base_model_dir, selected_model):
    model_dir = os.path.join(base_model_dir, selected_model)
    config_names = _get_config_names(model_dir)
    display_map = {
        "2D_Torus": "2D Torus",
        "3D_Torus": "3D Torus",
        "Dragonfly": "Dragonfly",
        "FoldedClos": "Folded-Clos"
    }
    display_names = [display_map.get(name, name) for name in config_names]
    selected_display = st.selectbox("Select a Network Topology", display_names)
    reverse_map = {v: k for k, v in display_map.items()}
    selected_config = reverse_map.get(selected_display, selected_display)
    return selected_config

def _model_and_config_selection(base_model_dir):
    col_model, col_config = st.columns([1, 1])
    with col_model:
        selected_model = model_selector(base_model_dir)

    with col_config:
        selected_config = None
        if selected_model:
            selected_config = config_selector(base_model_dir, selected_model)

    return selected_model, selected_config


def _parallelism_startegy_form(selected_model, selected_config, base_model_dir):
    st.write("Select the parallelism strategy:")

    # Directory containing trace files
    base_dir = os.path.join(base_model_dir, selected_model, selected_config)
    # List all files ending with _trace_matched_timing.csv
    trace_files = [
        f for f in os.listdir(base_dir)
        if f.endswith("_trace_matched_timing.csv")
    ]

    # Extract strategy part before .seq (e.g., "4_1_2_8_0")
    strategies = []
    strategy_map = {}
    seq_batch_map = {}
    for f in trace_files:
        strategy = f.split('.seq')[0]
        if strategy not in strategies:
            strategies.append(strategy)
            strategy_map[strategy] = []
        strategy_map[strategy].append(f)
        # Extract seq and batch values
        parts = f.split('.')
        seq_val = None
        batch_val = None
        for part in parts:
            if part.startswith('seq_'):
                seq_val = part.split('_')[1]
            if part.startswith('batch_'):
                batch_val = part.split('_')[1]
        if strategy not in seq_batch_map:
            seq_batch_map[strategy] = []
        seq_batch_map[strategy].append((seq_val, batch_val, f))

    # Format strategies for display
    display_strategies = []
    strategy_display_map = {}
    for strat in strategies:
        dp, tp, sp, pp, fsdp = strat.split("_")
        display = f"DP:{dp}, TP:{tp}, SP:{sp}, PP:{pp}, FSDP:{fsdp}"
        display_strategies.append(display)
        strategy_display_map[display] = strat

    if not display_strategies:
        st.warning("No trace files found for this model/config.")
        return

    selected_display_strategy = st.selectbox("Parallelism Strategy", display_strategies)
    selected_strategy = strategy_display_map[selected_display_strategy]

    # Get all seq/batch for selected strategy
    seq_batch_list = seq_batch_map[selected_strategy]

    # If only one file, just show it
    if len(seq_batch_list) == 1:
        seq_val, batch_val, selected_file = seq_batch_list[0]
        st.write(f"Only one trace file found: seq={seq_val}, batch={batch_val}")
    else:
        # Show all possible seq/batch values
        options = [
            f"seq={seq}, batch={batch}" for seq, batch, _ in seq_batch_list
        ]
        selected_option = st.selectbox("Select seq/batch", options)
        idx = options.index(selected_option)
        selected_file = seq_batch_list[idx][2]

    #submitted = st.button("Submit")

    #if submitted and selected_file:
    # Clear session state (if needed) - preserving important values
    #preserved_keys = {'temp_dir', 'session_id', 'df_matched', 'peak_perf', 'peak_bw', 'show_npu_plots'}
    #preserved_values = {}
    #
    ## Save values we want to keep
    #for key in preserved_keys:
    #    if key in st.session_state:
    #        preserved_values[key] = st.session_state[key]
    #
    ## Clear all session state
    #for key in list(st.session_state.keys()):
    #    del st.session_state[key]
    #
    ## Restore preserved values
    #for key, value in preserved_values.items():
    #    st.session_state[key] = value

    # Parse parallelism degrees from strategy
    dp, tp, sp, pp, sharding_val = selected_strategy.split("_")
    trace_file_name = selected_file
    csv_trace_file = os.path.join(base_dir, trace_file_name)
    file_base = selected_file.split("_trace_matched_timing")[0]
    res_path = os.path.abspath(
        os.path.join(os.getcwd(), f"../results/{selected_model}/{selected_config}/")
    )
    res_file = os.path.join(res_path, f"{file_base}_res.csv")
    log_file = os.path.join(base_dir, f"{file_base}.log")

    _set_session_df(base_dir, file_base, csv_trace_file)

    st.session_state.update(
        {
            "csv_trace_file": csv_trace_file,
            "trace_file_name": trace_file_name,
            "res_file": res_file,
            "log_file": log_file,
            "res_path": res_path,
            "file_base": file_base,
        }
    )

def _detect_file_change(csv_trace_file):
    if (
        "last_trace_file" not in st.session_state
        or st.session_state.last_trace_file != csv_trace_file
    ):
        st.session_state.last_trace_file = csv_trace_file
        st.session_state.timestep_idx = 0
        st.session_state.selected_timestep = None

def _set_session_df(base_dir, file_base, csv_trace_file):
    timed_file_name = f"{file_base}_trace_matched_timiming.csv"
    timed_csv = os.path.join(base_dir, timed_file_name)
    # Always update df_matched when a new trace file is selected
    st.session_state.df_matched = pd.read_csv(csv_trace_file)  

def set_session_peak_perf_bw(selected_config):
    CONFIGS_DIR = _get_configs_dir()
    config_file = os.path.join(CONFIGS_DIR, f"{selected_config}_sys.json")
    with open(config_file, "r") as f:
        config_file_content = f.read()
        config_file_content = json.loads(config_file_content)
        st.session_state.peak_perf = config_file_content.get("peak-perf", 300)
        st.session_state.peak_bw = config_file_content.get("local-mem-bw", 2000)
        

def get_model_and_config():
    BASE_MODELS_DIR = _get_output_dir()

    selected_model, selected_config = _model_and_config_selection(BASE_MODELS_DIR)

    return selected_model, selected_config

@st.cache_data
def get_all_parallelism_strategies_data(model, config, option):
    csv_files = get_files_list(os.path.join(_get_output_dir(), model, config), "_trace_matched_timing.csv")
    records = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        df.fillna(0, inplace=True) #TODO: Remove after fixing the file generation
       
        if option == 'slowest':
            mem, comp, comm = get_slowest_npu(df)
        elif option == 'average':
            mem, comp, comm = get_averaged_npus(df)
        elif option == 'fastest':
            mem, comp, comm = get_fastest_npu(df)
        else:
            return []
        
        base_name = os.path.basename(file)
        clean_name = base_name.split('_trace')[0]
        dp, tp, sp, pp, fsdp = clean_name.split('.')[0].split('_')
        total = mem + comp + comm
        records.append({
            'file_name': clean_name,
            'dp': int(dp),
            'tp': int(tp),
            'sp': int(sp),
            'pp': int(pp),
            'fsdp': int(fsdp),
            'mem': mem,
            'comp': comp,
            'comm': comm,
            'mem_percent': mem/total*100,
            'comp_percent': comp/total*100,
            'comm_percent': comm/total*100,
            'total': total
        })

    return pd.DataFrame(records)

@st.cache_data
def get_files_list(base_dir, end_with_str):
    files = os.listdir(base_dir)
    filtered = list()
    for file in files:
        if file.endswith(end_with_str):
            filtered.append(os.path.join(base_dir, file))
    return filtered

@st.cache_data
def get_slowest_npu(df):
    max_tick = df['callback_tick'].max()
    slowest_npu = df.loc[df['callback_tick'] == max_tick, 'sys_id'].iloc[0]
        
    return rv.get_info(
            df,
            npu=slowest_npu,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )

@st.cache_data
def get_fastest_npu(df):
    max_callback = df.groupby('sys_id')['callback_tick'].max().reset_index()
    fastest_npu = max_callback.loc[max_callback['callback_tick'].idxmin(), 'sys_id']
        
    return rv.get_info(
            df,
            npu=fastest_npu,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )

@st.cache_data
def get_averaged_npus(df):
    max_npu = df["sys_id"].max() + 1
    mem_tot = 0
    comp_tot = 0
    comm_tot = 0
    for npu in range(max_npu):
        mem, comp, comm = rv.get_info(
            df,
            npu=npu,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )
        mem_tot = mem_tot + mem
        comp_tot = comp_tot + comp
        comm_tot = comm_tot + comm

    return mem_tot/max_npu, comp_tot/max_npu, comm_tot/max_npu


def get_max_comm_npu(df):
    max_npu = df["sys_id"].max()
    max_comm = 0
    for npu in range(max_npu):
        _, _, comm = rv.get_info(
            df,
            npu=npu,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )
        max_comm = max(comm, max_comm)
    return max_comm

def get_max_comp_npu(df):
    max_npu = df["sys_id"].max()
    max_comp = 0
    for npu in range(max_npu):
        _, comp, _ = rv.get_info(
            df,
            npu=npu,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )
        max_comp = max(comp, max_comp)
    return max_comp

def get_max_mem_npu(df):
    max_npu = df["sys_id"].max()
    max_mem = 0
    for npu in range(max_npu):
        mem, _, _ = rv.get_info(
            df,
            npu=npu,
            perf=st.session_state.peak_perf,
            bw=st.session_state.peak_bw,
        )
        max_mem = max(mem, max_mem)

    return max_mem



#def plot_experiments_bound_breakdown(df, chunk_size=40):
#    plots = []
#
#    # Sort by 'comm' ascending
#    df_sorted = df.sort_values(by=['dp','tp','sp','pp','fsdp'], ascending=True).reset_index(drop=True)
#    min_comm_idx = df_sorted['comm'].idxmin()
#    # Total number of plots needed
#    num_chunks = math.ceil(len(df_sorted) / chunk_size)
#
#    for i in range(num_chunks):
#        start = i * chunk_size
#        end = min((i + 1) * chunk_size, len(df_sorted))
#
#        chunk = df_sorted.iloc[start:end]
#
#        file_names = chunk['file_name']
#        mem_values = chunk['mem']/chunk['total']*100
#        comp_values = chunk['comp']/chunk['total']*100
#        comm_values = chunk['comm']/chunk['total']*100
#
#        x = np.arange(len(chunk))
#
#        fig, ax = plt.subplots(figsize=(12, 6))
#
#        ax.bar(x, mem_values, label='Memory Bound OPs', color='skyblue')
#        ax.bar(x, comp_values, bottom=mem_values, label='Compute Bound OPs', color='lightgreen')
#        if (min_comm_idx <= end):
#            comm_colors = ['salmon'] * len(df_sorted)
#            comm_colors[min_comm_idx-start] = 'red'
#            ax.bar(x, comm_values, bottom=mem_values + comp_values, label='Communication', color=comm_colors)
#        else:
#            ax.bar(x, comm_values, bottom=mem_values + comp_values, label='Communication', color='salmon')
#
#        # Add labels on top of Communication Bound
#        for xi, mem, comm, comp in zip(x, mem_values, comm_values, comp_values):
#            ax.text(xi, comp + mem + comm / 2, f"{comm:.1f}", ha='center', va='center', fontsize=8, color='black')
#
#        ax.set_xticks(x)
#        ax.set_xticklabels(file_names, rotation=45, ha='right')
#        ax.set_xlabel('Parallelism strategy: DP,TP,SP,PP,FSDP')
#        ax.set_ylabel('Time (%)')
#        ax.set_title(f'Bound Breakdown per Experiment (Bars {start + 1} to {end})')
#        ax.legend()
#
#        fig.tight_layout()
#        plots.append(fig)
#
#    return plots
@st.cache_data
def plot_experiments_bound_breakdown(df, chunk_size=40):
    plt.rcParams.update({
        'font.size': constants.FONT_SIZE,  # change this value as needed
        'axes.titlesize': constants.TITLE_SIZE,
        'axes.labelsize': constants.LABEL_SIZE,
        'xtick.labelsize': constants.XTICK_SIZE,
        'ytick.labelsize': constants.YTICK_SIZE,
        'legend.fontsize': constants.LEGEND_SIZE
    })
    
    plots = []

    # Sort by parallelism columns
    df_sorted = df#.sort_values(by=['dp','tp','sp','pp','fsdp'], ascending=True).reset_index(drop=True)
    num_npus = int(df_sorted['dp'].head(1)) * int(df_sorted['tp'].head(1)) * int(df_sorted['sp'].head(1)) * int(df_sorted['pp'].head(1))
    # Normalize 'total' globally and map to colors
    norm = plt.Normalize(df_sorted['total'].min(), df_sorted['total'].max())
    cmap = plt.cm.Reds
    total_colors = cmap(norm(df_sorted['total']))  # Array of RGBA color

    low, high = np.percentile(df_sorted['total'], [10, 100])
    norm = plt.Normalize(low, high)
    cmap = plt.cm.Reds
    total_colors = cmap(norm(df_sorted['total'].clip(low, high)))

    num_chunks = math.ceil(len(df_sorted) / chunk_size)
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df_sorted))

        chunk = df_sorted.iloc[start:end]

        file_names = chunk['file_name'].str.split('.').str[0]  # Use only the base name without suffix
        mem_values = chunk['mem'] #/ chunk['total'] * 100
        comp_values = chunk['comp']# / chunk['total'] * 100
        comm_values = chunk['comm'] #/ chunk['total'] * 100

        x = np.arange(len(chunk))

        fig, ax = plt.subplots(figsize=(32, 8))

        ax.bar(x, mem_values, label='Memory\nBound OPs (%)', color='blue')
        ax.bar(x, comp_values, bottom=mem_values, label='Compute\nBound OPs (%)', color='lightgreen')
        ax.bar(x, comm_values, bottom=mem_values + comp_values, label='Exposed\nComm. (%)', color='lightcoral')

        for xi, mem, comm, comp in zip(x, mem_values, comm_values, comp_values):
            total = mem + comp + comm
            
            # Memory label
            if mem > 0:
                ax.text(float(xi), mem / 2, f"{mem/total * 100:.1f}", ha='center', va='center', 
                       fontsize=constants.IN_PLOT_LABEL_SIZE, color='black')
            
            # Compute label
            if comp > 0:
                ax.text(float(xi), mem + comp / 2, f"{comp/total * 100:.1f}", ha='center', va='center', 
                       fontsize=constants.IN_PLOT_LABEL_SIZE, color='black')
            
            # Communication label
            if comm > 0:
                ax.text(float(xi), mem + comp + comm / 2, f"{comm/total * 100:.1f}", ha='center', va='center', 
                       fontsize=constants.IN_PLOT_LABEL_SIZE, color='black')

        ax.set_xticks(x)
        ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=constants.XTICK_SIZE)
        ax.tick_params(axis='y', labelsize=constants.YTICK_SIZE)
        ax.yaxis.get_offset_text().set_fontsize(constants.YTICK_SIZE)
        ax.set_xlabel('Combination of parallelism strategies: DP,TP,SP,PP,FSDP', fontsize=constants.LABEL_SIZE)
        ax.set_ylabel('Time (Cycles)', fontsize=constants.LABEL_SIZE)
        ax.set_title(f'Execution Breakdown per Experiment - #NPUs is {num_npus} - (Experiments {start + 1} to {end})', fontsize=constants.TITLE_SIZE)
        if (i == 0):
            ax.legend(
                fontsize=constants.LEGEND_SIZE,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )
        fig.tight_layout()
        #plt.rcParams['svg.fonttype'] = 'none'
        #fig.savefig(f"parallel_st{i}.svg", dpi=600, format='svg')
        plots.append(fig)

    return plots

def plot_with_total_color(df, degree, option):
    fig, ax = plt.subplots(figsize=(8, 8))
    unique_vals = df[degree].unique()
    if option == 'average':
        totals = [df[df[degree] == val]['total'].mean() for val in unique_vals]
    elif option == 'slowest':
        totals = [df[df[degree] == val]['total'].max() for val in unique_vals]
    elif option == 'fastest':
        totals = [df[df[degree] == val]['total'].min() for val in unique_vals]
    else:
        return
    # Normalize comm for color mapping
    norm = plt.Normalize(min(totals), max(totals))
    cmap = plt.cm.Reds
    # Build a palette dict mapping each unique value to a color
    palette = {val: cmap(norm(total)) for val, total in zip(unique_vals, totals)}
    sns.barplot(x=degree, y='comm_percent', hue=degree, data=df, ax=ax, palette=palette, legend=False)
    ax.set_ylabel('Communication (%)', fontsize=18)
    ax.set_xlabel(degree.upper(), fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(f'Communication percent vs {degree.upper()} Degree\nwith lighter coloer for faster simulations', fontsize=18)
    return fig

def plot_with_total_color_total(df, degree):
    fig, ax = plt.subplots(figsize=(8, 8))    
    sns.barplot(x=degree, y='total', data=df, ax=ax)
    ax.set_ylabel('Total cycles', fontsize=18)
    ax.set_xlabel(degree.upper(), fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(f'Total Cycles vs {degree.upper()} Degree', fontsize=18)
    return fig


def set_sim_input(selected_model, selected_config):
    base_dir = _get_output_dir()
    _parallelism_startegy_form(selected_model, selected_config, base_dir)

    if "csv_trace_file" in st.session_state and os.path.isfile(
        st.session_state["csv_trace_file"]
    ):
        csv_trace_file = st.session_state["csv_trace_file"]
        trace_file_name = st.session_state["trace_file_name"]
        res_file = st.session_state["res_file"]
        log_file = st.session_state["log_file"]
        res_path = st.session_state["res_path"]

        _detect_file_change(csv_trace_file)

        st.success(f"✅ Trace file for parallelism strategy `{trace_file_name.split('.')[0]}` is found")

        set_session_peak_perf_bw(selected_config)

        dp, tp, sp, pp, sharding_val = Path(csv_trace_file).stem.split('.')[0].split("_")

        st.session_state.show_npu_plots = True

        return {
            "sim_dir": os.path.join(
                base_dir, selected_model, selected_config
            ),
            "log": log_file,
            "res_log": res_file,
            "dp": dp,
            "tp": tp,
            "sp": sp,
            "pp": pp,
            "sharding_val": sharding_val,
        }
    else:
        st.warning("Please submit the configuration to proceed.")