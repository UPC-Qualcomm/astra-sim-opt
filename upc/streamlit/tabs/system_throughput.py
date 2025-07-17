import streamlit as st


def render_system_throughput(df):
    parallelism_strategies = parallelism_strategies_selector(df["filename"].unique())
    df_filtered = get_filtered_data(df, parallelism_strategies)
    compute_system_thoughput(df_filtered, )


def get_filtered_data(df, parallelism_strategies):
    return df[df["filename"] == parallelism_strategies]

def parallelism_strategies_selector(df):
    return st.multiselect(
        "Select Parallelsim Strategies (dp_tp_sp_pp_fsdp)", df, key="system_ps"
    )
