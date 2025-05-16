import streamlit as st
import pandas as pd
import numpy as np
import os
import trace_visualization as tv
import roofline_visualization as rv
import simulation_res as sr
import time
import subprocess
import json
import streamlit as st
import sections.generate_workload as gen
import sections.astrasim as astra
import sections.trace_viewer as viewer
import sections.trace_picker as picker
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Bulk analysis", layout="wide")

st.title("Parallelism Strategy - Execution time Breakdown")


selected_model, selected_config = picker.get_model_and_config()
picker.set_session_peak_perf_bw(selected_config)

st.markdown("---")

col1, col2 = st.columns([11, 2])
with col2:
    option = st.radio(
    "Select Bound Analysis Option:",
    options=['slowest', 'fastest', 'average'],
    format_func=lambda x: {
        'slowest': 'Slowest NPU',
        'fastest': 'Fastest NPU',
        'average': 'Average Across all NPUs'
    }[x])

with col1:

    st.info(f"""
        **Peak performance**: {st.session_state.peak_perf} TFLOPs, \t
        **Peak bandwidth**: {st.session_state.peak_bw} GB/s
    """)
df_result = picker.get_all_parallelism_strategies_data(selected_model, selected_config, option)

#from scipy.stats import gmean
#
#def geom_mean_agg(group):
#    return pd.Series({
#        'total': gmean(group['total']),
#        'comm_percent': gmean(group['comm_percent'])
#    })
#
#st.subheader("Geometric Mean Communication Bound per DP degree")
#st.dataframe(
#    df_result.groupby('dp').apply(geom_mean_agg).reset_index().sort_values('total')
#)
#
#st.subheader("Geometric Mean Communication Bound per TP degree")
#st.dataframe(
#    df_result.groupby('tp').apply(geom_mean_agg).reset_index().sort_values('total')
#)
#
#st.subheader("Geometric Mean Communication Bound per SP degree")
#st.dataframe(
#    df_result.groupby('sp').apply(geom_mean_agg).reset_index().sort_values('total')
#)
#
#st.subheader("Geometric Mean Communication Bound per PP degree")
#st.dataframe(
#    df_result.groupby('pp').apply(geom_mean_agg).reset_index().sort_values('total')
#)
#
#st.subheader("Geometric Mean Communication Bound per FSDP degree")
#st.dataframe(
#    df_result.groupby('fsdp').apply(geom_mean_agg).reset_index().sort_values('total')
#)
#
#import matplotlib.pyplot as plt
#import seaborn as sns
#from scipy.stats import gmean
#import numpy as np
#
#def plot_with_total_color_geom(df, degree):
#    fig, ax = plt.subplots(figsize=(8, 4))
#    unique_vals = df[degree].unique()
#    # Calculate geometric mean of 'total' for each unique value
#    totals = [gmean(df[df[degree] == val]['total']) for val in unique_vals]
#    norm = plt.Normalize(min(totals), max(totals))
#    cmap = plt.cm.Reds
#    palette = {val: cmap(norm(total)) for val, total in zip(unique_vals, totals)}
#    sns.barplot(x=degree, y='comm_percent', hue=degree, data=df, ax=ax, palette=palette, legend=False)
#    ax.set_title(f'Communication percent vs {degree.upper()} Degree with color reflecting geometric mean total')
#    return fig
#
#for degree in ['dp', 'tp', 'sp', 'pp']:
#    fig = plot_with_total_color_geom(df_result, degree)
#    st.pyplot(fig)


###st.subheader("Average Communication Bound per DP degree")
###st.dataframe(df_result.groupby('dp')[['total', 'comm_percent']].mean().reset_index().sort_values('total'))
#####
###st.subheader("Average Communication Bound per TP degree")
###st.dataframe(df_result.groupby('tp')[['total', 'comm_percent']].mean().reset_index().sort_values('total'))
#####
#####
###st.subheader("Average Communication Bound per SP degree")
###st.dataframe(df_result.groupby('sp')[['total', 'comm_percent']].mean().reset_index().sort_values('total'))
#####
#####
###st.subheader("Average Communication Bound per PP degree")
###st.dataframe(df_result.groupby('pp')[['total', 'comm_percent']].mean().reset_index().sort_values('total'))
#####
#####
###st.subheader("Average Communication Bound per FSDP degree")
###st.dataframe(df_result.groupby('fsdp')[['total', 'comm_percent']].mean().reset_index().sort_values('total'))


figs = picker.plot_experiments_bound_breakdown(df_result,chunk_size=36)
for fig in figs:
    st.pyplot(fig)

st.markdown("---")

option2 = st.radio(
    "Select Analysis Option:",
    options=['slowest', 'fastest', 'average'],
    horizontal=True,
    format_func=lambda x: {
        'slowest': 'Slowest Experiment',
        'fastest': 'Fastest Experiment',
        'average': 'Average All Experidments'
    }[x])

st.subheader(f'parallelsim degree vs communication percentage ({option2})')
cols = st.columns([3,3,3,3,3])
degrees = ['dp', 'tp', 'sp', 'pp', 'fsdp']
for i in range(len(degrees)):
    with cols[i]:
        fig = picker.plot_with_total_color(df_result, degrees[i], option2)
        st.pyplot(fig)

st.subheader(f'parallelsim degree vs total cycles ({option2})')
cols = st.columns([3,3,3,3,3])
degrees = ['dp', 'tp', 'sp', 'pp', 'fsdp']
for i in range(len(degrees)):
    with cols[i]:
        fig = picker.plot_with_total_color_total(df_result, degrees[i])
        st.pyplot(fig)



#TODO: Remove
def get_coeffs(df_results):
    X = df_result[['dp', 'tp', 'sp', 'pp', 'fsdp']].values
    y = df_result['total'].values

    model = LinearRegression().fit(X, y)

    return  pd.DataFrame({
        'factor': ['dp', 'tp', 'sp', 'pp', 'fsdp'],
        'coefficient': model.coef_
    })
    ###st.subheader("Linear regression coefficients")
    ###st.dataframe(coeffs)



def get_cdf_df(df_col, ccdf = True):
    # Calculate CDF for comm_percent
    value_counts_comm = df_col.value_counts().sort_index()
    pdf_comm = value_counts_comm / value_counts_comm.sum()
    cdf_comm = pdf_comm.cumsum()
    if ccdf:
        ccdf_comm = 1 - cdf_comm
        return pd.DataFrame({'percent': pdf_comm.index, 'cdf': ccdf_comm.values})
    return pd.DataFrame({'percent': pdf_comm.index, 'cdf': cdf_comm.values})

cdf_df_comm = get_cdf_df(df_result['comm_percent'], ccdf = True)
cdf_df_mem = get_cdf_df(df_result['mem_percent'], ccdf = True)
cdf_df_comp = get_cdf_df(df_result['comp_percent'], ccdf = True)


col0, col1 = st.columns([2,4])
with col0:
    corr = df_result[['dp', 'tp', 'sp', 'pp', 'fsdp', 'total']].corr()

    st.subheader("Correlation Matrix")
    ###st.dataframe(corr)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
with col1:
    st.subheader('CDFs of Communication, Memory, and Computation Percentages')
    col0, col1 = st.columns(2)
    with col0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(cdf_df_comm['percent'], cdf_df_comm['cdf'], where='post', label='communication')
        ax.set_xlabel('Communication time (%)', fontsize=18)
        ax.set_ylabel('CDF', fontsize=18)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Cumulative Distribution Function (CDF)', fontsize=18)
        ax.grid(True)
        ax.legend(fontsize=18)
        st.pyplot(fig)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(cdf_df_mem['percent'], cdf_df_mem['cdf'], where='post', label='memory')
        ax.step(cdf_df_comp['percent'], cdf_df_comp['cdf'], where='post', label='computation')
        ax.set_xlabel('Mem/Comp Bounded OPs time (%)', fontsize=18)
        ax.set_ylabel('CDF', fontsize=18)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Cumulative Distribution Function (CDF)', fontsize=18)
        ax.grid(True)
        ax.legend(fontsize=18)
        st.pyplot(fig)