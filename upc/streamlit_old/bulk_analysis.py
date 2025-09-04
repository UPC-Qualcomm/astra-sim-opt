import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Custom Modules
import sections.trace_picker as picker

st.set_page_config(page_title="Bulk analysis", layout="wide")

st.title("Parallelism Strategy - Execution time Breakdown")


selected_model, selected_config = picker.get_model_and_config()
picker.set_session_peak_perf_bw(selected_config)

st.markdown("---")

col1, col2 = st.columns([11, 2])
with col2:
    option = st.radio(
        "Select Bound Analysis Option:",
        options=["slowest", "fastest", "average"],
        format_func=lambda x: {
            "slowest": "Slowest NPU",
            "fastest": "Fastest NPU",
            "average": "Average Across all NPUs",
        }[x],
    )

with col1:
    st.info(f"""
        **Peak performance**: {st.session_state.peak_perf} TFLOPs, \t
        **Peak bandwidth**: {st.session_state.peak_bw} GB/s
    """)
df_result = picker.get_all_parallelism_strategies_data(
    selected_model, selected_config, option
)

# df_0 = df_result[(df_result["fsdp"] == 0)]
# df_1 = df_result[(df_result["fsdp"] == 1)]
# df_0[['dp', 'tp', 'sp', 'pp', 'total']].to_excel(f'{selected_model}data_total_0.xlsx', index=False)
# df_1[['dp', 'tp', 'sp', 'pp', 'total']].to_excel(f'{selected_model}data_total_1.xlsx', index=False)
# df_0[['dp', 'tp', 'sp', 'pp', 'comm']].to_excel(f'{selected_model}data_comm_0.xlsx', index=False)
# df_1[['dp', 'tp', 'sp', 'pp', 'comm']].to_excel(f'{selected_model}data_comm_1.xlsx', index=False)
# from scipy.stats import gmean
#
# def geom_mean_agg(group):
#    return pd.Series({
#        'total': gmean(group['total']),
#        'comm_percent': gmean(group['comm_percent'])
#    })
#
# st.subheader("Geometric Mean Communication Bound per DP degree")
# st.dataframe(
#    df_result.groupby('dp').apply(geom_mean_agg).reset_index().sort_values('total')
# )
#
# st.subheader("Geometric Mean Communication Bound per TP degree")
# st.dataframe(
#    df_result.groupby('tp').apply(geom_mean_agg).reset_index().sort_values('total')
# )
#
# st.subheader("Geometric Mean Communication Bound per SP degree")
# st.dataframe(
#    df_result.groupby('sp').apply(geom_mean_agg).reset_index().sort_values('total')
# )
#
# st.subheader("Geometric Mean Communication Bound per PP degree")
# st.dataframe(
#    df_result.groupby('pp').apply(geom_mean_agg).reset_index().sort_values('total')
# )
#
# st.subheader("Geometric Mean Communication Bound per FSDP degree")
# st.dataframe(
#    df_result.groupby('fsdp').apply(geom_mean_agg).reset_index().sort_values('total')
# )
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import gmean
# import numpy as np
#
# def plot_with_total_color_geom(df, degree):
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
# for degree in ['dp', 'tp', 'sp', 'pp']:
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

df_result_0 = df_result[df_result["fsdp"] == 0].sort_values(by="total", ascending=True)
figs = picker.plot_experiments_bound_breakdown(df_result_0, chunk_size=32)
for fig in figs:
    st.pyplot(fig)

df_result_1 = df_result[df_result["fsdp"] == 1].sort_values(by="total", ascending=True)
figs = picker.plot_experiments_bound_breakdown(df_result_1, chunk_size=32)
for fig in figs:
    st.pyplot(fig)

st.markdown("---")
st.subheader("Best 10 experiments in terms of time.")
st.write(df_result.sort_values(["total"]).head(20))
st.markdown("---")
st.write(len(df_result[(df_result["comm_percent"] > 70.0)]))
st.write(len(df_result[(df_result["comm_percent"] <= 70.0)]))
st.write(len(df_result))
st.markdown("---")
option2 = st.radio(
    "Select Analysis Option:",
    options=["slowest", "fastest", "average"],
    horizontal=True,
    format_func=lambda x: {
        "slowest": "Slowest Experiment",
        "fastest": "Fastest Experiment",
        "average": "Average All Experidments",
    }[x],
)

df_result = picker.get_all_parallelism_strategies_data(
    selected_model, selected_config, option2
)

st.subheader(f"parallelsim degree vs communication percentage ({option2})")
cols = st.columns([3, 3, 3, 3, 3])
degrees = ["dp", "tp", "sp", "pp", "fsdp"]
for i in range(len(degrees)):
    with cols[i]:
        fig = picker.plot_with_total_color(df_result, degrees[i], option2)
        st.pyplot(fig)

st.subheader(f"parallelsim degree vs total cycles ({option2})")
cols = st.columns([3, 3, 3, 3, 3])
degrees = ["dp", "tp", "sp", "pp", "fsdp"]
for i in range(len(degrees)):
    with cols[i]:
        fig = picker.plot_with_total_color_total(df_result, degrees[i])
        st.pyplot(fig)


# TODO: Remove
def get_coeffs(df_results):
    X = df_result[["dp", "tp", "sp", "pp", "fsdp"]].values
    y = df_result["total"].values

    model = LinearRegression().fit(X, y)

    return pd.DataFrame(
        {"factor": ["dp", "tp", "sp", "pp", "fsdp"], "coefficient": model.coef_}
    )
    ###st.subheader("Linear regression coefficients")
    ###st.dataframe(coeffs)


def get_cdf_df(df_col, ccdf=True):
    # Calculate CDF for comm_percent
    value_counts_comm = df_col.value_counts().sort_index()
    pdf_comm = value_counts_comm / value_counts_comm.sum()
    cdf_comm = pdf_comm.cumsum()
    if ccdf:
        ccdf_comm = 1 - cdf_comm
        return pd.DataFrame({"percent": pdf_comm.index, "ccdf": ccdf_comm.values})
    return pd.DataFrame({"percent": pdf_comm.index, "cdf": cdf_comm.values})


ccdf = st.checkbox("Use ccdf?", value=False)
cdf_df_comm = get_cdf_df(df_result["comm_percent"], ccdf=ccdf)
cdf_df_mem = get_cdf_df(df_result["mem_percent"], ccdf=ccdf)
cdf_df_comp = get_cdf_df(df_result["comp_percent"], ccdf=ccdf)

from scipy.stats import gmean

# Compute stats
comm_mean, comm_std, comm_gmean = (
    df_result["comm_percent"].mean(),
    df_result["comm_percent"].std(),
    gmean(df_result["comm_percent"][df_result["comm_percent"] > 0]),
)
mem_mean, mem_std, mem_gmean = (
    df_result["mem_percent"].mean(),
    df_result["mem_percent"].std(),
    gmean(df_result["mem_percent"][df_result["mem_percent"] > 0]),
)
comp_mean, comp_std, comp_gmean = (
    df_result["comp_percent"].mean(),
    df_result["comp_percent"].std(),
    gmean(df_result["comp_percent"][df_result["comp_percent"] > 0]),
)

col0, col1 = st.columns([2, 4])
with col0:
    corr = df_result[["dp", "tp", "sp", "pp", "fsdp", "total"]].corr()

    st.subheader("Correlation Matrix")

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with col1:
    if ccdf:
        st.subheader("CCDFs of Communication, Memory, and Computation Percentages")
    else:
        st.subheader("CDFs of Communication, Memory, and Computation Percentages")

    col0, col1 = st.columns(2)

    with col0:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        if ccdf:
            ax.step(
                cdf_df_comm["percent"],
                cdf_df_comm["ccdf"],
                where="post",
                label="communication",
            )
            ax.set_ylabel("Experiments\nratio", fontsize=25)
            # ax.set_title(f'Exposed Comm Time (μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)', fontsize=20)
        else:
            ax.step(
                cdf_df_comm["percent"],
                cdf_df_comm["cdf"],
                where="post",
                label="communication",
            )
            ax.set_ylabel("Experiments\nratio", fontsize=25)
            # ax.set_title(f'CDF of Comm Time (μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)', fontsize=20)

        ax.set_xlabel(
            f"Exposed communication time (%)\n(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)",
            fontsize=25,
        )
        ax.grid(True)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # Vertical dashed lines
        # for p in [50, 75, 90]:
        #    ax.axvline(p, color='gray', linestyle='--', alpha=0.6)
        #    ax.text(p, 0.05, f'{p}%', rotation=90, va='bottom', ha='center', fontsize=12)

        ax.legend(fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.tick_params(axis="both", which="minor", labelsize=24)

        fig.savefig("comm_cdf.pdf", format="pdf", bbox_inches="tight")
        st.pyplot(fig)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        if ccdf:
            ax.step(
                cdf_df_mem["percent"],
                cdf_df_mem["ccdf"],
                where="post",
                label=f"memory (μ={mem_mean:.2f}%, σ={mem_std:.2f}%, gμ={mem_gmean:.2f}%)",
            )
            ax.step(
                cdf_df_comp["percent"],
                cdf_df_comp["ccdf"],
                where="post",
                label=f"computation (μ={comp_mean:.2f}%, σ={comp_std:.2f}%, gμ={comp_gmean:.2f}%)",
            )
            ax.step(
                cdf_df_comm["percent"],
                cdf_df_comm["ccdf"],
                where="post",
                label=f"communication (μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)",
            )
            ax.set_ylabel("Probability", fontsize=22)
            ax.set_title("Time Distribution CCDF", fontsize=22)
        else:
            ax.step(
                cdf_df_mem["percent"],
                cdf_df_mem["cdf"],
                where="post",
                label=f"memory (μ={mem_mean:.2f}%, σ={mem_std:.2f}%, gμ={mem_gmean:.2f}%)",
            )
            ax.step(
                cdf_df_comp["percent"],
                cdf_df_comp["cdf"],
                where="post",
                label=f"computation (μ={comp_mean:.2f}%, σ={comp_std:.2f}%, gμ={comp_gmean:.2f}%)",
            )
            ax.set_ylabel("Probability", fontsize=22)
            ax.set_title("Time Distribution CDF", fontsize=22)

        ax.set_xlabel("Mem/Comp Bounded OPs time (%)", fontsize=22)
        ax.grid(True)

        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.tick_params(axis="both", which="minor", labelsize=22)

        # for p in [50, 75, 90]:
        #    ax.axvline(p, color='gray', linestyle='--', alpha=0.6)
        #    ax.text(p, 0.05, f'{p}%', rotation=90, va='bottom', ha='center', fontsize=12)

        ax.legend(fontsize=20)
        fig.savefig("time_distribution_cdf.pdf", format="pdf", bbox_inches="tight")
        st.pyplot(fig)


comm_mean, comm_std, comm_gmean = (
    df_result["comm_percent"].mean(),
    df_result["comm_percent"].std(),
    gmean(df_result["comm_percent"][df_result["comm_percent"] > 0]),
)


fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_result["comm_percent"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_xlabel("Communication Percentage (%)")
ax.set_ylabel("Number of Experiments")
ax.set_title(
    f"Distribution of Communication Overhead\n(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)"
)
ax.grid(True)
st.pyplot(fig)


sorted_comm = np.sort(df_result["comm_percent"].values)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(sorted_comm, marker="o", linestyle="-", color="blue")
ax.set_xlabel("Experiment Number", fontsize=25)
ax.set_ylabel("Exposed\nComm. Time\n(%)", fontsize=25)
ax.set_title(
    f"Sorted - Exposed Communication Percentage per Experiment\n(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)",
    fontsize=25,
)
ax.grid(True)

ax.tick_params(axis="both", which="major", labelsize=24)
ax.tick_params(axis="both", which="minor", labelsize=24)

fig.savefig("comm_percentage.pdf", format="pdf", bbox_inches="tight")
st.pyplot(fig)

sorted_comm = np.sort(df_result["comm_percent"].values)
cdf = np.arange(1, len(sorted_comm) + 1) / len(sorted_comm)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sorted_comm, cdf, marker=".", linestyle="-", color="green")
ax.set_xlabel("Communication Percentage (%)")
ax.set_ylabel("Cumulative Probability")
ax.set_title(
    f"CDF of Communication Overhead\nM(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)"
)
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(y=df_result["comm_percent"], color="lightcoral", ax=ax)
ax.set_ylabel("Communication Percentage (%)")
ax.set_title(
    f"Communication Overhead Summary\n(μ={comm_mean:.2f}%, σ={comm_std:.2f}%, gμ={comm_gmean:.2f}%)"
)
st.pyplot(fig)


st.subheader("test ratios.")

npus = 64
df_ratios = df_result.copy()
selected_fsdp = st.checkbox("Use FSDP?", value=True)
df_ratios = df_ratios[(df_ratios["fsdp"] == selected_fsdp)]
df_ratios["dp_ratio"] = df_ratios["dp"] / npus
df_ratios["tp_ratio"] = df_ratios["tp"] / npus
df_ratios["sp_ratio"] = df_ratios["sp"] / npus
df_ratios["pp_ratio"] = df_ratios["pp"] / npus


st.subheader(f"parallelsim degree ratio vs communication percentage ({option2})")
cols = st.columns([3, 3, 3, 3, 3])
degrees = ["dp_ratio", "tp_ratio", "sp_ratio", "pp_ratio", "fsdp"]
for i in range(len(degrees)):
    with cols[i]:
        fig = picker.plot_with_total_color(df_ratios, degrees[i], option2)
        st.pyplot(fig)

st.subheader(f"parallelsim degree ratio vs total cycles ({option2})")
cols = st.columns([3, 3, 3, 3, 3])
degrees = ["dp_ratio", "tp_ratio", "sp_ratio", "pp_ratio", "fsdp"]
for i in range(len(degrees)):
    with cols[i]:
        fig = picker.plot_with_total_color_total(df_ratios, degrees[i])
        st.pyplot(fig)
