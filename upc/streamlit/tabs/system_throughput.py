import streamlit as st
import matplotlib.pyplot as plt

def render_system_throughput(df):
    df_sys = df.copy()
    calculate_num_tokens(df_sys)
    compute_sys_throughput(df_sys)
    set_parallelism_strategy(df_sys)
    fig = get_sys_throughput_plots(df_sys)
    st.pyplot(fig)
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666;'>System throughput measured in tokens per second for different parallelism strategies. Higher values indicate better performance. The top 50 performing configurations are shown, sorted by throughput in descending order.</p>", unsafe_allow_html=True)


def calculate_num_tokens(df):
    df["num_tokens"] = df["file_name"].apply(lambda x: int(x.split(".")[1].split("_")[1]) * int(x.split(".")[2].split("_")[1]))
    return df

def set_parallelism_strategy(df):
    df["parallelism_strategy"] = df["file_name"].apply(lambda x: str(x.split(".")[0]))
    return df

def get_sys_performance():
    return st.session_state.peak_perf

def compute_sys_throughput(df):
    df["throughput"] = (df["num_tokens"] / (df["total"] * 1e-9))

def get_sys_throughput_plots(df):
    df_sorted = df.sort_values(by="throughput", ascending=False).head(50)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(df_sorted["parallelism_strategy"], df_sorted["throughput"])

    ax.set_xlabel("Parallelism Strategy", fontsize=14)
    ax.set_ylabel("Throughput (Token/sec)", fontsize=14)
    ax.set_title(f"System Throughput by Parallelism Strategy (Top {len(df_sorted)} Experiments)", fontsize=14)
    ax.tick_params(axis='x', labelsize=10, rotation=90)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.2f}K"))

    plt.close(fig)
    return fig
    