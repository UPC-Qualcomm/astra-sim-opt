import os
import json
import pandas as pd
import streamlit as st
import trace_visualization as tv
from pathlib import Path


def trace_picker():
    st.title("📊 Trace Picker")

    BASE_MODELS_DIR = _get_output_dir()
    CONFIGS_DIR = _get_configs_dir()

    selected_model, selected_config = _model_and_config_selection(BASE_MODELS_DIR)

    if selected_model and selected_config:
        _parallelism_startegy_form(selected_model, selected_config, BASE_MODELS_DIR)

        if "csv_trace_file" in st.session_state and os.path.isfile(
            st.session_state["csv_trace_file"]
        ):
            csv_trace_file = st.session_state["csv_trace_file"]
            trace_file_name = st.session_state["trace_file_name"]
            res_file = st.session_state["res_file"]
            log_file = st.session_state["log_file"]
            res_path = st.session_state["res_path"]

            _detect_file_change(csv_trace_file)

            st.success(f"✅ Found trace file: `{trace_file_name}`")

            config_file = os.path.join(CONFIGS_DIR, f"{selected_config}_sys.json")
            with open(config_file, "r") as f:
                config_file_content = f.read()
                config_file_content = json.loads(config_file_content)

                st.session_state.peak_perf = config_file_content.get("peak-perf", 300)
                st.session_state.peak_bw = config_file_content.get("local-mem-bw", 2000)

            dp, tp, sp, pp, sharding_val, _ = Path(csv_trace_file).stem.split("_")

            st.session_state.show_npu_plots = True
            return {
                "sim_dir": os.path.join(
                    BASE_MODELS_DIR, selected_model, selected_config
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
    else:
        st.error(
            f"❌ The combination you entered does not correspond to an existing trace file: `{trace_file_name}`"
        )

    return -1


def _get_output_dir():
    base_model_output = os.path.abspath(os.path.join(os.getcwd(), "../output"))
    return base_model_output


def _get_configs_dir():
    configs_dir = os.path.abspath(os.path.join(os.getcwd(), "../configuration"))
    return configs_dir


def _get_config_names(model_dir):
    return [
        d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))
    ]


def _get_model_names(base_model_dir):
    return [
        d
        for d in os.listdir(base_model_dir)
        if os.path.isdir(os.path.join(base_model_dir, d))
    ]


def _model_and_config_selection(base_model_dir):
    col_model, col_config = st.columns([1, 1])
    with col_model:
        model_names = _get_model_names(base_model_dir)
        selected_model = st.selectbox("Select a Model", model_names)

    with col_config:
        selected_config = None
        if selected_model:
            model_dir = os.path.join(base_model_dir, selected_model)
            config_names = _get_config_names(model_dir)
            selected_config = st.selectbox("Select a Configuration", config_names)

    return selected_model, selected_config


def _parallelism_startegy_form(selected_model, selected_config, base_model_dir):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        dp = st.text_input("Data Parallelism (DP)", "1")
    with col2:
        tp = st.text_input("Tensor Parallelism (TP)", "8")
    with col3:
        sp = st.text_input("Pipeline Parallelism (SP)", "2")
    with col4:
        pp = st.text_input("Placement Parallelism (PP)", "4")
    with col5:
        sharding = st.checkbox("Sharding", value=False)

    sharding_val = 0
    if st.button("Submit"):
        for key in st.session_state.keys():
            del st.session_state[key]

        sharding_val = "1" if sharding else "0"
        file_base = f"{dp}_{tp}_{sp}_{pp}_{sharding_val}"
        trace_file_name = f"{file_base}_trace.csv"
        timed_file_name = f"{file_base}_trace_matched_timiming.csv"

        base_dir = os.path.join(base_model_dir, selected_model, selected_config)
        output_file = os.path.join(base_dir, timed_file_name)
        csv_trace_file = os.path.join(base_dir, trace_file_name)
        res_path = os.path.abspath(
            os.path.join(os.getcwd(), f"../results/{selected_model}/{selected_config}/")
        )
        res_file = os.path.join(res_path, f"{file_base}_res.csv")
        log_file = os.path.join(base_dir, f"{file_base}.log")

        if "df_matched" not in st.session_state:
            st.session_state.df_matched = tv.get_timings_df(csv_trace_file, output_file)

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
