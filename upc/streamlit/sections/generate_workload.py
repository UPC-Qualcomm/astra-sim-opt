import streamlit as st
import subprocess
import time
import scripts.generate_single_workload as gen
import os
import uuid

def generate_workload():
    
    st.header(
        "Generate Workload Trace",
        help=(
            "This section allows you to generate a workload trace for simulation.\n"
            "Click **Run Model** to generate the workload trace."
        )
    )
    if "temp_dir" not in st.session_state:
        # Create a temporary directory for the session
        st.session_state.temp_dir = f"temp/{uuid.uuid4()}/"
        os.makedirs(st.session_state.temp_dir, exist_ok=True)
    temp_dir = st.session_state.temp_dir

    # --- Form Section ---
    models_names = _get_models_names()
    selected_model_name = st.selectbox("Choose a pre-defined model:", models_names)
    (params_list, dp, tp, sp, pp, sharding, submitted) = (
        _workload_form(selected_model_name)
    )

    sharding_val = "1" if sharding else "0"

    # --- Submission Section ---
    if submitted:
        _clear_session_state()
        _clear_temp_dir(temp_dir)
        _run_trace_generation(
            selected_model_name, [dp, tp, sp, pp, sharding], params_list, temp_dir
        )
        if 'submitted' not in st.session_state:
            st.session_state['submitted'] = True
    # --- Return parameters for downstream use ---
    return {
        "dp": dp,
        "tp": tp,
        "sp": sp,
        "pp": pp,
        "sharding": sharding,
        "sharding_val": sharding_val,
        "temp_dir": temp_dir,
        "selected_model_name": selected_model_name,
    }

def _get_models_names():   
    return list(gen.model_display_names.values())

def _workload_form(selected_model_name):
    with st.form("model_config_form"):
        st.subheader("Model parameters")
        display_to_model = {v: k for k, v in gen.model_display_names.items()}
        selected_model = display_to_model[selected_model_name]

        params_list = gen.Model.get_model_params(selected_model)
        param_names = [
            "din",
            "dout",
            "dmodel",
            "dff",
            "batch",
            "seq",
            "head",
            "num_stacks",
        ]   

        params_labels = [                 
            "Input Embedding Size",            
            "Output Embedding Size",            
            "Model Feature Size",
            "FFN Feature Size",
            "Batch Size",
            "Sequance Length",
            "Number of Heads",
            "Number of Layers"
        ]
        cols = st.columns(len(param_names))
        for i, col in enumerate(cols):
            with col:
                params_list[i] = st.number_input(
                    label=params_labels[i], value=params_list[i], key=f"{param_names[i]}_input"
                )

        st.subheader(
            "Parallelism strategy",
            help=(
            "Set the parallelism parameters for the model.\n"
            "- **Data Parallelism Degree (DP)**.\n"
            "- **Tensor Parallelism Degree (TP)**.\n"
            "- **Sequence Parallelism Degree (SP)**.\n"
            "- **Pipeline Parallelism Degree (PP)**.\n"
            "- **FSDP**: Enable Fully Sharded Data Parallelism."
            )
        )
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            dp = st.text_input("Data Parallelism (DP)", 1)
        with col2:
            tp = st.text_input("Tensor Parallelism (TP)", 8)
        with col3:
            sp = st.text_input("Sequence Parallelism (SP)", 2)
        with col4:
            pp = st.text_input("Pipeline Parallelism (PP)", 4)
        with col5:
            sharding = st.checkbox("Sharding (FSDP)", value=False)

        submitted = st.form_submit_button("🚀 Run Model")
        params_list[4] = [params_list[4]]   
    return (params_list, dp, tp, sp, pp, sharding, submitted)


def _clear_session_state():
    # Preserve important session variables that should persist
    preserved_keys = {'temp_dir', 'session_id'}
    preserved_values = {}
    
    # Save values we want to keep
    for key in preserved_keys:
        if key in st.session_state:
            preserved_values[key] = st.session_state[key]
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value

def _clear_temp_dir(temp_dir):
    subprocess.run(f"rm -rf {temp_dir}*", shell=True, cwd=None)


def _run_trace_generation(selected_model_name, parallelism, params_list, temp_dir):
    with st.spinner(f"Generating a trace for `{selected_model_name}`..."):
        start_time = time.time()
        gen.generate_trace(parallelism, params_list, temp_dir)
        elapsed_time = time.time() - start_time
    st.success(f"✅ Trace generation completed in {elapsed_time:.2f} seconds.")
