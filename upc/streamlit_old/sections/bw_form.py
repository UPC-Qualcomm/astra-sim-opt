import streamlit as st
from pathlib import Path


def sweep_bw_form(selected_config, selected_model):
    with st.form("bw_sweep_form"):
        st.subheader("Bandwidth Ranges Configuration")

        col1, col2 = st.columns(2)
        default_intra_bw = "600, 900, 1800, 3600, 7200"
        default_inter_bw = "200, 400, 800, 1600, 3200"
        config_names = get_configuration_options()

        col1, col2 = st.columns(2)

        with col1:
            intra_bw_input = st.text_input(
                "Intra BW (GB/s) — comma-separated", value=default_intra_bw
            )

        with col2:
            inter_bw_input = st.text_input(
                "Inter BW (GB/s) — comma-separated", value=default_inter_bw
            )

        st.write("Select the set of network configurations:")
        selected_configs = []
        cols = st.columns(len(config_names))

        for i, config in enumerate(config_names):
            val = config == selected_config
            with cols[i]:
                if st.checkbox(config, value=val):
                    selected_configs.append(config)

        try:
            intra_bw_list = [
                int(x.strip()) for x in intra_bw_input.split(",") if x.strip()
            ]
            inter_bw_list = [
                int(x.strip()) for x in inter_bw_input.split(",") if x.strip()
            ]

            if not selected_configs:
                st.error("Please select at least one network configuration.")
                run_button = False 

        except ValueError:
            st.error("Please enter valid comma-separated numbers.")
            intra_bw_list, inter_bw_list, selected_configs = [], [], []

        run_button = st.form_submit_button("Run Simulations")
    
    return run_button, intra_bw_list, inter_bw_list, selected_configs


def get_configuration_options():
    config_dir = Path(__file__).parent / "../../configuration/"
    config_dir = config_dir.resolve() 
    config_files = [file.stem for file in config_dir.glob("*.yml")]
    return sorted(config_files)