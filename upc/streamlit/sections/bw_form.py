import streamlit as st

def sweep_bw_form():
    with st.form("bw_sweep_form"):
        st.subheader("Bandwidth Ranges Configuration")

        col1, col2 = st.columns(2)
        default_intra_bw = "600, 900, 1800, 3600, 7200"
        default_inter_bw = "200, 400, 800, 1600, 3200"

        col1, col2 = st.columns(2)

        with col1:
            intra_bw_input = st.text_input(
                "Intra BW (GB/s) — comma-separated", value=default_intra_bw
            )

        with col2:
            inter_bw_input = st.text_input(
                "Inter BW (GB/s) — comma-separated", value=default_inter_bw
            )

        try:
            intra_bw_list = [
                int(x.strip()) for x in intra_bw_input.split(",") if x.strip()
            ]
            inter_bw_list = [
                int(x.strip()) for x in inter_bw_input.split(",") if x.strip()
            ]
        except ValueError:
            st.error("Please enter valid comma-separated numbers.")
            intra_bw_list, inter_bw_list = [], []

        run_button = st.form_submit_button("Run Simulations")

    return run_button, intra_bw_list, inter_bw_list

