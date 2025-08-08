import streamlit as st
import sections.generate_workload as gen
import sections.astrasim as astra
import sections.trace_viewer as tv

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
# Generate Workload and Run AstraSim in a pipeline
params = gen.generate_workload()

if st.session_state.get("submitted") is not None:
    st.markdown("---")
    sim_outputs = astra.run_astrasim(params)
    st.markdown("---")
    if "df_matched" not in st.session_state:
        st.warning("Data is not loaded yet. Please run the simulation first.")
    else:
        # Visualization Tabs
        tv.render_sim_output_section(sim_outputs)
