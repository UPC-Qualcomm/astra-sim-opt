import streamlit as st
import sections.generate_workload as gen
import sections.astrasim as astra
import sections.trace_viewer as tv

st.set_page_config(layout="wide")

# Generate Workload
params = gen.generate_workload()


st.markdown("---")

# Run AstraSim
sim_outputs = astra.run_astrasim(params)


st.markdown("---")

if "df_matched" not in st.session_state:
    st.warning("Data is not loaded")

else:
    # Visualization Tabs
    tv.render_sim_ouput_section(sim_outputs)
