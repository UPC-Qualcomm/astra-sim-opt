import streamlit as st
import sections.generate_workload as gen
import sections.astrasim as astra
import sections.trace_viewer as viewer

st.set_page_config(page_title="Roofline Viewer", layout="wide")

# Section 1: Generate Workload
params = gen.generate_workload()

st.markdown("---")

# Section 2: Run AstraSim
sim_outputs = astra.run_astrasim(params)

st.markdown("---")

# Section 3: Visualize Simulation Results
astra.visualize_simulation_results(sim_outputs)

st.markdown("---")

# Section 4: Trace Visualization Viewer
viewer.trace_viewer(sim_outputs)
