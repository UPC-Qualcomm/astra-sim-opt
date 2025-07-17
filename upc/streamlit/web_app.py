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
import upc.ref_streamlit.sections.generate_workload as gen
import sections.astrasim as astra
import sections.trace_viewer as viewer
import sections.trace_picker as picker

st.set_page_config(page_title="Roofline Viewer", layout="wide")

sim_outputs = picker.trace_picker()

if sim_outputs != -1:
    st.markdown("---")

    # Section 3: Visualize Simulation Results
    astra.visualize_simulation_results(sim_outputs)

    st.markdown("---")

    # Section 4: Trace Visualization Viewer
    viewer.trace_viewer(sim_outputs)
