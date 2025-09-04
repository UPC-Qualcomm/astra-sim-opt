import streamlit as st
import sections.generate_workload as gen
import sections.astrasim_updated as astra
import sections.trace_viewer as tv
import hashlib
import json

def _get_config_hash(params):
    """Generate a hash of the current configuration for comparison"""
    config_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()

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
st.markdown("""
<div style='background-color: #fde8e8; border-left: 4px solid #F94F4F; padding: 1em; margin: 1.5em 0; border-radius: 4px;'>
    <p style='margin: 0; color: #822c2c; font-size: 0.95em;'>
        <strong>Note:</strong> DAI is a work in progress. This version has limited many of the functionalities for the demo purposes.<br>
        <strong>Note:</strong> The demo is designed to work on a PC. Thus, it may not work as expected on a mobile device.
    </p>
</div>
""", unsafe_allow_html=True)
st.title("Modeling and Simulation")

# Initialize session state for persistent results
if "simulation_completed" not in st.session_state:
    st.session_state.simulation_completed = False
if "last_sim_params" not in st.session_state:
    st.session_state.last_sim_params = None
if "last_sim_outputs" not in st.session_state:
    st.session_state.last_sim_outputs = None

# Generate Workload and Run AstraSim in a unified pipeline
params = gen.generate_workload_and_run_simulation()

# Check if this is a new simulation run
is_new_simulation = st.session_state.get("submitted") is not None

if is_new_simulation:
    #st.markdown("---")
    sim_outputs = astra.run_astrasim_unified(params)
    
    # Store simulation results in session state for persistence
    if sim_outputs:
        st.session_state.simulation_completed = True
        st.session_state.last_sim_params = params.copy()
        st.session_state.last_sim_outputs = sim_outputs.copy()
        # Clear the submitted flag to prevent re-running
        st.session_state.submitted = None
    
    st.markdown("---")

# Display results if available (either from current run or previous session)
if st.session_state.simulation_completed and "df_matched" in st.session_state:
    # Check if current configuration differs from the one used for results
    current_config_hash = _get_config_hash(params)
    stored_config_hash = st.session_state.get('last_config_hash', '')
    
    
    # Render the visualization tabs directly
    tv.render_sim_output_section(st.session_state.last_sim_outputs)
    
    # Add option to clear results
    st.markdown("---")
    col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 2])
    with col_clear2:
        if st.button("Clear Previous Results", help="Clear stored simulation results to start fresh"):
            # Clear simulation results from session state
            st.session_state.simulation_completed = False
            st.session_state.last_sim_params = None
            st.session_state.last_sim_outputs = None
            if "df_matched" in st.session_state:
                del st.session_state.df_matched
            st.rerun()

elif st.session_state.simulation_completed and "df_matched" not in st.session_state:
    st.warning("Simulation data was found in session but trace data is missing. Please run a new simulation.")
    # Option to clear corrupted session
    if st.button("Clear Session and Start Fresh"):
        st.session_state.simulation_completed = False
        st.session_state.last_sim_params = None
        st.session_state.last_sim_outputs = None
        st.rerun()
elif not st.session_state.simulation_completed:
    st.info("Configure your parameters above and click 'Generate Workload and Run Simulation' to see results here.")
