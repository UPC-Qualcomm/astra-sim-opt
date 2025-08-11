import streamlit as st
import os

st.set_page_config(
    page_title="DAI",
    layout="wide",
)

# Hide the the streamlit menu, deploy button and footer
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align: center; color: #4F8BF9; font-size: 1.75em;'>
        Welcome to <b>DAI</b>:<br>
        <span style='font-size: 1.1em; color: #222;'>A Simulation Driven Recommendation Framework for Exploring Parallelism Strategies and Optimizing Distributed AI Workloads Performance.</span>
    </h1>
    <hr style='border: 1px solid #4F8BF9; margin-top: 2em; margin-bottom: 2em;'>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style='background-color: #fde8e8; border-left: 4px solid #F94F4F; padding: 1em; margin: 1.5em 0; border-radius: 4px;'>
    <p style='margin: 0; color: #822c2c; font-size: 0.95em;'>
        <strong>Note:</strong> DAI is a work in progress. This version has limited many of the functionalities for the demo purposes.<br>
        <strong>Note:</strong> The demo is designed to work on a PC. Thus, it may not work as expected on a mobile device.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## Overview

**DAI (Distributed AI Workload Optimizer)** is a comprehensive simulation-driven framework built on top of the [Symbolic Tensor Graph (STG)](https://github.com/astra-sim/symbolic_tensor_graph) and [ASTRA-sim 2.0](https://github.com/astra-sim/astra-sim) that enables researchers to systematically explore, analyze, and optimize distributed deep learning workloads. The framework provides an intuitive web-based interface for investigating the complex interaction between parallelism strategies, hardware configurations, and network topologies in large-scale AI training scenarios.
""")


st.markdown("""
<div style='text-align: center; margin: 2em 0;'>
    <h3 style='color: #4F8BF9; margin-bottom: 1em;'>DAI Framework Architecture</h3>
</div>
""", unsafe_allow_html=True)

# Check if we should use mobile layout
st.markdown("""
<style>
@media (max-width: 768px) {
    .stColumns > div {
        width: 100% !important;
        flex: none !important;
    }
    .responsive-text {
        font-size: 0.9em !important;
        line-height: 1.5 !important;
    }
    .responsive-image {
        max-width: 100% !important;
        height: auto !important;
    }
}
@media (min-width: 769px) {
    .responsive-text {
        font-size: 1em;
        line-height: 1.6;
    }
}
.workflow-section {
    background-color: #f8f9fa;
    padding: 1.5em;
    border-radius: 8px;
    margin: 1em 0;
    border-left: 4px solid #4F8BF9;
}
</style>
""", unsafe_allow_html=True)

# Use responsive columns that stack on mobile
col1, col2 = st.columns([1, 1])

with col1:
    svg_path = os.path.join(os.path.dirname(__file__), "images/overview.svg")
    if os.path.exists(svg_path):
        st.image(svg_path, caption="Complete DAI workflow showing the integration of workload generation, simulation, and analysis components", use_container_width=True)
    else:
        alt_svg_path = "images/overview.svg"
        if os.path.exists(alt_svg_path):
            st.image(alt_svg_path, caption="Complete DAI workflow showing the integration of workload generation, simulation, and analysis components", use_container_width=True)
        else:
            st.markdown("""
            <div style='background-color: #f0f2f6; border: 2px dashed #4F8BF9; padding: 2em; text-align: center; margin: 1em 0; border-radius: 8px; width: 100%;'>
                <h4 style='color: #4F8BF9; margin-bottom: 0.5em; font-size: 1.1em;'>DAI Framework Architecture Diagram</h4>
                <p style='color: #666; margin: 0; font-size: 0.9em;'>Overview diagram will be displayed here<br><small>(overview.svg not found in current directory)</small></p>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="responsive-text workflow-section">
    <p>
    The DAI framework follows a comprehensive end-to-end workflow that integrates multiple sophisticated components:
    </p>
    <ul>
        <li>
            <strong>Input Layer</strong>:
            <ul>
                <li><strong>Workload Generation</strong>: Takes GPT-model architecture specifications (layer count, attention heads, embedding dimensions, etc.), parallelism strategy configuration (Data, Tensor, Sequence, Pipeline, and Fully Sharded Data Parallel - DP, PP, TP, SP, FSDP respectively), and the number of NPUs to generate computational workloads.</li>
                <li><strong>System Specification</strong>: Defines the hardware environment including interconnect network topology, links capacity, NPU peak performance characteristics, and local memory bandwidth specifications.</li>
            </ul>
        </li>
        <li>
            <strong>Trace Generation</strong>: The <a href="https://github.com/astra-sim/symbolic_tensor_graph" target="_blank">Symbolic Tensor Graph (STG)</a> component translates model architectures and parallelism configurations into <a https://mlcommons.org/working-groups/research/chakra/" target="_blank">Chakra</a> execution traces, creating detailed computational graphs that capture the precise sequence of operations required for distributed training.
        </li>
        <li>
            <strong>Simulation Engine</strong>: At the core lies the "Modeling & Simulation" environment powered by <a href="https://github.com/astra-sim/astra-sim" target="_blank">ASTRA-sim 2.0</a>, which processes the traces through four critical analysis dimensions:
            <ul>
                <li><strong>Computation</strong>: Models the execution time of the workload based on the defined system specifications and parallelism strategies.</li>
                <li><strong>Communication</strong>: Simulates communication patterns of the distributed workload, accounting for intra-node and inter-node bandwidths.</li>
                <li><strong>Memory</strong>: Tracks memory consumption.</li>
                <li><strong>Power</strong>: Estimates power consumption based on hardware specifications and parallelism strategies.</li>
            </ul>
        </li>
        <li>
            <strong>Analysis Output</strong>: The framework generates comprehensive performance metrics, timing breakdowns, and optimization recommendations, with additional capabilities for memory consumption estimation and bottleneck identification.
        </li>
    </ul>
    <p>
    This integrated approach enables researchers to systematically explore how different parallelism strategies, hardware configurations, and network topologies impact distributed AI workload performance.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## Explore DAI's Capabilities")

with st.expander("**Workload Modeling & Simulation**", expanded=False):
    st.markdown("""
    - **Interactive Workload Generation**: Create custom transformer-based workloads with configurable model parameters.
    - **Real-time Simulation**: Execute ASTRA-sim simulations directly through the web interface.
    - **Trace Visualization**: Comprehensive visualization of execution traces and performance metrics.
    """)

with st.expander("**Parallelism Strategy Exploration**", expanded=False):
    st.markdown("""
    - **Multi-dimensional Analysis**: Systematic exploration of Data Parallel (DP), Pipeline Parallel (PP), Tensor Parallel (TP), and Sequence Parallel (SP) strategies.
    - **Performance Comparison**: Side-by-side analysis of different parallelism configurations.
    - **Machine Learning Insights**: Advanced clustering and dimensionality reduction techniques.
    """)

with st.expander("**Interconnect Network Design Exploration**", expanded=False):
    st.markdown("""
    - **Topology Comparison**: Analyze performance across different interconnect designs (2D/3D topologies).
    - **Bandwidth Sensitivity Analysis**: Interactive exploration of intra-node and inter-node bandwidth effects.
    - **Scalability Studies**: Investigate performance scaling characteristics.
    """)

with st.expander("**Batch Size Exploration**", expanded=False):
    st.markdown("""
    - **Batch Size Analysis**: Study how batch size affects training efficiency of a single training step across parallelism strategies. Testing with multiple steps is ongoing.
    - **Performance Breakdown**: Detailed visualization of computation, communication, and overlap cycles.
    """)

with st.expander("**Solver**", expanded=False):
    st.markdown("""
    - **Automated Optimization**: Random search-based solver for optimal parallelism strategies.
    - **Configurable Search Space**: User-defined constraints for each parallelism dimension.
    - **Performance Ranking**: Comprehensive comparison with detailed cycle breakdowns.
    """)

##st.markdown("## Technical Foundation")
##col1, col2, col3 = st.columns(3)
##
##with col1:
##    st.markdown("""
##    **[ASTRA-sim 2.0](https://github.com/astra-sim/astra-sim)**
##    
##    State-of-the-art distributed ML system simulator
##    """)
##
##with col2:
##    st.markdown("""
##    **[Symbolic Tensor Graph](https://github.com/astra-sim/symbolic_tensor_graph)**
##    
##    Generates synthetic LLM workloads with Chakra Execution Traces
##    """)
##
##with col3:
##    st.markdown("""
##    **User-friendly Interface**
##    
##    Streamlit-based web application for intuitive exploration
##    """)

##st.markdown("## Applications & Use Cases")
##col1, col2 = st.columns(2)
##
##with col1:
##    st.markdown("""
##    **Research & Development**
##    - Explore novel parallelism strategies
##    - Test hardware configurations for emerging AI workloads
##    
##    **System Design**
##    - Evaluate network topology decisions
##    - Assess hardware specifications before deployment
##    """)
##
##with col2:
##    st.markdown("""
##    **Performance Optimization**
##    - Identify bottlenecks in distributed training
##    - Optimize existing training setups
##    
##    """)

st.markdown("## Getting Started")

st.markdown("""
**Navigate through the sidebar to access different analysis modules:**

1. **Modeling & Simulation**: Generate custom workloads and run simulations
2. **Explore Parallelism Strategies**: Compare different parallelism approaches  
3. **Explore Network Design**: Analyze interconnect topology and bandwidth impact
4. **Explore Batch Size**: Compare batch sizes impact on performance
5. **Solver**: Automatically discover optimal parallelism strategies

*Each module provides guided workflows with helpful tooltips and explanations.*
""")

st.markdown("""
---

<div style='text-align: center;'>
<h3 style='color: #4F8BF9;'>Our Team</h3>
</div>
""", unsafe_allow_html=True)

# Developer data
developers = [
    {"name": "Mohammad Nasser", "role": "Ph.D Student at Universitat Politècnica de Catalunya", "link": "https://es.linkedin.com/in/mohyna", "image": ""},
    {"name": "Tomás Gadea", "role": "Former Researcher at Universitat Politècnica de Catalunya", "link": "https://ch.linkedin.com/in/tomas-gadea", "image": ""},
    {"name": "Xavier Querol Bassols", "role": "Masters Student at Universitat Politècnica de Catalunya", "link": "https://es.linkedin.com/in/xavier-querol", "image": ""},
    {"name": "Abhijit Das", "role": "Director of Research and Group Leader at the N3Cat at Universitat Politècnica de Catalunya", "link": "https://abhijitcse.github.io/", "image": ""},
    {"name": "Àlex Batlle", "role": "Researcher at Qualcomm Europe, Inc.", "link": "https://es.linkedin.com/in/atellas23", "image": ""},
    {"name": "Adrián Pérez", "role": "Researcher at Qualcomm Technologies, Inc.", "link": "https://www.linkedin.com/in/aperezdieguez", "image": ""},
    {"name": "Jordi Cortadella", "role": "Professor in the Computer Science Department at the Universitat Politècnica de Catalunya", "link": "https://www.cs.upc.edu/~jordicf/", "image": ""},
    {"name": "Sergi Abadal", "role": "Distinguished Researcher at Universitat Politècnica de Catalunya", "link": "https://sergiabadal.com/", "image": ""},
    {"name": "Jordi Ros", "role": "Director of Engineering at Qualcomm Europe, Inc.", "link": "https://www.linkedin.com/in/jordi-ros-giralt-phd", "image": ""}
]

# Display developers in responsive grid using Streamlit columns
st.markdown("""
<style>
.team-member-card {
    text-align: center;
    padding: 0.6em;
    background-color: #f8f9fa;
    border-radius: 6px;
    margin: 0.3em 0;
    border: 1px solid #e9ecef;
    min-height: 110px;
}
.team-member-card img {
    border-radius: 50%;
    width: 60px;
    height: 60px;
    margin-bottom: 0.6em;
    object-fit: cover;
}
.team-member-card h4 {
    margin: 0.3em 0 0.1em 0;
    color: #333;
    font-size: 0.95em;
}
.team-member-card p {
    margin: 0 0 0.3em 0;
    color: #666;
    font-size: 0.78em;
    line-height: 1.3;
}
.team-member-card a {
    color: #4F8BF9;
    text-decoration: none;
    font-size: 0.88em;
}
@media (max-width: 768px) {
    .team-member-card img {
        width: 48px;
        height: 48px;
    }
    .team-member-card h4 {
        font-size: 0.9em;
    }
    .team-member-card p {
        font-size: 0.7em;
    }
}
</style>
""", unsafe_allow_html=True)

# Create responsive team grid using Streamlit columns
# Display in two rows: 5 in top row, 4 in bottom row
# First row - 5 team members
first_row_developers = developers[:5]
cols1 = st.columns(5)
for j, dev in enumerate(first_row_developers):
    with cols1[j]:
        img_src = dev["image"] if dev["image"] else f"https://via.placeholder.com/60x60/4F8BF9/white?text={dev['name'].replace(' ', '+')}"
        st.markdown(f"""
        <div class="team-member-card">
            <!--<img src="{img_src}" alt="{dev['name']}">-->
            <h4><a href='{dev["link"]}' target="_blank">{dev["name"]}</a></h4>
            <p>{dev["role"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Second row - 4 team members
second_row_developers = developers[5:]
cols2 = st.columns(4)

for j, dev in enumerate(second_row_developers):
    with cols2[j]:
        img_src = dev["image"] if dev["image"] else f"https://via.placeholder.com/60x60/4F8BF9/white?text={dev['name'].replace(' ', '+')}"
        st.markdown(f"""
        <div class="team-member-card">
            <!--<img src="{img_src}" alt="{dev['name']}">-->
            <h4><a href='{dev["link"]}' target="_blank">{dev["name"]}</a></h4>
            <p>{dev["role"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Add simple footer to sidebar with contact information
# Add contact information at the bottom of the sidebar
with st.sidebar:
    # Push content to bottom with spacer
    st.markdown("<div style='height: 50vh;'></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1em; background-color: #f8f9fa; border-radius: 8px;'>
        <p style='margin: 0; color: #666; font-size: 0.9em;'>
            <strong>Contact Info:</strong> mohammad.nasser@upc.edu
        </p>
    </div>
    """, unsafe_allow_html=True)

