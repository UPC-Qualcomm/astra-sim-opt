import streamlit as st
import sections.trace_viewer as tv
import sections.trace_picker as picker
import sections.report as report
import sections.bw_form as bw_form
import helper.bw_run_sim as bw_run
import sections.bw_show_res as bw_show

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
st.subheader(
    "Exploration: Interconnect Network Topology",
    help=(
        "Explore how different interconnect network designs affect simulation time for the selected GPT model."
    )
)
# Generate Workload
base_dir = picker._get_output_dir()
selected_model = picker.model_selector(base_dir)

#TODO: Switch variables rather than hardcoded values
st.info(f"""
        **Peak performance for Single NPU**: {989} TFLOPs, \t
        **Peak memory bandwidth**: {3350} GB/s, \n
        **Inter node Bandwidth**: {200} GB/s, \t
        **Intra node Bandwidth**: {900} GB/s, \t
        **Number of NPUs**: {32}
    """)
fig, selected_topologies, df = (
    report.analysis_across_topologies(selected_model)
)


st.pyplot(fig)
st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666; margin-top: 1em;'>Detailed comparison of selected parallelism strategies across different network topologies. Each subplot shows the execution time breakdown (overlap, exposed computation, and exposed communication) for a specific parallelism configuration, enabling direct performance comparison between topologies for optimal system design.</p>", unsafe_allow_html=True)

st.markdown("---")

###selected_config = picker.config_selector(base_dir, selected_model)
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        background-color: #e0e7ff !important;  /* Light blue */
        color: #222 !important;
        font-weight: bold;
        font-size: 1.2em;
        border-radius: 8px 8px 0 0;
        margin-right: 4px;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important; /* Indigo */
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

tabs = st.tabs(
    ["Explore Various Interconnect Network Bandwidths", "Detailed Trace Visualization"]
)

with tabs[0]:
    all_configs = bw_form.get_configuration_options()

    df_sorted = df.sort_values(by="exec_cycles", ascending=True).iloc[0:10]
    parallelism_options = df_sorted["dp_mp_sp_pp_sharded"].tolist()
    selected_parallelism = st.multiselect(
        f"Select Parallelism Strategies (dp_mp_sp_pp_sharded) - Top {len(parallelism_options)} is available (max 4 you can select):",
        parallelism_options,
        default=parallelism_options[:4],
        max_selections=4,
    )
    if len(selected_parallelism) == 0:
        st.warning(
            "Please select the parallelism strategies you want to include in the exploration."
        )
    else:
        parallelism_strategies = df[df["dp_mp_sp_pp_sharded"].isin(selected_parallelism)]["file_name"].unique()
        #### TODO: Restore this when the demo is done
        col0, _ = st.columns(2)
        #### with col0:
        ####     sim_mode = st.radio(
        ####         "Select Exploration Mode", ["Saved Data", "Generate New Data"]
        ####     )
        with col0:
            plot_type = st.radio("Select Plot Type", ["2D", "3D"])
            is_3d = plot_type == "3D"
        sim_mode = "Saved Data"  # For demo purposes, always use saved data
        if sim_mode == "Saved Data":
            #### st.subheader("Exploration Using Saved Data")

            all_res_dirs_configs = []

            st.subheader(
                "Study the effect of the link capacity over parallelism strategies", help=(
                    "Analyze how varying link capacities impact the performance of different parallelism strategies on a given topology."
                )
            )
            
            for config in all_configs:
                if config in selected_topologies:
                    result_dir = bw_run.get_results_dir(selected_model, config)
                    if result_dir.exists() and result_dir.is_dir():
                        bw_show.show_inter_intra_sim_res(
                            parallelism_strategies, result_dir, config, is_3d
                        )
                        all_res_dirs_configs.append((result_dir, config))

                        st.markdown("---")

            if all_res_dirs_configs:
                st.subheader(
                    "Study the effect of the link capacity over network topologies", help=(
                        "Analyze how varying link capacities impact the performance of different network topologies given a specific configuration."
                    )
                )
                for parallelism_strategy in parallelism_strategies:
                    bw_show.show_res_across_configs(
                        parallelism_strategy, all_res_dirs_configs, is_3d
                    )

        else:
            st.subheader("Exploration With New Generated Data")
            run_button, intra_bw_list, inter_bw_list, selected_config_names_list = (
                bw_form.sweep_bw_form(selected_topologies, selected_model)
            )
            all_res_dirs_configs = []
            for config in all_configs:
                if config in selected_config_names_list:
                    bw_run.intra_inter_simulations_run(
                        parallelism_strategies,
                        config,
                        selected_model,
                        run_button,
                        intra_bw_list,
                        inter_bw_list,
                    )

                result_dir = bw_run.get_results_dir(selected_model, config)
                if result_dir.exists() and result_dir.is_dir():
                    bw_show.show_inter_intra_sim_res(
                        parallelism_strategies, result_dir, config, is_3d
                    )
                    all_res_dirs_configs.append((result_dir, config))

                st.markdown("---")
            st.subheader(
                "Study the effect of the bandwidth over network configurations"
            )
            for parallelism_strategy in parallelism_strategies:
                bw_show.show_res_across_configs(
                    parallelism_strategy, all_res_dirs_configs, is_3d
                )
with tabs[1]:
    try:
        selected_config = picker.config_selector(base_dir, selected_model)
        sim_outputs = picker.set_sim_input(selected_model, selected_config)
    except:
        st.error("File Not Found.")

    
    st.markdown("---")
    if "df_matched" in st.session_state:
        tv.render_sim_output_section(sim_outputs)
