#!/bin/bash

#folder_name="T5_Small"
#model_num=0

#folder_name="T5_Base"
#model_num=1

#folder_name="T5_Large"
#model_num=2

#folder_name="GPT_2_Small"
#model_num=3

#folder_name="GPT_2_Medium"
#model_num=4

folder_name="GPT_3_1300M"
model_num=5

#folder_name="GPT_Neo_2700M"
#model_num=6

#folder_name="FLAN_T5_XXL_11B"
#model_num=7

#folder_name="OPT_13B"
#model_num=8

# folder_name="GPT_NeoX_20B"
# model_num=9

#folder_name="GPT_3_175B"
#model_num=10

#folder_name="PaLM_540B"
#model_num=11

#folder_name="GPT_4_Estimated_over_1T"
#model_num=12

#folder_name="Default"
#model_num=13

output="./output/"${folder_name}
result="./results/"${folder_name}
arch="2D_Torus"
for arch in "2D_Torus" "3D_Torus" "DGX1" "DGX_H100" "Dragonfly" "FullyConnected" "Ring" "Switch"; do
    echo "Current architecture: $arch"
    rm -rf "${result}/${arch}/"

    python gather_node_timings.py --sim_logfilename "${output}/${arch}/1_8_2_4_0.log" --output_filename "${result}/${arch}/trace_1_8_2_4_0.csv"
    python gather_node_timings.py --sim_logfilename "${output}/${arch}/2_1_32_1_0.log" --output_filename "${result}/${arch}/trace_2_1_32_1_0.csv"
    python gather_node_timings.py --sim_logfilename "${output}/${arch}/4_1_16_1_0.log" --output_filename "${result}/${arch}/trace_4_1_16_1_0.csv"
    python gather_node_timings.py --sim_logfilename "${output}/${arch}/4_2_2_4_0.log" --output_filename "${result}/${arch}/trace_4_2_2_4_0.csv"
    python gather_node_timings.py --sim_logfilename "${output}/${arch}/4_8_2_1_0.log" --output_filename "${result}/${arch}/trace_4_8_2_1_0.csv"
    python gather_node_timings.py --sim_logfilename "${output}/${arch}/16_1_1_4_0.log" --output_filename "${result}/${arch}/trace_16_1_1_4_0.csv"
    python gather_node_timings.py --sim_logfilename "${output}/${arch}/1_1_64_1_0.log" --output_filename "${result}/${arch}/trace_1_1_64_1_0.csv"

    python gather_roofline_metrics.py --sim_logfilename "${output}/${arch}/1_8_2_4_0.log" --output_filename "${result}/${arch}/roofline_1_8_2_4_0.csv"
    python gather_roofline_metrics.py --sim_logfilename "${output}/${arch}/2_1_32_1_0.log" --output_filename "${result}/${arch}/roofline_2_1_32_1_0.csv"
    python gather_roofline_metrics.py --sim_logfilename "${output}/${arch}/4_1_16_1_0.log" --output_filename "${result}/${arch}/roofline_4_1_16_1_0.csv"
    python gather_roofline_metrics.py --sim_logfilename "${output}/${arch}/4_2_2_4_0.log" --output_filename "${result}/${arch}/roofline_4_2_2_4_0.csv"
    python gather_roofline_metrics.py --sim_logfilename "${output}/${arch}/4_8_2_1_0.log" --output_filename "${result}/${arch}/roofline_4_8_2_1_0.csv"
    python gather_roofline_metrics.py --sim_logfilename "${output}/${arch}/16_1_1_4_0.log" --output_filename "${result}/${arch}/roofline_16_1_1_4_0.csv"
    python gather_roofline_metrics.py --sim_logfilename "${output}/${arch}/1_1_64_1_0.log" --output_filename "${result}/${arch}/roofline_1_1_64_1_0.csv"

done


