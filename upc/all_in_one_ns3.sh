#!/bin/bash

# AstraSim with NS3 Network Model Runner Script
# This script runs AstraSim with NS3 network backend instead of analytical models

set -e  # Exit on any error

BASE_DIR="/media/mohammad/extension/experiments"
# Python executable path
PYTHON_EXEC=$BASE_DIR"/astraenv39/bin/python"

echo "=== AstraSim NS3 Integration Runner ==="
echo "Using Python: $PYTHON_EXEC"

# Model configuration - choose one of the following
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

folder_name="GPT_3_1300M_test"
model_num=5

#folder_name="GPT_Neo_2700M"
#model_num=6

#folder_name="FLAN_T5_XXL_11B"
#model_num=7

#folder_name="OPT_13B"
#model_num=8

#folder_name="GPT_NeoX_20B"
#model_num=9

#folder_name="GPT_3_175B"
#model_num=10

#folder_name="PaLM_540B"
#model_num=11

#folder_name="GPT_4_Estimated_over_1T"
#model_num=12

#folder_name="Default"
#model_num=13

#folder_name="LLaMA_3_70B"
#model_num=14

#folder_name="Model_100B"
#model_num=15

#folder_name="Model_120B"
#model_num=16

#folder_name="llama_8B"
#model_num=17

#folder_name="GPT_30B"
#model_num=18

#folder_name="GPT_40B"
#model_num=19

#folder_name="simple"
#model_num=20

# Configuration paths
workload_configuration="./workload/"${folder_name}
memory_config="./configuration/RemoteMemory.json"
network_log="./network_log/"${folder_name}"/"
output="./output/"${folder_name}"/"
result="./results/"${folder_name}"/"

# NS3 Network Model Configuration
sim_type="ns3"

# Clean up previous runs
rm -rf $output
rm -rf $result
rm -rf $workload_configuration
rm -rf $network_log

# Generate workloads
time $PYTHON_EXEC generate_workloads.py --model $model_num --folder_name $folder_name


# You can choose different topologies by commenting/uncommenting the sections below

# 64-Node Ring Topology (Compatible with current workload generation)

time $PYTHON_EXEC run_astrasim_ns3.py \
    --workload_dir $workload_configuration \
    --system ./configuration/FoldedClos_my_BARRIER_sys.json \
    --network_config $BASE_DIR/astra-sim/configuration/ns3/configs/FoldedClos_16_config3.txt  \
    --logical_topology ./configuration/ns3/128_nodes_logical_me.json \
    --memory $memory_config \
    --output_dir ${output}testns3 \
    --network_log ${network_log}testns3

time python gather_all_NPUs_results.py --sim_logfile ${output}FoldedClos  --output_filename ${result}FoldedClos


# 128-Node Clos Topology (Uncomment to use)
# echo "Running AstraSim with NS3 128-node Clos topology..."
# echo "⚠️  Note: Large-scale NS3 simulations take significantly longer!"
# time $PYTHON_EXEC run_astrasim_ns3.py \
#     --workload_dir $workload_configuration \
#     --system ./configuration/ns3/128_nodes_sys.json \
#     --network_config ./configuration/ns3/astrasim_128nodes_clos_config.txt \
#     --logical_topology ./configuration/ns3/128_nodes_logical.json \
#     --memory $memory_config \
#     --output_dir ${output}128_nodes_clos \
#     --network_log ${network_log}128_nodes_clos

 
