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


# folder_name="GPT_40B"
# model_num=19

workload_configuration="./workload/"${folder_name}
memory_config="./configuration/RemoteMemory.json"
network_log="./network_log/"${folder_name}"/"

output="./output/"${folder_name}"/"
result="./results/"${folder_name}"/"

#rm -rf $output
#rm -rf $result
#rm -rf $workload_configuration
#rm -rf $network_log
#
#
#python generate_workloads.py --model $model_num --folder_name $folder_name

#FullyConnected
python run_astrasim_aware.py \
    --workload_dir $workload_configuration \
    --system ./configuration/FullyConnected_aware_sys.json \
    --network ./configuration/FullyConnected.yml \
    --memory $memory_config  \
    --output_dir ${output}FullyConnected_aware \
    --network_log ${network_log}FullyConnected_aware

##Ring
python run_astrasim_aware.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Ring_aware_sys.json \
    --network ./configuration/Ring.yml \
    --memory $memory_config  \
    --output_dir ${output}Ring_aware \
    --network_log ${network_log}Ring_aware

#Switch
python run_astrasim_aware.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Switch_aware_sys.json \
    --network ./configuration/Switch.yml \
    --memory $memory_config  \
    --output_dir ${output}Switch_aware \
    --network_log ${network_log}Switch_aware


#Collect results

#FullyConnected
#python gather_results.py --sim_logfiles_dir ${output}FullyConnected --output_filename ${result}FullyConnected.csv
python gather_all_NPUs_results.py --sim_logfile ${output}FullyConnected_aware  --output_filename ${result}FullyConnected_aware

#Ring
#python gather_results.py --sim_logfiles_dir ${output}Ring --output_filename ${result}Ring.csv
python gather_all_NPUs_results.py --sim_logfile ${output}Ring_aware  --output_filename ${result}Ring_aware

#Switch
#python gather_results.py --sim_logfiles_dir ${output}Switch --output_filename ${result}Switch.csv
python gather_all_NPUs_results.py --sim_logfile ${output}Switch_aware  --output_filename ${result}Switch_aware
