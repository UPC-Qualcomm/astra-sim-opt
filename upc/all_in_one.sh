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

workload_configuration="./workload/"${folder_name}
memory_config="./configuration/RemoteMemory.json"
network_log="./network_log/"${folder_name}"/"

output="./output/"${folder_name}"/"
result="./results/"${folder_name}"/"

rm -rf $output
rm -rf $result
rm -rf $workload_configuration
rm -rf $network_log
#
#
python generate_workloads.py --model $model_num --folder_name $folder_name

#2D_Torus
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/2D_Torus_sys.json \
    --network ./configuration/2D_Torus.yml \
    --memory $memory_config \
    --output_dir ${output}2D_Torus \
    --network_log ${network_log}2D_Torus

#3D_Torus
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/3D_Torus_sys.json \
    --network ./configuration/3D_Torus.yml \
    --memory $memory_config  \
    --output_dir ${output}3D_Torus \
    --network_log ${network_log}3D_Torus

#DGX_H100
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/DGX_H100_sys.json \
    --network ./configuration/DGX_H100.yml \
    --memory $memory_config  \
    --output_dir ${output}DGX_H100 \
    --network_log ${network_log}DGX_H100

#DGX1
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/DGX1_sys.json \
    --network ./configuration/DGX1.yml \
    --memory $memory_config  \
    --output_dir ${output}DGX1 \
    --network_log ${network_log}DGX1

#Dragonfly
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Dragonfly_sys.json \
    --network ./configuration/Dragonfly.yml \
    --memory $memory_config  \
    --output_dir ${output}Dragonfly \
    --network_log ${network_log}Dragonfly

#FullyConnected
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/FullyConnected_sys.json \
    --network ./configuration/FullyConnected.yml \
    --memory $memory_config  \
    --output_dir ${output}FullyConnected \
    --network_log ${network_log}FullyConnected

#Ring
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Ring_sys.json \
    --network ./configuration/Ring.yml \
    --memory $memory_config  \
    --output_dir ${output}Ring \
    --network_log ${network_log}Ring

#Switch
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Switch_sys.json \
    --network ./configuration/Switch.yml \
    --memory $memory_config  \
    --output_dir ${output}Switch \
    --network_log ${network_log}Switch


#Collect results
#2D_Torus
python gather_results.py --sim_logfiles_dir ${output}2D_Torus --output_filename ${result}2D_Torus.csv

#3D_Torus
python gather_results.py --sim_logfiles_dir ${output}3D_Torus --output_filename ${result}3D_Torus.csv

#DGX_H100
python gather_results.py --sim_logfiles_dir ${output}DGX_H100 --output_filename ${result}DGX_H100.csv

#DGX1
python gather_results.py --sim_logfiles_dir ${output}DGX1 --output_filename ${result}DGX1.csv

#Dragonfly
python gather_results.py --sim_logfiles_dir ${output}Dragonfly --output_filename ${result}Dragonfly.csv

#FullyConnected
python gather_results.py --sim_logfiles_dir ${output}FullyConnected --output_filename ${result}FullyConnected.csv

#Ring
python gather_results.py --sim_logfiles_dir ${output}Ring --output_filename ${result}Ring.csv

#Switch
python gather_results.py --sim_logfiles_dir ${output}Switch --output_filename ${result}Switch.csv
