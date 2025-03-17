#!/bin/bash

#generate workloads


model_name="GPT_3_175B"
model_num=5
workload_configuration="./workload/"${model_name}
memory_config="./configuration/RemoteMemory.json"

output="./output/"${model_name}"/"
result="./results/"${model_name}"/"

rm -rf $output
rm -rf $result
rm -rf $workload_configuration
#
#
python generate_workloads.py --model $model_num --folder_name $model_name

#2D_Torus
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/2D_Torus_sys.json \
    --network ./configuration/2D_Torus.yml \
    --memory $memory_config \
    --output_dir ${output}2D_Torus

#3D_Torus
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/3D_Torus_sys.json \
    --network ./configuration/3D_Torus.yml \
    --memory $memory_config  \
    --output_dir ${output}3D_Torus

#DGX_H100
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/DGX_H100.json \
    --network ./configuration/DGX_H100.yml \
    --memory $memory_config  \
    --output_dir ${output}DGX_H100

#DGX1
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/DGX1_sys.json \
    --network ./configuration/DGX1.yml \
    --memory $memory_config  \
    --output_dir ${output}DGX1

#Dragonfly
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Dragonfly_sys.json \
    --network ./configuration/Dragonfly.yml \
    --memory $memory_config  \
    --output_dir ${output}Dragonfly

#FullyConnected
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/FullyConnected_sys.json \
    --network ./configuration/FullyConnected.yml \
    --memory $memory_config  \
    --output_dir ${output}FullyConnected

#Ring
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Ring_sys.json \
    --network ./configuration/Ring.yml \
    --memory $memory_config  \
    --output_dir ${output}Ring

#Switch
python run_astrasim.py \
    --workload_dir $workload_configuration \
    --system ./configuration/Switch_sys.json \
    --network ./configuration/Switch.yml \
    --memory $memory_config  \
    --output_dir ${output}Switch


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