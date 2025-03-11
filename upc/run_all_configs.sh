#!/bin/bash


workload_configuration="./workload/GPT3"
memory_config="./configuration/RemoteMemory.json"


#2D_Torus
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/2D_Torus_sys.json \
--network ./configuration/2D_Torus.yml \
--memory $memory_config \
--output_dir ./output/GPT3/2D_Torus

#3D_Torus
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/3D_Torus_sys.json \
--network ./configuration/3D_Torus.yml \
--memory memory_config \
--output_dir ./output/GPT3/3D_Torus


#DGX_H100
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/DGX_H100.json \
--network ./configuration/DGX_H100.yml \
--memory memory_config \
--output_dir ./output/GPT3/DGX_H100


#DGX1
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/DGX1_sys.json \
--network ./configuration/DGX1.yml \
--memory memory_config \
--output_dir ./output/GPT3/DGX1


#Dragonfly
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/Dragonfly_sys.json \
--network ./configuration/Dragonfly.yml \
--memory memory_config \
--output_dir ./output/GPT3/Dragonfly


#FullyConnected
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/FullyConnected_sys.json \
--network ./configuration/FullyConnected.yml \
--memory memory_config \
--output_dir ./output/GPT3/FullyConnected

#Ring
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/Ring_sys.json \
--network ./configuration/Ring.yml \
--memory memory_config \
--output_dir ./output/GPT3/Ring

#Switch
python run_astrasim.py \
--workload_dir $workload_configuration \
--system ./configuration/Switch_sys.json \
--network ./configuration/Switch.yml \
--memory memory_config \
--output_dir ./output/GPT3/Switch


