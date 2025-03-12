#!/bin/bash


output="./output/GPT3/"
result="./results/GPT3/"

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
