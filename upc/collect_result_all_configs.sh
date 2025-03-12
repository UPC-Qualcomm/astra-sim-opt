#!/bin/bash

#2D_Torus
python gather_results.py --sim_logfiles_dir ./output/GPT3/2D_Torus --output_filename ./results/GPT3/2D_Torus.csv

#3D_Torus
python gather_results.py --sim_logfiles_dir ./output/GPT3/3D_Torus --output_filename ./results/GPT3/3D_Torus.csv

#DGX_H100
python gather_results.py --sim_logfiles_dir ./output/GPT3/DGX_H100 --output_filename ./results/GPT3/DGX_H100.csv

#DGX1
python gather_results.py --sim_logfiles_dir ./output/GPT3/DGX1 --output_filename ./results/GPT3/DGX1.csv

#Dragonfly
python gather_results.py --sim_logfiles_dir ./output/GPT3/Dragonfly --output_filename ./results/GPT3/Dragonfly.csv

#FullyConnected
python gather_results.py --sim_logfiles_dir ./output/GPT3/FullyConnected --output_filename ./results/GPT3/FullyConnected.csv

#Ring
python gather_results.py --sim_logfiles_dir ./output/GPT3/Ring --output_filename ./results/GPT3/Ring.csv

#Switch
python gather_results.py --sim_logfiles_dir ./output/GPT3/Switch --output_filename ./results/GPT3/Switch.csv
