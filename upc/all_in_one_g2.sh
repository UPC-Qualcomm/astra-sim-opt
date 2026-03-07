#!/bin/bash
source /media/mohammad/extension/experiments/311_astraenv/bin/activate

npus="_64"
model="GPT_40B_16L"
workload_configuration_file_name=${model}${npus}
npus="_64"

folder_name=${model}"_test_g2_Astra_sync"${npus}
model_num=28

workload_configuration="./workload/"${workload_configuration_file_name}
memory_config="./configuration/RemoteMemory.json"
sim_type="g2"
export PYTHONPATH=/media/mohammad/extension/experiments/astra-sim/extern/network_backend/g2:$PYTHONPATH

rm -rf $output
rm -rf $result
rm -rf $workload_configuration
rm -rf $network_log

time python generate_workloads_g2.py --model $model_num --folder_name $workload_configuration_file_name

# Run the simulation 100 times
#for iteration in {2..2}
#do
#    echo "Start: GPT_1.3B_test_g2_Astra_sync"${npus}
#    
#    network_log="./network_log/"${folder_name}"_iter${iteration}/"
#    output="./output/"${folder_name}"/"
#    result="./results/"${folder_name}"/"
#
#    rm -rf $output
#    rm -rf $result
#    rm -rf $network_log
#
#    #FoldedClos
#    time python run_astrasim.py \
#        --workload_dir $workload_configuration \
#        --system /media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/Switch_sys.json \
#        --network /media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/FoldedClos_16_config.yml \
#        --memory $memory_config  \
#        --output_dir ${output}FoldedClos_iter${iteration} \
#        --network_log ${network_log}FoldedClos_iter${iteration} \
#        --sim_type ${sim_type} > ${folder_name}_log.txt 2>&1
#
#    time python gather_all_NPUs_results.py --sim_logfile ${output}FoldedClos_iter${iteration} --output_filename ${result}FoldedClos_iter${iteration} 
#
#    echo "Finish: GPT_1.3B_test_g2_Astra_sync"${npus}
#done


folder_name=${model}"_test_g2_my_sync_fix"${npus}
network_config="/media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/FoldedClos_16_config_untracked.yml"
power_config="./power_model/a100_config.json"
## Run the simulation 100 times
for iteration in {2..2}
do
    echo "Start: " $folder_name
    
    network_log="./network_log/"${folder_name}"_iter${iteration}/"
    output="./output/"${folder_name}"/"
    result="./results/"${folder_name}"/"

    rm -rf $output
    rm -rf $result
    rm -rf $network_log

    #FoldedClos
    time python run_astrasim.py \
        --workload_dir $workload_configuration \
        --system /media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/switch_my_BARRIER_sys.json \
        --network ${network_config} \
        --memory $memory_config  \
        --output_dir ${output}FoldedClos_iter${iteration} \
        --network_log ${network_log}FoldedClos_iter${iteration} \
        --sim_type ${sim_type} > ${folder_name}_log.txt 2>&1

    echo "Post-processing: Estimate the power"
    time python power_model/estimate_power.py \
        --output-dir ${output}FoldedClos_iter${iteration} \
        --network    ${network_config} \
        --config     ${power_config} \
        --result-dir ${output}FoldedClos_iter${iteration} \
        --mode compare

    time python gather_all_NPUs_results.py --sim_logfile ${output}FoldedClos_iter${iteration} --output_filename ${result}FoldedClos_iter${iteration} --include_power true

done

#folder_name="GPT_1.3B_test_g2_dep_inject"${npus}
#
#
#
## Run the simulation 100 times
#for iteration in {2..2}
#do
#    echo "Start: GPT_1.3B_test_g2_dep_inject"${npus}
#    
#    network_log="./network_log/"${folder_name}"_iter${iteration}/"
#    output="./output/"${folder_name}"/"
#    result="./results/"${folder_name}"/"
#
#    rm -rf $output
#    rm -rf $result
#    rm -rf $network_log
#
#    #FoldedClos
#    time python run_astrasim.py \
#        --workload_dir $workload_configuration \
#        --system /media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/switch_ORDER_INJECTION_sys.json \
#        --network /media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/FoldedClos_16_config.yml \
#        --memory $memory_config  \
#        --output_dir ${output}FoldedClos_iter${iteration} \
#        --network_log ${network_log}FoldedClos_iter${iteration} \
#        --sim_type ${sim_type} > ${folder_name}_log.txt 2>&1
#
#    
#    echo "Finish: GPT_1.3B_test_g2_dep_inject"${npus}
#done
#
#
#folder_name="GPT_1.3B_test_g2_disabled_sync"${npus}
# Run the simulation 100 times
#for iteration in {2..2}
#do
#    echo "Start: GPT_1.3B_test_g2_disabled_sync"${npus}
#
#    network_log="./network_log/"${folder_name}"_iter${iteration}/"
#    output="./output/"${folder_name}"/"
#    result="./results/"${folder_name}"/"
#
#    rm -rf $output
#    rm -rf $result
#    rm -rf $network_log
#
#    #FoldedClos
#    time python run_astrasim.py \
#        --workload_dir $workload_configuration \
#        --system /media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/Switch_NO_SYNC_sys.json \
#        --network /media/mohammad/extension/experiments/astra-sim/upc/configuration/g2/FoldedClos_16_config.yml \
#        --memory $memory_config  \
#        --output_dir ${output}FoldedClos_iter${iteration} \
#        --network_log ${network_log}FoldedClos_iter${iteration} \
#        --sim_type ${sim_type} > ${folder_name}_log.txt 2>&1
#
#    time python gather_all_NPUs_results.py --sim_logfile ${output}FoldedClos_iter${iteration} --output_filename ${result}FoldedClos_iter${iteration} 
#    
#    echo "Finish: GPT_1.3B_test_g2_disabled_sync"${npus}
#done
#