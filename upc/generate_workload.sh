# python3 /app/astra-sim/upc/generate_workloads.py \
#     --model 5 \
#     --folder_name ../comparing_networks/workload/GPT_3_1300M \
python3 /app/astra-sim/upc/generate_workloads.py \
    --model 20 \
    --folder_name ../comparing_networks/workload/T5_Base_multiple

# Postprocessing: move files from subfolder to parent folder with prefix, then group by basepath
cd /app/astra-sim/upc/comparing_networks/workload/
for file in T5_Base_multiple/*; do
    if [ -f "$file" ]; then
        mv "$file" "T5_Base_multiple_$(basename "$file")"
    fi
done
mv T5_Base_multiple_* ./
rmdir T5_Base_multiple

# Create a parent folder for the grouped folders
mkdir -p T5_Base_grouped

# Group files by basepath into folders inside the parent folder
for file in T5_Base_multiple_*; do
    base=$(echo "$file" | sed 's/\.[0-9]*\.et$//' | sed 's/\.json$//')
    mkdir -p "T5_Base_grouped/$base"
    mv "$file" "T5_Base_grouped/$base/"
done