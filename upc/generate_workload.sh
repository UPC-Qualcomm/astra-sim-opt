# python3 /app/astra-sim/upc/generate_workloads.py \
#     --model 5 \
#     --folder_name ../comparing_networks/workload/GPT_3_1300M \
python3 /app/astra-sim/upc/generate_workloads.py \
    --model 5 \
    --folder_name ../comparing_networks/workload/T5_Small_multiple_last

# Postprocessing: move files from subfolder to parent folder with prefix, then group by basepath
cd /app/astra-sim/upc/comparing_networks/workload/
for file in T5_Small_multiple_last/*; do
    if [ -f "$file" ]; then
        mv "$file" "T5_Small_multiple_last_$(basename "$file")"
    fi
done
mv T5_Small_multiple_last_* ./
rmdir T5_Small_multiple_last

# Create a parent folder for the grouped folders
mkdir -p T5_Small_grouped_128

# Group files by basepath into folders inside the parent folder
for file in T5_Small_multiple_last_*; do
    base=$(echo "$file" | sed 's/\.[0-9]*\.et$//' | sed 's/\.json$//')
    mkdir -p "T5_Small_grouped_128/$base"
    mv "$file" "T5_Small_grouped_128/$base/"
done