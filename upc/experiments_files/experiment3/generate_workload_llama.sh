python3 /app/astra-sim/upc/experiments_files/experiment3/generate_workloads.py \
    --model 17 \
    --folder_name /app/astra-sim/upc/experiments_files/experiment3/workload/Llama8B_last \
    --num_samples 30

cd /app/astra-sim/upc/experiments_files/experiment3/workload/
for file in Llama8B_last/*; do
    if [ -f "$file" ]; then
        mv "$file" "Llama8B_last_$(basename "$file")"
    fi
done
mv Llama8B_last_* ./
rmdir Llama8B_last

mkdir -p Llama8B

for file in Llama8B_last_*; do
    base=$(echo "$file" | sed 's/\.[0-9]*\.et$//' | sed 's/\.json$//')
    mkdir -p "Llama8B/$base"
    mv "$file" "Llama8B/$base/"
done