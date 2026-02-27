python3 /app/astra-sim/upc/experiments_files/experiment7/generate_workloads.py \
    --model 20 \
    --folder_name /app/astra-sim/upc/experiments_files/experiment7/workload/GPT13B_last \
    --num_samples 5 

cd /app/astra-sim/upc/experiments_files/experiment7/workload/
for file in GPT13B_last/*; do
    if [ -f "$file" ]; then
        mv "$file" "GPT13B_last_$(basename "$file")"
    fi
done
mv GPT13B_last_* ./
rmdir GPT13B_last

mkdir -p GPT13B

for file in GPT13B_last_*; do
    base=$(echo "$file" | sed 's/\.[0-9]*\.et$//' | sed 's/\.json$//')
    mkdir -p "GPT13B/$base"
    mv "$file" "GPT13B/$base/"
done