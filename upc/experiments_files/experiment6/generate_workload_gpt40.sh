python3 /app/astra-sim/upc/experiments_files/experiment6/generate_workloads.py \
    --model 19 \
    --folder_name /app/astra-sim/upc/experiments_files/experiment6/workload/GPT40B_last \
    --num_samples 5

cd /app/astra-sim/upc/experiments_files/experiment6/workload/
for file in GPT40B_last/*; do
    if [ -f "$file" ]; then
        mv "$file" "GPT40B_last_$(basename "$file")"
    fi
done
mv GPT40B_last_* ./
rmdir GPT40B_last

mkdir -p GPT40B

for file in GPT40B_last_*; do
    base=$(echo "$file" | sed 's/\.[0-9]*\.et$//' | sed 's/\.json$//')
    mkdir -p "GPT40B/$base"
    mv "$file" "GPT40B/$base/"
done