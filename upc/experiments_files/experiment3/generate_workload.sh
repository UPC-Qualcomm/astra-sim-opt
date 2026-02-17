# python3 /app/astra-sim/upc/generate_workloads.py \
#     --model 5 \
#     --folder_name ../comparing_networks/workload/GPT_3_1300M \
python3 /app/astra-sim/upc/experiments_files/experiment3/generate_workloads.py \
    --model 5 \
    --folder_name /app/astra-sim/upc/experiments_files/experiment3/workload/GPT_3_1300M_last \

cd /app/astra-sim/upc/experiments_files/experiment3/workload/
for file in GPT_3_1300M_last/*; do
    if [ -f "$file" ]; then
        mv "$file" "GPT_3_1300M_last_$(basename "$file")"
    fi
done
mv GPT_3_1300M_last_* ./
rmdir GPT_3_1300M_last

mkdir -p GPT_3_1300M

for file in GPT_3_1300M_last_*; do
    base=$(echo "$file" | sed 's/\.[0-9]*\.et$//' | sed 's/\.json$//')
    mkdir -p "GPT_3_1300M/$base"
    mv "$file" "GPT_3_1300M/$base/"
done

cd GPT_3_1300M
folder_count=$(ls -1d */ 2>/dev/null | wc -l)
if [ "$folder_count" -gt 30 ]; then
    ls -1d */ | shuf | head -30 > /tmp/keep_folders
    for folder in */; do
        if ! grep -q "$folder" /tmp/keep_folders; then
            rm -rf "$folder"
        fi
    done
    rm -f /tmp/keep_folders
fi