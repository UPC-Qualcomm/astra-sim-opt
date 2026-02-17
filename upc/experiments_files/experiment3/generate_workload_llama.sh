python3 /app/astra-sim/upc/experiments_files/experiment3/generate_workloads.py \
    --model 17 \
    --folder_name /app/astra-sim/upc/experiments_files/experiment3/workload/Llama8B_last \

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

cd Llama8B
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