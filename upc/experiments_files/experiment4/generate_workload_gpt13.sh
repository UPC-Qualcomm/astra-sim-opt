python3 /app/astra-sim/upc/experiments_files/experiment4/generate_workloads.py \
    --model 8 \
    --folder_name /app/astra-sim/upc/experiments_files/experiment4/workload/GPT13B_last \

cd /app/astra-sim/upc/experiments_files/experiment4/workload/
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

cd GPT13B
folder_count=$(ls -1d */ 2>/dev/null | wc -l)
if [ "$folder_count" -gt 4 ]; then
    ls -1d */ | shuf | head -4 > /tmp/keep_folders
    for folder in */; do
        if ! grep -q "$folder" /tmp/keep_folders; then
            rm -rf "$folder"
        fi
    done
    rm -f /tmp/keep_folders
fi