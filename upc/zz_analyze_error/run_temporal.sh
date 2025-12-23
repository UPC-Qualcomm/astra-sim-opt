for i in $(seq 0 15)
do
  INPUT_FILE="/app/astra-sim/upc/workload/GPT_3_1300M_teest/4_4_1_1_0.seq_2048.batch_1024.${i}.et"
  OUTPUT_FILE="/app/astra-sim/upc/zz_analyze_error/jsons/${i}.et.txt"
  
  CMD=$(printf "chakra_jsonizer --input_filename=%q --output_filename=%q" "$INPUT_FILE" "$OUTPUT_FILE")
  eval "$CMD"
done