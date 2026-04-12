#!/bin/bash

# Configuration Parameters (Update these with your real paths!)
INPUT_FILE="qwen64.jsonl"
MODEL_PATH="/data/home/jianfeng/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base"
OUTPUT_DIR="experiment_results"

mkdir -p $OUTPUT_DIR
LOG_FILE="$OUTPUT_DIR/experiment_logs.txt"
# Clear or create the log file
echo "Starting Comprehensive Grid Search at $(date)" > $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE
echo "Phase 1: Greedy Baseline (Temperature = 0.0)" | tee -a $LOG_FILE
echo "Since temperature is 0, multiple samples will be identical." | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE
# Loop through different top_k values for Greedy
# Run Greedy without Top-K constraint (k=-1 means all candidates)
for k in -1; do
    echo -e "\n>>> Starting Greedy Verification: T=0 | top_k=ALL | n=1" | tee -a $LOG_FILE
    python $(dirname "$0")/verify_all_candidates_simple.py \
        --model_path "$MODEL_PATH" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_DIR/greedy_top_k_ALL.jsonl" \
        --top_k $k \
        --num_return_sequences 1 \
        --temperature 0.0 2>&1 | tee -a $LOG_FILE
done
echo "" | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE
echo "Phase 2: Full Hyperparameter Grid Search" | tee -a $LOG_FILE
echo "- Temperatures : 0.3, 0.6, 0.9" | tee -a $LOG_FILE
echo "- N (samples)  : 8, 16, 32" | tee -a $LOG_FILE
echo "- Top K levels : ALL_CANDIDATES" | tee -a $LOG_FILE
echo "Total Runs in Phase 2: 9 configurations (no top-k variation)" | tee -a $LOG_FILE
echo "==========================================================" | tee -a $LOG_FILE

for n in 8 16 32; do
    for t in 0.6 0.3 0.9; do
        for k in -1; do
            echo -e "\n>>> Starting Grid Run: Temp=$t | n=$n | top_k=ALL" | tee -a $LOG_FILE
            
            # Format temperature for filename (e.g., 0.3 -> 0p3) so parsing is safe
            clean_t=$(echo $t | tr '.' 'p')
            out_file="$OUTPUT_DIR/grid_t${clean_t}_n${n}_topk_ALL.jsonl"
            
            python $(dirname "$0")/verify_all_candidates_simple.py \
                --model_path "$MODEL_PATH" \
                --input_file "$INPUT_FILE" \
                --output_file "$out_file" \
                --top_k $k \
                --num_return_sequences $n \
                --temperature $t 2>&1 | tee -a $LOG_FILE
                
        done
    done
done
echo "" | tee -a $LOG_FILE
echo "All 10 experiments (1 Greedy + 9 Grid Search) completed successfully!" | tee -a $LOG_FILE
echo "Results are saved in $OUTPUT_DIR." | tee -a $LOG_FILE
echo "Terminal output logs are permanently saved in $LOG_FILE" | tee -a $LOG_FILE