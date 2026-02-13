set -e

CATEGORY="Industrial_and_Scientific"

# sft
exp_name="saves/qwen2.5-0.5b/full/${CATEGORY}-sft-dsz0/checkpoint-120"
# exp_name="saves/qwen2.5-1.5b/full/${CATEGORY}-sft-dsz2"

# rl
# exp_name="rl_outputs/${CATEGORY}-qwen2.5-0.5b-instruct-grpo/checkpoint-1155"
# exp_name="rl_outputs/${CATEGORY}-qwen2.5-1.5b-instruct-grpo/checkpoint-1155"


# multi-GPU config
cuda_list="0 1 2 3 4 5 6 7"  # 8 GPUs
# cuda_list="0"                    # single GPU
# cuda_list="0 1 2 3"

# Test data path
test_data_path="data/${CATEGORY}/sft/test.json"

# Temp and output directories
temp_dir="temp/eval-${CATEGORY}"
output_dir="results/${exp_name#*/}"

# evaluation params
index_path="data/${CATEGORY}/id2sid.json"
batch_size=8
max_new_tokens=256
num_beams=50
temperature=1.0
do_sample=False

# main
exp_name_clean=$(basename "$exp_name")
echo "=========================================="
echo "Industry Recommendation Evaluation"
echo "Model: $exp_name_clean"
echo "Test data: $test_data_path"
echo "=========================================="

# check test data
if [[ ! -f "$test_data_path" ]]; then
    echo "Error: Test file not found: $test_data_path"
    exit 1
fi

# check index file
if [[ ! -f "$index_path" ]]; then
    echo "Error: Index file not found: $index_path"
    exit 1
fi

# check model
if [[ ! -d "$exp_name" ]]; then
    echo "Error: Model dir not found: $exp_name"
    exit 1
fi

# clean temp dir to avoid stale splits or partial results
rm -rf "$temp_dir"
mkdir -p "$temp_dir"
mkdir -p "$output_dir"

# single GPU mode
if [[ -z "$cuda_list" ]] || [[ "$cuda_list" == "0" ]]; then
    echo "Single GPU evaluation..."
    CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
        --model_name_or_path "$exp_name" \
        --test_data_path "$test_data_path" \
        --result_json_path "$temp_dir/result.json" \
        --index_path "$index_path" \
        --batch_size $batch_size \
        --max_new_tokens $max_new_tokens \
        --num_beams $num_beams \
        --temperature $temperature \
        --do_sample $do_sample \
        --length_penalty $length_penalty \
        --compute_metrics_flag True

    if [[ -f "$temp_dir/result.json" ]]; then
        cp "$temp_dir/result.json" "$output_dir/final_result.json"
        echo "Results saved to: $output_dir/final_result.json"
    fi
    if [[ -f "$temp_dir/metrics.json" ]]; then
        cp "$temp_dir/metrics.json" "$output_dir/metrics.json"
        cp "$temp_dir/metrics.csv" "$output_dir/metrics.csv"
        echo "Metrics saved to: $output_dir/metrics.json, $output_dir/metrics.csv"
    fi
    rm -rf "$temp_dir"
    echo "Evaluation done!"
    exit 0
fi

# multi-GPU mode: requires split_json.py and merge_json.py
echo "Multi-GPU parallel evaluation..."
echo "GPUs: $cuda_list"

# check split_json.py
if [[ ! -f "split_json.py" ]]; then
    echo "Warning: split_json.py not found, falling back to single GPU"
    CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
        --model_name_or_path "$exp_name" \
        --test_data_path "$test_data_path" \
        --result_json_path "$temp_dir/result.json" \
        --index_path "$index_path" \
        --batch_size $batch_size \
        --max_new_tokens $max_new_tokens \
        --num_beams $num_beams \
        --temperature $temperature \
        --do_sample $do_sample \
        --length_penalty $length_penalty \
        --compute_metrics_flag True
    cp "$temp_dir/result.json" "$output_dir/final_result.json"
    if [[ -f "$temp_dir/metrics.json" ]]; then
        cp "$temp_dir/metrics.json" "$output_dir/metrics.json"
        cp "$temp_dir/metrics.csv" "$output_dir/metrics.csv"
        echo "Metrics saved to: $output_dir/metrics.json, $output_dir/metrics.csv"
    fi
    rm -rf "$temp_dir"
    echo "Evaluation done!"
    exit 0
fi

# split data
echo "Splitting test data..."
python split_json.py --input_path "$test_data_path" --output_path "$temp_dir" --cuda_list "$cuda_list"

# parallel evaluation
for i in $cuda_list; do
    if [[ -f "$temp_dir/${i}.json" ]]; then
        echo "Starting evaluation on GPU $i..."
        CUDA_VISIBLE_DEVICES=$i python -u evaluate.py \
            --model_name_or_path "$exp_name" \
            --test_data_path "$temp_dir/${i}.json" \
            --result_json_path "$temp_dir/${i}_result.json" \
            --index_path "$index_path" \
            --batch_size $batch_size \
            --max_new_tokens $max_new_tokens \
            --num_beams $num_beams \
            --temperature $temperature \
            --do_sample $do_sample \
            --length_penalty $length_penalty \
            --compute_metrics_flag False &
    fi
done

echo "Waiting for all evaluation processes..."
wait

# merge results (metrics computed later by evaluate.py --metrics_only)
echo "Merging results..."
python merge_json.py --input_path "$temp_dir" --output_path "$output_dir/final_result.json" --cuda_list "$cuda_list" --compute_metrics False

if [[ -f "$output_dir/final_result.json" ]]; then
    echo "Results saved to: $output_dir/final_result.json"
else
    echo "Error: Merge failed"
    exit 1
fi

# compute metrics on merged results
python -u evaluate.py \
    --result_json_path "$output_dir/final_result.json" \
    --metrics_only True

rm -rf "$temp_dir"
echo "Evaluation done!"
