set -e

export DATA_DIR="data/Amazon18"
export CATEGORY="Industrial_and_Scientific"
# output directory (sft/, rl/, new_tokens.json); default data/<category>
export OUTPUT_DIR="data/${CATEGORY}"

export TASK4_SAMPLE=10000 # sample all if -1
export SEED=42

python preprocess_data_sft_rl.py \
    --data_dir $DATA_DIR \
    --category $CATEGORY \
    --output_dir $OUTPUT_DIR \
    --seq_sample $TASK4_SAMPLE \
    --seed $SEED \
    --data_source $CATEGORY
