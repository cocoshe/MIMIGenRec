set -e

export CATEGORY="Industrial_and_Scientific"

# export MODEL_PATH="saves/qwen2.5-1.5b/full/${CATEGORY}-sft-dsz2"
export MODEL_PATH="saves/qwen2.5-0.5b/full/${CATEGORY}-sft-dsz0"
export DATA_DIR="data/${CATEGORY}/rl"
export INDEX_PATH="data/${CATEGORY}/id2sid.json"
# export OUTPUT_DIR="rl_outputs/${CATEGORY}-qwen2.5-1.5b-instruct-grpo"
export OUTPUT_DIR="rl_outputs/${CATEGORY}-qwen2.5-0.5b-instruct-grpo"

# wandb
export WANDB_PROJECT="MIMIGenRec-GRPO"
# export WANDB_RUN_NAME="${CATEGORY}-qwen2.5-1.5b-instruct"
export WANDB_RUN_NAME="${CATEGORY}-qwen2.5-0.5b-instruct"
export WANDB_MODE=offline  # set to "online" to use wandb
export WANDB_API_KEY=""
export REPORT_TO="wandb"

export SAVE_TOTAL_LIMIT=1

# deepspeed
NUM_PROCESSES=8
MAIN_PORT=29503
ds_config=config/zero2.yaml

# rollout
export NUM_BEAMS=16

accelerate launch \
  --config_file $ds_config \
  --num_processes $NUM_PROCESSES \
  --main_process_port $MAIN_PORT \
  trl_trainer.py \
  --model "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --index_path "$INDEX_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_beams $NUM_BEAMS \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 2 \
  --learning_rate 1e-5 \
  --eval_step 20 \
  --max_completion_length 128 \
  --beta 1e-3 \
  --temperature 1.0 \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --report_to $REPORT_TO
