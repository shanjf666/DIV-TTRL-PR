#!/bin/bash
"""
CUDA_VISIBLE_DEVICES=2,3,4,5 bash examples/labelfree/aime_explore_train.sh --task AIME --backbone /data/home/jianfeng/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base 2>&1 | tee train_$(date +%Y%m%d_%H%M).log
python /data/home/jianfeng/DIV-TTRL-PR/verl/scripts/model_merger.py \
    --backend fsdp \
    --local_dir /data/home/jianfeng/model/TTRL-AIME24/AIME-TTT-Qwen3-4B-Base/diversity-RL-Ent0.000/230602/global_step_45/actor \
    --hf_model_path /data/home/jianfeng/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base \
    --target_dir /data/home/jianfeng/model/aime_step_45_passk
"""
# AIME-TTT Exploration Training Script
# Mirroring logic from examples/labelfree/aime25_train.sh

# Usage:
#     bash examples/labelfree/aime_explore_train.sh --backbone /path/to/model --clip-high --ent 0.003

export WANDB_ENTITY=2691454060-ucla

# API Self Verification configuration
export USE_API_SELF_VERIFY=0
export AUTODL_API_KEY="EMPTY"
export AUTODL_MODEL="qwen3-4b-base"
export AUTODL_BASE_URL="https://u630113-8ba4-8da84932.westc.seetacloud.com:8443/v1"
export API_VERIFY_TOP_K=5
export API_VERIFY_SC_THRESHOLD=0.3
export API_VERIFY_MAX_WORKERS=8

# === Parse command line arguments ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --clip-high)
            CLIP_HIGH="true"
            CLIP_SPECIFIED="true"
            CLIP_MODE="high"
            if [[ -n "$2" && "$2" != --* ]]; then
              CLIP_VALUE="$2"
              shift 2
            else
              shift 1
            fi
            ;;
        --clip-high=*)
            CLIP_HIGH="true"
            CLIP_SPECIFIED="true"
            CLIP_MODE="high"
            CLIP_VALUE="${1#--clip-high=}"
            shift 1
            ;;
        --ent)
            if [[ -z "$2" || "$2" == --* ]]; then
              shift 1
            else
              ENT="$2"
              shift 2
            fi
            ;;
        --ent=*)
            ENT="${1#--ent=}"
            shift 1
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --temp=*)
            TEMP="${1#--temp=}"
            shift 1
            ;;
        --pseudo_label_file)
            PSEUDO_LABEL_FILE="$2"
            shift 2
            ;;
        --explore_threshold)
            EXPLORE_THRESHOLD="$2"
            shift 2
            ;;
        --enable_hybrid)
            ENABLE_HYBRID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--backbone BACKBONE] [--clip-high[=VAL]] [--ent COEFF] [--temp TEMP]"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# === Set default values ===
TASK="AIME"
BACKBONE=${BACKBONE:-"Qwen3-4B-Base"}
CLIP_HIGH=${CLIP_HIGH:-"true"}
CLIP_SPECIFIED=${CLIP_SPECIFIED:-"false"}
CLIP_VALUE=${CLIP_VALUE:-""}
CLIP_MODE=${CLIP_MODE:-""}
ENT=${ENT:-"0.000"}
TEMP=${TEMP:-"1.0"}
ENTROPY_COEFF=$ENT

PSEUDO_LABEL_FILE=${PSEUDO_LABEL_FILE:-""}
EXPLORE_THRESHOLD=${EXPLORE_THRESHOLD:-"0.3"}
ENABLE_HYBRID=${ENABLE_HYBRID:-"false"}

# Clean up any existing processes
pkill -f "python.*main_explore_ppo" || true
ray stop --force 2>/dev/null || true
sleep 2

# === Configuration ===
echo "=== AIME-TTT Exploration Training Configuration ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE"

# Set K value and sequence length
K=4
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * $K))
MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKEN_LEN2=$((MAX_TOKEN_LEN * 2))

# Training parameters
EPISODE=40
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=64
N_SAMPLES_PER_PROMPT=32
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2

DATA_LOCAL_DIR="data"

# Parse backbone model path
if [[ "$BACKBONE" == *"/"* ]]; then
  BACKBONE_PATH="$BACKBONE"
  BACKBONE_NAME="${BACKBONE##*/}"
else
  BACKBONE_PATH="/root/autodl-tmp/model/${BACKBONE}"
  BACKBONE_NAME="$BACKBONE"
fi

MODEL="${TASK}-${BACKBONE_NAME}"
EXPERIMENT="explore-hybrid-RL-AIME"

# Set clip_ratio_high
if [ "$CLIP_SPECIFIED" = "true" ]; then
  if [ -n "$CLIP_VALUE" ]; then
    CLIP_RATIO_HIGH=$CLIP_VALUE
  else
    CLIP_RATIO_HIGH=0.28
  fi
  if [ "$CLIP_HIGH" = "true" ]; then
    EXPERIMENT="${EXPERIMENT}-ClipHigh"
  fi
else
  CLIP_RATIO_HIGH=0.2
fi

# Data files
TRAIN_FILES="train-simplerl.parquet"
TEST_FILES="test-simplerl.parquet"

# WandB configuration
# Set WANDB_PROJECT based on TASK
if [ "$RAW_TASK" = "math_train" ]; then
  WANDB_PROJECT="TTRL_MATH_TRAIN"
  EXPERIMENT="${EXPERIMENT}-MATH_TRAIN"
elif [ "$TASK" = "AIME-TTT" ]; then
  WANDB_PROJECT="TTRL-AIME24"
else
  WANDB_PROJECT="TTRL-MATH500"
fi
# EXPERIMENT="${EXPERIMENT}-Ent${ENTROPY_COEFF}"
LOG_NAME="${EXPERIMENT}-${MODEL}"
OUTPUT_DIR="/root/autodl-tmp/model/${WANDB_PROJECT}/${MODEL}/${EXPERIMENT}/${TIME_TAG}"

# === Run Training ===
python -m verl.trainer.main_explore_ppo \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  +reward_model.reward_kwargs.enable_hybrid=$ENABLE_HYBRID \
  +reward_model.reward_kwargs.pseudo_label_file="$PSEUDO_LABEL_FILE" \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/$TRAIN_FILES"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/$TEST_FILES"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_TOKEN_LEN2)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.max_model_len=$((MAX_TOKEN_LEN)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_TOKEN_LEN2)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=pass_grpo \
  algorithm.diversity_density_k=4 \
  +algorithm.use_explore_rollout=True \
  +algorithm.explore_threshold=$EXPLORE_THRESHOLD \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=15 \
  trainer.test_freq=2 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "=== Training Completed ==="
