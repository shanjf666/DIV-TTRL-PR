#!/bin/bash
"""
bash examples/labelfree/update_first.sh --backbone /root/autodl-tmp/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base --clip-high --ent 0.003
python /root/autodl-tmp/DIV-TTRL/verl/scripts/model_merger.py \
    --backend fsdp \
    --local_dir /root/autodl-tmp/model/TTRL-MATH500/MATH-TTT-Qwen3-4B-Base/diversity-RL-Ent0.000/220024/global_step_45/actor \
    --hf_model_path /root/autodl-tmp/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base \
    --target_dir /root/autodl-tmp/model/math_step_45_adaptive_passk
"""
export WANDB_ENTITY=2691454060-ucla
export USE_API_SELF_VERIFY=0
# === TTRL Training Script ===
# Usage: ./evol_rl_no_embedding.sh [--task TASK] [--backbone BACKBONE] [--clip-high] [--temp TEMP]
#
# Parameters:
#   --task      Task name (default: AIME-TTT)
#               Options: AIME-TTT, MATH-TTT, AIME25, etc.
#   --backbone  Backbone model (default: Qwen3-4B-Base)
#               Options: Qwen3-4B-Base, Qwen3-8B-Base
#   --clip-high[=VAL]
#               Clip ratio control:
#               - Not specified: use default 0.2
#               - Specified without value: use 0.28
#               - Specified with value: use that value
#   --ent       Entropy regularization coefficient (float), e.g. 0.000/0.001/0.003 (default: 0.000)
#   --temp      Temperature parameter (default: 1.0)
#               Controls generation randomness, typically range 0.1-2.0
#   -h, --help  Show help information
#
# Examples:
#   ./evol_rl_no_embedding.sh                                    # Use default parameters
#   ./evol_rl_no_embedding.sh --task MATH                   # Specify task
#   ./evol_rl_no_embedding.sh --task AIME --backbone Qwen3-4B-Base  # Specify task and model
#   ./evol_rl_no_embedding.sh --clip-high                       # High clip ratio mode
#   ./evol_rl_no_embedding.sh --temp 0.8                        # Set temperature parameter
#
# =======================

#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
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
            # Read entropy regularization coefficient; use default if no value provided
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
        -h|--help)
            echo "Usage: $0 [--task TASK] [--backbone BACKBONE] [--clip-high[=VAL]] [--ent COEFF] [--temp TEMP]"
            echo "  --task      Task name (default: AIME)"
            echo "  --backbone  Backbone model (default: Qwen3-4B-Base)"
            echo "  --clip-high[=VAL] set clip ratio: not specified=0.2; flag only=0.28; with value use that value"
            echo "  --ent       Entropy regularization coefficient (float), e.g. 0.000/0.001/0.005 (default: 0.000)"
            echo "  --temp      Temperature parameter (float), e.g. 0.6/0.8/1.0 (default: 1.0)"
            echo "  -h, --help  Show help information"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Set default values
TASK=${TASK:-"MATH"}
BACKBONE=${BACKBONE:-"Qwen3-4B-Base"}
CLIP_HIGH=${CLIP_HIGH:-"false"}
CLIP_SPECIFIED=${CLIP_SPECIFIED:-"false"}
CLIP_VALUE=${CLIP_VALUE:-""}
CLIP_MODE=${CLIP_MODE:-""}
ENT=${ENT:-"0.000"}
TEMP=${TEMP:-"1.0"}

## ENT as numerical coefficient, no longer perform boolean normalization



# Set entropy coefficient (numerical) based on --ent
ENTROPY_COEFF=$ENT
RAW_TASK="$TASK"
if [ "$RAW_TASK" = "math_train" ]; then
  TASK="MATH-TTT"
else
  TASK="$TASK-TTT"
fi

pkill -f "python.*main_ppo" || true
pkill -f "python.*main_dapo" || true
pkill -f "multiprocessing.spawn" || true
pkill -f "test_three_datasets.sh" || true
pkill -f "python.*scripts.model_merger" || true
ray stop --force 2>/dev/null || true
sleep 2
echo "========================="

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
# TIME_TAG=182434

ADVANTAGE="pass_grpo_penalized"

echo "=== Basic Configuration Information ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE"
echo "Advantage estimator: $ADVANTAGE"
echo "====================================="

# Set K value
K=4
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * $K))
# Pre-calculate required values to avoid type errors - use arithmetic expansion to ensure numerical type
MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKEN_LEN2=$((MAX_TOKEN_LEN * 2))
if [ "$K" -gt 13 ]; then
  N=4
else
  N=16
fi

# Set EPISODE
EPISODE=4
DATA_TRAIN_BATCH_SIZE=16
N_VOTES_PER_PROMPT=32 # Reduce candidates to balance computational overhead
N_SAMPLES_PER_PROMPT=32 # Keep training sample count
MINI_BATCH_SIZE=1 # Actual mini batch size is MINI_BATCH_SIZE * N_SAMPLES_PER_PROMPT - increase mini batch
MICRO_BATCH_SIZE=2 # Increase micro batch to better utilize GPU

DATA_LOCAL_DIR="data"
# Parse backbone model path and safe name (avoid directory names containing slashes)
CHAT_TEMPLATE=""
if [[ "$BACKBONE" == *"/"* ]]; then
  BACKBONE_PATH="$BACKBONE"
  BACKBONE_NAME="${BACKBONE##*/}"
else
  BACKBONE_PATH="/root/autodl-tmp/model/${BACKBONE}"
  BACKBONE_NAME="$BACKBONE"
fi

echo "Parsed model path: $BACKBONE_PATH"
echo "Parsed model name: $BACKBONE_NAME"

MODEL="${TASK}-${BACKBONE_NAME}"

EXPERIMENT="diversity-RL"

# Set clip_ratio_high value and experiment name suffix
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

# if RAW_TASK is math_train, use our preprocessed parquet; else follow original logic
if [ "$RAW_TASK" = "math_train" ]; then
  TRAIN_FILES="math_train_ttrl.parquet"
else
  if [[ "$TASK" == *"AIME"* ]]; then
    TRAIN_FILES="train-simplerl-16.parquet"
  else
    TRAIN_FILES="train-simplerl.parquet"
  fi
fi

# Set WANDB_PROJECT based on TASK
if [ "$RAW_TASK" = "math_train" ]; then
  WANDB_PROJECT="TTRL_MATH_TRAIN"
  EXPERIMENT="${EXPERIMENT}-MATH_TRAIN"
elif [ "$TASK" = "AIME-TTT" ]; then
  WANDB_PROJECT="TTRL-AIME24"
elif [ "$TASK" = "AMC-TTT" ]; then
  WANDB_PROJECT="TTRL-AMC"
else
  WANDB_PROJECT="TTRL-MATH500"
fi


if [ "$CLIP_HIGH" = "true" ]; then
  EXPERIMENT="${EXPERIMENT}-ClipHigh"
fi

# Always include specific entropy coefficient in experiment name for ablation tracking
EXPERIMENT="${EXPERIMENT}-Ent${ENTROPY_COEFF}"


LOG_NAME="${EXPERIMENT}-${MODEL}"
OUTPUT_DIR="/root/autodl-tmp/model/${WANDB_PROJECT}/${MODEL}/${EXPERIMENT}/${TIME_TAG}"



echo "=== TTRL Training Configuration ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE"
echo "Advantage estimator: $ADVANTAGE"
# Print entropy regularization switch (based on whether coefficient is 0)
if [[ "$ENTROPY_COEFF" != "0" && "$ENTROPY_COEFF" != "0.0" && "$ENTROPY_COEFF" != "0.00" && "$ENTROPY_COEFF" != "0.000" ]]; then
  ENT_ENABLED="true"
else
  ENT_ENABLED="false"
fi
echo "Enable entropy regularization: $ENT_ENABLED"
echo "Entropy coefficient: $ENTROPY_COEFF"
echo "Output directory: $OUTPUT_DIR"
echo "Experiment name: $LOG_NAME"
echo "==============================="

  # reward_model.reward_manager=diversity_ttrl \
  # reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  # reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  # reward_model.reward_kwargs.mode="train" \

# # ------------------------------------------------------------
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train-simplerl.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test-simplerl.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
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
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_TOKEN_LEN2)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.top_p=0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_TOKEN_LEN)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_TOKEN_LEN2)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  +two_stage_verify=True \
  +two_stage_n=-1 \
  +two_stage_max_candidates=5 \
  algorithm.k=4 \
  +algorithm.lam_div=0.05 \
  +algorithm.c_max=2 \
  +algorithm.div_sc_threshold=0.5 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=15 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "=== Training Completed ==="
echo "Output directory: $OUTPUT_DIR"
echo "Project name: $WANDB_PROJECT"
echo "Experiment name: $LOG_NAME"
# echo "========================"