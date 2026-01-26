#!/bin/bash

export PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA  # change to your own path
export OBJAVERSE_HOUSES_DIR=/root/data/objaverse_houses  # change to your own path
export OBJAVERSE_DATA_DIR=/root/data/objaverse_assets  # change to your own path
export HF_HOME=/root/huggingface
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Default values
task_type=""
il_ckpt_path=""
resume_checkpoint=""
wandb_project=""
wandb_entity=""
num_train_processes=32
output_dir=""
dataset_dir=""
cost_limit=2.31
max_stage_steps=""
tag=""
max_houses=""
grpo_num_generations=4
max_steps=""
step_penalty=""

# Function to print usage
print_usage() {
    echo "Usage: $0 --task_type <type> --il_ckpt_path <path> --output_dir <path> --dataset_dir <path> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --task_type           Task type: objectnav | pickup | fetch"
    echo "  --il_ckpt_path        Path to initial pretrained model"
    echo "  --output_dir          Output directory for checkpoints"
    echo "  --dataset_dir         Dataset directory"
    echo ""
    echo "Optional arguments:"
    echo "  --checkpoint          Path to checkpoint for resuming training"
    echo "  --wandb_project       WandB project name"
    echo "  --wandb_entity        WandB entity name"
    echo "  --num_train_processes Number of training processes (default: 32)"
    echo "  --cost_limit          Cost limit (default: 2.31)"
    echo "  --max_stage_steps     Total training steps across pipeline stages (default: use config)"
    echo "  --max_steps           Max steps per episode (default: use config)"
    echo "  --tag                 Experiment tag override (default: derived from task_type)"
    echo "  --max_houses          Max houses to sample (default: use config)"
    echo "  --grpo_num_generations Number of GRPO generations (default: use config)"
    echo "  --step_penalty        Step penalty reward (default: use config)"
    echo "  --help                Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_type)
            task_type="$2"
            shift 2
            ;;
        --il_ckpt_path)
            il_ckpt_path="$2"
            shift 2
            ;;
        --checkpoint)
            resume_checkpoint="$2"
            shift 2
            ;;
        --wandb_project)
            wandb_project="$2"
            shift 2
            ;;
        --wandb_entity)
            wandb_entity="$2"
            shift 2
            ;;
        --num_train_processes)
            num_train_processes="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --dataset_dir)
            dataset_dir="$2"
            shift 2
            ;;
        --cost_limit)
            cost_limit="$2"
            shift 2
            ;;
        --max_stage_steps)
            max_stage_steps="$2"
            shift 2
            ;;
        --tag)
            tag="$2"
            shift 2
            ;;
        --max_steps)
            max_steps="$2"
            shift 2
            ;;
        --max_houses)
            max_houses="$2"
            shift 2
            ;;
        --grpo_num_generations)
            grpo_num_generations="$2"
            shift 2
            ;;
        --step_penalty)
            step_penalty="$2"
            shift 2
            ;;
        --help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Check required arguments
if [ -z "$task_type" ] || [ -z "$il_ckpt_path" ] || [ -z "$output_dir" ] || [ -z "$dataset_dir" ]; then
    echo "Error: Missing required arguments"
    echo ""
    print_usage
fi

# Convert task type to internal format
if [ "$task_type" == "objectnav" ]; then
    task_type_internal="ObjectNavType"
elif [ "$task_type" == "pickup" ]; then
    task_type_internal="PickupType"
elif [ "$task_type" == "fetch" ]; then
    task_type_internal="FetchType"
else
    echo "Error: Invalid task type '$task_type'"
    echo "Valid options: objectnav, pickup, fetch"
    exit 1
fi

if [ -z "$tag" ]; then
    tag="$task_type_internal"
fi

# Auto-detect latest checkpoint if not provided
if [ -z "$resume_checkpoint" ]; then
    base_exp_dir=""
    for candidate in "$output_dir/$tag" "$output_dir/checkpoints/$tag"; do
        if [ -d "$candidate" ]; then
            base_exp_dir="$candidate"
            break
        fi
    done

    if [ -n "$base_exp_dir" ]; then
        latest_ckpt=""
        while IFS= read -r -d '' dir; do
            ckpt_candidate="$(find "$dir" -maxdepth 1 -type f -name "exp*.pt" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
            if [ -n "$ckpt_candidate" ]; then
                latest_ckpt="$ckpt_candidate"
                break
            fi
        done < <(find "$base_exp_dir" -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\0" 2>/dev/null | sort -zr | cut -z -d' ' -f2-)

        if [ -n "$latest_ckpt" ]; then
            resume_checkpoint="$latest_ckpt"
            echo -e "\nAuto-resuming from latest checkpoint: $resume_checkpoint\n"
        else
            echo -e "\nNo checkpoints found under $base_exp_dir\n"
        fi
    fi
fi

# Build the base command
cmd="python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path $il_ckpt_path \
    --num_train_processes $num_train_processes \
    --output_dir $output_dir \
    --dataset_dir $dataset_dir/$task_type_internal \
    --cost_limit $cost_limit \
    --tag $tag \
    --use_grpo True \
    --grpo_num_generations ${grpo_num_generations} \
    --seed 42 \
    --shaping_weight 1.0 \
    --step_penalty -0.01"
    # --enable_lagrange True"
    # --shaping_weight 0.1 \

# Add checkpoint parameter if provided
if [ -n "$resume_checkpoint" ]; then
    cmd="$cmd --checkpoint $resume_checkpoint"
fi

# Get wandb project and entity from environment variables if not provided
if [ -z "$wandb_project" ] && [ "$WANDB_DISABLED" != "True" ]; then
    wandb_project="${WANDB_PROJECT:-}"
fi
if [ -z "$wandb_entity" ] && [ "$WANDB_DISABLED" != "True" ]; then
    wandb_entity="${WANDB_ENTITY:-}"
fi

# Add wandb parameters if provided
if [ -n "$wandb_project" ] && [ -n "$wandb_entity" ]; then
    cmd="$cmd \
    --wandb_project \"$wandb_project\" \
    --wandb_entity \"$wandb_entity\" \
    --callbacks \"wandb_logging_callback\""
fi

# Add max_stage_steps if provided
if [ -n "$max_stage_steps" ]; then
    cmd="$cmd --max_stage_steps $max_stage_steps"
fi

# Add max_steps if provided
if [ -n "$max_steps" ]; then
    cmd="$cmd --max_steps $max_steps"
fi

# Add step_penalty if provided
if [ -n "$step_penalty" ]; then
    cmd="$cmd --step_penalty $step_penalty"
fi

# Add max_houses if provided
if [ -n "$max_houses" ]; then
    cmd="$cmd --max_houses $max_houses"
fi

# Execute the command
echo "Executing command:"
echo "$cmd"
echo ""
eval $cmd


# TODO: params to add
# --test_expert

# https://github.com/rdesc/allenact/blob/5c7db0c8b9b425881064768f3f6eb60ad96ecbc4/allenact/main.py
