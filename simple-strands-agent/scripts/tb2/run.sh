#!/bin/bash

export AWS_PROFILE="default"
CONFIG_NAME="tb2_bedrock_opus_46_think_adaptive_effort_max.yaml"
DATASET="tb2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$(cd "$SCRIPT_DIR/../../resources" && pwd)"
export TB2_ECR_MAP="$RESOURCES_DIR/tb2_docker_public.json"
echo "TB2_ECR_MAP=$TB2_ECR_MAP"
if [ ! -f "$TB2_ECR_MAP" ]; then
    echo "ERROR: TB2_ECR_MAP not found at $TB2_ECR_MAP"
    exit 1
fi
echo "  -> Found"

export TB2_INSTRUCTIONS_MAP="$RESOURCES_DIR/tb2_instruction_map.json"
echo "TB2_INSTRUCTIONS_MAP=$TB2_INSTRUCTIONS_MAP"
if [ ! -f "$TB2_INSTRUCTIONS_MAP" ]; then
    echo "ERROR: TB2_INSTRUCTIONS_MAP not found at $TB2_INSTRUCTIONS_MAP"
    exit 1
fi
echo "  -> Found"

TB2_REPO_DIR="$SCRIPT_DIR/terminal-bench-2"
if [ -d "$TB2_REPO_DIR" ]; then
    echo "TB2 repo already exists at $TB2_REPO_DIR"
else
    echo "Cloning terminal-bench-2 into $TB2_REPO_DIR..."
    git clone https://github.com/harbor-framework/terminal-bench-2.git "$TB2_REPO_DIR"
    echo "Clone complete."
fi
export TB2_REPO_PATH="$TB2_REPO_DIR"
INSTANCE_ID="adaptive-rejection-sampler"
# 
uv run python -m ssa.run \
    --config-name=$CONFIG_NAME \
    dataset.name=$DATASET \
    dataset.identifier=$INSTANCE_ID \
    env.env_type=docker