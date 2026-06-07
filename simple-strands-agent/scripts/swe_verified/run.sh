#!/bin/bash

export AWS_PROFILE="default"
CONFIG_NAME="sbv_bedrock_opus_46_think_adaptive_effort_max.yaml"
DATASET="sbv"
INSTANCE_ID="pylint-dev__pylint-8898"

uv run python -m ssa.run \
    --config-name=$CONFIG_NAME \
    dataset.name=$DATASET \
    dataset.identifier=$INSTANCE_ID \
    env.env_type=docker 
