#!/bin/bash

export AWS_PROFILE="default"
CONFIG_NAME="sbp_bedrock_opus_46_think_adaptive_effort_max.yaml"
DATASET="sbpro"
INSTANCE_ID="instance_future-architect__vuls-407407d306e9431d6aa0ab566baa6e44e5ba2904"

uv run python -m ssa.run \
    --config-name=$CONFIG_NAME \
    dataset.name=$DATASET \
    dataset.identifier=$INSTANCE_ID \
    env.env_type=docker