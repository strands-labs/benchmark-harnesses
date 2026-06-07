#!/bin/bash

export AWS_PROFILE="default"
# Disable aiohttp transport in litellm — aiohttp sessions are bound to a single
# event loop, which breaks when multiple trials each run asyncio.run() in their
# own thread.  httpx's native transport doesn't have this limitation.
export DISABLE_AIOHTTP_TRANSPORT=True

TASK_ID="break-filter-js-from-html"
export CONFIG_NAME="tb2_bedrock_opus_46_think_adaptive_effort_max"
export DATE_TAG=$(date "+%y-%m-%d-%H-%M-%S")

uv run harbor run \
    --agent-import-path "ssa.utils.harbor_plugin.ssa_native:SSANative" \
    --environment-import-path "ssa.utils.harbor_plugin.ssa_docker:SSADockerEnvironment" \
    --ak "config_name=${CONFIG_NAME}"  \
    --dataset terminal-bench@2.0 \
    --jobs-dir ~/harbor_logs/ssa_harbor \
    --job-name $DATE_TAG \
    -n 1 \
    -t $TASK_ID