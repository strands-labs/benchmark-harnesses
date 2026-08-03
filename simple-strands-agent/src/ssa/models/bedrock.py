from botocore.exceptions import ClientError, ReadTimeoutError, EventStreamError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import ProtocolError
import logging
from collections.abc import AsyncGenerator
from typing import Callable, Optional, Any
import traceback

from strands.models.bedrock import BedrockModel
from strands.types.content import Messages, ContentBlock, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec, ToolChoice
from strands.types.exceptions import ModelThrottledException


LOG = logging.getLogger(__name__)

BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "model is getting throttled",
    "read timed out"
]
BEDROCK_SERVER_ERROR_MESSAGES = [
    "system encountered an unexpected error during processing",
]


class SRBedrockModel(BedrockModel):
    """
    Handle exceptions in addition to already present in the bedrock model
    """

    def _format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Override to inject TTL into system and toolConfig cachePoints when messages use 1h TTL."""
        request = super()._format_request(messages, tool_specs, system_prompt_content, tool_choice)

        # Check if any message cachePoint has a "1h" TTL
        has_1h = any(
            block.get("cachePoint", {}).get("ttl") == "1h"
            for msg in messages
            for block in msg.get("content", [])
            if isinstance(block, dict)
        )

        if has_1h:
            for block in request.get("system", []):
                if "cachePoint" in block:
                    block["cachePoint"]["ttl"] = "1h"
            for tool in request.get("toolConfig", {}).get("tools", []):
                if "cachePoint" in tool:
                    tool["cachePoint"]["ttl"] = "1h"

        return request

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any] | None:
        """Override to preserve TTL in cachePoint blocks."""
        result = super()._format_request_message_content(content)
        if result and "cachePoint" in result and "cachePoint" in content and "ttl" in content["cachePoint"]:
            result["cachePoint"]["ttl"] = content["cachePoint"]["ttl"]
        return result

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Override to estimate reasoning tokens from output tokens and output text length."""
        output_chars = 0
        has_reasoning = False

        async for event in super().stream(
            messages, tool_specs, system_prompt,
            tool_choice=tool_choice, system_prompt_content=system_prompt_content, **kwargs
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                # Track if reasoning content appears in the stream
                if "reasoningContent" in delta:
                    has_reasoning = True
                # Accumulate visible output chars (text + tool use input)
                elif "text" in delta:
                    output_chars += len(delta["text"])
                elif "toolUse" in delta:
                    output_chars += len(delta["toolUse"].get("input", ""))

            # Only estimate reasoning tokens if reasoning content was present
            if "metadata" in event and has_reasoning:
                usage = event["metadata"].get("usage", {})
                output_tokens = usage.get("outputTokens", 0)
                estimated_visible_tokens = output_chars // 3
                reasoning_tokens = max(0, output_tokens - estimated_visible_tokens)
                if reasoning_tokens > 0:
                    event["metadata"]["usage"]["reasoningTokens"] = reasoning_tokens

            yield event

    def _stream(
        self,
        callback: Callable[..., None],
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
    ) -> None:
        try:
            super()._stream(callback, messages, tool_specs, system_prompt, tool_choice)
        except ClientError as e:
            LOG.error(traceback.format_exc())
            error_message = str(e)

            if any(throttle_message in error_message.lower() for throttle_message in BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES):
                LOG.warning("bedrock threw throttling/timeout error")
                raise ModelThrottledException(error_message) from e

            if any(server_err_msg in error_message.lower() for server_err_msg in BEDROCK_SERVER_ERROR_MESSAGES):
                LOG.warning("bedrock threw system stream error")
                raise ModelThrottledException(error_message) from e
        except (
            ReadTimeoutError,
            ProtocolError,
            URLLib3ReadTimeoutError,
            EventStreamError,
            ) as e:
            error_message = str(e)
            raise ModelThrottledException(error_message) from e
