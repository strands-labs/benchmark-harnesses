"""Tests for StrandsResolverAgent._execute_event_loop_cycle retry behavior."""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException

from ssa.agent import StrandsResolverAgent


def _make_agent(messages: list | None = None) -> StrandsResolverAgent:
    """Bypass Agent.__init__ to build the minimum surface that
    _execute_event_loop_cycle touches."""
    agent = StrandsResolverAgent.__new__(StrandsResolverAgent)
    agent.messages = list(messages or [])
    agent._session_manager = None
    agent.tool_registry = MagicMock()
    agent.hooks = MagicMock()
    agent.conversation_manager = MagicMock()
    return agent


async def _drain(gen: AsyncGenerator) -> list:
    return [event async for event in gen]


@pytest.mark.asyncio
async def test_agent_context_overflow_retries_with_reduce(monkeypatch):
    """ContextWindowOverflowException triggers reduce_context(from_overflow=True)
    and a fresh event loop cycle."""

    call_count = {"n": 0}

    async def fake_cycle(*, agent, invocation_state, structured_output_context=None, limits=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ContextWindowOverflowException("context too big")
        # Second call succeeds with a single sentinel event.
        yield "final-event"

    monkeypatch.setattr("ssa.agent.event_loop_cycle", fake_cycle)

    agent = _make_agent(messages=[{"role": "user", "content": [{"text": "hi"}]}])

    events = await _drain(agent._execute_event_loop_cycle({}))

    # Cycle was invoked twice.
    assert call_count["n"] == 2
    # Reduce was called with from_overflow=True.
    agent.conversation_manager.reduce_context.assert_called_once()
    _, kwargs = agent.conversation_manager.reduce_context.call_args
    assert kwargs.get("from_overflow") is True
    # The final event from the retry reached the caller.
    assert events == ["final-event"]
    # AgentCompletedEvent fired via hooks.invoke_callbacks (in finally).
    assert agent.hooks.invoke_callbacks.called


@pytest.mark.asyncio
async def test_agent_max_tokens_retries_with_last_message_popped(monkeypatch):
    """MaxTokensReachedException drops the last message and retries."""

    call_count = {"n": 0}

    async def fake_cycle(*, agent, invocation_state, structured_output_context=None, limits=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise MaxTokensReachedException("max tokens")
        yield "final"

    monkeypatch.setattr("ssa.agent.event_loop_cycle", fake_cycle)

    agent = _make_agent(
        messages=[
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "truncated"}]},
        ]
    )

    events = await _drain(agent._execute_event_loop_cycle({}))

    assert call_count["n"] == 2
    # Last message popped between calls.
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "user"
    assert events == ["final"]
