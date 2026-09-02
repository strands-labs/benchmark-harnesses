"""StrandsAgent: a PRO-LONG analyzer backend built on the Strands SDK.

PRO-LONG's two shipped backends shell out to third-party coding-agent CLIs
(`codex`, `claude`) running inside Docker. This backend replaces that with
`strands.Agent` — a ReAct loop driving Claude on Amazon Bedrock — so the whole
system runs on Strands + Bedrock with no external CLI and no container.

Everything else is PRO-LONG's, unchanged:
  * the append-all `logs.txt` and its markers            (environment/runner.py)
  * the ARC-AGI-3 env wrapper                            (environment/arcagi3.py)
  * the ~30-line system prompt and user prompts          (agent/prompts.py)
  * `actions.json` parsing, session state, log windowing  (agent/base.py)

What this file supplies is the analyzer: a system prompt, a user prompt, and an
`analyze()` that runs the loop until the agent has written `actions.json`.

The tool space is deliberately the same six PRO-LONG relies on — their Table 1
shows it spans the entire effect (Read-only 23.1 -> +Python 38.3 -> +Write 41.2),
so it is the one thing that must not be approximated.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

from strands import Agent
from strands.models.bedrock import BedrockModel, CacheConfig

from prolong_agent.agent.base import BaseAgent
from prolong_agent.agent.prompts import (
    ASCII_COLOR_MAP,
    HEX_COLOR_MAP,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_INPROMPT,
    format_actions_block,
)
from prolong_agent.agent.strands_tools import make_prolong_tools

log = logging.getLogger(__name__)

_ACTIONS_FILE = "actions.json"

# Per-lane fallback ladder, used only after a call hangs.
#
# On 0813, 77 analyzer calls hung on a dead Bedrock socket at a median 1013s each.
# A fresh connection is already built per call (_build_model runs inside analyze),
# so the connection is not the problem -- all 5 runner retries re-used the same
# route, producing ~83 minutes of silence per action, which outlived the game
# session (bp35 died in a 51-minute gap).
#
# Every lost level across both full runs (0812, 0813) was a lane being kicked this
# way, never a puzzle the agent could not solve -- so silence is the only failure
# mode that has ever cost score, and shortening it is the only lever that matters.
#
# Order: exhaust opus-5 endpoints only. A weaker model is a real quality cost (the
# 0811 opus-4-7 run scored 46.82% vs 96.87% on opus-5) while a different endpoint
# costs nothing -- but more importantly, deeper rungs are unreachable. At
# read_timeout=600 a hang costs ~680s, and the game session dies after ~51 min of
# silence (bp35, 0813), so only ~4 rungs can be walked before the lane is gone
# anyway. Adding opus-4-8 / opus-4-7 tiers below these would be dead weight: they
# could only ever be reached after the session was already lost.
#
# Within the ladder: configured route first (warm prompt cache), then same region
# with a different router, then another US region (cold cache), then EU last.
#
# All four were probed and answered with our exact config (adaptive thinking +
# output_config.effort). `apac.` is NOT entitled and is excluded.
_FALLBACK_ROUTES: tuple[tuple[str, str], ...] = (
    ("global.anthropic.claude-opus-5", "us-west-2"),
    ("us.anthropic.claude-opus-5", "us-west-2"),
    ("us.anthropic.claude-opus-5", "us-east-1"),
    ("us.anthropic.claude-opus-5", "us-east-2"),
)

# Substrings that mark a call as *hung* rather than merely unsuccessful. Only
# these demote a lane. A throttle has its own backoff and is not a stall, and a
# reply that omits actions.json is a content problem a new route will not fix.
_HANG_MARKERS = (
    "readtimeout",
    "read timed out",
    "connectionerror",
    "connection aborted",
    "connection reset",
    "endpointconnectionerror",
    "incompleteread",
)

# The runner appends a board after every action, so a turn that writes no
# actions.json wastes a model call. Bound the ReAct loop per call.
_MAX_TOOL_ITERATIONS = 60

# Bedrock on-demand list prices, USD per 1M tokens, as (input, output, cache_read,
# cache_write_5m). Confirmed against the AWS Bedrock pricing page 2026-08-10; the
# pricing API is not readable from this role, so these are a maintained constant.
# Override per-run with PROLONG_PRICE_IN / _OUT / _CACHE_READ / _CACHE_WRITE.
_PRICES: dict[str, tuple[float, float, float, float]] = {
    "opus-4-6": (5.50, 27.50, 0.55, 6.875),
    "opus-4-7": (5.50, 27.50, 0.55, 6.875),
    "opus-4-8": (5.50, 27.50, 0.55, 6.875),
    "opus-5": (5.50, 27.50, 0.55, 6.875),
    "fable-5": (11.00, 55.00, 1.10, 13.75),
    "sonnet-5": (2.20, 11.00, 0.22, 2.75),
    "sonnet-4-6": (3.30, 16.50, 0.33, 4.125),
}


def _short_model(model_id: str) -> str:
    """`global.anthropic.claude-opus-4-8` -> `opus-4-8`, for readable logs."""
    return model_id.rsplit("claude-", 1)[-1] if "claude-" in model_id else model_id


def _short_route(route: tuple[str, str]) -> str:
    """`(global.anthropic.claude-opus-5, us-west-2)` -> `opus-5@global/us-west-2`."""
    model_id, region = route
    prefix = model_id.split(".", 1)[0] if "." in model_id else "?"
    return f"{_short_model(model_id)}@{prefix}/{region}"


def _prices_for(model_id: str) -> tuple[float, float, float, float] | None:
    env = [os.environ.get(k) for k in
           ("PROLONG_PRICE_IN", "PROLONG_PRICE_OUT",
            "PROLONG_PRICE_CACHE_READ", "PROLONG_PRICE_CACHE_WRITE")]
    if all(v is not None for v in env):
        return tuple(float(v) for v in env)  # type: ignore[return-value]
    for key, row in _PRICES.items():
        if key in model_id:
            return row
    return None

# Throttle handling for high-concurrency runs (25 games in parallel).
_MAX_THROTTLE_RETRIES = 6
_THROTTLE_BASE_DELAY = 8.0


class StrandsAgent(BaseAgent):
    """PRO-LONG analyzer driven by strands.Agent on Bedrock."""

    BACKEND_ID = "strands"

    def __init__(
        self,
        model_id: str = "global.anthropic.claude-opus-4-7",
        region: str = "us-west-2",
        action_cap: int = 20,
        log_window: int | None = None,
        effort: str = "high",
        max_tokens: int = 32_000,
        grid_mode: str = "hex",
        in_prompt_board: bool = False,
        cross_turn_hint: bool = True,
    ) -> None:
        self._model_id = model_id
        self._region = region
        self._action_cap = int(action_cap)
        # PRO-LONG's convention: None = full history (the default), -1 = board
        # injected in-prompt (the no-log baseline), 0 = current only, N>0 = last N.
        self._log_window = log_window
        self._effort = effort
        self._max_tokens = int(max_tokens)
        self._grid_mode = grid_mode
        self._in_prompt_board = bool(in_prompt_board) or log_window == -1
        self._cross_turn_hint = bool(cross_turn_hint)

        # BaseAgent's session helpers expect these.
        self._session_ids: dict[str, str] = {}
        self._call_count: dict[str, int] = {}

        # Per-lane position on _FALLBACK_MODELS, keyed by log path exactly like
        # _call_count. One StrandsAgent instance is shared by all 25 lanes
        # (swarm.py passes `agent.analyze` as the analyzer hook to every runner),
        # so this MUST be keyed per lane: a hang on one game must not demote the
        # other 24. Each lane only ever touches its own key.
        self._model_level: dict[str, int] = {}

        self._available_actions: list[str] = []
        self._usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    # ----- per-lane route fallback ------------------------------------------

    def _route_for(self, path_key: str) -> tuple[str, str]:
        """The (model_id, region) this lane should use for its next call.

        Level 0 is always the configured model and region, so a run on a different
        backbone is unaffected until it actually hangs.
        """
        level = self._model_level.get(path_key, 0)
        if level <= 0:
            return self._model_id, self._region
        return _FALLBACK_ROUTES[level % len(_FALLBACK_ROUTES)]

    def _demote(self, path_key: str, action_num: int, failed: tuple[str, str]) -> None:
        """Rotate this lane to the next route after a hung call.

        Wraps rather than clamping. Clamping pinned a lane on the last rung for as
        long as the runner's unbounded retry loop ran -- once the ladder had been
        walked, every subsequent attempt re-used the same endpoint and no other was
        ever tried again, even after it recovered. A hang is a transient endpoint
        fault, so the right behaviour is to keep cycling.
        """
        level = self._model_level.get(path_key, 0)
        new_level = (level + 1) % len(_FALLBACK_ROUTES)
        self._model_level[path_key] = new_level
        nxt = self._route_for(path_key)
        wrapped = " (wrapped)" if new_level == 0 else ""
        log.warning(
            "hang at action %d on %s — next attempt for this game uses %s%s",
            action_num, _short_route(failed), _short_route(nxt), wrapped,
        )

    def _promote(self, path_key: str, action_num: int, used: tuple[str, str]) -> None:
        """Reset this lane to the configured route after any successful call."""
        if self._model_level.get(path_key, 0) == 0:
            return
        self._model_level[path_key] = 0
        log.info(
            "recovered at action %d on %s — back to %s",
            action_num, _short_route(used),
            _short_route((self._model_id, self._region)),
        )

    # ----- model -----------------------------------------------------------

    def _build_model(self, model_id: str | None = None,
                     region: str | None = None) -> BedrockModel:
        """Bedrock model with thinking configured for this model family.

        opus-4-7 and newer reject `reasoning_config.type=enabled` and require
        `thinking.type=adaptive` + `output_config.effort`; 4.6 and older are the
        reverse. Getting this wrong fails every call, so dispatch on the id.

        `model_id`/`region` override the configured route for a single call, which
        is how the per-lane fallback ladder re-routes after a hang.
        """
        model_id = model_id or self._model_id
        region = region or self._region
        legacy = ("opus-4-6", "opus-4-5", "sonnet-4-5", "sonnet-4-6", "haiku-4-5")
        wants_effort = not any(t in model_id.lower() for t in legacy)
        extra: dict[str, Any] = (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": self._effort}}
            if wants_effort
            else {"reasoning_config": {"type": "enabled", "budget_tokens": 8192}}
        )
        from botocore.config import Config as BotocoreConfig

        return BedrockModel(
            model_id=model_id,
            region_name=region,
            max_tokens=self._max_tokens,
            additional_request_fields=extra,
            # 25 concurrent lanes throttle. `adaptive` adds client-side rate
            # limiting on top of retries, which is what actually helps under
            # sustained concurrency.
            #
            # 600, between the 900 the 96.87% run used and the 300 tried on 0814.
            # 300 did not help (stall rate 7.8% -> 9.5%) because it cut off calls
            # that would still have delivered; 900 leaves a hang sitting for ~17
            # min. 600 halves that while staying above the p90 of successful calls
            # (505s on 0813). Note it is per HTTP request (per LLM call), not per
            # analyze(), and with streaming on it bounds the gap between chunks
            # rather than the call's total duration -- so a long healthy call that
            # keeps streaming is never cut. Every hang also triggers a route switch,
            # so a lower value trades more false switches for faster detection.
            boto_client_config=BotocoreConfig(
                read_timeout=600,
                connect_timeout=30,
                retries={"max_attempts": 8, "mode": "adaptive"},
            ),
            # A single analyze() call is a ReAct loop of many model invocations over
            # a growing conversation. Without caching each iteration re-bills the
            # whole prefix at full input price, which is most of the cost.
            cache_config=CacheConfig(strategy="auto"),
        )

    # ----- prompts (PRO-LONG's, verbatim templates) -------------------------

    def _log_window_desc(self) -> str:
        if self._log_window is None:
            return "It contains the full game history."
        if (self._log_window or 0) <= 0:
            return "It contains only the current board."
        return f"It contains the last {self._log_window} actions."

    def _build_system_prompt(self) -> str:
        template = SYSTEM_PROMPT_INPROMPT if self._in_prompt_board else SYSTEM_PROMPT
        hint = (
            " Cross-turn parsing (diffs between distant boards, greps of a fixed "
            "cell across board sections) is tractable and can be useful for "
            "understanding mechanics, including long-horizon ones."
            if self._cross_turn_hint and not self._in_prompt_board
            else ""
        )
        prompt = template.format(
            log_window_desc=self._log_window_desc(),
            cross_turn_hint=hint,
            action_cap=self._action_cap,
            actions_section=format_actions_block(self._available_actions),
        )
        colour = HEX_COLOR_MAP if self._grid_mode == "hex" else ASCII_COLOR_MAP
        # Tool names here are ours, not Claude Code's; keep the mapping explicit
        # so the prompt's "Tools:" line is not a lie.
        tool_note = (
            "\n**Your tools**: `read_file`, `write_file`, `edit_file`, `grep`, "
            "`glob_files`, `bash`. `bash` runs in the workspace with python3 and "
            "the standard library; there is no network and no package installs.\n"
        )
        return prompt + colour + tool_note

    def _build_prompt(
        self,
        log_path: Path,
        action_num: int,
        is_first: bool,
        retry_nudge: str = "",
        **kwargs: Any,
    ) -> str:
        if is_first:
            body = (
                f"Read the full game log at {log_path.name}\n\n"
                "This is the first analysis. Analyze the board state and write "
                f"{_ACTIONS_FILE} with your first set of actions."
            )
        else:
            body = (
                f"Read {log_path.name}"
                + (
                    f" (last {self._log_window} actions)."
                    if (self._log_window or 0) > 0
                    else "."
                )
                + " Recent actions and boards are at the end of the log; what changed "
                "since the last call (new moves, score transitions, plan adherence) "
                "can be informative.\n\n"
                f"Score: {kwargs.get('score', 0)} | Action: {action_num} | "
                f"Level: {kwargs.get('level', 1)}\n"
                f"Last actions: {kwargs.get('last_actions', 'none')}\n\n"
                f"Check the workspace for anything you saved previously, then write a "
                f"new {_ACTIONS_FILE}."
            )
        return f"{body}\n\n{retry_nudge}" if retry_nudge else body

    # ----- the analyzer call ----------------------------------------------

    def analyze(
        self,
        log_path: Path,
        action_num: int,
        retry_nudge: str = "",
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        log_path = Path(log_path)
        workspace = log_path.parent

        avail = kwargs.get("available_actions_list")
        if avail:
            self._available_actions = list(avail)

        path_key = str(log_path)
        is_first = self._call_count.get(path_key, 0) == 0
        self._call_count[path_key] = self._call_count.get(path_key, 0) + 1

        actions_path = workspace / _ACTIONS_FILE
        if actions_path.exists():
            actions_path.unlink()  # PRO-LONG clears it every call

        active_route = self._route_for(path_key)

        agent = Agent(
            model=self._build_model(*active_route),
            tools=make_prolong_tools(workspace),
            system_prompt=self._build_system_prompt(),
            callback_handler=None,
        )
        prompt = self._build_prompt(
            log_path, action_num, is_first, retry_nudge=retry_nudge, **kwargs
        )

        t0 = time.time()
        result = None
        for attempt in range(_MAX_THROTTLE_RETRIES):
            try:
                result = agent(prompt)
                break
            except Exception as exc:
                msg = repr(exc)
                throttled = any(
                    k in msg.lower()
                    for k in ("throttl", "too many requests", "rate exceeded",
                              "429", "serviceunavailable", "slow down")
                )
                if not throttled or attempt == _MAX_THROTTLE_RETRIES - 1:
                    # A hung socket demotes this lane one rung so the runner's next
                    # retry uses a different model. Anything else (throttling that
                    # outlived its backoff, a bad request, a tool error) is not a
                    # stall and leaves the ladder alone.
                    if any(k in msg.lower() for k in _HANG_MARKERS):
                        self._demote(path_key, action_num, active_route)
                    log.warning(
                        "strands analyze failed at action %d (attempt %d) model=%s: %s",
                        action_num, attempt + 1, _short_route(active_route), msg[:200],
                    )
                    return None
                # Exponential backoff with jitter; with 25 lanes, un-jittered
                # retries re-collide and throttle again.
                delay = min(_THROTTLE_BASE_DELAY * (2 ** attempt), 120.0)
                delay *= 0.5 + random.random()
                log.info(
                    "throttled at action %d (attempt %d), backing off %.1fs",
                    action_num, attempt + 1, delay,
                )
                time.sleep(delay)
        if result is None:
            return None
        elapsed = time.time() - t0

        text = str(result)
        meta = self._extract_usage(result, active_route[0])

        if not actions_path.exists():
            # The model finished its turn without writing the file. PRO-LONG handles
            # this by retrying the whole call, which costs another 60-175s and a cold
            # cache. Nudging the SAME agent instance keeps the conversation (and the
            # prompt cache) and usually completes in seconds.
            log.info(
                "no %s after analyze (action=%d, %.1fs) — nudging in-context",
                _ACTIONS_FILE, action_num, elapsed,
            )
            try:
                agent(
                    f"You have not written {_ACTIONS_FILE} yet. Write it now with "
                    'the shape {"actions": ["ACTION1", "ACTION6(30,40)"]} — a list of '
                    f"1-{self._action_cap} actions to execute in order. Write the file; "
                    "do not reply with the JSON only."
                )
            except Exception as exc:
                log.warning("in-context nudge failed: %r", exc)
            elapsed = time.time() - t0
            if not actions_path.exists():
                # Last resort: the model often emits the JSON inline instead of
                # writing the file. The decision is the model's either way, so
                # recover it from the text rather than burning another 60-175s call.
                recovered = self._actions_from_text(text)
                if recovered:
                    log.info(
                        "recovered %d actions from response text (action=%d)",
                        len(recovered), action_num,
                    )
                    actions_path.write_text(recovered)
                else:
                    dump = workspace / f"FAILED_response_action{action_num}.txt"
                    try:
                        dump.write_text(text)
                    except Exception:
                        pass
                    log.warning(
                        "still no %s after nudge (action=%d, %.1fs) tail=%r",
                        _ACTIONS_FILE, action_num, elapsed, text[-400:],
                    )
                    return None
            meta = self._extract_usage(result, active_route[0])

        actions = self._parse_actions_json_text(
            actions_path.read_text(errors="replace"), cap=self._action_cap
        )
        if not actions:
            log.warning("%s parsed to zero actions at action %d", _ACTIONS_FILE, action_num)
            return None

        # The call produced a usable plan, so whatever was stalling has cleared.
        # Reset to the configured model for this lane's next call: the fallback is
        # for recovery only and must never become the steady state.
        self._promote(path_key, action_num, active_route)

        log.info(
            "strands analyze action=%d actions=%d %.1fs model=%s in=%s out=%s cached=%s "
            "cycles=%s tools=[%s]",
            action_num, len(actions), elapsed, _short_route(active_route),
            meta.get("input_tokens"), meta.get("output_tokens"),
            meta.get("cached_tokens"),
            meta.get("cycles"), meta.get("tools", ""),
        )
        self._save_session_state(log_path, self._session_ids.get(path_key, "strands"), action_num)

        return {
            "actions": actions,
            "hint": self._extract_tag(text, "PLAN") or text[-1500:],
            "plan": self._extract_tag(text, "PLAN") or "",
            "meta": meta,
            "cost": 0.0,
        }

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _actions_from_text(text: str) -> str:
        """Extract an {"actions": [...]} object from free text, or "".

        Handles fenced code blocks and bare objects, taking the last match so a
        revised plan wins over an earlier draft.
        """
        import json as _json
        import re as _re

        best = ""
        for m in _re.finditer(r'\{[^{}]*"actions"\s*:\s*\[[^\]]*\][^{}]*\}', text, _re.S):
            blob = m.group(0)
            try:
                obj = _json.loads(blob)
            except Exception:
                continue
            if isinstance(obj.get("actions"), list) and obj["actions"]:
                best = _json.dumps({"actions": obj["actions"]})
        return best

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        marker = f"[{tag}]"
        idx = text.rfind(marker)
        return text[idx + len(marker):].strip() if idx >= 0 else ""

    def _extract_usage(self, result: Any, model_id: str | None = None) -> dict[str, Any]:
        """Pull token usage off the Strands result, tolerating shape changes.

        `model_id` is the model that actually served the call, which may be a
        fallback rung rather than the configured one -- pricing must follow it.
        """
        usage: dict[str, Any] = {"cumulative": False}
        metrics = getattr(result, "metrics", None)
        raw = getattr(metrics, "accumulated_usage", None) if metrics else None
        if isinstance(raw, dict):
            usage["input_tokens"] = raw.get("inputTokens") or raw.get("input_tokens") or 0
            usage["output_tokens"] = raw.get("outputTokens") or raw.get("output_tokens") or 0
            usage["cached_tokens"] = (
                raw.get("cacheReadInputTokens") or raw.get("cached_tokens") or 0
            )
        else:
            usage.update(input_tokens=0, output_tokens=0, cached_tokens=0)
        # Tool-call distribution, so we can compare our agent's working style with
        # PRO-LONG's reported mix (python3 60.6%, log parsing 20.3%, workspace 19.1%).
        try:
            summary = metrics.get_summary() if metrics else {}
            tools = summary.get("tool_usage", {}) or {}
            counts = {
                name: (info.get("execution_stats", {}) or {}).get("call_count", 0)
                for name, info in tools.items()
            }
            usage["tools"] = " ".join(
                f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v
            )
            usage["cycles"] = getattr(metrics, "cycle_count", None)
        except Exception:
            usage["tools"] = ""
            usage["cycles"] = None

        # Price the call. The 0809 run left this at 0 for all 1,907 calls, so the
        # whole 25-game run had no cost figure and no local token totals at all.
        if isinstance(raw, dict):
            usage["cache_write_tokens"] = (
                raw.get("cacheWriteInputTokens") or raw.get("cache_write_tokens") or 0
            )
        else:
            usage["cache_write_tokens"] = 0
        pr = _prices_for(model_id or self._model_id)
        if pr:
            p_in, p_out, p_cr, p_cw = pr
            usage["call_cost_usd"] = (
                usage.get("input_tokens", 0) * p_in
                + usage.get("output_tokens", 0) * p_out
                + usage.get("cached_tokens", 0) * p_cr
                + usage.get("cache_write_tokens", 0) * p_cw
            ) / 1_000_000.0
        else:
            usage["call_cost_usd"] = 0.0
            log.warning("no price row for model %s — cost will read 0", self._model_id)
        return usage
