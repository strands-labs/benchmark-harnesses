import asyncio
import logging
from typing import Any, AsyncGenerator
from strands.agent import Agent
from strands.agent.agent import AgentResult, AgentInput
from strands.event_loop.event_loop import event_loop_cycle
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException
from strands.types._events import TypedEvent
from strands.types.agent import Limits
from strands.tools.structured_output._structured_output_context import StructuredOutputContext
from pydantic import BaseModel

from ssa.hooks.events import AgentCompletedEvent


LOG = logging.getLogger(__file__)


class StrandsResolverAgent(Agent):
    """Wrap event loop execution for handling max-tokens exception and max recursions exception"""

    def __call__(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: type[BaseModel] | None = None,
        structured_output_prompt: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Override to run async event loop in the main thread so signals interrupt it directly."""
        return asyncio.run(
            self.invoke_async(
                prompt,
                invocation_state=invocation_state,
                structured_output_model=structured_output_model,
                structured_output_prompt=structured_output_prompt,
                **kwargs,
            )
        )

    async def _execute_event_loop_cycle(
        self, invocation_state: dict[str, Any], structured_output_context: StructuredOutputContext | None = None,
        limits: Limits | None = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute the event loop cycle with retry logic for context window limits.

        This internal method handles the execution of the event loop cycle and implements
        retry logic for handling context window overflow exceptions by reducing the
        conversation context and retrying.

        Yields:
            Events of the loop cycle.
        """
        # Add `Agent` to invocation_state to keep backwards-compatibility
        invocation_state["agent"] = self
        invocation_state["_sra_depth"] = invocation_state.get("_sra_depth", 0) + 1
        is_outermost = invocation_state["_sra_depth"] == 1

        if structured_output_context:
            structured_output_context.register_tool(self.tool_registry)

        try:
            # Execute the main event loop cycle
            events = event_loop_cycle(
                agent=self,
                invocation_state=invocation_state,
                structured_output_context=structured_output_context,
                limits=limits,
            )
            async for event in events:
                yield event

        except ContextWindowOverflowException as e:
            # Try reducing the context size and retrying
            self.conversation_manager.reduce_context(self, e=e, from_overflow=True)

            # Sync agent after reduce_context to keep conversation_manager_state up to date in the session
            if self._session_manager:
                self._session_manager.sync_agent(self)

            events = self._execute_event_loop_cycle(invocation_state, structured_output_context)
            async for event in events:
                yield event

        except MaxTokensReachedException:
            # Re-try by declaring this as throttling event upto max-retry for max-tokens exception 
            LOG.info("Re-trying for max-tokens exception by strands-resolver agent by removing the last received assistant message")
            LOG.info(f"removing the last message: {self.messages[-1]}")
            self.messages = self.messages[:-1]
            events = self._execute_event_loop_cycle(invocation_state, structured_output_context)
            async for event in events:
                yield event
        
        except Exception as e:
            if str(e).startswith("Recursion depth exceed the maximum set limit"):
                # TODO: (vatshank) make exception str constant
                LOG.info("Re-init event loop recursions due to set limit hit")
                events = self._execute_event_loop_cycle(invocation_state, structured_output_context)
                async for event in events:
                    yield event
            else:
                LOG.error(f"Exception in event loop: {e}")
                raise e
        finally:
            if structured_output_context:
                structured_output_context.cleanup(self.tool_registry)
            invocation_state["_sra_depth"] -= 1
            if is_outermost:
                self.hooks.invoke_callbacks(AgentCompletedEvent(agent=self))
