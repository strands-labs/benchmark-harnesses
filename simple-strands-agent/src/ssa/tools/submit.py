import logging
from typing import Dict, Any
import traceback

from rich.console import Console
from rich.panel import Panel
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment


LOG = logging.getLogger(__file__)
console = Console()


TOOL_SPEC = {
    "name": "submit",
    "description": (
        "Submit the final result and conclude the task.\n"
        "Call this tool ONLY when all assigned tasks have been completed successfully. \n"
        "This signals that the agent has finished its work and is ready to return the final output. \n"
        "Do not call this tool if there are remaining steps, unresolved errors, or incomplete objectives."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A brief summary of what was accomplished, including key actions taken and any important decisions made during execution."
                },
                "status": {
                    "type": "string",
                    "enum": ["success", "partial_success", "failure"],
                    "description": "The completion status of the task. Use 'success' if all objectives were met, 'partial_success' if some but not all objectives were achieved, and 'failure' if the task could not be completed."
                },
                "paths": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of file paths that were created or modified and need to be submitted as part of the final result. IMPORTANT: Only include files that are strictly necessary and directly related to the task. Do not include unrelated or tangentially modified files. Broader or unwanted changes may break hidden tests."
                },
            },
            "required": ["summary", "status", "paths"]
        }
    }
}


def submit(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    String replace tool for the file editing
    Args:
        tool (Any): Tool information containing toolUseId and input
    Returns:
        Dict[str, Any]: Tool execution result
    """
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})
    show_panel: bool = kwargs.pop("show_panel", True)
    request_state: Dict = kwargs.get("request_state", {})
    environment: Environment = kwargs["environment"]

    # Per-tool config from YAML (agent.tools.submit)
    _tool_cfg = kwargs.get("tool_params", {}).get("submit", {})

    try:
        # Validate required parameters
        if not tool_input.get("summary"):
            raise ValueError("summary parameter is required")
        
        if not tool_input.get("status"):
            raise ValueError("status parameter is required")

        if "paths" not in tool_input or not isinstance(tool_input["paths"], list):
            raise ValueError("paths parameter is required and must be a list of file paths")

        missing_paths = [p for p in tool_input["paths"] if not environment.file_exists(p)]
        if missing_paths:
            raise ValueError(f"The following paths do not exist: {missing_paths}")

        request_state["stop_event_loop"] = True
        request_state["submit_paths"] = tool_input["paths"] 

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": "Successfully terminated."}],
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if show_panel:
            try:
                console.print(Panel(error_msg, title="[bold red]Error", border_style="red"))
            except Exception as e:
                LOG.warning(f"Error message failed to render. Msg={error_msg}. Details: {e}")
        LOG.error(f"{traceback.format_exc()}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }