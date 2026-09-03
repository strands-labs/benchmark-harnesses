import os
import difflib
import logging
import shlex
import traceback
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment
from ssa.tools.utils import detect_language
from ssa.tools.str_replace_editor import RefineLLMResponse

LOG = logging.getLogger(__name__)
console = Console()
MAX_CHARS_LIMIT: int = 20_000


TOOL_SPEC = {
    "name": "file_edit",
    "description": (
        "Tool for editing existing files.\n"
        "* The `str_replace` command replaces an exact string in a file with a new string.\n"
        "* The `undo_edit` command reverts the last edit made to the file.\n"
        "\n"
        "Notes for using the `str_replace` command:\n"
        "* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. "
        "Be mindful of whitespaces!\n"
        "* If the `old_str` parameter is not unique in the file, the replacement will not be performed. "
        "Make sure to include enough context in `old_str` to make it unique.\n"
        "* The `new_str` parameter should contain the edited lines that should replace the `old_str`.\n"
        "* To create new files, use the `file_write` tool instead."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["str_replace", "undo_edit"],
                    "description": "The command to run. Allowed options are: `str_replace`, `undo_edit`."
                },
                "description": {
                    "description": "Why I'm making this edit",
                    "type": "string"
                },
                "path": {
                    "description": "Absolute path to file, e.g. `/repo/file.py`.",
                    "type": "string"
                },
                "old_str": {
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    "type": "string"
                },
                "new_str": {
                    "description": "Required parameter of `str_replace` command containing the new string.",
                    "type": "string"
                },
            },
            "required": ["command", "path", "description"]
        }
    }
}


def _execute_bash(command: str, workdir: str, environment: Environment, verbose: bool = False) -> str:
    try:
        bash_result = environment.execute_bash(command=command, workdir=workdir, verbose=verbose)
        return bash_result.get("output", "")
    except Exception as e:
        LOG.warning(f"Bash execution failed for command: {command} in workdir: {workdir}. Details: {e}")
        return ""


def _display_code_edit_blocks(
    old_str: str, new_str: str, path: str, language: str, grid: Table
) -> None:
    old_panel = Panel(
        Syntax(str(old_str), language, theme="monokai", line_numbers=True, word_wrap=True),
        title="[bold red]Original Content",
        subtitle=f"{len(old_str.splitlines())} lines, {len(old_str)} characters",
        border_style="red",
        box=box.ROUNDED,
    )
    new_panel = Panel(
        Syntax(str(new_str), language, theme="monokai", line_numbers=True, word_wrap=True),
        title="[bold green]New Content",
        subtitle=f"{len(new_str.splitlines())} lines, {len(new_str)} characters",
        border_style="green",
        box=box.ROUNDED,
    )
    grid.add_row(old_panel, Text("\n\n➔", justify="center", style="bold yellow"), new_panel)
    preview_panel = Panel(
        grid,
        title=f"[bold blue]🔄 Text Replacement Preview ({os.path.basename(path)})",
        subtitle=f"{os.path.abspath(path)}",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print()
    try:
        console.print(preview_panel)
    except Exception as e:
        LOG.warning(f"Failed to render panel with old_str:{old_str}\nnew_str: {new_str}\nDetails: {e}")
    console.print()


def file_edit(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Create and edit files via str_replace, create, and undo_edit commands."""
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    show_panel: bool = kwargs.pop("show_panel", True)
    verbose = kwargs.get("verbose", True)
    environment: Environment = kwargs["environment"]
    workdir = environment.workdir or os.getcwd()

    # Per-tool config from YAML (agent.tools.xai.file_edit)
    _tool_cfg = kwargs.get("tool_params", {}).get("xai.file_edit", {})
    reply_detail: bool = _tool_cfg.get("reply_detail", kwargs.pop("str_replace_reply_detail", True))
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    allow_duplicate_edits: bool = _tool_cfg.get("allow_duplicate_edits", kwargs.get("allow_duplicate_edits", False))
    duplicate_locations_detail: bool = _tool_cfg.get("duplicate_locations_detail", kwargs.get("duplicate_locations_detail", False))
    provide_fuzzy_feedback: bool = _tool_cfg.get("provide_fuzzy_feedback", kwargs.get("provide_fuzzy_feedback", False))

    try:
        path: str = tool_input.get("path", "")
        if not path:
            raise ValueError("path parameter is required")

        mode = tool_input.get("command", None)
        if mode is None:
            old_str = tool_input.get("old_str")
            new_str = tool_input.get("new_str")
            if old_str is not None and new_str is not None and old_str != new_str:
                mode = "str_replace"
        if mode is None:
            raise ValueError("command parameter is required")

        grid = Table.grid(expand=True)
        grid.add_column("Original", justify="left", ratio=1)
        grid.add_column("Arrow", justify="center", width=5)
        grid.add_column("New", justify="left", ratio=1)

        language = detect_language(path)

        if mode == "view":
            raise ValueError("Use the `file_read` tool for viewing files and directories.")
        if mode == "create":
            raise ValueError("Use the `file_write` tool for creating new files.")
        if mode == "insert":
            raise ValueError("command=`insert` is not supported, instead use `str_replace`")
        if mode not in ["str_replace", "undo_edit"]:
            raise ValueError(f"Unknown command={mode}. Allowed options are: `str_replace`, `undo_edit`.")

        # ── str_replace ──
        if mode == "str_replace":
            old_str = tool_input.get("old_str")
            new_str = tool_input.get("new_str")

            if old_str is None or new_str is None:
                raise ValueError("old_str and new_str are both required for mode `str_replace`")
            if old_str == new_str:
                raise ValueError(f"Provided old_str and new_str are exactly same. No edit performed for path: {path}")
            if environment.dir_exists(path):
                raise ValueError(f"`str_replace` mode not suitable for provided path: {path} which is a dir")
            if not environment.file_exists(path):
                raise ValueError(
                    f"Provided file path: {path} does not exist. "
                    "Make sure to use absolute path only, e.g., `/repo/file.py`"
                )
            if not path.startswith(workdir):
                path = os.path.join(workdir, path)

            # Make backup
            backup_path = f"{path}.bak"
            command = f"cp {shlex.quote(path)} {shlex.quote(backup_path)}"
            _ = environment.execute_bash(command, workdir=workdir)

            command = f"cat {shlex.quote(path)}"
            content = _execute_bash(command, workdir=workdir, environment=environment)

            old_str, new_str = RefineLLMResponse._strip_equal_newlines(old_str, new_str)

            if show_panel:
                _display_code_edit_blocks(old_str, new_str, path, language, grid)

            edit_result = RefineLLMResponse.normalize_and_apply(content, old_str, new_str)

            if edit_result.partial_lines:
                LOG.warning(f"old_str does not carry complete text lines from path: {path}. Expand your current context")
                raise ValueError(
                    f"old_str does not carry complete text lines from path: {path}. Expand your current context"
                )

            if edit_result.current_counts == 0:
                fuzzy_feedback = ""
                if provide_fuzzy_feedback:
                    fuzzy_feedback = RefineLLMResponse.fuzzy_find_closest_block(content, old_str, threshold=0.8)
                if fuzzy_feedback:
                    LOG.warning(f"old_str not found in {path}. Fuzzy match feedback:\n{fuzzy_feedback}")
                    raise ValueError(fuzzy_feedback)
                else:
                    LOG.warning(f"Note: old_str not found in {path}")
                    raise ValueError(f"old_str not found in {path}")

            counts = edit_result.current_counts
            if counts > 1 and not allow_duplicate_edits:
                # Show duplicate locations
                locations = []
                offset = 0
                for _ in range(counts):
                    n_match = RefineLLMResponse.get_normalized_match(content, old_str, offset=offset)
                    locations.append((n_match.start_line, n_match.end_line))
                    offset = n_match.end_line

                cat_command = f"cat -n {shlex.quote(path)}"
                numbered_content = _execute_bash(cat_command, workdir=workdir, environment=environment)
                numbered_content_lines = numbered_content.splitlines()
                locations_detail = []
                for ct in range(counts):
                    start_line, end_line = locations[ct]
                    locations_detail.extend([
                        f"<--- location-{ct + 1} --->\n",
                        "\n".join(numbered_content_lines[start_line:end_line]),
                        "\n",
                    ])
                multiple_locations_feedback = (
                    f"Provided old_str is present at {counts} locations in the file:\n"
                    + ("\n".join(locations_detail) if duplicate_locations_detail else "")
                    + "\nExpand your current context of `old_str` such that it is unique in the file"
                )
                LOG.warning(multiple_locations_feedback)
                raise ValueError(multiple_locations_feedback)

            new_content = edit_result.new_content
            if content == new_content:
                LOG.warning(f"Failed to apply feasible diff to path: {path}")

            diff_path = path.replace(workdir, "").lstrip("/")
            diffs = difflib.unified_diff(
                content.splitlines(),
                new_content.splitlines(),
                lineterm="",
                fromfile=f"a/{diff_path}",
                tofile=f"b/{diff_path}",
                n=3,
            )
            diff_str = "\n".join(diffs)

            environment.write_file(new_content, path)

            result = (
                "Text replacement complete."
                + (f"\nFile: {path}" if not reply_detail else "")
                + (f"\nReplaced {counts} occurrence{'s' if counts > 1 else ''}\n" if counts > 1 else "")
                + (f"\nApplied diff:\n{diff_str}\n" if reply_detail else "")
            )
            if show_panel:
                try:
                    console.print(Panel(result, title="[bold green]Output", border_style="blue"))
                except Exception as e:
                    LOG.warning(f"Failed to render text replacement result: {result}\nDetails:{e}")

        # ── undo_edit ──
        elif mode == "undo_edit":
            backup_path = f"{path}.bak"
            if not environment.file_exists(backup_path):
                raise ValueError(f"No backup file found for {path}")

            command = f"cp {shlex.quote(backup_path)} {shlex.quote(path)}"
            _ = environment.execute_bash(command, workdir=workdir)
            command = f"rm {shlex.quote(backup_path)}"
            _ = environment.execute_bash(command, workdir=workdir)

            result = f"Successfully reverted changes to {path}. No further backup exists for this file"

        if len(result) > max_chars_limit:
            LOG.info(
                f"Clipping file_edit output for mode: {mode} due to chars len ({len(result)}) "
                f"exceeding limit ({max_chars_limit})"
            )
            result = result[:max_chars_limit] + "\n<response clipped>"

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result}],
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if show_panel:
            try:
                console.print(Panel(error_msg, title="[bold red]Error", border_style="red"))
            except Exception:
                pass
        if verbose:
            LOG.error(f"{traceback.format_exc()}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }
