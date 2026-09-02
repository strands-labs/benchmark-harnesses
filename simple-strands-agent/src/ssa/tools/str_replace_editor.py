import os
from dataclasses import dataclass
import difflib
import logging
from typing import List, Any
import traceback
import shlex

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment
from .utils import format_output, create_rich_panel, get_tree_from_files, detect_language


LOG = logging.getLogger(__name__)
console = Console()
MAX_CHARS_LIMIT: int = 20_000


TOOL_SPEC = {
    "name": "str_replace_editor",
    "description": (
        "Custom editing tool for viewing, creating, and editing files.\n"
        "* State is persistent across command calls and discussions with the user\n"
        "Supported features:\n"
        "* If `path` is a file, `view` displays the result of applying `cat -n`.\n"
        "  If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n"
        "* The `create` command cannot be used if the specified `path` already exists as a file\n"
        "* If a `command` generates a long output, it will be truncated and marked with `<response clipped>` \n"
        "* The `undo_edit` command will revert the last edit made to the file at `path`\n"
        "\n"
        "Notes for using the `str_replace` command:\n"
        "* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!\n"
        "* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique\n"
        "* The `new_str` parameter should contain the edited lines that should replace the `old_str`"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "undo_edit"],
                    "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `undo_edit`."
                },
                "description": {
                    "description": "Why I'm making this edit",
                    "type": "string"
                },
                "file_text": {
                    "description": "Required parameter of `create` command, with the content of the file to be created.",
                    "type": "string"
                },
                "new_str": {
                    "description": "Required parameter of `str_replace` command containing the new string.",
                    "type": "string"
                },
                "old_str": {
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    "type": "string"
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                    "type": "string"
                },
                "view_range": {
                    "description": (
                        "Optional parameter of `view` command when `path` points to a file. If none is given, "
                        "the full file is shown. If provided, the file will be shown in the indicated line number range, "
                        "e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows "
                        "all lines from `start_line` to the end of the file."
                    ),
                    "items": {
                        "type": "integer"
                    },
                    "type": "array"
                }
            },
            "required": ["description", "command", "path"]
        }
    }
}


@dataclass
class RefineLLMResult:
    current_counts: int
    new_content: str
    mod_old_str: str
    partial_lines: bool = False
    extra_blanks: int = 0


@dataclass
class NormalizedMatchResult:
    normalized_str: str
    found: bool
    start_line: int = -1 # format: [start_line, end_line)
    end_line: int = -1
    extra_blanks: int = 0  # blank-only lines in content absorbed by the match


class RefineLLMResponse:
    """Handles whitespace normalization, indentation adjustment, and
    fuzzy matching when applying LLM-generated str_replace edits."""

    @staticmethod
    def _count_leading_whitespace(line: str) -> int:
        """Return the number of leading whitespace units, treating tabs as 4 spaces."""
        count = 0
        for char in line:
            if char == "\t":
                count += 4
            elif char == " ":
                count += 1
            else:
                break
        return count

    @staticmethod
    def _find_min_indentation(block: str) -> int:
        """Return the minimum indentation across all non-empty lines in block."""
        indents = []
        for line in block.splitlines():
            if line.strip():
                indents.append(RefineLLMResponse._count_leading_whitespace(line))
        return min(indents) if indents else 0

    @staticmethod
    def find_indentation_diff(block1: str, block2: str) -> int:
        """
        Get indentation difference between block2 and block1 by looking
        at non-empty lines. Returns block2_min_indent - block1_min_indent.
        """
        if not block1 or not block2:
            return 0
        return RefineLLMResponse._find_min_indentation(
            block2
        ) - RefineLLMResponse._find_min_indentation(block1)

    @staticmethod
    def adjust_indentation(text: str, indent_diff: int) -> str:
        """
        Adjust indentation of all lines in text by indent_diff spaces.
        Treats tabs as 4 spaces. If any line has insufficient indentation
        to remove, returns the original text unchanged.
        """
        if indent_diff == 0:
            return text
        lines = text.splitlines()
        if indent_diff < 0:
            for line in lines:
                if line.strip() and RefineLLMResponse._count_leading_whitespace(
                    line
                ) < abs(indent_diff):
                    return text
        adjusted = []
        for line in lines:
            if indent_diff > 0:
                adjusted.append(" " * indent_diff + line)
            else:
                to_remove = abs(indent_diff)
                removed = 0
                idx = 0
                while removed < to_remove and idx < len(line):
                    if line[idx] == "\t":
                        removed += 4
                    elif line[idx] == " ":
                        removed += 1
                    else:
                        break
                    idx += 1
                adjusted.append(line[idx:])
        return "\n".join(adjusted)

    @staticmethod
    def get_normalized_match(content: str, old_str: str, offset: int = 0) -> NormalizedMatchResult:
        """Find old_str in content with whitespace tolerance.

        Tries three branches in order:
          1. Trailing-whitespace-normalized match (preserves length & blanks).
          2. Fully-stripped match (indentation-agnostic, same length).
          3. Blank-tolerant match: strip blank/whitespace-only lines.

        Partial lines (mid-line matches) are not supported.
        """
        content_lines = content.split("\n") # preserve end \n
        old_lines = old_str.split("\n")

        if not old_lines:
            return NormalizedMatchResult(normalized_str="", found=False)

        norm_content = [line.rstrip() for line in content_lines]
        norm_old = [line.rstrip() for line in old_lines]
        stripped_old = [line.strip() for line in norm_old]

        # Branches 1 & 2: same length, position-by-position.
        for i in range(offset, len(content_lines) - len(old_lines) + 1):
            if norm_content[i : i + len(old_lines)] == norm_old:
                return NormalizedMatchResult(
                    normalized_str="\n".join(content_lines[i : i + len(old_lines)]),
                    found=True,
                    start_line=i,
                    end_line=i + len(old_lines),
                )
            stripped_window = [line.strip() for line in norm_content[i : i + len(old_lines)]]
            if stripped_window == stripped_old:
                return NormalizedMatchResult(
                    normalized_str="\n".join(content_lines[i : i + len(old_lines)]),
                    found=True,
                    start_line=i,
                    end_line=i + len(old_lines),
                )

        # Branch 3: blank-tolerant. Match on the non-blank subsequence
        old_nonblank_idx = [k for k, line in enumerate(norm_old) if line.strip()]
        if not old_nonblank_idx:
            return NormalizedMatchResult(normalized_str="", found=False)
        content_nonblank_idx = [k for k, line in enumerate(norm_content) if line.strip()]

        old_text = [norm_old[k].strip() for k in old_nonblank_idx]
        n = len(old_text)

        for j in range(len(content_nonblank_idx) - n + 1):
            first = content_nonblank_idx[j]
            if first < offset:
                continue
            window_text = [
                norm_content[content_nonblank_idx[j + k]].strip() for k in range(n)
            ]
            if window_text == old_text:
                start_line = first
                end_line = content_nonblank_idx[j + n - 1] + 1
                # Marker only — exact count doesn't matter, just that splice
                # path activates downstream.
                extra_blanks = max(1, abs((end_line - start_line) - len(norm_old)))
                return NormalizedMatchResult(
                    normalized_str="\n".join(content_lines[start_line:end_line]),
                    found=True,
                    start_line=start_line,
                    end_line=end_line,
                    extra_blanks=extra_blanks,
                )

        return NormalizedMatchResult(normalized_str="", found=False)

    @staticmethod
    def get_normalized_counts(content: str, old_str: str) -> int:
        """Count line-level normalized matches of old_str in content.
        Uses get_normalized_match to avoid substring false positives."""
        counts = 0
        offset = 0
        while True:
            m = RefineLLMResponse.get_normalized_match(content, old_str, offset)
            if not m.found:
                break
            counts += 1
            offset = m.end_line
        return counts

    @staticmethod
    def _strip_equal_newlines(old_str: str, new_str: str) -> tuple:
        """Strip equal number of leading and trailing newlines from both strings."""
        # Count leading newlines
        old_leading = len(old_str) - len(old_str.lstrip("\n"))
        new_leading = len(new_str) - len(new_str.lstrip("\n"))
        leading = min(old_leading, new_leading)

        # Count trailing newlines
        old_trailing = len(old_str) - len(old_str.rstrip("\n"))
        new_trailing = len(new_str) - len(new_str.rstrip("\n"))
        trailing = min(old_trailing, new_trailing)

        if leading > 0:
            old_str = old_str[leading:]
            new_str = new_str[leading:]
        if trailing > 0:
            old_str = old_str[:-trailing]
            new_str = new_str[:-trailing]

        return old_str, new_str

    @staticmethod
    def fuzzy_find_closest_block(content: str, old_str: str, threshold: float = 0.5) -> str:
        """When old_str is not found, find the closest matching block in content
        and return a human-readable diff with ^^ markers under mismatched characters.

        Returns a feedback string showing the closest match and where it differs,
        or empty string if no reasonable match found.
        """
        content_lines = content.split("\n")
        old_lines = old_str.split("\n")
        num_old = len(old_lines)

        if num_old == 0 or len(content_lines) == 0:
            return ""

        best_ratio = 0.0
        best_start = 0

        # Slide a window of len(old_lines) over the content to find the best match
        stripped_old = [line.strip() for line in old_lines]
        old_joined = "\n".join(stripped_old)

        for i in range(max(1, len(content_lines) - num_old + 1)):
            window = content_lines[i : i + num_old]
            stripped_window = [line.strip() for line in window]
            window_joined = "\n".join(stripped_window)
            ratio = difflib.SequenceMatcher(None, old_joined, window_joined).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio < threshold:
            return ""

        # Build character-level diff feedback
        best_block = content_lines[best_start : best_start + num_old]
        feedback_lines = []
        feedback_lines.append(
            f"old_str not found. Closest match at lines {best_start + 1}-{best_start + num_old}:"
        )
        feedback_lines.append("")

        for line_idx, (got, expected) in enumerate(zip(old_lines, best_block)):
            got_stripped = got.rstrip()
            expected_stripped = expected.rstrip()
            if got_stripped == expected_stripped:
                continue
            # Show the mismatched line pair
            file_line_num = best_start + line_idx + 1
            feedback_lines.append(f"  Line {file_line_num}:")
            feedback_lines.append(f"    actual: {expected_stripped}")
            feedback_lines.append(f"    yours:  {got_stripped}")
            # Build ^^ marker line
            markers = []
            sm = difflib.SequenceMatcher(None, got_stripped, expected_stripped)
            # Mark characters in 'yours' that don't match 'actual'
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    markers.append(" " * (i2 - i1))
                elif tag == "replace":
                    markers.append("^" * max(i2 - i1, j2 - j1))
                elif tag == "delete":
                    markers.append("^" * (i2 - i1))
                elif tag == "insert":
                    markers.append("^" * (j2 - j1))
            feedback_lines.append(f"            {''.join(markers)}")

        # Handle length mismatch
        if len(old_lines) != len(best_block):
            feedback_lines.append(
                f"  (your old_str has {len(old_lines)} lines, "
                f"closest match has {len(best_block)} lines)"
            )

        return "\n".join(feedback_lines)

    @staticmethod
    def normalize_and_apply(content: str, old_str: str, new_str: str) -> RefineLLMResult:
        """Attempt to find and replace old_str in content, handling whitespace
        normalization and indentation adjustment."""

        partial_lines = False
        # Normalized matching
        n_match = RefineLLMResponse.get_normalized_match(content, old_str)
        mod_old_str, found = n_match.normalized_str, n_match.found

        if not found:
            if content.count(old_str) > 0:
                # old_str exists but normalized string not found => partial lines in old_str
                partial_lines = True
            return RefineLLMResult(
                current_counts=0,
                new_content="",
                mod_old_str=old_str,
                partial_lines=partial_lines,
            )

        # Compute indentation difference and adjust new_str
        indent_diff = RefineLLMResponse.find_indentation_diff(old_str, mod_old_str)
        if indent_diff > 0:
            LOG.info(f"detected non-zero indentation difference between llm old_str and file-located block: {indent_diff}. new_str will be adjusted accordingly.")
        mod_new_str = RefineLLMResponse.adjust_indentation(new_str, indent_diff)

        counts = RefineLLMResponse.get_normalized_counts(content, old_str)
        if n_match.extra_blanks > 0 and counts == 1:
            # Blank-tolerant match: mod_old_str includes extra blank lines that
            # don't appear in new_str, so content.replace would silently drop
            # them elsewhere. Splice via line range instead.
            LOG.info(
                f"blank-tolerant match absorbed {n_match.extra_blanks} blank "
                f"line(s) at lines {n_match.start_line}-{n_match.end_line}"
            )
            content_lines = content.split("\n")
            new_content_lines = (
                content_lines[: n_match.start_line]
                + mod_new_str.split("\n")
                + content_lines[n_match.end_line :]
            )
            new_content = "\n".join(new_content_lines)
        else:
            new_content = content.replace(mod_old_str, mod_new_str)

        return RefineLLMResult(
            current_counts=counts,
            new_content=new_content,
            mod_old_str=mod_old_str,
            extra_blanks=n_match.extra_blanks,
        )

def execute_bash(command: str, workdir: str, environment: Environment, verbose: bool = False) -> str:
    try:
        bash_result = environment.execute_bash(
            command=command,
            workdir=workdir,
            verbose=verbose,
        )
        return bash_result.get("output", "")
    except Exception as e:
        LOG.warning(f"Bash execution failed for command: {command} in workdir: {workdir}. Details: {e}")
        return ""

def list_files(path: str, environment: Environment, max_depth: int=2) -> List[str]:
    command = f"find {shlex.quote(path)} -mindepth 1 -maxdepth {max_depth} -not -path '*/.*'"
    result = execute_bash(command=command, workdir=path, environment=environment, verbose=False)
    return result.splitlines()

def display_code_edit_blocks(
    old_str: str,
    new_str: str,
    path: str,
    language: str,
    grid: Table,
) -> None:
    old_panel = Panel(
        Syntax(
            str(old_str),
            language,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        ),
        title="[bold red]Original Content",
        subtitle=f"{len(old_str.splitlines())} lines, {len(old_str)} characters",
        border_style="red",
        box=box.ROUNDED,
    )

    new_panel = Panel(
        Syntax(
            str(new_str),
            language,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        ),
        title="[bold green]New Content",
        subtitle=f"{len(new_str.splitlines())} lines, {len(new_str)} characters",
        border_style="green",
        box=box.ROUNDED,
    )

    # Add panels with arrow between
    grid.add_row(
        old_panel,
        Text("\n\n➔", justify="center", style="bold yellow"),
        new_panel,
    )

    # Wrap everything in a container panel for consistent look
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

def str_replace_editor(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    String replace tool for the file editing
    Args:
        tool (Any): Tool information containing toolUseId and input
    Returns:
        Dict[str, Any]: Tool execution result
    """
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    include_summary: bool = kwargs.pop("include_summary", True)
    show_panel: bool = kwargs.pop("show_panel", True)
    verbose = kwargs.get("verbose", True)
    environment: Environment = kwargs["environment"]
    workdir = environment.workdir or os.getcwd()

    # Per-tool config from YAML (agent.tools.str_replace_editor)
    _tool_cfg = kwargs.get("tool_params", {}).get("str_replace_editor", {})
    show_numbered: bool = _tool_cfg.get("show_numbered", kwargs.get("show_numbered", True))
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    reply_detail: bool = _tool_cfg.get("reply_detail", kwargs.pop("str_replace_reply_detail", True))
    allow_overwrite: bool = _tool_cfg.get("allow_overwrite", kwargs.get("allow_overwrite", False))
    restrict_workspace_create: bool = _tool_cfg.get("restrict_workspace_create", kwargs.get("restrict_workspace_create", True))
    allow_duplicate_edits: bool = _tool_cfg.get("allow_duplicate_edits", kwargs.get("allow_duplicate_edits", False))
    duplicate_locations_detail: bool = _tool_cfg.get("duplicate_locations_detail", kwargs.get("duplicate_locations_detail", False))
    provide_fuzzy_feedback: bool = _tool_cfg.get("provide_fuzzy_feedback", kwargs.get("provide_fuzzy_feedback", False))

    try:
        # Validate required parameters
        if not tool_input.get("path"):
            raise ValueError("path parameter is required: include the absolute path to the file, e.g. /repo/pkg/file.py, or dir, e.g. /repo/src")
        
        mode = tool_input.get("command", None)
        if mode is None:
            old_str = tool_input.get("old_str")
            new_str = tool_input.get("new_str")
            if (
                old_str is not None and 
                new_str is not None and 
                old_str != new_str
            ):
                mode = "str_replace"
        if mode is None:
            raise ValueError("mode parameter is required")

        # Create table grid for side-by-side display
        grid = Table.grid(expand=True)
        grid.add_column("Original", justify="left", ratio=1)
        grid.add_column("Arrow", justify="center", width=5)
        grid.add_column("New", justify="left", ratio=1)
        
        path: str = tool_input["path"]
        language = detect_language(path) 

        if mode == "insert":
            raise ValueError("command=`insert` is not supported, instead use `str_replace`")
        if mode not in ["view", "create", "str_replace", "undo_edit"]:
            raise ValueError(f"Unknown command={mode}. Allowed options are: `view`, `create`, `str_replace`.")

        if mode == "view":
            if environment.dir_exists(path):
                all_files = list_files(path, environment, max_depth=2)
                tree, tree_str = get_tree_from_files(all_files)
                result = (
                    f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n" +
                    tree_str
                )

                if show_panel:
                    try:
                        console.print(Panel(result, title="[bold green]File Tree", border_style="blue"))
                    except Exception as e:
                        LOG.warning(f"Failed to render file tree. Details: {e} ")
            elif environment.file_exists(path):
                cat_command = "cat " + ("-n " if show_numbered else "") + f"{shlex.quote(path)}"
                numbered_content = execute_bash(cat_command, workdir=workdir, environment=environment)
                len_lines = len(numbered_content.splitlines())
                view_range = tool_input.get("view_range", None)

                if view_range is not None:
                    start_line=view_range[0]
                    end_line=view_range[1]
                    if end_line == -1:
                        end_line = None
                    
                    if start_line > 0:
                        start_line -= 1 # normalize to 0-idx
                    if end_line is not None:
                        end_line -= 1
                    else:
                        end_line = len_lines
                    if end_line is not None:
                        if start_line > end_line:
                            raise ValueError(f"Incorrect `view_range`={view_range}, `start_line` must be less than `end_line`")


                    lines_range = numbered_content.splitlines()[start_line:end_line+1]

                    if len(lines_range) > 0:
                        result = f"Here's the content of {path} (which has {len_lines} total lines) with view_range=({start_line+1}, {end_line+1}):\n" + "\n".join(lines_range)
                    elif len_lines > 0:
                        raise ValueError(f"view_range is outside the file contents. The file has {len_lines} lines.")
                    else:
                        result = f"{path} is empty file contents"
                else:
                    if show_panel:
                        # Create rich panel with syntax highlighting
                        view_panel = create_rich_panel(
                            numbered_content,
                            f"📄 {os.path.basename(path)}",
                            path,
                        )
                        try:
                            console.print(f"Valid file workdir: {workdir}")
                            console.print(view_panel)
                        except Exception as e:
                            LOG.warning(f"Failed to render file-contents\n{numbered_content}\nDetails: {e}")
                    result = (f"Here's the content of {path} (which has {len_lines} lines):\n" if include_summary else "") + numbered_content
            else:
                raise ValueError(f"Provided path: {path} does not exist. Make sure to use absolute path only")

        elif mode == "create":
            if not allow_overwrite and environment.file_exists(path):
                raise ValueError(f"Path you are trying to create {path} already exists. Create mode is only for creating new files") 
            if restrict_workspace_create and workdir is not None and not path.startswith(workdir):
                raise ValueError(f"Path `{path}` is not within current workspace: {workdir}. Only create new files within workspace directory by providing full absolute paths, e.g., `/repo/file.py`")
            if not os.path.isabs(path):
                raise ValueError(f"Path `{path}` is not an absolute path. Provide the full absolute path, e.g. `/repo/src/file.py`.")
            if not environment.dir_exists(os.path.dirname(path)):
                raise ValueError(f"Directory containing provided path: {path} does not exist.")
            if environment.dir_exists(path):
                raise ValueError(f"Path `{path}` is a directory. `create` command is only for creating new files") 
            file_text = tool_input.get("file_text", None)
            if file_text is None:
                raise ValueError("file_text is required for `create` command")

            # Create rich panel with syntax highlighting
            if show_panel:
                view_panel = create_rich_panel(
                    file_text,
                    f"📄 {os.path.basename(path)}",
                    path,
                )
                console.print(f"Valid file workdir: {workdir}")
                try:
                    console.print(view_panel)
                except Exception as e:
                    LOG.warning(f"Failed to render file contents:\n{file_text}\nDetails: {e}")
        
            # Write new content
            environment.write_file(file_text, path)
            
            result = f"New file created with contents from file_text.\nFile: {path}"

        elif mode == "str_replace":
            old_str = tool_input.get("old_str")
            new_str = tool_input.get("new_str")

            if old_str is None or new_str is None:
                raise ValueError("old_str and new_str are both required for mode `str_replace`")

            if old_str == new_str:
                raise ValueError(f"Provided old_str and new_str are exactly same. No edit performed for path: {path}")

            if environment.dir_exists(path):
                raise ValueError(f"`str_replace` mode not suitable for provided path: {path} which is a dir")

            if not environment.file_exists(path):
                raise ValueError(f"Provided file path: {path} does not exist. Make sure to use absolute path only, e.g., `/repo/file.py`")

            if not path.startswith(workdir):
                path = os.path.join(workdir, path)

            # Make backup
            backup_path = f"{path}.bak"
            command = f"cp {shlex.quote(path)} {shlex.quote(backup_path)}"
            _ = environment.execute_bash(command, workdir=workdir)
            
            command = f"cat {shlex.quote(path)}"
            content = execute_bash(command, workdir=workdir, environment=environment)

            # Strip equal leading/trailing newlines early so the same
            old_str, new_str = RefineLLMResponse._strip_equal_newlines(old_str, new_str)

            if show_panel:
                display_code_edit_blocks(old_str, new_str, path, language, grid)
            edit_result = RefineLLMResponse.normalize_and_apply(
                content, old_str, new_str
            )

            if edit_result.partial_lines:
                LOG.warning(f"old_str does not carry complete text lines from path: {path}. Expand your current context")
                raise ValueError(f"old_str does not carry complete text lines from path: {path}. Expand your current context")
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
            else:
                counts = edit_result.current_counts
                if counts > 1:
                    locations = []
                    _content = content
                    offset = 0
                    for _ct in range(counts):
                        n_match = RefineLLMResponse.get_normalized_match(_content, old_str, offset=offset)
                        start_line, end_line = n_match.start_line, n_match.end_line
                        assert start_line > 0 and end_line > 0, f"Unexpected start-line ({start_line}) or end-line ({end_line}) for valid string patch. Normalized string patch:\n{n_match.normalized_str}"
                        locations.append((start_line, end_line))
                        offset = end_line
                    
                    command = f"cat -n {shlex.quote(path)}"
                    numbered_content = execute_bash(command, workdir=workdir, environment=environment)
                    numbered_content_lines = numbered_content.splitlines()
                    locations_detail = []
                    for _ct in range(counts):
                        start_line, end_line = locations[_ct]
                        locations_detail.extend(
                            [
                                f"<--- location-{_ct+1} --->\n",
                                "\n".join(numbered_content_lines[start_line:end_line]),
                                "\n"
                            ]
                        )
                    multiple_locations_feedback = (
                        f"Provided old_str is present at {counts} locations in the file:\n"
                        + ("\n".join(locations_detail) if duplicate_locations_detail else "")
                        + "\nExpand your current context of `old_str` such that it is unique in the file"
                    )
                    if not allow_duplicate_edits:
                        complete_logging = (
                            f"Provided old_str is present at {counts} locations in the file:\n"
                            + "\n".join(locations_detail)
                            + "\nExpand your current context of `old_str` such that it is unique in the file"
                        )
                        LOG.warning(complete_logging)
                        raise ValueError(multiple_locations_feedback)
                    else:
                        new_content = edit_result.new_content
                        if content == new_content:
                            LOG.warning(f"Failed to apply feasible diff to path: {path}")

                        diff_path=path.replace(workdir, "").lstrip("/")
                        diffs = difflib.unified_diff(
                            content.splitlines(),
                            new_content.splitlines(),
                            lineterm="",
                            fromfile=f'a/{diff_path}',
                            tofile=f'b/{diff_path}',
                            n=3, 
                        )
                        diff_str = "\n".join(diffs)
                        # Write new content
                        environment.write_file(new_content, path)

                        result = (
                            "Text replacement complete."
                            +
                            (f"\nFile: {path}" if not reply_detail else "")
                            +
                            f"\nReplaced {counts} occurrence{'s' if counts > 1 else ''}\n"
                            +
                            (
                                f"\nApplied diff:\n{diff_str}\n" if reply_detail else ""
                            )
                        )
                        if show_panel:
                            try:
                                console.print(Panel(result, title="[bold green]Output", border_style="blue"))
                            except Exception as e:
                                LOG.warning(f"Failed to render text replacement result: {result}\nDetails:{e}")

                else:
                    new_content = edit_result.new_content
                    if content == new_content:
                        LOG.warning(f"Failed to apply feasible diff to path: {path}")

                    diff_path=path.replace(workdir, "").lstrip("/")
                    diffs = difflib.unified_diff(
                        content.splitlines(),
                        new_content.splitlines(),
                        lineterm="",
                        fromfile=f"a/{diff_path}",
                        tofile=f"b/{diff_path}",
                        n=3, 
                    )
                    diff_str = "\n".join(diffs)
                    # Write new content
                    environment.write_file(new_content, path)

                    result = (
                        "Text replacement complete."
                        +
                        (f"\nFile: {path}" if not reply_detail else "")
                        +
                        (
                            f"\nApplied diff:\n{diff_str}\n" if reply_detail else ""
                        )
                    )
                    if show_panel:
                        try:
                            console.print(Panel(result, title="[bold green]Output", border_style="blue"))
                        except Exception as e:
                            LOG.warning(f"Failed to render text replacement: {result}\nDetails: {e}")

        elif mode == "undo_edit":
            backup_path = f"{path}.bak"
            if not environment.file_exists(backup_path):
                raise ValueError(f"No backup file found for {path}")

            # Restore from backup
            command = f"cp {shlex.quote(backup_path)} {shlex.quote(path)}"
            _ = environment.execute_bash(command, workdir=workdir)
            command = f"rm  {shlex.quote(backup_path)}"
            _ = environment.execute_bash(command, workdir=workdir)

            formatted_output = format_output("↩️ Undo Complete", f"Successfully reverted changes to {path}", "yellow")
            console.print(formatted_output)
            result = f"Successfully reverted changes to {path}. No further backup exists for this file"

        if len(result) > max_chars_limit:
            LOG.info(f"Clipping str_replace output for mode: {mode} due to chars len ({len(result)}) exceeding limit ({max_chars_limit})")
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
            except Exception as e:
                LOG.warning(f"Error message failed to render. Msg={error_msg}. Details: {e}")
        if verbose:
            LOG.error(f"{traceback.format_exc()}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }
