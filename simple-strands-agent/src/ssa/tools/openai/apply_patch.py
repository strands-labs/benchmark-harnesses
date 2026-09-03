import os
import difflib
import logging
from typing import Any
import traceback

from rich.console import Console
from rich.panel import Panel
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment


import enum
import dataclasses
from typing import Optional
import pathlib


LOG = logging.getLogger(__name__)
console = Console()
MAX_CHARS_LIMIT: int = 20_000


TOOL_SPEC = {
    "name": "apply_patch",
    "description": (
        "Apply a patch to one or more files using the apply_patch format. "
        "The `patch` value MUST start with `*** Begin Patch` and end with `*** End Patch`. "
        "Example `patch` value:\n"
        "*** Begin Patch\n"
        "*** Update File: path/to/file.py\n"
        "@@\n"
        "- old line\n"
        "+ new line\n"
        "*** End Patch"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "description": {
                    "description": "Why I'm making this edit",
                    "type": "string"
                },
                "patch": {
                    "description": "The apply_patch command that you wish to execute.",
                    "type": "string"
                },
            },
            "required": ["patch", "description"]
        }
    }
}

class DiffError(ValueError):
    """Raised when a patch is invalid or cannot be applied."""


class ActionType(str, enum.Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclasses.dataclass
class FileChange:
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


@dataclasses.dataclass
class Commit:
    changes: dict[str, FileChange] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Chunk:
    # Line index of the first line in the original file for this chunk.
    orig_index: int = -1
    del_lines: list[str] = dataclasses.field(default_factory=list)
    ins_lines: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PatchAction:
    type: ActionType
    new_file: Optional[str] = None
    chunks: list[Chunk] = dataclasses.field(default_factory=list)
    move_path: Optional[str] = None


@dataclasses.dataclass
class Patch:
    actions: dict[str, PatchAction] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FileSnapshot:
    text: str          # normalized to LF
    newline: str       # original preferred newline style
    existed: bool = True


class SafeFileSystem:
    def __init__(
        self,
        root: str | pathlib.Path,
        env: Environment,
        restrict_workspace_create: bool = True
    ) -> None:
        self.root = pathlib.Path(root).resolve()
        self.env = env
        self.restrict_workspace_create = restrict_workspace_create

    def resolve_path(self, rel_path: str) -> pathlib.Path:
        if not rel_path:
            raise DiffError("Empty paths are not allowed")
        path = pathlib.Path(rel_path)
        if not path.is_absolute():
            resolved = (self.root / path).resolve()
        else:
            resolved = path.resolve()
        if self.restrict_workspace_create:
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise DiffError(f"Path escapes root: {rel_path}") from exc
        return resolved

    def read_snapshot(self, rel_path: str) -> FileSnapshot:
        path = self.resolve_path(rel_path)
        if self.env.dir_exists(path):
            raise DiffError(f"Expected file but found directory: {rel_path}")
        if not self.env.file_exists(path):
            raise DiffError(f"Missing file: {rel_path}. Make sure to only use absolute path, e.g., /repo/file.py. Current workdir: {self.root}")

        command = f"cat {path}"
        result = self.env.execute_bash(command=command)
        raw = result.get("output", "")

        return FileSnapshot(
            text=normalize_newlines(raw),
            newline=detect_newline_style(raw),
            existed=True,
        )

    def write_text_atomic(self, rel_path: str, content: str) -> None:
        path = self.resolve_path(rel_path)
        command = f"mkdir -p {path.parent}"
        _ = self.env.execute_bash(command=command)
        # path.parent.mkdir(parents=True, exist_ok=True)
        self.env.write_file(content, path)

    def remove_file(self, rel_path: str) -> None:
        path = self.resolve_path(rel_path)
        if not self.env.file_exists(path):
        # if not path.exists():
            raise DiffError(f"Cannot delete missing file: {rel_path}. Make sure to only use absolute path, e.g., /repo/file.py")
        if self.env.dir_exists(path):
        # if path.is_dir():
            raise DiffError(f"Cannot delete directory as file: {rel_path}")
        command = f"rm {path}"
        _ = self.env.execute_bash(command=command)


@dataclasses.dataclass
class Parser:
    current_files: dict[str, str] = dataclasses.field(default_factory=dict)
    lines: list[str] = dataclasses.field(default_factory=list)
    index: int = 0
    patch: Patch = dataclasses.field(default_factory=Patch)
    fuzz: int = 0

    def is_done(self, prefixes: Optional[tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if prefixes and self.lines[self.index].startswith(prefixes):
            return True
        return False

    def startswith(self, prefix: str | tuple[str, ...]) -> bool:
        if self.index >= len(self.lines):
            return False
        return self.lines[self.index].startswith(prefix)

    def read_str(self, prefix: str = "", return_everything: bool = False) -> str:
        if self.index >= len(self.lines):
            raise DiffError(f"Unexpected end of patch at line index {self.index}")
        line = self.lines[self.index]
        if line.startswith(prefix):
            self.index += 1
            if return_everything:
                return line
            return line[len(prefix):]
        return ""

    def parse(self, fs: SafeFileSystem) -> None:
        path = ""
        while not self.is_done(("*** End Patch",)):
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Update File Error: duplicate path: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Update File Error: missing file: {path}. Make sure to only use absolute path, e.g., /repo/file.py")

                move_to = self.read_str("*** Move to: ")
                action = self.parse_update_file(path, self.current_files[path])
                action.move_path = move_to or None
                self.patch.actions[path] = action
                continue

            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Delete File Error: duplicate path: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error: missing file: {path}. Make sure to only use absolute path, e.g., /repo/file.py")

                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            path = self.read_str("*** Add File: ")
            if path:
                # Delete and Add same path is acceptable. Equivalent to overwrite the contents
                if path in self.patch.actions:
                    if self.patch.actions[path].type == ActionType.DELETE:
                        pass
                    else:
                        raise DiffError(f"Add File Error: duplicate path: {path}")

                self.patch.actions[path] = self.parse_add_file()
                continue

            raise DiffError(f"Unknown line: {self.lines[self.index]}")

        if not self.startswith("*** End Patch"):
            raise DiffError("Missing End Patch")
        self.index += 1
        if not path:
            raise DiffError("Invalid patch. Must contain one of *** Update File, *** Delete File, *** Add File")

    def parse_update_file(self, path: str, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0

        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            def_str = self.read_str("@@ ")
            section_str = ""

            if not def_str and self.index < len(self.lines) and self.lines[self.index] == "@@":
                section_str = self.lines[self.index]
                self.index += 1

            if not (def_str or section_str or index == 0):
                raise DiffError(f"Invalid line in update hunk: {self.lines[self.index]}")

            if def_str.strip():
                found = False

                # Skip-ahead by exact section marker.
                if not any(s == def_str for s in lines[:index]):
                    for i in range(index, len(lines)):
                        if lines[i] == def_str:
                            index = i + 1
                            found = True
                            break

                # Skip-ahead by stripped section marker.
                if not found and not any(s.strip() == def_str.strip() for s in lines[:index]):
                    for i in range(index, len(lines)):
                        if lines[i].strip() == def_str.strip():
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break

            next_chunk_context, chunks, end_patch_index, eof = peek_next_section(self.lines, self.index)
            next_chunk_text = "\n".join(next_chunk_context)

            new_index, fuzz = find_context(lines, next_chunk_context, index, eof)
            if new_index == -1:
                section_header = f"@@ {def_str}" if def_str else "@@"
                # chunk_details = format_chunk_context_details(chunks)
                if eof:
                    raise DiffError(
                        f"Failed to match update chunk in {path} at file line {index + 1} onwards.\n"
                        f"Section: {section_header}\n"
                        f"Context:\n{next_chunk_text}\n"
                        "Reason: EOF context did not match."
                    )
                raise DiffError(
                    f"Failed to match update chunk in {path} at file line {index + 1} onwards.\n"
                    f"Section: {section_header}\n"
                    f"Context:\n{next_chunk_text}"
                )

            self.fuzz += fuzz
            for chunk in chunks:
                chunk.orig_index += new_index
                if fuzz >= 100:
                    _adjust_chunk_indentation(chunk, lines)
                action.chunks.append(chunk)

            index = new_index + len(next_chunk_context)
            self.index = end_patch_index

        return action

    def parse_add_file(self) -> PatchAction:
        lines: list[str] = []

        while not self.is_done(("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")):
            s = self.read_str("", return_everything=True)
            if not s.startswith("+"):
                raise DiffError(f"Invalid Add File line: {s}")
            lines.append(s[1:])

        if not lines:
            raise DiffError("Add File hunk must contain at least one '+' line")

        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def detect_newline_style(text: str) -> str:
    for i, ch in enumerate(text):
        if ch == "\n":
            if i > 0 and text[i - 1] == "\r":
                return "\r\n"
            return "\n"
        if ch == "\r":
            if i + 1 < len(text) and text[i + 1] == "\n":
                return "\r\n"
            return "\r"
    return "\n"

def restore_newlines(text: str, newline: str) -> str:
    if newline == "\n":
        return text
    return text.replace("\n", newline)

def split_patch_lines(text: str) -> list[str]:
    normalized = normalize_newlines(text)
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    if not normalized:
        return []
    return normalized.split("\n")

def collapse_adjacent_patch_envelopes(lines: list[str]) -> list[str]:
    """Merge repeated patch envelopes into one logical patch.

    Protection against models emitting multiple envelopes in a single tool call.
    The parser accepts multiple file actions inside one envelope, so adjacent
    ``End``/``Begin`` boundaries (optionally separated by blank lines) can be
    removed without changing the requested edits.
    """
    if not lines:
        return lines

    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index] == "*** End Patch":
            lookahead = index + 1
            while lookahead < len(lines) and lines[lookahead].strip() == "":
                lookahead += 1
            if lookahead < len(lines) and lines[lookahead] == "*** Begin Patch":
                index = lookahead + 1
                continue
        collapsed.append(lines[index])
        index += 1
    return collapsed

def _adjust_chunk_indentation(chunk: Chunk, file_lines: list[str]) -> None:
    """When a fuzzy (strip) match occurred, adjust ins_lines indentation to
    match the actual file indentation instead of the patch's indentation."""
    if not chunk.del_lines or not chunk.ins_lines:
        return
    if chunk.orig_index >= len(file_lines):
        return

    # Find the leftmost (least indented) non-empty del_line and its
    # corresponding file line to compute the indentation delta.
    min_indent_len = None
    min_idx = None
    for i, dl in enumerate(chunk.del_lines):
        stripped = dl.lstrip()
        if not stripped:
            continue
        indent_len = len(dl) - len(stripped)
        if min_indent_len is None or indent_len < min_indent_len:
            min_indent_len = indent_len
            min_idx = i

    if min_idx is None:
        return

    patch_line = chunk.del_lines[min_idx]
    file_idx = chunk.orig_index + min_idx
    if file_idx >= len(file_lines):
        return
    actual_line = file_lines[file_idx]

    actual_indent = actual_line[: len(actual_line) - len(actual_line.lstrip())]
    patch_indent = patch_line[: len(patch_line) - len(patch_line.lstrip())]

    if actual_indent == patch_indent:
        return

    LOG.info(f"Found mismatched indentation between model and file context, actual (#{len(actual_indent)}), model (#{patch_indent}). Correcting for insertion.")
    adjusted: list[str] = []
    for ins_line in chunk.ins_lines:
        if ins_line.startswith(patch_indent):
            adjusted.append(actual_indent + ins_line[len(patch_indent):])
        else:
            adjusted.append(ins_line)
    chunk.ins_lines = adjusted

    # Also fix del_lines so they reflect the actual file content for
    # any downstream validation that may compare them.
    adjusted_del: list[str] = []
    for del_line in chunk.del_lines:
        if del_line.startswith(patch_indent):
            adjusted_del.append(actual_indent + del_line[len(patch_indent):])
        else:
            adjusted_del.append(del_line)
    chunk.del_lines = adjusted_del

def find_context_core(lines: list[str], context: list[str], start: int) -> tuple[int, int]:
    if not context:
        return start, 0

    # Exact match.
    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0

    # Tolerate trailing whitespace differences.
    stripped_right = [s.rstrip() for s in context]
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == stripped_right:
            return i, 1

    # Tolerate surrounding whitespace differences.
    stripped_full = [s.strip() for s in context]
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == stripped_full:
            return i, 100

    return -1, 0

def find_context(lines: list[str], context: list[str], start: int, eof: bool) -> tuple[int, int]:
    if eof:
        new_index, fuzz = find_context_core(lines, context, max(0, len(lines) - len(context)))
        if new_index != -1:
            return new_index, fuzz

        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10000

    return find_context_core(lines, context, start)

def peek_next_section(lines: list[str], index: int) -> tuple[list[str], list[Chunk], int, bool]:
    old: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    chunks: list[Chunk] = []
    mode = "keep"
    orig_index = index

    while index < len(lines):
        s = lines[index]

        if s.startswith(
            (
                "@@",
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            break

        if s == "***":
            break
        if s.startswith("***"):
            raise DiffError(f"Invalid line inside chunk: {s}")

        index += 1
        last_mode = mode

        if s == "":
            s = " "

        prefix = s[0]
        if prefix == "+":
            mode = "add"
        elif prefix == "-":
            mode = "delete"
        elif prefix == " ":
            mode = "keep"
        else:
            raise DiffError(f"Invalid change line: {s}")

        payload = s[1:]

        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines = []
            ins_lines = []

        if mode == "delete":
            del_lines.append(payload)
            old.append(payload)
        elif mode == "add":
            ins_lines.append(payload)
        else:
            old.append(payload)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True

    if index == orig_index:
        bad = lines[index] if index < len(lines) else "<end>"
        raise DiffError(f"Nothing in this section: index={index}, line={bad}")

    return old, chunks, index, False

def format_chunk_context_details(chunks: list[Chunk]) -> str:
    if not chunks:
        return "<no chunk details available>"

    rendered: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        old_side = chunk.del_lines if chunk.del_lines else ["<no removed lines>"]
        rendered.append(
            f"Chunk {idx} (orig_index={chunk.orig_index}, removes={len(chunk.del_lines)}, adds={len(chunk.ins_lines)}):\n"
            + "\n".join(old_side)
        )
    return "\n\n".join(rendered)

def text_to_patch(
    text: str,
    orig: dict[str, str],
    fs: SafeFileSystem,
    collapse_envelopes: bool = False,
) -> tuple[Patch, int]:
    lines = split_patch_lines(text)
    # Tolerate two common model formatting mistakes at the tail:
    #   1) a stray heredoc terminator like "PATCH" after *** End Patch
    #   2) a duplicated "*** End Patch" line
    if len(lines) >= 2 and lines[-1] == "PATCH" and lines[-2] == "*** End Patch":
        lines = lines[:-1]
    if len(lines) >= 2 and lines[-1] == "*** End Patch" and lines[-2] == "*** End Patch":
        lines = lines[:-1]

    if collapse_envelopes:
        lines = collapse_adjacent_patch_envelopes(lines)

    if len(lines) < 2 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise DiffError("Invalid patch text")

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse(fs)

    if parser.index != len(lines):
        raise DiffError(f"Unexpected trailing patch content starting at line index {parser.index}")

    return parser.patch, parser.fuzz

def identify_files_needed(text: str) -> list[str]:
    lines = split_patch_lines(text)
    result: set[str] = set()
    for line in lines:
        if line.startswith("*** Update File: "):
            result.add(line[len("*** Update File: "):])
        elif line.startswith("*** Delete File: "):
            result.add(line[len("*** Delete File: "):])
    return sorted(result)

def get_updated_file(text: str, action: PatchAction, path: str) -> str:
    if action.type != ActionType.UPDATE:
        raise DiffError(f"Internal error: expected update action for {path}")

    orig_lines = text.split("\n")
    dest_lines: list[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"{path}: chunk.orig_index {chunk.orig_index} > file length {len(orig_lines)}"
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                f"{path}: orig_index {orig_index} > chunk.orig_index {chunk.orig_index}"
            )

        dest_lines.extend(orig_lines[orig_index:chunk.orig_index])
        orig_index = chunk.orig_index

        if chunk.ins_lines:
            dest_lines.extend(chunk.ins_lines)

        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)

def patch_to_commit(patch: Patch, orig: dict[str, str]) -> Commit:
    commit = Commit()

    for path, action in patch.actions.items():
        if action.type == ActionType.DELETE:
            commit.changes[path] = FileChange(type=ActionType.DELETE, old_content=orig[path])
        elif action.type == ActionType.ADD:
            commit.changes[path] = FileChange(type=ActionType.ADD, new_content=action.new_file or "")
        elif action.type == ActionType.UPDATE:
            new_content = get_updated_file(orig[path], action, path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
        else:
            raise DiffError(f"Unknown action type for {path}: {action.type}")

    return commit

def load_files(paths: list[str], fs: SafeFileSystem) -> dict[str, FileSnapshot]:
    snapshots: dict[str, FileSnapshot] = {}
    for path in paths:
        snapshots[path] = fs.read_snapshot(path)
    return snapshots

def apply_commit(commit: Commit, fs: SafeFileSystem, snapshots: dict[str, FileSnapshot]) -> None:
    for path, change in commit.changes.items():
        if change.type == ActionType.DELETE:
            fs.remove_file(path)
            continue

        if change.type == ActionType.ADD:
            fs.write_text_atomic(path, restore_newlines(change.new_content or "", "\n"))
            continue

        if change.type != ActionType.UPDATE:
            raise DiffError(f"Unknown commit change type for {path}: {change.type}")

        source_snapshot = snapshots.get(path)
        newline = source_snapshot.newline if source_snapshot else "\n"
        rendered = restore_newlines(change.new_content or "", newline)

        if change.move_path:
            src = fs.resolve_path(path)
            dst = fs.resolve_path(change.move_path)

            if dst.exists() and dst != src:
                raise DiffError(f"Move destination already exists: {change.move_path}")

            fs.write_text_atomic(change.move_path, rendered)
            if dst != src:
                fs.remove_file(path)
        else:
            fs.write_text_atomic(path, rendered)

def apply_patch(tool: ToolUse, **kwargs: Any) -> ToolResult:
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
    verbose = kwargs.get("verbose", True)
    environment: Environment = kwargs["environment"]
    workdir = environment.workdir or os.getcwd()
    _tool_cfg = kwargs.get("tool_params", {}).get("openai.apply_patch", {})
    restrict_workspace_create: bool = _tool_cfg.get("restrict_workspace_create", kwargs.get("restrict_workspace_create", True))
    collapse_envelopes: bool = _tool_cfg.get("collapse_envelopes", kwargs.get("collapse_envelopes", False))

    try:
        # Validate required parameters
        if not tool_input.get("patch"):
            raise ValueError("patch string is required")

        patch_text = tool_input.get("patch")
        fs = SafeFileSystem(root=workdir, env=environment, restrict_workspace_create=restrict_workspace_create)
        paths = identify_files_needed(patch_text)
        snapshots = load_files(paths, fs)
        orig = {path: snapshot.text for path, snapshot in snapshots.items()}
        patch, fuzz = text_to_patch(patch_text, orig, fs, collapse_envelopes=collapse_envelopes)
        commit = patch_to_commit(patch, orig)
        apply_commit(commit, fs, snapshots)
        result = "Success! Updated the following files:"
        
        for path, change in commit.changes.items():
            if change.type == ActionType.DELETE:
                result += f"\n- Deleted path: {path}"
            elif change.type == ActionType.ADD:
                result += f"\n- Added a new file with provided contents at path: {path}"
            elif change.type == ActionType.UPDATE:
                if change.move_path:
                    result += f"- Moved the file from old path {path} to new path: {change.move_path}"
                else:
                    new_snapshot = load_files([path], fs)[path]
                    diff_path=path.replace(workdir, "").lstrip("/")
                    diffs = difflib.unified_diff(
                        snapshots[path].text.splitlines(),
                        new_snapshot.text.splitlines(),
                        lineterm="",
                        fromfile=f'a/{diff_path}',
                        tofile=f'b/{diff_path}',
                        n=3, 
                    )
                    diff_str = "\n".join(diffs)
                    if len(diff_str) == 0:
                        raise DiffError(f"Failed to update path: {path}. Old context and new context in update hunk are exactly same.")
                    # result += f"\n- Text replacement complete at path: {path}\nApplied diff:\n{diff_str}"
                    result += f"\n- Text replacement complete at path: {path}"

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
        