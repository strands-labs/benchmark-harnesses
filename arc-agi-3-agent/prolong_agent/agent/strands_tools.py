"""The PRO-LONG tool space, implemented as Strands tools.

PRO-LONG's headline ablation is over the tool space, not the prompt:

    Read only ............. 23.1
    + Grep / Regex ........ 27.2
    + Python .............. 38.3   <- the biggest single jump
    + Write, Edit ......... 41.2

So these six functions *are* the independent variable. Three properties matter as
much as their existence:

1. **Per-agent binding.** `make_prolong_tools(workspace)` closes the workspace over
   each tool. It cannot be a module global (PRO-LONG's Swarm runs one thread per
   game, so parallel games would write into each other's directories) and it cannot
   be thread-local either (Strands runs tools on its own worker threads, which never
   see the caller's binding).
2. **Bounded output.** Reference logs reach 320,000 lines. A tool that returned a
   whole file would blow the context and defeat keeping history out of the prompt.
3. **Sealed shell.** `bash` runs under bubblewrap with only the workspace bound and
   no network, so it cannot read the ARC game implementations or fetch a walkthrough.

`/workspace/...` is accepted everywhere and mapped to the real directory, because
that is the path the prompt uses and what the shell sees inside the mount.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from strands import tool

MAX_OUTPUT_CHARS = 30_000
MAX_READ_LINES = 2_000
MAX_GREP_MATCHES = 200
BASH_TIMEOUT_S = 120

# `python3` is deliberately allowed: it is the tool the ablation shows carries the effect.
_BASH_DENY = re.compile(
    r"\b(curl|wget|nc|ncat|telnet|ssh|scp|sftp|pip|pip3|uv|npm|apt|apt-get|"
    r"docker|git|aws|sudo)\b"
)

_BWRAP = shutil.which("bwrap")
_BWRAP_BASE = [
    "--ro-bind", "/usr", "/usr",
    "--ro-bind", "/bin", "/bin",
    "--ro-bind", "/lib", "/lib",
    "--ro-bind", "/lib64", "/lib64",
    "--proc", "/proc",
    "--dev", "/dev",
    "--tmpfs", "/tmp",
    "--unshare-all",
    "--die-with-parent",
    "--new-session",
]


def _clip(text: str, what: str = "output") -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        f"{text[:MAX_OUTPUT_CHARS]}\n\n[... {what} truncated: {len(text):,} chars total, "
        f"showing first {MAX_OUTPUT_CHARS:,}. Narrow the request — grep for a pattern, "
        f"or read a line range.]"
    )


def make_prolong_tools(workspace: str | Path) -> list:
    """Build the six tools bound to `workspace`. One set per agent instance."""
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)

    def resolve(path: str) -> Path:
        raw = str(path)
        for prefix in ("/workspace/", "/workspace"):
            if raw == prefix.rstrip("/") or raw.startswith(prefix):
                raw = raw[len(prefix):].lstrip("/")
                break
        p = Path(raw)
        full = (ws / p).resolve() if not p.is_absolute() else p.resolve()
        if full != ws and ws not in full.parents:
            raise PermissionError(
                f"path escapes workspace: {path!r}. Everything you need is in the "
                f"workspace; use relative paths or /workspace/..."
            )
        return full

    @tool
    def read_file(path: str, offset: int = 0, limit: int = MAX_READ_LINES) -> str:
        """Read a text file from the workspace, with line numbers.

        Args:
            path: File path, e.g. "logs.txt" or "/workspace/logs.txt".
            offset: 0-based line to start at. Page through big files with limit.
            limit: Maximum lines to return (capped at 2000).
        """
        full = resolve(path)
        if not full.is_file():
            return f"no such file: {path}"
        lim = max(1, min(int(limit), MAX_READ_LINES))
        off = max(0, int(offset))
        out, total = [], 0
        with full.open(errors="replace") as fh:
            for i, line in enumerate(fh):
                total = i + 1
                if i >= off and len(out) < lim:
                    out.append(f"{i + 1}\t{line.rstrip()}")
        return _clip(f"{path}: lines {off + 1}-{off + len(out)} of {total}\n" + "\n".join(out))

    @tool
    def write_file(path: str, content: str) -> str:
        """Write (or overwrite) a text file in the workspace.

        Args:
            path: File path, e.g. "actions.json" or "/workspace/actions.json".
            content: Full file contents.
        """
        full = resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"wrote {path} ({len(content):,} chars)"

    @tool
    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace an exact substring in a workspace file.

        Args:
            path: File path in the workspace.
            old: Exact text to find; must be unique unless replace_all.
            new: Replacement text.
            replace_all: Replace every occurrence.
        """
        full = resolve(path)
        if not full.is_file():
            return f"no such file: {path}"
        text = full.read_text(errors="replace")
        n = text.count(old)
        if n == 0:
            return f"not found in {path}: {old[:120]!r}"
        if n > 1 and not replace_all:
            return f"{old[:80]!r} appears {n} times; pass replace_all=True or add context"
        full.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1))
        return f"edited {path}"

    @tool
    def grep(pattern: str, path: str = ".", max_matches: int = MAX_GREP_MATCHES) -> str:
        """Search workspace files for a regular expression, returning matching lines.

        The primary way to query a long game log — e.g. every score change, or every
        board where a given cell held a given value.

        Args:
            pattern: Python regular expression.
            path: File or directory in the workspace.
            max_matches: Maximum matching lines (capped at 200).
        """
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"bad regex: {exc}"
        root = resolve(path)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        cap = max(1, min(int(max_matches), MAX_GREP_MATCHES))
        hits, total = [], 0
        for f in files:
            try:
                with f.open(errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            total += 1
                            if len(hits) < cap:
                                hits.append(f"{f.relative_to(ws)}:{i}:{line.rstrip()[:400]}")
            except OSError:
                continue
        if not hits:
            return f"no matches for {pattern!r} in {path}"
        note = f"\n[{total:,} matches total; showing {len(hits)}]" if total > len(hits) else ""
        return _clip("\n".join(hits) + note, "matches")

    @tool
    def glob_files(pattern: str = "*") -> str:
        """List workspace files matching a glob pattern, with sizes and line counts.

        Args:
            pattern: Glob pattern, e.g. "*.py" or "**/*.json".
        """
        found = sorted(p for p in ws.glob(pattern) if p.is_file())
        if not found:
            return f"no files match {pattern!r}"
        return _clip("\n".join(
            f"{p.relative_to(ws)}\t{p.stat().st_size:,} bytes" for p in found
        ))

    @tool
    def bash(command: str) -> str:
        """Run a shell command in the workspace. Use for python3, wc, head, tail, sed, awk.

        There is no network and no package installation, and paths outside the
        workspace are not readable.

        Args:
            command: Shell command, e.g. "wc -l logs.txt" or "python3 analyse.py".
        """
        denied = _BASH_DENY.search(command)
        if denied:
            return (
                f"blocked: {denied.group(0)!r} is not available. No network or package "
                "installs. Use python3 with the standard library on workspace files."
            )
        if not _BWRAP:
            return "blocked: sandbox unavailable (bwrap missing); use the file tools."
        sandbox_args = [*_BWRAP_BASE, "--bind", str(ws), "/workspace",
                        "--chdir", "/workspace", "/bin/sh", "-c", command]
        # The program executed is always bwrap (`executable=_BWRAP`, resolved once by
        # shutil.which at import); `command` is not run by a host shell. shell=False
        # and the arguments are a list, so nothing re-parses them here -- the command
        # is first interpreted by /bin/sh *inside* the sandbox, which has no network,
        # a read-only system, and only this game's workspace writable.
        try:
            proc = subprocess.run(
                ["bwrap", *sandbox_args], executable=_BWRAP,
                cwd=str(ws), capture_output=True, text=True,
                timeout=BASH_TIMEOUT_S,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/workspace", "LC_ALL": "C"},
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {BASH_TIMEOUT_S}s: {command}"
        parts = []
        if proc.stdout:
            parts.append(proc.stdout.rstrip())
        if proc.stderr:
            parts.append("[stderr]\n" + proc.stderr.rstrip())
        if proc.returncode != 0:
            parts.append(f"[exit {proc.returncode}]")
        return _clip("\n".join(parts) or "(no output)")

    return [read_file, write_file, edit_file, grep, glob_files, bash]
