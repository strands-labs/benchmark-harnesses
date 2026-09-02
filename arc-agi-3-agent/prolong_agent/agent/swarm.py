# Modified from PRO-LONG (https://github.com/alexisfox7/PRO-LONG), MIT licensed.
"""Run one scorecard across multiple games in parallel threads.

Usage:
    prolong_agent-swarm --suite all --max-actions 500
    prolong_agent-swarm --game ls20,ft09
"""
from __future__ import annotations

import argparse
import json
import logging

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

import arc_agi
from arc_agi import OperationMode

from prolong_agent.environment.runner import GameRunner
from prolong_agent.environment import ArcAgi3Env
from prolong_agent.environment.config import EVALUATION_GAMES
from prolong_agent.metrics.structures import GameMetrics, Status
from prolong_agent.metrics.reporting import generate_console_report, save_summary_report, calculate_stats

log = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
load_dotenv(dotenv_path=_project_root / ".env", override=False)

# Must be the `three.` host. The bare apex proxies /api/cmd/* (so actions appear to
# work) but 404s /api/scorecard/close, which silently loses the scorecard at the end
# of a run: the 0812 run's 98.33 was only recoverable from the web dashboard because
# of this. runner.py has always had the correct default; this one disagreed.
ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")


class Swarm:
    """Manages a single scorecard and runs one agent per game in daemon threads."""

    def __init__(
        self,
        inner_agent_kwargs: dict[str, Any],
        arcade: arc_agi.Arcade,
        games: list[str],
        tags: list[str],
        max_actions: int = 500,
        analyzer_hook: Any = None,
        prompts_log_dir: Path | None = None,
        log_post_board: bool = True,
        analyzer_retries: int = 5,
        baseline: bool = False,
        resume: bool = False,
        analyzer_agent: Any = None,
        stateless: bool = False,
    ) -> None:
        self.inner_agent_kwargs = inner_agent_kwargs
        self._arcade = arcade
        self.games = games
        self.tags = tags
        self.max_actions = max_actions
        self.analyzer_hook = analyzer_hook
        self.prompts_log_dir = prompts_log_dir
        self.log_post_board = log_post_board
        self.baseline = baseline
        self.analyzer_retries = analyzer_retries
        self.resume = resume
        self.analyzer_agent = analyzer_agent
        self.stateless = stateless

        self.card_id: str | None = None
        self.scorecard: Any = None
        self.results: dict[str, GameMetrics] = {}
        self._lock = threading.Lock()

    def run(self) -> dict[str, GameMetrics]:
        self.card_id = self._arcade.open_scorecard(tags=self.tags)
        log.info("Opened scorecard %s for %d game(s)", self.card_id, len(self.games))

        # Persist the API session cookies next to the run: arcprize scopes
        # scorecard GET/close to the cookies issued at open, so a crashed run's
        # card can only be recovered/closed with these (analysis/close_card.py).
        try:
            jar = getattr(self._arcade, "_master_cookie_jar", None)
            if jar is not None and self.prompts_log_dir:
                (Path(self.prompts_log_dir) / "card_session.json").write_text(
                    json.dumps({"card_id": self.card_id, "cookies": dict(jar)}))
        except Exception:
            log.warning("could not persist scorecard session cookies", exc_info=True)

        threads = [
            threading.Thread(target=self._run_game, args=(self.card_id, gid), daemon=True)
            for gid in self.games
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.scorecard = self._arcade.close_scorecard(self.card_id)
        log.info("Closed scorecard %s", self.card_id)
        return self.results

    def _run_game(self, card_id: str, game_id: str) -> None:
        try:
            env = ArcAgi3Env.from_arcade(
                arcade=self._arcade, game_id=game_id,
                scorecard_id=card_id, max_actions=self.max_actions,
            )

            prompts_log_path = None
            if self.prompts_log_dir:
                game_dir = self.prompts_log_dir / game_id.split("-")[0]
                game_dir.mkdir(parents=True, exist_ok=True)
                prompts_log_path = game_dir / "logs.txt"
                if not self.resume:
                    prompts_log_path.write_text("")
                elif not prompts_log_path.exists():
                    log.warning("resume: %s has no logs.txt — nothing to resume", game_dir)
                    prompts_log_path.write_text("")

            # When resuming, restore any persisted analyzer session before the
            # runner starts so the first analyzer call uses --session --continue.
            if self.resume and prompts_log_path is not None and self.analyzer_hook is not None:
                owner = getattr(self.analyzer_hook, "__self__", None)
                prime = getattr(owner, "prime_session_from_disk", None) if owner else None
                if callable(prime):
                    try:
                        prime(prompts_log_path)
                    except Exception as exc:
                        log.warning("resume: prime_session_from_disk failed: %s", exc)

            runner = GameRunner(
                env=env,
                game_id=game_id,
                agent_name=self.inner_agent_kwargs.get("name", "swarm_agent"),
                max_actions_per_game=self.max_actions,
                tags=self.tags,
                prompts_log_path=prompts_log_path,
                analyzer=self.analyzer_hook,
                analyzer_agent=self.analyzer_agent,
                log_post_board=self.log_post_board,
                analyzer_retries=self.analyzer_retries,
                agent_kwargs=self.inner_agent_kwargs,
                baseline=self.baseline,
                resume=self.resume,
                stateless=self.stateless,
            )
            metrics = runner.run()

            with self._lock:
                self.results[game_id] = metrics

        except Exception as exc:
            log.error("Game %s failed: %s", game_id, exc, exc_info=True)
            with self._lock:
                self.results[game_id] = GameMetrics(
                    game_id=game_id,
                    agent_name=self.inner_agent_kwargs.get("name", "swarm_agent"),
                    start_time=time.time(),
                    status=Status.ERROR,
                    error_message=str(exc),
                )
        finally:
            try:
                env.close()
            except Exception:
                pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("arc_agi").propagate = False

    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 Swarm evaluation.")
    parser.add_argument("--agent", "-a", default="prolong_agent")
    parser.add_argument("--game", "-g",
                        help="Comma-separated game IDs (e.g. ls20-cb3b57cc,ft09-9ab2447a).")
    parser.add_argument("--suite", "-s", choices=list(EVALUATION_GAMES.keys()))
    parser.add_argument("--tags", "-t", help="Comma-separated tags.")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--operation-mode", default="online",
                        choices=["normal", "online", "offline", "competition"],
                        help="competition = OperationMode.COMPETITION, required for the ARC "
                             "community leaderboard. Sets competition_mode=True when the "
                             "scorecard is opened, so it must be chosen before the run starts. "
                             "Forces: single make() per environment, level-resets only, one "
                             "scorecard, scored against all available environments, and no "
                             "in-flight get_scorecard (so SIGUSR2 stops reporting a score).")
    parser.add_argument("--model", "-m", dest="analyzer_model", default="claude-opus-4-6",
                        help="Analyzer model name")
    parser.add_argument("--retries", dest="analyzer_retries", type=int, default=5,
                        help="Max analyzer retry attempts")
    parser.add_argument("--backend", default="strands",
                        choices=["strands"],
                        help="Agent backend: strands (Strands Agents SDK on Amazon Bedrock)")
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="Reasoning effort passed to the model")
    parser.add_argument("--reasoning-effort", default="none",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort level")
    parser.add_argument("--session-mode", dest="session_mode", default="resume",
                        choices=["resume"],
                        help="Fixed to resume (codex exec once, then exec resume). "
                             "The other modes (fresh/reprime/clear/summary) were "
                             "removed as a confusing footgun — kept only for back-compat.")
    parser.add_argument("--workspace", dest="workspace", default="persistent",
                        choices=["persistent", "stateless"],
                        help="persistent=default (/workspace files accumulate across "
                             "turns). stateless=ablation: every turn wipe all sandbox "
                             "files except logs.txt/AGENTS.md AND suppress [PLAN] from "
                             "logs.txt -> the only cross-turn memory is the objective "
                             "game trace (boards/actions/scores).")
    parser.add_argument("--clear-every", dest="clear_every", type=int, default=15,
                        help="For session_mode=clear/summary: force a fresh "
                             "codex exec every N analyzer calls. Default 15.")
    parser.add_argument("--clear-every-actions", dest="clear_every_actions",
                        type=int, default=None,
                        help="For session_mode=clear/summary: force a fresh "
                             "session every N actions (overrides --clear-every).")
    parser.add_argument("--extra-system-prompt", dest="extra_system_prompt", default=None,
                        help="Append this text to the system prompt on every analyzer call")
    parser.add_argument("--user-prompt-prepend", dest="user_prompt_prepend", default=None,
                        help="Prepend this text to the user prompt on every analyzer call")
    parser.add_argument("--user-prompt-inject-every", dest="user_prompt_inject_every", type=int, default=None,
                        help="If set with --user-prompt-prepend, only prepend once every N actions")
    parser.add_argument("--grid-mode", default="hex", choices=["ascii", "hex", "num"],
                        help="Board representation: hex (default), ascii, or num")
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline mode: agent sees current board only, no full log")
    parser.add_argument("--log-window", dest="log_window", type=int, default=None,
                        help="Log context mode: None=full log, 0=log with current board only, N>0=last N action sections, -1=inject board into prompt (no log file in workspace)")
    parser.add_argument("--action-cap", dest="action_cap", type=int, default=15,
                        help="Max actions per analyzer plan (default 15)")
    parser.add_argument("--note", default="",
                        help="Short description of this run's purpose (saved to run_info.txt)")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None,
                        help="Resume an interrupted run. Path to an existing "
                             "evaluation_results/<timestamp>_swarm_* directory. "
                             "Replays logged actions on a fresh offline env and "
                             "restores any persisted analyzer session. Forces "
                             "--operation-mode offline.")

    args = parser.parse_args()

    resume_run_dir: Path | None = None
    if args.resume:
        resume_run_dir = Path(args.resume).resolve()
        if not resume_run_dir.exists() or not resume_run_dir.is_dir():
            log.error("resume: %s does not exist or is not a directory", resume_run_dir)
            sys.exit(1)
        # Force offline mode — online resume is unsupported because arcprize.org
        # does not let us re-attach to an existing guid.
        if args.operation_mode != "offline":
            log.info("resume: forcing --operation-mode offline")
            args.operation_mode = "offline"
        # If the user didn't explicitly pass --game or --suite, infer the games
        # from the subdirectories of the resume dir.
        if not args.game and not args.suite:
            detected = sorted(
                p.name for p in resume_run_dir.iterdir()
                if p.is_dir() and (p / "logs.txt").exists()
            )
            if detected:
                args.game = ",".join(detected)
                log.info("resume: detected games %s", args.game)

    # Resolve game list — support short names (e.g. "ls20" -> "ls20-cb3b57cc")
    all_known = {gid for ids in EVALUATION_GAMES.values() for gid in ids}
    prefix_map = {gid.split("-")[0]: gid for gid in all_known}

    games: list[str] = []
    if args.game:
        raw = [g.strip() for g in args.game.split(",") if g.strip()]
        games = [prefix_map.get(g, g) for g in raw]
    elif args.suite:
        games = EVALUATION_GAMES[args.suite]
    else:
        api_key = os.getenv("ARC_API_KEY", "")
        try:
            resp = requests.get(
                f"{ROOT_URL}/api/games",
                headers={"X-API-Key": api_key, "Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            games = [g["game_id"] for g in resp.json()]
            log.info("Fetched %d games from API", len(games))
        except Exception as exc:
            log.error("Failed to fetch games from API: %s", exc)
            sys.exit(1)

    if not games:
        log.error("No games to run. Provide --game, --suite, or set ARC_API_KEY.")
        sys.exit(1)

    # Competition Mode scores against EVERY available environment, "even if you choose
    # not to interact with them" — so any game we skip counts as a zero rather than
    # being excluded. The default path (no --game/--suite) fetches the full set and is
    # therefore safe, but a subset run silently dilutes the score, which is expensive
    # to discover after a 10-hour run. Check the set we are about to play against what
    # the API currently offers. Note the offered set is not frozen: 16 of the 25 games
    # the 0809 run played have since been re-published under new instance hashes.
    if args.operation_mode == "competition":
        try:
            _resp = requests.get(
                f"{ROOT_URL}/api/games",
                headers={"X-API-Key": os.getenv("ARC_API_KEY", ""),
                         "Accept": "application/json"},
                timeout=15,
            )
            _resp.raise_for_status()
            _available = {g["game_id"] for g in _resp.json()}
        except Exception as exc:
            log.warning("competition mode: could not verify the available game set: %s", exc)
            _available = set()
        if _available:
            _missing = sorted(_available - set(games))
            if _missing:
                log.warning(
                    "COMPETITION MODE SCORE DILUTION: playing %d of %d available "
                    "environments. The %d unplayed will score 0 and are still counted, "
                    "so the ceiling is %.1f%%, not 100%%. Unplayed: %s",
                    len(games), len(_available), len(_missing),
                    100.0 * len(games) / len(_available), ", ".join(_missing),
                )
            else:
                log.info("competition mode: playing all %d available environments",
                         len(_available))

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    tags.append(f"swarm-{args.agent}")

    arcade = arc_agi.Arcade(
        arc_api_key=os.getenv("ARC_API_KEY", ""),
        arc_base_url=ROOT_URL,
        operation_mode=OperationMode(args.operation_mode),
    )

    _wl = "full" if args.log_window is None else f"last {args.log_window}" if args.log_window > 0 else "current only"
    _clear_desc = (
        f"every {args.clear_every_actions} actions"
        if args.clear_every_actions is not None
        else f"every {args.clear_every} calls"
    )

    from prolong_agent.agent.strands_agent import StrandsAgent
    agent = StrandsAgent(
        model_id=args.analyzer_model,
        region=os.environ.get("AWS_REGION", "us-west-2"),
        action_cap=args.action_cap,
        log_window=args.log_window,
        effort=args.effort,
        grid_mode=args.grid_mode,
        in_prompt_board=args.baseline,
    )
    log.info("backend=strands model=%s effort=%s grid=%s log_window=%s",
             args.analyzer_model, args.effort, args.grid_mode, args.log_window)

    if _HAS_WANDB and os.environ.get("WANDB_API_KEY"):
        _wb_tags = [args.backend, args.analyzer_model, args.grid_mode]
        if args.note:
            _wb_tags.append(args.note)
        wandb.init(
            project="rgb-agent",
            name=args.note or None,
            config={
                "agent": args.agent,
                "backend": args.backend,
                "model": args.analyzer_model,
                "games": ",".join(games),
                "max_actions": args.max_actions,
                "grid_mode": args.grid_mode,
                "log_window": args.log_window,
                "baseline": args.baseline,
                "operation_mode": args.operation_mode,
                "note": args.note,
            },
            tags=_wb_tags,
        )
        # Make `action` the x-axis for the top-level score/level/max_score_so_far
        # panels so the workspace default view shows score-vs-action curves.
        wandb.define_metric("action")
        wandb.define_metric("score", step_metric="action")
        wandb.define_metric("level", step_metric="action")
        wandb.define_metric("max_score_so_far", step_metric="action")
        log.info("wandb initialized")

    if resume_run_dir is not None:
        run_dir = resume_run_dir
        log.info("resume: reusing run directory %s", run_dir)
    else:
        timestamp = datetime.now().strftime("%m%dT%H%M%S")
        note_slug = f"__{args.note.replace(' ', '-')}" if args.note else ""
        run_dir = Path("evaluation_results") / f"{timestamp}_swarm_{args.agent}{note_slug}"
        run_dir.mkdir(parents=True, exist_ok=True)

    inner_agent_kwargs: dict[str, Any] = {
        "name": args.agent,
        "grid_mode": args.grid_mode,
    }

    # Record exactly which game instances this run resolved to. ARC re-publishes
    # games under new instance hashes and withdraws the old sources (16 of the 25
    # the 0809 run played now 404), so without this a run cannot be interpreted
    # after the fact: the baselines that set the score are per-instance.
    try:
        manifest = {
            "resolved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model": args.analyzer_model,
            "backend": args.backend,
            "max_actions": args.max_actions,
            "resolved_from": ("--game" if args.game else
                              "--suite" if args.suite else "api:/api/games"),
            "game_count": len(games),
            "games": [],
        }
        by_id = {}
        try:
            for e in arcade.available_environments:
                gid = getattr(e, "game_id", None)
                if gid:
                    by_id[gid] = list(getattr(e, "baseline_actions", []) or [])
        except Exception:
            pass
        for gid in games:
            bl = by_id.get(gid, [])
            manifest["games"].append({
                "game_id": gid,
                "levels": len(bl),
                "baseline_actions": bl,
            })
        manifest["total_levels"] = sum(g["levels"] for g in manifest["games"])
        (run_dir / "games_manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("manifest: %d games, %d levels -> %s",
                 manifest["game_count"], manifest["total_levels"],
                 run_dir / "games_manifest.json")
    except Exception as exc:
        log.warning("could not write games_manifest.json: %s", exc)

    swarm = Swarm(
        inner_agent_kwargs=inner_agent_kwargs,
        arcade=arcade, games=games, tags=tags,
        max_actions=args.max_actions,
        analyzer_hook=agent.analyze,
        analyzer_agent=agent,
        prompts_log_dir=run_dir,
        log_post_board=True,
        analyzer_retries=args.analyzer_retries,
        baseline=args.baseline,
        resume=resume_run_dir is not None,
        stateless=(args.workspace == "stateless"),
    )

    runner = threading.Thread(target=swarm.run, daemon=True)
    runner.start()

    def sigint_handler(sig: int, frame: Any) -> None:
        print("[Swarm] SIGINT received — cleaning up...", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    def sigusr1_handler(sig: int, frame: Any) -> None:
        """Close scorecard early and save to disk. Game continues."""
        if not swarm.card_id:
            print("[Swarm] SIGUSR1: no scorecard to close", flush=True)
            return
        try:
            sc = swarm._arcade.close_scorecard(swarm.card_id)
            if sc:
                swarm.scorecard = sc
                sc_path = run_dir / "scorecard.json"
                sc_path.write_text(sc.model_dump_json(indent=2))
                print(f"[Swarm] SIGUSR1: scorecard closed & saved (score={sc.score:.1f})", flush=True)
            else:
                print("[Swarm] SIGUSR1: close_scorecard returned None", flush=True)
        except Exception as e:
            print(f"[Swarm] SIGUSR1: failed to close scorecard: {e}", flush=True)

    signal.signal(signal.SIGUSR1, sigusr1_handler)

    def sigusr2_handler(sig: int, frame: Any) -> None:
        """Health-check scorecard (GET, non-destructive)."""
        if not swarm.card_id:
            print("[Swarm] SIGUSR2: no scorecard", flush=True)
            return
        try:
            sc = swarm._arcade.get_scorecard(swarm.card_id)
            print(f"[Swarm] SIGUSR2: scorecard alive (score={sc.score:.1f}, actions={sc.total_actions})", flush=True)
        except Exception as e:
            print(f"[Swarm] SIGUSR2: scorecard dead or unreachable: {e}", flush=True)

    signal.signal(signal.SIGUSR2, sigusr2_handler)

    while runner.is_alive():
        runner.join(timeout=1)

    results_list = list(swarm.results.values())

    print(f"\nScorecard ID: {swarm.card_id}")
    print(f"Results:      {run_dir}")
    for m in sorted(results_list, key=lambda r: r.game_id):
        if m.replay_url:
            print(f"  Replay:     {m.replay_url}")

    if swarm.scorecard:
        sc = swarm.scorecard
        print(f"\n{'='*60}")
        print(f"ARC Scorecard  —  overall score: {sc.score:.1f}")
        print(f"  Environments: {sc.total_environments_completed}/{sc.total_environments}")
        print(f"  Levels:       {sc.total_levels_completed}/{sc.total_levels}")
        print(f"  Actions:      {sc.total_actions}")
        for env in sc.environments:
            run = env.runs[0] if env.runs else None
            if not run:
                continue
            label = env.id or "unknown"
            state = run.state.name if run.state else "?"
            print(f"\n  {label}  score={run.score:.1f}  state={state}  actions={run.actions}")
            if run.level_scores:
                for i, (ls, la, lb) in enumerate(zip(
                    run.level_scores,
                    run.level_actions or [],
                    run.level_baseline_actions or [],
                )):
                    baseline = str(lb) if lb >= 0 else "n/a"
                    print(f"    Level {i+1}: efficiency={ls:.1f}  actions={la}  baseline={baseline}")
            if run.message:
                print(f"    Note: {run.message}")
        print(f"{'='*60}")

        scorecard_path = run_dir / "scorecard.json"
        scorecard_path.write_text(sc.model_dump_json(indent=2))
        log.info("Scorecard saved to %s", scorecard_path)

    if results_list:
        generate_console_report(results_list, "swarm", args.agent, 1, scorecard=swarm.scorecard)
        game_stats, overall = calculate_stats(results_list)
        summary_path = run_dir / "summary.txt"
        save_summary_report(
            str(summary_path), game_stats, overall, results_list,
            args.agent, "swarm", 1, scorecard=swarm.scorecard,
        )
        log.info("Summary saved to %s", summary_path)
    else:
        log.error("No results collected.")


if __name__ == "__main__":
    main()
