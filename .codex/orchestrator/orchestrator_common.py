#!/usr/bin/env python3
from __future__ import annotations

import datetime as _dt
import fcntl
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


class OrchestratorError(RuntimeError):
    pass


def load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise OrchestratorError(f"invalid TOML root: {path}")
    return data


def section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    raw = cfg.get(name)
    if isinstance(raw, dict):
        return raw
    return {}


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


def as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def as_list_of_int(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out or list(default)

    s = str(value).strip()
    if not s:
        return list(default)
    s = s.removeprefix("[").removesuffix("]")
    out = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out or list(default)


def as_list_of_str(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or list(default)
    s = str(value).strip()
    if not s:
        return list(default)
    s = s.removeprefix("[").removesuffix("]")
    out = []
    for token in s.split(","):
        token = token.strip().strip('"').strip("'")
        if token:
            out.append(token)
    return out or list(default)


def ensure_command(cmd: str) -> None:
    if shutil.which(cmd) is not None:
        return
    raise OrchestratorError(f"required command not found: {cmd}")


def run_cmd(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        capture_output=capture_output,
        cwd=str(cwd) if cwd else None,
    )


def run_cmd_text(args: list[str], *, check: bool = True, default: str = "") -> str:
    cp = run_cmd(args, check=False, capture_output=True)
    if cp.returncode != 0:
        if check:
            raise OrchestratorError(
                f"command failed ({cp.returncode}): {' '.join(shlex.quote(x) for x in args)}\n{cp.stderr.strip()}"
            )
        return default
    return cp.stdout.strip()


def safe_key(text: str) -> str:
    return re.sub(r"[\/ :@]", "_", text)


def extract_first_url(text: str) -> str:
    m = re.search(r"https://\S+", text)
    return m.group(0) if m else ""


def extract_pr_number_from_url(url: str) -> str:
    m = re.search(r"/pull/(\d+)", url)
    return m.group(1) if m else ""


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        data[k.strip()] = v
    return data


def has_state_in_env(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("state="):
                return True
    return False


def file_mtime_epoch(path: Path) -> int:
    try:
        return int(path.stat().st_mtime)
    except FileNotFoundError:
        return 0


def now_epoch() -> int:
    return int(_dt.datetime.now().timestamp())


def iso_now() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def log_event(path: Path, level: str, event: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{iso_now()}\t{level}\t{event}\t{detail}\n")


def send_tmux_command(pane_id: str, args: list[str]) -> None:
    command = shlex.join(args)
    run_cmd(["tmux", "send-keys", "-t", pane_id, "-l", "--", command], check=True)
    run_cmd(["tmux", "send-keys", "-t", pane_id, "Enter"], check=True)


def tmux_interrupt_pane(pane_id: str) -> None:
    run_cmd(["tmux", "send-keys", "-t", pane_id, "C-c"], check=False)


def acquire_lock(lock_path: Path) -> Any:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise OrchestratorError(f"failed to acquire lock: {lock_path}") from exc
    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()}\n")
    fh.flush()
    return fh
