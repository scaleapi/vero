"""Build the vendored opencode source into a single Linux binary, once per tree.

The editable target vendors the opencode monorepo under ``opencode/``. Each
evaluation runs the agent class in the harbor process (the evaluation host), so
the build happens there: install the workspace with Bun, compile the CLI for the
host platform, and cache the binary under a content hash of the source tree. Every
trial then uploads the cached binary into its task container. A tree that does not
compile raises here, before any trial spends model budget on a broken agent.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

BUN_VERSION = "1.3.14"  # opencode's packageManager pin
EXCLUDED_DIRS = {"node_modules", "dist", ".git", ".turbo", ".sst"}


def repo_dir() -> Path:
    """The vendored opencode checkout that ships inside this target."""
    return Path(__file__).resolve().parents[2] / "opencode"


def cache_root() -> Path:
    return Path(os.environ.get("VERO_OPENCODE_BUILD_CACHE", Path.home() / ".cache" / "vero-opencode"))


def tree_hash(root: Path) -> str:
    """Content hash of the source tree, ignoring build outputs and dependencies."""
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()[:16]


def _bun() -> str:
    found = shutil.which("bun") or str(Path.home() / ".bun" / "bin" / "bun")
    if Path(found).exists():
        return found
    subprocess.run(
        ["bash", "-c", f"curl -fsSL https://bun.sh/install | bash -s bun-v{BUN_VERSION}"],
        check=True, capture_output=True, text=True,
        env={**os.environ, "BUN_INSTALL": str(Path.home() / ".bun")},
    )
    return str(Path.home() / ".bun" / "bin" / "bun")


def _log(message: str) -> None:
    print(f"[opencode-build] {message}", flush=True)


def build_binary(root: Path | None = None) -> Path:
    """Return a compiled opencode binary for the host platform, building if needed."""
    prebuilt = os.environ.get("VERO_OPENCODE_BINARY")
    if prebuilt:
        return Path(prebuilt)
    root = root or repo_dir()
    if not (root / "package.json").exists():
        raise RuntimeError(f"no opencode source at {root}; is the submodule checked out?")
    key = tree_hash(root)
    out_dir = cache_root() / key
    binary = out_dir / "opencode"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if binary.exists():
                return binary
            bun = _bun()
            env = {
                **os.environ,
                "BUN_INSTALL_CACHE_DIR": str(cache_root() / "bun-cache"),
                # husky's `prepare` hook and interactive prompts have no place here
                "HUSKY": "0", "CI": "1",
            }
            _log(f"tree {key}: bun install")
            subprocess.run([bun, "install", "--frozen-lockfile"], cwd=root, check=True,
                           capture_output=True, text=True, env=env)
            _log(f"tree {key}: compile for {platform.system().lower()}-{platform.machine()}")
            subprocess.run([bun, "run", "script/build.ts", "--single", "--skip-embed-web-ui", "--skip-install"],
                           cwd=root / "packages" / "opencode", check=True, capture_output=True, text=True, env=env)
            built = sorted((root / "packages" / "opencode" / "dist").glob("opencode-*/bin/opencode"))
            if not built:
                raise RuntimeError("opencode build produced no binary under packages/opencode/dist")
            shutil.copy2(built[0], binary)
            binary.chmod(0o755)
            _log(f"tree {key}: cached {binary}")
            return binary
        except subprocess.CalledProcessError as error:
            tail = (error.stderr or error.stdout or "")[-4000:]
            raise RuntimeError(f"opencode build failed ({error.cmd[0]} {error.cmd[1]}):\n{tail}") from error
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
