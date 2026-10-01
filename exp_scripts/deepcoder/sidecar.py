"""Sidecar for the formal SAO DeepCoder run. Two jobs, both read-only toward Reef.

1. Progress file for stream.py's TrainerPacer (SAO_PROGRESS_FILE). The upstream
   driver leaves writing it to "whoever can see the trainer's checkpoints"; the
   number it needs is optimizer steps done, which is exactly the count
   stream.wait_for_training uses: releases with operation == "training" on the
   scenario's chain. Written atomically, only when it changes, so its mtime is
   the pacer's idle clock.

2. Keep checkpoints past Reef's retention. The backend writes a full bundle
   (adapter + actor + critic Megatron states, ~16.3 GB) every step and Reef's
   'latest' policy, under the storage cap, deletes all but the newest ~15. So:
     - every ADAPTER_EVERY-th step's LoRA adapter (127 MB) -> KEEP/adapters/hf_rollout_NNNNN
     - every FULL_EVERY-th step's full bundle (16.3 GB)   -> KEEP/full/step_NNN/{hf,actor,critic}
     - FINAL_STEP (if set) gets both, so the last step is kept off-cadence
   SIDECAR_FULL_EVERY=0 keeps adapters only: with the backend's adapter-only
   checkpoints (--reef-checkpoint-adapter-only) there is no Megatron state to copy.
   KEEP is outside the managed tree (inside it, the checkpoint preflight rejects
   unowned assets and refuses to start). Checkpoint files are root-owned and
   not world-readable, so discovery happens on the host but the copy runs as
   root inside the stack container, into KEEP mounted at /var/lib/reef-kept,
   and the copy is made world-readable for later evaluation.

   Step numbering: rollout_id is 0-based, so step = rollout_id + 1 and the
   "every 10 steps" bundles are rollout_ids 9, 19, ..., 189.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
SCENARIO = os.environ.get("SAO_SCENARIO", "sao-deepcoder")
PROGRESS_FILE = Path(os.environ["SAO_PROGRESS_FILE"])
HF_DIR = Path(os.environ["SIDECAR_HF_DIR"])            # <state>/checkpoints/hf
KEEP_DIR = Path(os.environ["SIDECAR_KEEP_DIR"])        # outside the checkpoint tree (host path)
CONTAINER = os.environ.get("SIDECAR_CONTAINER", "reef-sao-stack")
FULL_EVERY = int(os.environ.get("SIDECAR_FULL_EVERY", "10"))   # 0 = never (adapter-only checkpoints)
ADAPTER_EVERY = int(os.environ.get("SIDECAR_ADAPTER_EVERY", "1"))   # 1 = every step
FINAL_STEP = int(os.environ.get("SIDECAR_FINAL_STEP", "0"))          # also keep this step (0 = none)
C_CKPT = "/var/lib/reef/checkpoints"                    # the same trees, as the container sees them
C_KEEP = "/var/lib/reef-kept"
LOG = Path(os.environ["SIDECAR_LOG"])
POLL_S = 20


def log(msg: str) -> None:
    line = f"[sidecar {time.strftime('%F %T %Z')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def trained_steps() -> int | None:
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            rows = json.loads(response.read())["releases"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
        return None
    return sum(1 for r in rows if r.get("operation") == "training")


def write_progress(steps: int) -> None:
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(f"{steps}\n")
    os.replace(tmp, PROGRESS_FILE)


def _container_copy(sources: list[tuple[str, str]], dest: str) -> tuple[bool, str]:
    """Copy (container_src, name) pairs into container dir ``dest`` atomically."""
    cmds = [f"set -e", f"tmp={dest}.incoming", 'rm -rf "$tmp"', 'mkdir -p "$tmp"']
    for src, name in sources:
        cmds.append(f'cp -a "{src}" "$tmp/{name}"')
    cmds += ['chmod -R a+rX "$tmp"', f'mv "$tmp" "{dest}"']
    proc = subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", "; ".join(cmds)],
                          capture_output=True, text=True, timeout=1800)
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-300:]


def _latest_iteration(root: Path) -> int:
    try:
        return int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    except (OSError, ValueError):
        return -1


def mirror_checkpoints(kept_adapters: set[int], kept_full: set[int]) -> None:
    if not HF_DIR.is_dir():
        return
    rids = sorted(int(p.name) for p in HF_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    megatron_root, critic_root = HF_DIR.parent / "megatron", HF_DIR.parent / "megatron-critic"
    for rid in rids:
        if not (HF_DIR / str(rid) / "reef-adapter.json").is_file():
            continue                                   # adapter still being written
        step = rid + 1
        keep_step = step == FINAL_STEP
        if rid not in kept_adapters and (step % ADAPTER_EVERY == 0 or keep_step):
            dest = f"{C_KEEP}/adapters/hf_rollout_{rid:05d}"
            if (KEEP_DIR / "adapters" / f"hf_rollout_{rid:05d}").exists():
                kept_adapters.add(rid)
            else:
                ok, err = _container_copy([(f"{C_CKPT}/hf/{rid}", "hf")], dest)
                if ok:
                    kept_adapters.add(rid)
                    log(f"kept adapter step={rid + 1} (rollout_id {rid}) -> {KEEP_DIR}/adapters/hf_rollout_{rid:05d}")
                else:
                    log(f"adapter copy step={rid + 1} failed, will retry: {err}")
        if FULL_EVERY > 0 and (step % FULL_EVERY == 0 or keep_step) and rid not in kept_full:
            # The Megatron states are complete once both trackers reached this iteration.
            if _latest_iteration(megatron_root) < rid or _latest_iteration(critic_root) < rid:
                continue
            if (KEEP_DIR / "full" / f"step_{step:03d}").exists():
                kept_full.add(rid)
                continue
            ok, err = _container_copy(
                [(f"{C_CKPT}/hf/{rid}", "hf"),
                 (f"{C_CKPT}/megatron/iter_{rid:07d}", "actor"),
                 (f"{C_CKPT}/megatron-critic/iter_{rid:07d}", "critic")],
                f"{C_KEEP}/full/step_{step:03d}",
            )
            if ok:
                kept_full.add(rid)
                log(f"kept FULL bundle step={step} (rollout_id {rid}) -> {KEEP_DIR}/full/step_{step:03d}")
            else:
                log(f"full bundle copy step={step} failed, will retry: {err}")


def main() -> None:
    KEEP_DIR.mkdir(parents=True, exist_ok=True)
    kept_adapters = {int(p.name.split("_")[-1]) for p in (KEEP_DIR / "adapters").glob("hf_rollout_*")}
    kept_full = {int(p.name.split("_")[-1]) - 1 for p in (KEEP_DIR / "full").glob("step_*")}
    last = None
    log(f"started: scenario={SCENARIO} progress={PROGRESS_FILE} hf={HF_DIR} keep={KEEP_DIR}")
    while True:
        steps = trained_steps()
        if steps is not None and steps != last:
            write_progress(steps)
            log(f"trained steps: {steps}")
            last = steps
        mirror_checkpoints(kept_adapters, kept_full)
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
