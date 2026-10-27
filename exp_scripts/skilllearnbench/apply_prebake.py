"""Give already prepared task copies the verifier prebake (prepare_tasks.py verifier_prebake_step, guarded_test_sh) and
local base images (local_base_dockerfile; run tag_local_bases.sh on the host first)
without disturbing a run that is using them (user, 2026-10-20: cache everything the verifiers download).

Each instance's new image is built from a staging copy first, so the BuildKit cache holds the download layer, and
only then are its Dockerfile and tests/test.sh replaced (atomic rename). A trial that starts after the swap builds
from cache inside its build timeout; one that started before keeps its old image and verifier.

    python3 apply_prebake.py [--jobs 4] [--skip-family F ...] TASKS_DIR [TASKS_DIR ...]
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare_tasks import guarded_test_sh, local_base_dockerfile, prebaked_dockerfile


def replace(path: Path, text: str) -> None:
    staging = path.with_name(path.name + ".prebake")
    staging.write_text(text)
    shutil.copymode(path, staging)
    os.replace(staging, path)


def apply(instance: Path) -> str:
    family = instance.name.rsplit("-", 1)[0]
    dockerfile, test_sh = instance / "environment" / "Dockerfile", instance / "tests" / "test.sh"
    new_dockerfile = local_base_dockerfile(prebaked_dockerfile(family, dockerfile.read_text(), test_sh.read_text()))
    new_test_sh = guarded_test_sh(test_sh.read_text())
    if new_dockerfile != dockerfile.read_text():
        with tempfile.TemporaryDirectory(prefix="slb-prebake-") as stage:
            context = Path(stage) / "environment"
            shutil.copytree(instance / "environment", context)
            (context / "Dockerfile").write_text(new_dockerfile)
            built = subprocess.run(["docker", "build", "-q", "-t", f"slb-prebuild/{instance.name}", str(context)],
                                   capture_output=True, text=True)
        if built.returncode != 0:
            return f"FAILED {instance}: {built.stderr.strip()[-400:]}"
        replace(dockerfile, new_dockerfile)
    if new_test_sh != test_sh.read_text():
        replace(test_sh, new_test_sh)
    return f"ok {instance}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-family", action="append", default=[])
    args = parser.parse_args()
    instances = sorted(
        p for root in args.roots for p in Path(root).iterdir()
        if (p / "environment" / "Dockerfile").is_file() and p.name.rsplit("-", 1)[0] not in args.skip_family
    )
    with ThreadPoolExecutor(args.jobs) as pool:
        for line in pool.map(apply, instances):
            print(line, flush=True)


if __name__ == "__main__":
    main()
