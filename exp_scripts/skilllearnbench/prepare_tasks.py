"""Copy the SkillLearnBench instances of the split into runnable Harbor task dirs and write the split.

Split (user, 2026-10-14): nine office-style families; instance 1 of each family is the test
instance, every other instance trains. Each copy gets the family's human-authored skills in
environment/skills (the task Dockerfiles COPY that directory into the agents' skill paths) and an
[agent] timeout that is only a backstop: the harness bounds an episode itself (SLB_EPISODE_S of
non-model time, SLB_MAX_TURNS turns), and a model call may wait through a Reef training step.

    python3 prepare_tasks.py --out /works/yanan/slb-sao/tasks

Full-benchmark check (user, 2026-10-19: all 20 tasks, instance 1 test, the rest train): ``--all-families`` takes
every task directory instead of the nine office families, and ``--keep-solution`` keeps solution/ so Harbor's
oracle agent can run it (the training and eval copies drop it so the agent cannot read the answer).

    python3 prepare_tasks.py --out /mnt/disk1t/slb-all-tasks --all-families --keep-solution \
        --software-db /mnt/disk1t/slb-assets/skillbench-software-db/db.zip

github-repo-analytics queries GitHub live through the gh CLI and needs GH_TOKEN in the environment of the process
that starts the trials (user, 2026-10-19: token in ~/.config/slb/gh_token on .29). Its docker-compose.yaml also asks
for Claude Vertex variables and a ~/.config/gcloud mount, which only the authors' Claude Code agent used; the copy
keeps GH_TOKEN alone. dependency-vulnerability-check clones a HuggingFace dataset while its image builds, and
that clone fails TLS verification inside the build container on .29 (the host reaches it). --software-db gives the
dataset's db.zip, downloaded on the host (sha256 e67dd914...), and the copy's Dockerfile unpacks it from the build
context instead (user, 2026-10-19).
"""

import argparse
import json
import re
import shutil
from pathlib import Path

SLB = Path("/home/yanan/agents/SkillLearnBench")
FAMILIES = [
    "court-form-filling",
    "offer-letter-generator",
    "enterprise-information-search",
    "organize-messy-files",
    "weighted-gdp-calculation",
    "financial-analysis",
    "schedule-planning",
    "travel-planning",
    "stock-data-visualization",
]
TEST_INSTANCE = 1
#: Instances left out of the split (user, 2026-10-14): weighted-gdp-calculation-2 ships no
#: tests/test_outputs.py, so its verifier can never score it.
EXCLUDED = {"weighted-gdp-calculation-2"}
EXCLUDED_FAMILIES: set[str] = set()
GITHUB_COMPOSE = """services:
  main:
    build:
      context: ${CONTEXT_DIR}
    image: ${MAIN_IMAGE_NAME}
    command: [ "sh", "-c", "sleep infinity" ]
    environment:
      - GH_TOKEN=${GH_TOKEN}
"""
SOFTWARE_DB_CLONE = re.compile(r"RUN curl -s https://packagecloud\.io/install/repositories/github/git-lfs/script\.deb\.sh.*?rm -rf skillbench-software-db\n", re.S)
SOFTWARE_DB_COPY = """COPY db.zip /tmp/db.zip
RUN mkdir -p /root/.cache/trivy && \\
    unzip /tmp/db.zip -d /root/.cache/trivy && \\
    ln -s /root/.cache/trivy /root/trivy-cache && \\
    rm /tmp/db.zip
"""
AGENT_TIMEOUT_S = 14400.0


def verifier_prebake_step(test_sh: str) -> str:
    """One Dockerfile RUN step that performs a task's grading-time downloads while the image builds.

    Every SkillLearnBench verifier (tests/test.sh) fetches its tools when it grades: uv and its uvx test environment
    (nlp-paper-reproduction's holds torch 2.1.2 and the CUDA wheels, about 3 GB, and timed out at 900 s and 1800 s),
    pip3 pytest packages, or coursier for Scala. The user asked (2026-10-20) that everything downloaded be cached
    locally: the step runs the same commands (same uv version, uvx pins, pip3 arguments) as root, so grading finds
    them in /root/.cache/uv, the system site-packages, or the coursier directory. Empty when there is nothing to fetch.
    """
    parts = []
    if re.search(r"(?m)^\s*apt-get install -y curl\b", test_sh):
        parts.append("apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*")
    uv = re.search(r"curl -LsSf (https://astral\.sh/uv/[0-9.]+/install\.sh) \| sh", test_sh)
    if uv:
        parts.append(f"curl -LsSf {uv.group(1)} | sh")
    python = re.search(r"(?m)^\s*uv python install ([0-9.]+)", test_sh)
    if python:
        parts.append(f"/root/.local/bin/uv python install {python.group(1)}")
    block = re.search(r"(?ms)^uvx \\\n(.*?)^\s*(pytest|python) ", test_sh)
    if block:
        flags = " ".join(line.strip().rstrip("\\").strip() for line in block.group(1).splitlines() if line.strip())
        command = f"/root/.local/bin/uvx {flags} {block.group(2)} --version".replace("uvx  ", "uvx ")
        parts.append(command if block.group(2) == "pytest" else f"({command} || true)")
    coursier = re.search(r"curl -fL (https://github\.com/coursier/\S+) \| gzip -d > cs", test_sh)
    if coursier:
        parts.append(f"(curl -fL {coursier.group(1)} | gzip -d > /tmp/cs && chmod +x /tmp/cs && /tmp/cs setup --yes; rm -f /tmp/cs)")
    for pip in re.finditer(r"(?ms)^pip3 install (.*?[^\\])$", test_sh):
        parts.append("pip3 install " + " ".join(pip.group(1).replace("\\\n", " ").split()))
    if not parts:
        return ""
    return "RUN " + " \\\n    && ".join(parts)


def prebaked_dockerfile(family: str, dockerfile: str, test_sh: str) -> str:
    """The task's Dockerfile with verifier_prebake_step appended (before the first COPY for nlp-paper-reproduction,
    whose three instances then share the 3 GB layer; at the end elsewhere, where the whole image is in place)."""
    step = verifier_prebake_step(test_sh)
    if not step or "verifier_prebake_step" in dockerfile or "nlp_prebaked_dockerfile" in dockerfile:  # already prebaked
        return dockerfile
    block = "# Verifier downloads, fetched once at build time (prepare_tasks.py verifier_prebake_step)\n" + step + "\n"
    first_copy = re.search(r"(?m)^COPY ", dockerfile)
    if family == "nlp-paper-reproduction" and first_copy:
        return dockerfile[: first_copy.start()] + block + "\n" + dockerfile[first_copy.start() :]
    return dockerfile.rstrip("\n") + "\n\n" + block


LOCAL_BASE_REGISTRY = "slb-local.invalid"


def local_base_dockerfile(dockerfile: str) -> str:
    """The Dockerfile with each FROM image renamed to its local tag under an unresolvable registry host.

    Harbor runs `docker compose build` for every trial, and BuildKit looks the FROM image up on Docker Hub each time;
    a TLS timeout there failed 16 transfer-eval trials on 2026-10-20. tag_local_bases.sh pulls each base image once
    and tags it slb-local.invalid/<image>; a build from that name finds the local image and never contacts a
    registry (verified on .29). User, 2026-10-20: cache everything that is downloaded.
    """
    return re.sub(
        r"(?m)^(FROM\s+(?:--platform=\S+\s+)?)(?!" + re.escape(LOCAL_BASE_REGISTRY) + r"/)(\S+)",
        lambda m: f"{m.group(1)}{LOCAL_BASE_REGISTRY}/{m.group(2)}",
        dockerfile,
    )


def guarded_test_sh(test_sh: str) -> str:
    """tests/test.sh with its tool installs skipped when the image already has the tool (user, 2026-10-20: use the
    local cache for everything). Only the curl and uv install lines change; what the verifier tests is untouched."""
    only_curl = all(re.match(r"apt-get install -y curl\b", m.group(1)) for m in re.finditer(r"(?m)^\s*(apt-get install .*)$", test_sh))
    if only_curl:
        test_sh = re.sub(r"(?m)^(\s*)(apt-get update\b.*)$", r"\1command -v curl >/dev/null 2>&1 || \2", test_sh)
        test_sh = re.sub(r"(?m)^(\s*)(apt-get install -y curl\b.*)$", r"\1command -v curl >/dev/null 2>&1 || \2", test_sh)
    return re.sub(r"(?m)^(\s*)(curl -LsSf https://astral\.sh/uv/\S+ \| sh.*)$", r'\1[ -x "$HOME/.local/bin/uv" ] || \2', test_sh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--all-families", action="store_true", help="every SkillLearnBench task, not the 9 office ones")
    parser.add_argument("--keep-solution", action="store_true", help="keep solution/ for an oracle run")
    parser.add_argument("--software-db", help="db.zip of zodi1121/skillbench-software-db for dependency-vulnerability-check")
    args = parser.parse_args()
    if args.all_families:
        families = sorted(p.name for p in (SLB / "tasks").iterdir() if p.is_dir() and p.name not in EXCLUDED_FAMILIES)
    else:
        families = FAMILIES
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split: dict[str, list[str]] = {"train": [], "test": []}
    for family in families:
        skills = SLB / "skills" / "human_authored" / family
        if not skills.is_dir():
            raise SystemExit(f"no human-authored skills for {family}: {skills}")
        instances = sorted(
            (p for p in (SLB / "tasks" / family).iterdir() if p.is_dir() and p.name.startswith(f"{family}-")),
            key=lambda p: int(p.name.rsplit("-", 1)[1]),
        )
        for source in instances:
            if source.name in EXCLUDED:
                continue
            target = out / source.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, ignore=None if args.keep_solution else shutil.ignore_patterns("solution"))
            env_skills = target / "environment" / "skills"
            if env_skills.exists():
                shutil.rmtree(env_skills)
            shutil.copytree(skills, env_skills)
            if family == "dependency-vulnerability-check":
                if not args.software_db:
                    raise SystemExit("dependency-vulnerability-check needs --software-db (its build-time clone fails here)")
                dockerfile = target / "environment" / "Dockerfile"
                text, count = SOFTWARE_DB_CLONE.subn(SOFTWARE_DB_COPY, dockerfile.read_text())
                if count != 1:
                    raise SystemExit(f"{dockerfile}: expected one HuggingFace clone step, found {count}")
                dockerfile.write_text(text)
                shutil.copyfile(args.software_db, target / "environment" / "db.zip")
            if family == "github-repo-analytics":
                (target / "environment" / "docker-compose.yaml").write_text(GITHUB_COMPOSE)
            toml = target / "task.toml"
            text = toml.read_text()
            if re.search(r"(?m)^\[agent\]", text):
                text = re.sub(r"(?ms)(^\[agent\]\s*\n)timeout_sec\s*=\s*[0-9.]+", rf"\g<1>timeout_sec = {AGENT_TIMEOUT_S}", text)
            else:
                text += f"\n[agent]\ntimeout_sec = {AGENT_TIMEOUT_S}\n"
            # anthropic-poster-design and chinese-poem-generator ship a [task] name that is a title ("Brand-Compliant
            # Technical Infographic Generation"); Harbor requires 'org/name' and refuses to load the task.
            name = re.search(r'(?m)^name\s*=\s*"([^"]*)"', text.split("[task]", 1)[1]) if "[task]" in text else None
            if name and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name.group(1)):
                text = text.replace(name.group(0), f'name = "skilllearnbench/{source.name}"', 1)
            dockerfile, test_sh = target / "environment" / "Dockerfile", target / "tests" / "test.sh"
            dockerfile.write_text(local_base_dockerfile(prebaked_dockerfile(family, dockerfile.read_text(), test_sh.read_text())))
            test_sh.write_text(guarded_test_sh(test_sh.read_text()))
            if family == "nlp-paper-reproduction":
                # Its verifier installs trl/transformers with uvx at grading time and timed out at 900 s on 7 of 8 base
                # episodes of instance 1 (2026-10-20); user, 2026-10-20: raise it to 1800 s.
                text = re.sub(r"(?ms)(^\[verifier\]\s*\n)timeout_sec\s*=\s*[0-9.]+", r"\g<1>timeout_sec = 1800.0", text)
            toml.write_text(text)
            index = int(source.name.rsplit("-", 1)[1])
            split["test" if index == TEST_INSTANCE else "train"].append(source.name)
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(f"{len(split['train'])} train, {len(split['test'])} test instances in {out}")


if __name__ == "__main__":
    main()
