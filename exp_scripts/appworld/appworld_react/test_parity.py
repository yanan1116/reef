"""Parity of episode.run_episode with grpo_vanilla's collector (the measured protocol).

Replays episodes recorded by grpo_vanilla's collector (measure_base.py, which runs the
collector's own run_task) through run_episode: the recorded assistant replies are fed back
turn by turn from a scripted ChatModel against a live AppWorld server, and the whole
conversation (prompt, every tool output) and the grade must come out identical.

usage: test_parity.py EPISODES_DIR [N] [PORT]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from appworld_react.episode import AppWorldServer, ChatModel, run_episode  # noqa: E402

TAIL = Path("/home/yanan/agents/gitlab/tail")
APPWORLD_ROOT = Path(os.environ.get("APPWORLD_ROOT", str(TAIL / "appworld")))
APPWORLD_VENV = Path(os.environ.get("APPWORLD_VENV", str(APPWORLD_ROOT / ".venv")))


class Replay(ChatModel):
    def __init__(self, record: dict) -> None:
        self.replies = [m for m in record["messages"] if m["role"] == "assistant"]
        self.turns = record.get("turns") or []
        self.index = 0

    def complete(self, messages, tools, max_tokens):
        reply = self.replies[self.index]
        turn = self.turns[self.index] if self.index < len(self.turns) else {}
        self.index += 1
        return reply, {"completion_tokens": turn.get("completion_tokens"), "finish_reason": turn.get("finish_reason")}


def main() -> None:
    episodes = sorted(Path(sys.argv[1]).glob("*.json"))[: int(sys.argv[2]) if len(sys.argv) > 2 else 10]
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 7700
    failures = 0
    for path in episodes:
        record = json.loads(path.read_text())
        server = AppWorldServer(port, APPWORLD_ROOT, APPWORLD_VENV, Path("/tmp") / f"parity-server-{port}.log")
        try:
            server.start()
            episode = run_episode(Replay(record), server, record["task_id"], f"sao-parity/{record['task_id']}")
        finally:
            server.stop()
        same_messages = episode.messages == record["messages"]
        same_grade = (episode.reward, episode.tgc) == (record["corrected_reward"], record["official_tgc"])
        same_stop = episode.stop_reason == record["stop_reason"]
        ok = same_messages and same_grade and same_stop
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'} {record['task_id']}: messages {'=' if same_messages else '!='} "
              f"({len(episode.messages)} vs {len(record['messages'])}), reward {episode.reward} vs {record['corrected_reward']}, "
              f"stop {episode.stop_reason} vs {record['stop_reason']}", flush=True)
        if not same_messages:
            for i, (a, b) in enumerate(zip(episode.messages, record["messages"])):
                if a != b:
                    print(f"    first difference at message {i} ({a.get('role')}):\n    ours: {str(a)[:300]}\n    rec : {str(b)[:300]}")
                    break
    print(f"parity: {len(episodes) - failures}/{len(episodes)} identical")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
