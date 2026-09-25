"""End-to-end: does the served (unmerged, bf16) LoRA adapter change the policy?

Against a live vLLM server that serves `base` and an adapter. For N DeepCoder
test prompts:
  1. greedy generation (max 384 tokens) by base, by base again (control), and by
     the adapter -> fraction of outputs that differ, first divergence position;
  2. teacher-forced scoring of base's own greedy continuation under base, base
     again, and the adapter (completions API, prompt_logprobs) -> per-token
     logprob shift the adapter applies to identical text. base-vs-base is the
     numerical noise floor of concurrent bf16 serving.
"""
import glob, json, sys, statistics as st, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from transformers import AutoTokenizer

URL, ADAPTER, BASE_DIR, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(BASE_DIR)
root = glob.glob("/home/yanan/.cache/huggingface/hub/datasets--agentica-org--DeepCoder-Preview-Dataset/snapshots/*")[0]
dfs = []
for cfg in ("lcbv5", "codeforces"):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/{cfg}/test-*.parquet"))], ignore_index=True)
    dfs.append(df.sample(N // 2, random_state=1))
probs = [r.problem.strip() + "\n\nWrite a Python solution. Enclose your code within ```python delimiters." for d in dfs for r in d.itertuples()]

def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

def gen(model, p):
    r = post("/v1/chat/completions", dict(model=model, messages=[{"role": "user", "content": p}],
             temperature=0, max_tokens=384, seed=1234))
    return r["choices"][0]["message"]["content"]

def score(model, prompt_text, cont_text):
    n_prompt = len(tok(prompt_text, add_special_tokens=False).input_ids)
    r = post("/v1/completions", dict(model=model, prompt=prompt_text + cont_text, max_tokens=1,
             temperature=0, prompt_logprobs=0))
    pl = r["choices"][0]["prompt_logprobs"]
    return [list(d.values())[0]["logprob"] if isinstance(d, dict) else list(d)[0] for d in pl[n_prompt:] if d]

def one(p):
    rendered = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    b1, b2, a = gen("base", p), gen("base", p), gen(ADAPTER, p)
    sb1, sb2, sa = score("base", rendered, b1), score("base", rendered, b1), score(ADAPTER, rendered, b1)
    return b1, b2, a, sb1, sb2, sa

def first_div(x, y):
    tx, ty = tok(x).input_ids, tok(y).input_ids
    return next((i for i, (u, v) in enumerate(zip(tx, ty)) if u != v), None if len(tx) == len(ty) else min(len(tx), len(ty)))

with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(one, probs))

def pair(label, outs):
    diffs = [first_div(x, y) for x, y in outs]
    changed = [d for d in diffs if d is not None]
    return label, len(changed), len(outs), (st.median(changed) if changed else None)

def shift(label, pairs):
    d = [abs(u - v) for s1, s2 in pairs for u, v in zip(s1, s2)]
    signed = [v - u for s1, s2 in pairs for u, v in zip(s1, s2)]
    return label, len(d), st.mean(d), st.median(d), max(d), st.mean(signed), sum(x > 0.1 for x in d) / len(d)

print(json.dumps({
    "generation": [pair("base vs base (control)", [(r[0], r[1]) for r in res]),
                   pair(f"base vs {ADAPTER}", [(r[0], r[2]) for r in res])],
    "teacher_forced": [shift("base vs base (control)", [(r[3], r[4]) for r in res]),
                       shift(f"base vs {ADAPTER}", [(r[3], r[5]) for r in res])],
}, indent=1))
