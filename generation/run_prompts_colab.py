"""Colab/CUDA variant of run_prompts.py: batched vLLM generation (instead of an
OpenAI-compatible server) + the same Anthropic judge. Written to verify the
pipeline against the released llama3.1-8b factorial data on a free-tier T4,
and reused as-is for the EM-organism (reverse alignment faking) runs by
swapping --model/--lora-adapter and --queries/--prompts-dir.

Reuses CELL_FILES / system_prompt / JUDGE_SYSTEM / JUDGE_TEMPLATE / VALID from
run_prompts.py unchanged, so the judge rubric is identical to the original run
-- only the generation backend and the --queries/--prompts-dir/--model/
--lora-adapter choices differ.

Vanilla verification run:
  python run_prompts_colab.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
      --tag llama3.1-8b_verify --cells A A_prime A_instr B_minus B_rnt B --runs 10

EM organism run (base model + LoRA adapter, different query set/prompt wording):
  python run_prompts_colab.py --model meta-llama/Llama-3.1-8B-Instruct \
      --lora-adapter ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice \
      --quantization none --queries harmbench_50.json --prompts-dir factorial_em \
      --tag llama31-8b-em-risky-financial --cells A A_prime A_instr B_minus B_rnt B --runs 10

Writes data/generations/factorial/<tag>.jsonl, same schema as the released files.

Logging: uses logging.StreamHandler, which flushes on every record, so progress
shows up immediately in a Colab cell even though stdout isn't a real terminal
(plain print() can sit in a buffer for a while under subprocess.run). Each major
stage (load existing output, load model, generate, judge) is wrapped so a
failure's stage and full traceback are always visible, not just a bare
"non-zero exit status" from subprocess.run.
"""
import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_prompts import JUDGE_SYSTEM, JUDGE_TEMPLATE, VALID, system_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("run_prompts_colab")


def load_done(out):
    """Already-written (cell, query_id, run) keys, skipping any line that fails to
    parse (e.g. truncated by an interrupted previous run) instead of crashing."""
    if not out.exists():
        return set()
    done, bad = set(), 0
    for i, line in enumerate(out.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            done.add((r["cell"], r["query_id"], r["run"]))
        except Exception as e:
            bad += 1
            log.warning("skipping unparseable line %d in %s: %s", i, out, e)
    if bad:
        log.warning("%d unparseable line(s) in %s -- those jobs will be re-run", bad, out)
    return done


def main():
    from anthropic import Anthropic
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model id/path (e.g. hugging-quants/...-AWQ-INT4, or meta-llama/Llama-3.1-8B-Instruct when using --lora-adapter)")
    ap.add_argument("--lora-adapter", default=None, help="HF repo id/path of a LoRA adapter to apply on top of --model (e.g. an EM organism)")
    ap.add_argument("--quantization", default="awq", help="vLLM quantization method, or 'none' to disable (use for full-precision/LoRA-adapted models)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--queries", default="borderline_50.json", help="filename under queries/")
    ap.add_argument("--num-queries", type=int, default=None, help="use only the first N queries from --queries, instead of all of them")
    ap.add_argument("--prompts-dir", default="factorial", help="subfolder under prompts/ holding <cell>.txt files")
    ap.add_argument("--cells", nargs="+", default=["A", "A_prime", "A_instr", "B_minus", "B_rnt", "B"])
    ap.add_argument("--runs", type=int, default=10, help="repetitions per (cell, query)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=None, help="generation seed; unset = non-deterministic (default/original behavior)")
    ap.add_argument("--judge-model", default="claude-haiku-4-5", help="Anthropic model id used as the judge (original paper used claude-opus-4-6)")
    ap.add_argument("--judge-workers", type=int, default=16)
    a = ap.parse_args()
    log.info("args: %s", vars(a))

    out = ROOT / "data" / "generations" / "factorial" / f"{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)

    Q = json.loads((ROOT / "queries" / a.queries).read_text())
    if a.num_queries is not None:
        Q = Q[: a.num_queries]
    jobs = [(c, q, n) for c in a.cells for q in Q for n in range(a.runs) if (c, q["id"], n) not in done]
    log.info("%d queries, %d generations to run (%d already done)", len(Q), len(jobs), len(done))
    if not jobs:
        log.info("nothing to do -- exiting")
        return

    log.info("loading model=%s lora_adapter=%s quantization=%s", a.model, a.lora_adapter, a.quantization)
    quant = None if a.quantization.lower() == "none" else a.quantization
    lora_kwargs = dict(enable_lora=True, max_lora_rank=64) if a.lora_adapter else {}
    try:
        llm = LLM(model=a.model, quantization=quant, dtype="float16", max_model_len=4096, **lora_kwargs)
    except Exception:
        log.exception("FAILED while loading the model/engine (see traceback above)")
        raise
    log.info("model loaded")
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, seed=a.seed)
    lora_req = LoRARequest("adapter", 1, a.lora_adapter) if a.lora_adapter else None

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": system_prompt(c, prompts_dir=a.prompts_dir)}, {"role": "user", "content": q["query"]}],
        add_generation_prompt=True, tokenize=False) for c, q, _ in jobs]

    log.info("generating %d prompts...", len(prompts))
    try:
        results = llm.generate(prompts, sp, lora_request=lora_req)
    except Exception:
        log.exception("FAILED during generation (see traceback above)")
        raise
    responses = [r.outputs[0].text for r in results]
    log.info("generation done")

    judge = Anthropic()
    lock = threading.Lock()
    counts = {"ok": 0, "error": 0}

    def judge_one(args):
        (c, q, n), resp = args
        try:
            raw = judge.messages.create(model=a.judge_model, max_tokens=5, temperature=0, system=JUDGE_SYSTEM, messages=[
                {"role": "user", "content": JUDGE_TEMPLATE.format(query=q["query"], response=resp[:600])}]).content[0].text
        except Exception as e:
            with lock:
                counts["error"] += 1
            log.error("judge call failed for cell=%s query_id=%s run=%d: %s: %s -- leaving unjudged, will retry on next run",
                       c, q["id"], n, type(e).__name__, e)
            return
        lab = (raw.strip().upper().split() or ["UNCLEAR"])[0]
        rec = dict(model=a.tag, cell=c, query_id=q["id"], run=n, query=q["query"], response=resp,
                   label=lab if lab in VALID else "UNCLEAR", judge=a.judge_model)
        with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            counts["ok"] += 1
            if counts["ok"] % 100 == 0:
                log.info("judging... (%d/%d, %d errors)", counts["ok"], len(jobs), counts["error"])

    log.info("judging %d responses...", len(jobs))
    with ThreadPoolExecutor(a.judge_workers) as ex:
        list(ex.map(judge_one, zip(jobs, responses)))
    log.info("done -> %s (%d written, %d errors)", out, counts["ok"], counts["error"])


if __name__ == "__main__":
    main()
