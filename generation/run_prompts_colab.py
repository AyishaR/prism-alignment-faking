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
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_prompts import JUDGE_MODEL, JUDGE_SYSTEM, JUDGE_TEMPLATE, VALID, system_prompt  # noqa: E402


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
    ap.add_argument("--prompts-dir", default="factorial", help="subfolder under prompts/ holding <cell>.txt files")
    ap.add_argument("--cells", nargs="+", default=["A", "A_prime", "A_instr", "B_minus", "B_rnt", "B"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--judge-workers", type=int, default=16)
    a = ap.parse_args()

    out = ROOT / "data" / "generations" / "factorial" / f"{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["cell"], r["query_id"], r["run"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()

    Q = json.loads((ROOT / "queries" / a.queries).read_text())
    jobs = [(c, q, n) for c in a.cells for q in Q for n in range(a.runs) if (c, q["id"], n) not in done]
    print(f"{len(jobs)} generations to run ({len(done)} already done)")
    if not jobs:
        return

    quant = None if a.quantization.lower() == "none" else a.quantization
    llm = LLM(model=a.model, quantization=quant, dtype="float16", max_model_len=4096,
              enable_lora=bool(a.lora_adapter), max_lora_rank=64)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, max_tokens=a.max_tokens)
    lora_req = LoRARequest("adapter", 1, a.lora_adapter) if a.lora_adapter else None

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": system_prompt(c, prompts_dir=a.prompts_dir)}, {"role": "user", "content": q["query"]}],
        add_generation_prompt=True, tokenize=False) for c, q, _ in jobs]

    print("generating...")
    results = llm.generate(prompts, sp, lora_request=lora_req)
    responses = [r.outputs[0].text for r in results]

    judge = Anthropic()
    lock = threading.Lock()

    def judge_one(args):
        (c, q, n), resp = args
        raw = judge.messages.create(model=JUDGE_MODEL, max_tokens=5, temperature=0, system=JUDGE_SYSTEM, messages=[
            {"role": "user", "content": JUDGE_TEMPLATE.format(query=q["query"], response=resp[:600])}]).content[0].text
        lab = (raw.strip().upper().split() or ["UNCLEAR"])[0]
        rec = dict(model=a.tag, cell=c, query_id=q["id"], run=n, query=q["query"], response=resp,
                   label=lab if lab in VALID else "UNCLEAR", judge=JUDGE_MODEL)
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print("judging...")
    with ThreadPoolExecutor(a.judge_workers) as ex:
        list(ex.map(judge_one, zip(jobs, responses)))
    print("done ->", out)


if __name__ == "__main__":
    main()
