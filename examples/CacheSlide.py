#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# -----------------------------
# Helpers: robust access to your custom vLLM hooks
# -----------------------------
def _try_get(obj: Any, chain: Sequence[str]) -> Any:
    cur = obj
    for key in chain:
        if cur is None:
            return None
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def get_vllm_core_model(llm: LLM) -> Any:
    """
    Try to locate the core model object that contains:
      - .layers (list of decoder layers)
      - .cache_fuse_metadata (dict)
      - .old_kvs (list-like)
    Your fork path may differ; we try a few common ones.
    """
    candidates = [
        ["llm_engine", "model_executor", "driver_worker", "model_runner", "model", "model"],
        ["llm_engine", "model_executor", "driver_worker", "model_runner", "model"],
        ["llm_engine", "model_executor", "worker", "model_runner", "model", "model"],
        ["llm_engine", "model_executor", "worker", "model_runner", "model"],
    ]
    for chain in candidates:
        m = _try_get(llm, chain)
        if m is not None:
            return m
    raise RuntimeError("Could not locate the vLLM core model object. "
                       "Update get_vllm_core_model() for your fork.")


def get_cache_fuse_metadata(core_model: Any) -> Dict[str, Any]:
    meta = getattr(core_model, "cache_fuse_metadata", None)
    if meta is None:
        # Some forks place it one level deeper
        meta = _try_get(core_model, ["model", "cache_fuse_metadata"])
    if meta is None or not isinstance(meta, dict):
        raise RuntimeError("cache_fuse_metadata dict not found on model. "
                           "Your vLLM fork must expose it.")
    return meta


def get_decoder_layers(core_model: Any) -> List[Any]:
    layers = getattr(core_model, "layers", None)
    if layers is None:
        layers = _try_get(core_model, ["model", "layers"])
    if layers is None:
        raise RuntimeError("Could not find .layers on model (decoder layers).")
    return list(layers)


# -----------------------------
# Dataset → (doc_chunks, query) builders
# -----------------------------
def build_from_hotpotqa(ex: Dict[str, Any],
                        max_chunks: int = 4,
                        use_supporting_titles: bool = True) -> Tuple[List[str], str]:
    """
    HotpotQA example typically includes:
      - question: str
      - context: list of [title, [sent1, sent2, ...]]
      - supporting_facts: list of [title, sent_idx]
    We build doc chunks from (supporting titles first) and then fill up.
    """
    question = ex.get("question", "")
    context = ex.get("context", [])
    supporting = ex.get("supporting_facts", [])

    # Normalize context to {title: paragraph_text}
    title2text: Dict[str, str] = {}
    for item in context:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sents = item[0], item[1]
        if isinstance(sents, (list, tuple)):
            para = " ".join([str(x) for x in sents])
        else:
            para = str(sents)
        title2text[str(title)] = f"Title: {title}\n{para}"

    chunks: List[str] = []

    if use_supporting_titles and isinstance(supporting, (list, tuple)):
        seen = set()
        for sf in supporting:
            if not isinstance(sf, (list, tuple)) or len(sf) < 1:
                continue
            t = str(sf[0])
            if t in seen:
                continue
            if t in title2text:
                chunks.append(title2text[t])
                seen.add(t)
            if len(chunks) >= max_chunks:
                break

    # Fill remaining with other context paragraphs (stable order)
    if len(chunks) < max_chunks:
        for t, txt in title2text.items():
            if txt in chunks:
                continue
            chunks.append(txt)
            if len(chunks) >= max_chunks:
                break

    if not chunks:
        # Fallback: at least one chunk
        chunks = ["(No context provided)"]

    return chunks, question


def _format_msc_session(session: Any) -> str:
    """
    Try to format a session into a readable chunk.
    We handle a few common shapes:
      - list of dicts: [{"role": "...", "content": "..."}, ...]
      - list of 2-tuples: [("user","..."), ("assistant","..."), ...]
      - string
    """
    if isinstance(session, str):
        return session

    if isinstance(session, (list, tuple)):
        lines = []
        for turn in session:
            if isinstance(turn, dict):
                role = str(turn.get("role", ""))
                content = str(turn.get("content", turn.get("text", "")))
                if role:
                    lines.append(f"{role}: {content}")
                else:
                    lines.append(content)
            elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                role = str(turn[0])
                content = str(turn[1])
                lines.append(f"{role}: {content}")
            else:
                lines.append(str(turn))
        return "\n".join(lines)

    return str(session)


def build_from_msc_memfuse(ex: Dict[str, Any],
                           max_sessions: int = 5) -> Tuple[List[str], str]:
    """
    MSC-derived HF benchmark (Percena/msc-memfuse-mc10) typically includes:
      - question: str
      - haystack_sessions: list[...]  (sessions of dialogue/history)
    We'll make one chunk per session.
    """
    question = ex.get("question", "")
    sessions = ex.get("haystack_sessions", ex.get("sessions", ex.get("history", [])))

    chunks: List[str] = []
    if isinstance(sessions, (list, tuple)):
        for s in sessions[:max_sessions]:
            chunks.append(_format_msc_session(s))
    if not chunks:
        chunks = ["(No sessions provided)"]
    return chunks, question


def build_from_swebench(ex: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    SWE-bench example typically includes:
      - problem_statement: str
      - hints_text: Optional[str]
      - repo, base_commit, etc.
    We'll use problem + hints as "doc chunks", and the "query" asks for a patch.
    """
    problem = ex.get("problem_statement", "")
    hints = ex.get("hints_text", "") or ""
    repo = ex.get("repo", "") or ""

    chunks = []
    if repo:
        chunks.append(f"Repository: {repo}")
    if problem:
        chunks.append(f"Problem:\n{problem}")
    if hints.strip():
        chunks.append(f"Hints:\n{hints}")

    if not chunks:
        chunks = ["(No SWE-bench fields found)"]

    query = "Produce a minimal, correct patch that makes the failing tests pass."
    return chunks, query


def build_sample(dataset_name: str, ex: Dict[str, Any],
                 max_chunks: int) -> Tuple[List[str], str]:
    if dataset_name == "hotpotqa":
        return build_from_hotpotqa(ex, max_chunks=max_chunks, use_supporting_titles=True)
    if dataset_name == "msc":
        return build_from_msc_memfuse(ex, max_sessions=max_chunks)
    if dataset_name == "swebench":
        return build_from_swebench(ex)
    raise ValueError(f"Unknown dataset preset: {dataset_name}")


# -----------------------------
# CacheSlide-style KV reuse harness (your template)
# -----------------------------
def run_one(llm: LLM,
            tokenizer: Any,
            doc_prompts: List[str],
            query_prompt: str,
            max_new_tokens: int = 10) -> Tuple[str, float, str, float]:
    """
    Returns:
      (cached_text, cached_ttft, normal_text, normal_ttft)
    """
    core_model = get_vllm_core_model(llm)
    cache_fuse_metadata = get_cache_fuse_metadata(core_model)

    # Build token IDs (no special tokens added here)
    def enc(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # Mistral-Instruct style markers (computed from tokenizer strings)
    # This matches your intent: one global [INST] ... [/INST] wrapping.
    s_start_full = enc("[INST]")
    s_start = []  # keep empty like your template
    s_end = enc("[/INST]")

    # NOTE: vLLM typically inserts BOS at runtime when tokenizing the prompt string.
    # Your slicing assumes that (hence the +1 offsets).
    s_start_len = len(s_start_full) + 1
    s_start_1_len = len(s_start) + 1

    # Prepare chunk token IDs (strip nothing; we already used add_special_tokens=False)
    doc_chunk_ids = [enc(doc) for doc in doc_prompts]
    q_ids = enc(query_prompt)

    # Wrap like your original structure:
    #   [chunk0: INST], [chunk1..N: docs], [last chunk: query + END]
    doc_chunk_ids = [s_start + ids for ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

    # Correct suffix length (your original had a TODO)
    suffix_len = len(q_ids) + len(s_end)

    # 1) Collect KV per chunk (prefill-only generation)
    cache_fuse_metadata["collect"] = True
    cache_fuse_metadata["check"] = False

    num_layers = len(get_decoder_layers(core_model))
    chunk_past_key_values: List[List[torch.Tensor]] = []

    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)

    for i in range(len(doc_chunk_ids)):
        prompt_str = tokenizer.decode(doc_chunk_ids[i])
        llm.generate([prompt_str], prefill_params)

        layers = get_decoder_layers(core_model)
        for j in range(num_layers):
            attn = layers[j].self_attn
            if not hasattr(attn, "hack_kv"):
                raise RuntimeError("layers[j].self_attn.hack_kv not found. "
                                   "Your fork must expose hack_kv=(K,V).")
            past_k, past_v = attn.hack_kv  # [seq, kv] each

            if i == 0:
                temp_k = past_k[:s_start_len].clone()
                temp_v = past_v[:s_start_len].clone()
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                # Skip BOS at position 0 (and any per-chunk s_start if you add it later)
                end = len(doc_chunk_ids[i]) + 1
                temp_k = past_k[s_start_1_len:end].clone()
                temp_v = past_v[s_start_1_len:end].clone()
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)

        # Inject concatenated KVs into model (your hook)
        core_model.old_kvs = chunk_past_key_values

    # 2) Build the full prompt string exactly matching the concatenated KV layout
    input_ids: List[int] = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            input_ids += doc_chunk_ids[i]
        else:
            input_ids += doc_chunk_ids[i][s_start_1_len - 1:]
    input_prompt = tokenizer.decode(input_ids)

    # 3) Cached generation
    gen_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata["collect"] = False
    cache_fuse_metadata["suffix_len"] = suffix_len

    out_cached = llm.generate([input_prompt], gen_params)[0]
    cached_text = out_cached.outputs[0].text
    cached_ttft = float(out_cached.metrics.first_token_time - out_cached.metrics.first_scheduled_time)

    # 4) Normal generation (no reuse)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata["collect"] = False

    out_normal = llm.generate([input_prompt], gen_params)[0]
    normal_text = out_normal.outputs[0].text
    normal_ttft = float(out_normal.metrics.first_token_time - out_normal.metrics.first_scheduled_time)

    return cached_text, cached_ttft, normal_text, normal_ttft


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--gpu_mem_util", type=float, default=0.5)

    parser.add_argument("--dataset", type=str, choices=["hotpotqa", "msc", "swebench"], default="hotpotqa")
    parser.add_argument("--hf_name", type=str, default=None,
                        help="Override HF dataset id. If unset, uses a preset for --dataset.")
    parser.add_argument("--hf_config", type=str, default=None,
                        help="Optional HF dataset config/subset (e.g., 'distractor' for HotpotQA).")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_chunks", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    # Dataset presets (all on HuggingFace)
    if args.hf_name is None:
        if args.dataset == "hotpotqa":
            hf_name = "hotpotqa/hotpot_qa"
            hf_config = args.hf_config or "distractor"
        elif args.dataset == "msc":
            hf_name = "Percena/msc-memfuse-mc10"
            hf_config = args.hf_config  # usually None
        elif args.dataset == "swebench":
            hf_name = "princeton-nlp/SWE-bench"
            hf_config = args.hf_config  # usually None
        else:
            raise ValueError(args.dataset)
    else:
        hf_name = args.hf_name
        hf_config = args.hf_config

    print(f"[Info] Loading dataset: name={hf_name} config={hf_config} split={args.split}")
    ds = load_dataset(hf_name, hf_config, split=args.split)

    # vLLM + tokenizer
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_util)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Some vLLM forks let you override tokenizer explicitly; ignore if missing.
    if hasattr(llm, "set_tokenizer"):
        llm.set_tokenizer(tokenizer)

    end_index = min(len(ds), args.start_index + args.num_samples)

    for idx in range(args.start_index, end_index):
        ex = ds[idx]
        doc_prompts, query_prompt = build_sample(args.dataset, ex, max_chunks=args.max_chunks)

        cached_text, cached_ttft, normal_text, normal_ttft = run_one(
            llm=llm,
            tokenizer=tokenizer,
            doc_prompts=doc_prompts,
            query_prompt=query_prompt,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"\n=== Sample {idx} ({args.dataset}) ===")
        print(f"Cached generation: {cached_text}")
        print(f"TTFT with cache: {cached_ttft:.6f}s")
        print(f"Normal generation: {normal_text}")
        print(f"TTFT with full prefill: {normal_ttft:.6f}s")
        print("------------")


if __name__ == "__main__":
    main()
