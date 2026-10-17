#!/usr/bin/env python3
"""Collect target-regenerated PerfectBlend data with one in-process SGLang engine per node.

Same contract as scripts/collect_perfectblend.py (which runs one HF worker per
GPU): read a byte range of the prompt JSONL, replay every user turn against the target,
write `part-<shard>-of-<total>.jsonl` rows whose `conversations` hold the regenerated
assistant turns. scripts/convert_perfectblend_to_h5.py consumes the output
unchanged.

The difference is the decode engine. The HF collector pads a batch to a common length
and steps it row-by-row, so a batch runs at the pace of its longest row and the padding
is wasted work. SGLang gives continuous batching, a radix prefix cache and CUDA graphs
behind one TP=8 engine per node, so the node's 8 GPUs serve one large in-flight pool
instead of 8 independent padded batches.

Prompt construction, the stop-token set and the answer detokenisation are imported from
the HF collector rather than re-derived: the chat-v2 control IDs (mask=2, im_start=7,
im_end=8, think_close=20, user=3946, assistant=100497, newline=1401) were recovered from
the Fleet SFT H5 records and any drift would silently produce a differently-tokenised
dataset.

Shard layout: `--shards-per-rank` sub-shards are processed sequentially per node so the
global part count (and therefore the H5 part count the trainer's dataloader sees) can
match the existing 96-part dataset even though there are only 6-12 engines.

Example (6 nodes, echo_mtp2-7_sft step_4800):
  mpirun -n 6 --allow-run-as-root -npernode 1 \
    bash <WORKSPACE>/SpecForge/examples/run_collect_perfectblend_sglang.sh
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# The HF collector owns the prompt protocol; import it instead of duplicating the IDs.
from collect_perfectblend import (  # noqa: E402  (same scripts/ dir)
    TARGET_MASK_ID,
    IM_END_ID,
    _build_target_chatv2_ids,
    _byte_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Zero matches the greedy HF/Qwen/DSV4 collectors.",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help=(
            "Engine-side concurrency cap. Unset, SGLang derives it from the KV pool, "
            "which can sit below --rows-in-flight and silently throttle collection: "
            "measured 14.96 rows/s / 5.3k tok/s at the default on one 8-GPU node, "
            "which is what a bs=16 serving sweep gets on the same target."
        ),
    )
    parser.add_argument(
        "--speculative-algorithm",
        default=None,
        help=(
            "Optional drafter for collection, e.g. EAGLE to drive the target's own "
            "native MTP block. Collection is throughput-bound, and a bs=16 serving "
            "sweep on this target still showed ~2.3x total tok/s with the MTP "
            "drafter, so this can roughly halve a multi-hour run. Off by default."
        ),
    )
    parser.add_argument("--speculative-draft-model-path", default=None)
    parser.add_argument("--speculative-num-steps", type=int, default=1)
    parser.add_argument("--speculative-eagle-topk", type=int, default=1)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=2)
    parser.add_argument(
        "--rows-in-flight",
        type=int,
        default=256,
        help="Conversations handed to the engine per round; SGLang schedules within it.",
    )
    parser.add_argument(
        "--shards-per-rank",
        type=int,
        default=8,
        help="Sub-shards per node, so the global part count can match the 96-part "
        "layout produced by the 12-node x 8-GPU HF collector.",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def rank_info() -> tuple[int, int]:
    rank = int(
        os.environ.get(
            "NODE_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        )
    )
    world = int(
        os.environ.get("NNODES", os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    )
    return rank, world


def build_engine(args: argparse.Namespace):
    """Start one in-process SGLang engine with the target port's fixes active.

    `register_target()` must run before ServerArgs/ModelConfig touch AutoConfig:
    sglang resolves the checkpoint through transformers and does not know the
    `echo_target` model_type otherwise. The the target checkpoint dirs also ship no
    tokenizer, hence the explicit --tokenizer-path.

    The 7 MTP layers in echo_mtp2-7 (model.layers.24..30) are dropped on load:
    deepseek_weight_loader.do_load_weights skips layer ids >= num_hidden_layers when
    is_nextn=False and num_nextn_predict_layers > 0, so the engine serves the plain
    24-layer target. No speculation is used here -- continuous batching is already
    the win, and a target-only decode keeps the collected text exactly equal to the
    greedy reference path.
    """
    import sglang
    from specforge.target import is_target_checkpoint, register_target

    if is_target_checkpoint(args.model_path):
        register_target()

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        tp_size=args.tp_size,
        dtype="bfloat16",
        mem_fraction_static=args.mem_fraction_static,
        attention_backend=args.attention_backend,
        trust_remote_code=True,
        disable_radix_cache=False,
        log_level="warning",
    )
    if args.context_length is not None:
        engine_kwargs["context_length"] = args.context_length
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.speculative_algorithm:
        # The draft path may point at the target checkpoint itself: sglang loads it
        # with is_nextn=True and keeps only the MTP block. ds_mtp1 has
        # num_nextn_predict_layers=1 and mtp_reuse_last_layer=False, so depth 1 is
        # all it has -- num_steps>1 would have no layer to extrapolate from.
        engine_kwargs.update(
            speculative_algorithm=args.speculative_algorithm,
            speculative_draft_model_path=(
                args.speculative_draft_model_path or args.model_path
            ),
            speculative_num_steps=args.speculative_num_steps,
            speculative_eagle_topk=args.speculative_eagle_topk,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        )
    return sglang.Engine(**engine_kwargs)


def sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        # Same stop set the HF collector's decode loop uses.
        "stop_token_ids": [TARGET_MASK_ID, IM_END_ID],
        "no_stop_trim": True,
    }
    if args.temperature > 0:
        params["top_p"] = args.top_p
    return params


def decode_answer(tokenizer, output_ids: list[int]) -> str:
    """Detokenise exactly as the HF collector does: drop the stop IDs, keep specials."""
    answer_ids = [
        int(token_id)
        for token_id in output_ids
        if int(token_id) not in (TARGET_MASK_ID, IM_END_ID)
    ]
    return tokenizer.decode(answer_ids, skip_special_tokens=False).strip()


def _expand_inputs(pattern: str) -> list[Path]:
    """Resolve --input-file-path to a sorted file list.

    Accepts a single JSONL, a directory, or a glob. The canonical source is the
    single-file DSV4-flash t0 PerfectBlend regeneration
    (perfectblend_prompts/perfectblend_deepseek_v4_flash_chat_t0_8k.jsonl, 1,420,572
    rows), which is byte-range sharded exactly as the HF collector does. A directory
    of collector output (e.g. perfectblend_step5000_raw/part-*.jsonl) also works
    and is sharded whole-file; either way only user/system turns are read, so the
    prompt set is the same and the assistant turns get regenerated.
    """
    import glob as _glob

    path = Path(pattern)
    if path.is_dir():
        found = sorted(path.glob("part-*.jsonl"))
    elif any(ch in pattern for ch in "*?["):
        found = sorted(Path(p) for p in _glob.glob(pattern))
    else:
        found = [path] if path.is_file() else []
    files = [p for p in found if not p.name.endswith(".error.jsonl")]
    if not files:
        raise FileNotFoundError(f"no input JSONL matched: {pattern}")
    return files


def _shard_inputs(
    files: list[Path], shard: int, total_shards: int
) -> list[tuple[Path, int, int]]:
    """Return (file, start_byte, end_byte) slices owned by one global shard.

    With several input files, whole files are handed out in contiguous blocks -- no
    byte splitting, so ids stay disjoint without any boundary handling. With a single
    file, fall back to the HF collector's byte-range split.
    """
    if len(files) == 1:
        start, end = _byte_range(str(files[0]), shard, total_shards)
        return [(files[0], start, end)]
    first = len(files) * shard // total_shards
    last = len(files) * (shard + 1) // total_shards
    return [(f, 0, f.stat().st_size) for f in files[first:last]]


def _read_shard_rows(
    slices: list[tuple[Path, int, int]],
    processed_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read this sub-shard's slices; return (usable rows, malformed rows)."""
    rows: list[dict[str, Any]] = []
    broken: list[dict[str, Any]] = []
    for input_path, start, end in slices:
        with input_path.open("rb") as source:
            source.seek(start)
            if start:
                source.readline()  # split landed mid-line; the previous shard owns it
            while source.tell() < end:
                line = source.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                    if not isinstance(row.get("conversations"), list):
                        raise ValueError("missing conversations list")
                except Exception as exc:  # malformed input line
                    broken.append(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw": line.decode("utf-8", errors="replace").rstrip("\n"),
                        }
                    )
                    continue
                if str(row.get("id", "")) in processed_ids:
                    continue
                # gen_stats belongs to whichever target produced the discarded
                # assistant turns; carrying it forward would mislabel this dataset.
                row.pop("gen_stats", None)
                row.pop("status", None)
                rows.append(row)
    return rows, broken



def _regenerate_chunk(
    engine,
    tokenizer,
    args: argparse.Namespace,
    chunk: list[dict[str, Any]],
) -> tuple[list[list[dict[str, str]] | None], list[str | None], int]:
    """Replay every user turn of every conversation in the chunk.

    One engine.generate call per *turn index*, not per conversation: all rows that are
    waiting on their k-th assistant turn go in together, so the in-flight pool stays
    large even though multi-turn rows must be walked sequentially.
    """
    states = [
        {"conversations": row["conversations"], "position": 0, "messages": [], "out": []}
        for row in chunk
    ]
    failures: list[str | None] = [None] * len(chunk)
    generated_tokens = 0

    while True:
        pending: list[tuple[int, list[int]]] = []
        for index, state in enumerate(states):
            if failures[index] is not None:
                continue
            conversations = state["conversations"]
            while state["position"] < len(conversations):
                message = conversations[state["position"]]
                state["position"] += 1
                role = message.get("role")
                content = str(message.get("content", ""))
                if role == "assistant":
                    continue  # the target regenerates these
                if role not in ("system", "user"):
                    failures[index] = f"unsupported conversation role: {role}"
                    break
                item = {"role": role, "content": content}
                state["messages"].append(item)
                state["out"].append(item)
                if role == "user":
                    prompt_ids = _build_target_chatv2_ids(tokenizer, state["messages"])
                    pending.append((index, prompt_ids[-args.max_input_tokens :]))
                    break
        if not pending:
            break

        outputs = engine.generate(
            input_ids=[prompt_ids for _, prompt_ids in pending],
            sampling_params=sampling_params(args),
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        for (index, _), output in zip(pending, outputs):
            output_ids = output.get("output_ids") or []
            generated_tokens += len(output_ids)
            answer = decode_answer(tokenizer, output_ids)
            if not answer:
                failures[index] = "target produced an empty assistant turn"
                continue
            assistant = {"role": "assistant", "content": answer}
            states[index]["messages"].append(assistant)
            states[index]["out"].append(assistant)

    results: list[list[dict[str, str]] | None] = []
    for index, state in enumerate(states):
        if failures[index] is not None:
            results.append(None)
            continue
        if not any(item["role"] == "assistant" for item in state["out"]):
            failures[index] = "conversation contains no user turn"
            results.append(None)
            continue
        results.append(state["out"])
    return results, failures, generated_tokens


def _load_processed_ids(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in row:
                processed.add(str(row["id"]))
    return processed


def collect_shard(
    engine, tokenizer, args, shard: int, total_shards: int, files: list[Path]
) -> tuple[int, int]:
    output_dir = Path(args.output_dir)
    stem = f"part-{shard:05d}-of-{total_shards:05d}"
    output_path = output_dir / f"{stem}.jsonl"
    error_path = output_dir / f"{stem}.error.jsonl"

    processed_ids = _load_processed_ids(output_path) if args.resume else set()
    rows, broken = _read_shard_rows(
        _shard_inputs(files, shard, total_shards), processed_ids
    )
    if args.num_samples is not None:
        rows = rows[: args.num_samples]

    file_mode = "a" if args.resume else "w"
    written = failed = tokens = 0
    started_at = time.perf_counter()
    with (
        output_path.open(file_mode, encoding="utf-8") as output,
        error_path.open(file_mode, encoding="utf-8") as errors,
    ):
        for row in broken:
            errors.write(json.dumps(row, ensure_ascii=False) + "\n")
            failed += 1
        errors.flush()
        for chunk_start in range(0, len(rows), args.rows_in_flight):
            chunk = rows[chunk_start : chunk_start + args.rows_in_flight]
            results, failures, chunk_tokens = _regenerate_chunk(
                engine, tokenizer, args, chunk
            )
            tokens += chunk_tokens
            for row, regenerated, failure in zip(chunk, results, failures):
                if regenerated is None:
                    errors.write(
                        json.dumps(
                            {
                                "status": "error",
                                "error": failure or "unknown failure",
                                "id": row.get("id"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    failed += 1
                    continue
                row["conversations"] = regenerated
                row["status"] = "success"
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            output.flush()
            errors.flush()
            if written and written % args.log_interval < args.rows_in_flight:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                print(
                    f"[collector-sglang] shard={shard}/{total_shards} "
                    f"rows={written}/{len(rows)} errors={failed} "
                    f"rows_per_sec={written / elapsed:.3f} "
                    f"tokens_per_sec={tokens / elapsed:.1f}",
                    flush=True,
                )
    return written, failed


def main() -> None:
    args = parse_args()
    rank, world = rank_info()
    total_shards = world * args.shards_per_rank
    my_shards = list(range(rank * args.shards_per_rank, (rank + 1) * args.shards_per_rank))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    files = _expand_inputs(args.input_file_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    engine = build_engine(args)
    print(
        f"[collector-sglang] node={rank}/{world} tp={args.tp_size} "
        f"shards={my_shards[0]}..{my_shards[-1]} of {total_shards} "
        f"inputs={len(files)} model={args.model_path}",
        flush=True,
    )

    total_written = total_failed = 0
    for shard in my_shards:
        written, failed = collect_shard(
            engine, tokenizer, args, shard, total_shards, files
        )
        total_written += written
        total_failed += failed
        print(
            f"[collector-sglang] node={rank} shard={shard} done "
            f"rows={written} errors={failed}",
            flush=True,
        )
    print(
        f"[collector-sglang] node={rank} ALL DONE rows={total_written} "
        f"errors={total_failed}",
        flush=True,
    )
    engine.shutdown()


if __name__ == "__main__":
    main()



