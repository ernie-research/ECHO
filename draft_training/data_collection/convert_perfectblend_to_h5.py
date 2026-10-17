#!/usr/bin/env python3
"""Convert the target-regenerated PerfectBlend JSONL shards to Fleet H5.

The source rows already contain the target-generated assistant turns. This
adapter only materializes the exact the target chat-v2 token stream and Fleet's
ds16/lossmask/offset schema; it does not call a model or alter the responses.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from transformers import AutoTokenizer

# Recovered from Fleet SFT H5 records and the local HF/Paddle alignment code.
# These control IDs belong to the 201216-token model space.
TARGET_MASK_ID = 2
IM_START_ID = 7
IM_END_ID = 8
THINK_CLOSE_ID = 20
ROLE_USER_ID = 3946
ROLE_ASSISTANT_ID = 100497
NEWLINE_ID = 1401


def _encode_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--max-record-tokens", type=int, default=32768)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _append_turn(
    ids: list[int],
    mask: list[int],
    tokenizer,
    role: str,
    content: str,
    final: bool,
) -> None:
    if role == "user":
        ids.extend([IM_START_ID, ROLE_USER_ID, NEWLINE_ID])
        mask.extend([0, 0, 0])
        content_ids = _encode_text(tokenizer, content)
        ids.extend(content_ids)
        mask.extend([0] * len(content_ids))
        ids.extend([IM_END_ID, NEWLINE_ID])
        mask.extend([0, 0])
        return

    if role != "assistant":
        raise ValueError(f"unsupported role: {role}")

    ids.extend([IM_START_ID, ROLE_ASSISTANT_ID, NEWLINE_ID, THINK_CLOSE_ID])
    mask.extend([0, 0, 0, 0])
    content_ids = _encode_text(tokenizer, content)
    if not content_ids:
        raise ValueError("assistant response is empty")
    ids.extend(content_ids)
    mask.extend([1] * len(content_ids))
    if final:
        ids.extend([IM_END_ID, TARGET_MASK_ID])
        mask.extend([1, 1])
    else:
        ids.extend([IM_END_ID, NEWLINE_ID])
        mask.extend([1, 0])


def row_to_features(row: dict[str, Any], tokenizer, max_tokens: int) -> tuple[list[int], list[int]]:
    if row.get("status", "success") != "success":
        raise ValueError("row status is not success")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("missing conversations")

    ids = [TARGET_MASK_ID]
    mask = [1]
    valid_turns = 0
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("conversation message is not an object")
        role = message.get("role")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"{role} content is not text")
        if role == "system":
            raise ValueError("system messages are not supported by this carrier")
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        if role == "user" and valid_turns % 2 != 0:
            raise ValueError("conversation does not alternate user/assistant")
        if role == "assistant" and valid_turns % 2 != 1:
            raise ValueError("conversation does not alternate user/assistant")
        _append_turn(
            ids,
            mask,
            tokenizer,
            role,
            content,
            final=role == "assistant" and message is conversations[-1],
        )
        valid_turns += 1

    if valid_turns < 2 or valid_turns % 2:
        raise ValueError("conversation must end with assistant")
    if max_tokens > 0 and len(ids) > max_tokens:
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
    if sum(mask) < 2:
        raise ValueError("record has fewer than two loss tokens after truncation")
    return ids, mask


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    input_paths = sorted(
        path
        for path in glob.glob(os.path.join(args.input_dir, "part-*.jsonl"))
        if not path.endswith(".error.jsonl")
    )
    if not input_paths:
        raise FileNotFoundError(f"no collector part-*.jsonl files in {args.input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[str] = []
    total_rows = total_written = total_errors = 0
    for input_path in input_paths:
        output_path = output_dir / (Path(input_path).stem + ".h5")
        if args.resume and output_path.exists():
            with h5py.File(output_path, "r") as handle:
                count = len(handle["ds16_offset"])
            manifest_rows.append(f"1.0 {output_path}\n")
            total_written += count
            continue

        temporary_path = output_path.with_suffix(".h5.tmp")
        token_buffer: list[int] = []
        mask_buffer: list[int] = []
        offset_buffer: list[int] = []
        written = 0
        errors = 0
        with h5py.File(temporary_path, "w") as handle:
            ds = handle.create_dataset(
                "ds16",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint32,
                chunks=(65536,),
            )
            loss = handle.create_dataset(
                "ds16_lossmask",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint8,
                chunks=(65536,),
            )
            offsets = handle.create_dataset(
                "ds16_offset",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint64,
                chunks=(65536,),
            )

            def flush() -> None:
                nonlocal token_buffer, mask_buffer, offset_buffer
                if not token_buffer:
                    return
                old_tokens = ds.shape[0]
                new_tokens = old_tokens + len(token_buffer)
                ds.resize((new_tokens,))
                loss.resize((new_tokens,))
                ds[old_tokens:new_tokens] = np.asarray(token_buffer, dtype=np.uint32)
                loss[old_tokens:new_tokens] = np.asarray(mask_buffer, dtype=np.uint8)
                old_offsets = offsets.shape[0]
                offsets.resize((old_offsets + len(offset_buffer),))
                offsets[old_offsets:] = np.asarray(offset_buffer, dtype=np.uint64)
                token_buffer = []
                mask_buffer = []
                offset_buffer = []

            for line in open(input_path, encoding="utf-8"):
                if not line.strip():
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                    ids, row_mask = row_to_features(
                        row, tokenizer, args.max_record_tokens
                    )
                except (ValueError, TypeError, json.JSONDecodeError):
                    errors += 1
                    total_errors += 1
                    continue
                token_buffer.extend(ids)
                mask_buffer.extend(row_mask)
                written += 1
                offset_buffer.append(written and len(token_buffer) + ds.shape[0])
                if len(token_buffer) >= 1_000_000:
                    flush()
            flush()
        os.replace(temporary_path, output_path)
        manifest_rows.append(f"1.0 {output_path}\n")
        total_written += written
        print(
            f"{input_path}: written={written} errors={errors} output={output_path}",
            flush=True,
        )

    manifest_path.write_text("".join(manifest_rows), encoding="utf-8")
    print(
        f"completed rows={total_rows} written={total_written} errors={total_errors} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
