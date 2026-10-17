# Aligning the target: HF reference against the sglang port

Training the drafters against an sglang TP8 engine instead of an HF eager copy
of the target cut an epoch from roughly 20h to 1.45h on twelve nodes, but it is
only sound if the two paths produce the same logits. They were made to agree:
the sglang path was initially missing the RoPE even/odd regroup that the HF
reference applies, and once that was fixed real-text argmax agreement reached
99.61%. Routed-scaling and fp32 router-GEMM fixes were found alongside it. This
directory holds the export code that depends on that agreement, and the weight
map it is built from.

## What is here

- `EAGLE3_EXPORT_WEIGHT_MAP.md` — the key-by-key map between the training-side
  draft (`TargetMTPEagleModel`) and the sglang drafter (`TargetModelNextN`). The
  two describe the same MTP block with different prefixes and expert layouts, so
  the export is a key and tensor rewrite with no new modelling. This table was
  written because the original export script was truncated by a disk ENOSPC
  before it was ever committed anywhere — the map is the authority, the code
  below is a rebuild from it.
- `export_eagle3_draft_for_sglang.py` — that rebuild. It is the complete
  script; the 46-line truncated original is not published, since it cannot run.
- `export_dspark_draft_for_sglang.py` — the same for the DSpark-family drafts,
  including the DFlash checkpoint, which exports through the DSpark path because
  its config declares `TargetDSparkModel` with `markov_rank=0`.
- `smoke_warmstart_alignment.py` — checks that a warm-started EAGLE-3 draft
  reproduces the target's own MTP block before any training step is taken.
- `eval_eagle3_alignment_probe.py` — the per-step probe used to compare draft
  and target distributions during training.

## What alignment does and does not transfer

Agreement was established on the target's logits. That is what the drafter is
trained against, and accepted length is a function of the logits and the
acceptance rule, so accepted length transfers.

Decode speedup does not. The sglang path described here was taken only as far as
training needed: enough kernel work to make the target forward cheap and to make
it agree numerically with the HF reference, and no further. It is not a
fully optimised serving stack — several of the optimisations a serving deployment
would carry are simply absent — so any decode rate or speedup measured on it
understates what the method can do and should not be read as a performance
claim. The paper's speedup numbers were not taken from this stack, and nothing
in this repository reproduces them.
