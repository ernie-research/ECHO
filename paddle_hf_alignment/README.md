# Paddle checkpoint, HF modelling code

The checkpoints are Paddle-trained and exported in HF format. Serving and
drafter training read them through an HF/PyTorch implementation of the model,
which is only sound if that implementation reproduces the Paddle one. This
directory is the pair of paths and the harness that compares them.

The comparison is run the same way on both sides: the same fifty GSM8K
completion prompts, the same checkpoint, the same decoding settings, once
through PyTorch and once through PaddleFleet.

```bash
./run_hf.sh      echo      0 1.0    # HF/PyTorch path
./run_paddle.sh  echo      0 1.0    # PaddleFleet path
./run_hf.sh      baseline  0 1.0    # the MoE-MTP baseline checkpoint
```

Arguments are `{echo|baseline} [gpu] [temp]`. `temp=0` is rejected rather than
silently wrong: the sampler divides by the temperature, so zero gives NaN, and
`1e-6` is used for greedy. Point `ECHO_CKPT` and `BASELINE_CKPT` at the two
checkpoints; cluster paths appear as `<WORKSPACE>`.

## The two sides

- `src/modeling/` — the HF/PyTorch implementation: the config, the
  model, and `torch_weight_loader.py`, which builds it and loads the exported
  weights. This is what serving and drafter training actually use.
- `reference/modeling_reference.py` — the Paddle-side model definition. This is
  the authority the HF implementation is checked against. The router and RoPE
  helpers it relies on are not duplicated here; they are the ones in
  `runtime/PaddleFleet/`, which is what actually runs.
- `runtime/PaddleFleet/src/` — **third-party.** A vendored, unmodified copy of
  the open-source PaddleFleet library, from the `mtp_reuse_last_layer` branch at
  commit `32e7980`, Apache 2.0. See `runtime/README.md`. It is left exactly as
  upstream wrote it, identifiers included, so it stays diffable against the
  original; the anonymisation applied elsewhere in this repository stops at this
  boundary.
- `src/fleet_model/target/` — first-party. The provider the Paddle evaluator
  constructs the model through, written against the PaddleFormers provider
  pattern. Included so the Paddle side runs from this tree alone.
- `src/hf_weight_loader.py` — the HF to PaddleFleet key mapping, used to drive
  the Paddle model from the exported weights so that both sides run the same
  tensors.
- `eval/eval_stochastic_hf.py`, `eval/eval_stochastic_paddle.py` — the two
  evaluators. `patch_baseline_mtp.py` adapts the baseline checkpoint's MTP
  weights, which are laid out differently from ECHO's.

## What you need that is not here

The checkpoints, the tokenizer, and a Paddle build. The HF path needs only
`torch` and `transformers`; the Paddle path additionally needs Paddle
`3.4.0.post20260608+f3f5f3fd150` or a build verified compatible with it. The
Python and CUDA binaries for Paddle are not vendored.

## A note on identifiers

The model classes here are named `Target*` and the config declares
`model_type = "echo_target"`. The internal family name they carried has been
replaced throughout, the same substitution the rest of this repository applies
to paths and hostnames. `torch_weight_loader.build_and_load_torch` constructs
the config and the model directly rather than going through `AutoConfig` /
`AutoModel`, so it does not read `model_type` or `architectures` from the
checkpoint and is unaffected by the rename. If you load a checkpoint through
`AutoModel.from_pretrained` instead, its `config.json` has to declare the same
`model_type` as the class registered here.
