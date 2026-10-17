# Third-party code

`PaddleFleet/` is not part of this work. It is a vendored copy of
[PaddleFleet](https://github.com/PaddlePaddle/PaddleFleet), the open-source
distributed-training library, taken from the `mtp_reuse_last_layer` branch at
commit `32e7980` because that branch carries the collective-free depth sampling
the Paddle evaluator needs.

Only `src/` is included; the build artefacts and the git history are not. The
code is unmodified: 163 of its 166 Python files carry the upstream
`Copyright (c) 2025 PaddlePaddle Authors` notice and are licensed under the
Apache License 2.0, whose terms are reproduced in each file header and available
at http://www.apache.org/licenses/LICENSE-2.0. Nothing here was renamed or
rewritten, including identifiers that name model families, so that the copy
stays diffable against upstream.

The first-party adapter that builds a model through this library is not in this
directory; it is `../src/fleet_model/`.
