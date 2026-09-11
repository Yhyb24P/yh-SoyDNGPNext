# SoyDNGPNext Reproduction

- Upstream repository: https://github.com/IndigoFloyd/SoyDNGPNext
- Frozen commit: see `UPSTREAM_COMMIT.txt`
- `upstream-legacy` tag: unmodified upstream baseline
- `repro` branch: auditable reproduction fixes and experiments

## Repro fixes (repro branch only, NOT in upstream-legacy)

Canonical record of compatibility-only patches: `COMPATIBILITY_PATCHES.md`.
Algorithmic or training-protocol fixes are NOT listed there; they use the
T-series (T1, T2, ...) in the corrected trainer (G4).

- **F1 (G2)**: `soydngpnext/reader.py` — replaced `df.index.values_host`
  (removed in cudf >= 23; the 26.x Index is also not iterable) with
  `df.index.to_numpy()`. Host-side ordered sample IDs, same semantics as the
  original `values_host`. Environment adaptation only. Triggered by G2
  parity test on cudf-cu13 26.8.1 / cupy-cuda13x 14.2.0.

## Paper model provenance

- Paper architecture (PAPER_MODEL_V1): `IndigoFloyd/SoybeanWebsite`,
  `AlexNet_206.py`, commit `f724d9b1974bb78b7832b191af99f589a2c5e549`
  (fetched 2026-09-11). Verbatim port at `src/paper_model/soydngp_ca.py`;
  contract at `results/package_baseline/g3_model_contract/paper_model_contract.json`.
- The 2025 residual variant (`soydngpnext/SoyDNGP_res.py`) is a
  post-publication secondary variant, NOT the paper model; it is not
  executable on the expected 206x206 input (CA_Block h/w=322, see
  `results/package_baseline/g3_model_contract/residual_model_status.json`).

## Stage status

- G0 PASS: upstream frozen @ ebda01d (tag upstream-legacy)
- G1 PASS: data-contract smoke on frozen baseline (results/package_baseline/g1_smoke/)
- G2 CLOSED: CPU/GPU parity verified on the adversarial fixture
  (tests/test_gpu_cpu_parity.py) and on the official full-SNP inference
  fixture (results/package_baseline/g2_full_parity/parity.log). No more
  reader-layer work unless the reader code changes.
- F1 ACCEPTED: cuDF compatibility-only patch, no intended algorithmic change
- G3 DONE: model contracts for the package yaml model, the paper CA model,
  and the 2025 residual variant (results/package_baseline/g3_model_contract/)
- G4 BLOCKED BY G3: corrected training implementation (new code, upstream
  Train untouched)
- G5 BLOCKED BY G4: synthetic micro-overfit
- D0 CAN RUN IN PARALLEL: scientific dataset recovery
