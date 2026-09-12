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
- G4 DONE: corrected trainer in src/soydngp_repro/ (T1-T4 contracts,
  upstream Train untouched); all 10 contract gates PASS
  (results/trainer_validation/g4/contract.json)
- G5 DONE: synthetic micro-overfit, both gates PASS
  (results/trainer_validation/g5/; regression PCC 0.9895, classification
  acc 1.0 / macro-F1 1.0)
- D0-Raw CLOSED: full SoySNP50K gnm2 VCF acquired + verified (20,087 samples
  x 41,726 variants; MD5+SHA-256 in data_manifest/provenance.yaml
  soysnp50k_gnm2). 41,726 is the authoritative raw fact (the 42,509 figure
  cited elsewhere does not match this file).
- D1 DONE: ordered 32,032 paper-SNP contract (results/data_contract/d1/).
  32,033 source lines - 1 (pd.read_csv consumes line 1 as the header) =
  32,032 effective (= the paper's figure; the old off-by-one is resolved).
  Joined to the raw 41,726 on (chrom, gnm2 pos): 31,673 matched, 359 missing
  (imputed './.' -> code 3, author default no-Beagle), 10,053 raw unselected.
- D2 DONE: sample x phenotype alignment (results/data_contract/d2/). 16,960
  exact matches; 1,441 zero-pad candidate pairs (PI/FC differ only by leading
  zeros); 1,686 vcf_only, 2,105 phenotype_only; 0 duplicates. Phenotype
  missingness is row-level (0 partial-missing), so all 23 traits share one
  non-missing cohort. Decision (d2_decision.json, user 2026-09-11): ACCEPT the
  1,441 zero-pad pairs -> matched cohort 18,401, non-missing 15,899.
- D3 DONE: paper input matrix (data/derived/d3_matrix/d3_matrix.npy,
  15,899 x 32,032 int8) + historical preprocessing parity PASS
  (results/data_contract/d3/d3_manifest.json). 31,673 SNPs read from the
  reduced VCF, 359 missing -> code 3; one-hot 206x206 wrap verified.
- G6 DONE: real-data smoke test on the D3 matrix (1,024 subset x 50 epochs,
  NOT 150). Pipeline-health gates PASS for both protein (regression) and
  maturity_group (classification): finite loss/preds/grads, checkpoint reloads
  to identical predictions (results/trainer_validation/g6/). Loss decrease is
  reported but not gated (slow bias-free head; convergence is G7's job).
- Pipeline: D0-Raw(CLOSED) -> D1(DONE) -> D2(DONE) -> D3(DONE) -> G6(DONE) ->
  G7(paper-faithful baseline) -> G8(corrected/fair baseline).
- G7 COMPLETE (2026-09-12): paper-faithful baseline, 2-trait pilot
  (protein regression + maturity_group classification, 10 levels), 10-fold x
  150 epochs, PAPER_REPRO, full D3 matrix (15,899). Result: protein PCC
  @150ep 0.6701 +/- 0.0153; maturity_group macro-F1 @150ep 0.5544 +/- 0.0173
  (best-epoch 0.5699 +/- 0.0159). Script scripts/g7_paper_baseline.py
  (quick = 2-fold x 10-ep harness check, full = 10-fold x 150-ep). Ran
  detached (setsid) with per-fold checkpointing
  (results/trainer_validation/g7/g7_progress.jsonl, resume skips done folds);
  watchdog tools/watch_g7_training.sh (crontab */10) auto-relaunched across a
  machine suspend. Per-epoch training-curve visualization:
  scripts/plot_g7_curves.py (g7_curves_full.png: protein 8 folds /
  maturity_group 10 folds) + rfig contract results/trainer_validation/g7/rfig/
  (g7_curves_rfig.png, double-column 183 mm, CJK labels via RFIG_FONT,
  8-fold intersection; CSV built by scripts/make_g7_rfig_csv.py). Protein
  folds 0,1 were resumed from checkpoint so their per-epoch curves were not
  re-captured (their final metrics remain in the 10-fold aggregate). Pushed
  to public repo Yhyb24P/yh-SoyDNGPNext (repro branch = default;
  upstream-legacy tag = frozen baseline).
- NEXT after G7: expand to all 23 traits (optional), then G8. ONNX export of
  the author's .pt weights (opset <= 26) can run in parallel.
