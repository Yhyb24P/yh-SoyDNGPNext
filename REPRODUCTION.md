# SoyDNGPNext Reproduction

- Upstream repository: https://github.com/IndigoFloyd/SoyDNGPNext
- Frozen commit: see `UPSTREAM_COMMIT.txt`
- `upstream-legacy` tag: unmodified upstream baseline
- `repro` branch: auditable reproduction fixes and experiments

## Repro fixes (repro branch only, NOT in upstream-legacy)

- **F1 (G2)**: `soydngpnext/reader.py` — replaced `df.index.values_host`
  (removed in cudf >= 23; the 26.x Index is also not iterable) with
  `df.index.to_numpy()`. Host-side ordered sample IDs, same semantics as the
  original `values_host`. Environment adaptation only. Triggered by G2
  parity test on cudf-cu13 26.8.1 / cupy-cuda13x 14.2.0.
