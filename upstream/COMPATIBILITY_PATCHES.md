# Compatibility patches

Environment-adaptation patches on the frozen upstream code. Every entry is
compatibility-only: no intended algorithmic change. Algorithmic or
training-protocol fixes are NOT recorded here; they use the T-series
(T1, T2, ...) when the corrected trainer (G4) is written.

## F1

file:
  soydngpnext/reader.py

upstream:
  df.index.values_host

repro:
  df.index.to_numpy()

reason:
  cuDF >= 23 removed Index.values_host (and the 26.x Index is not iterable);
  to_numpy() returns host-side ordered sample IDs on both cudf and pandas.

semantic impact:
  none intended (ordered sample IDs, same semantics as values_host)

verification:
  adversarial CPU/GPU parity PASS (tests/test_gpu_cpu_parity.py, G2)
