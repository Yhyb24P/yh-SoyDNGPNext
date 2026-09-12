"""Run the G4 contract tests, write results/trainer_validation/g4/.

Outputs:
    results/trainer_validation/g4/contract.json   (PASS gates)
    results/trainer_validation/g4/test.log         (full test output)

Run (CPU is enough, soydngp312 env):
    /home/yhshy/miniconda3/envs/soydngp312/bin/python scripts/run_g4_validation.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# contract gate -> test file
KEYS = [
    ("regression_float32_B1", "tests/test_regression_target_contract.py"),
    ("no_broadcasting", "tests/test_regression_target_contract.py"),
    ("classification_int64_B", "tests/test_classification_target_contract.py"),
    ("full_epoch_metrics", "tests/test_metrics_contract.py"),
    ("pcc_matches_reference", "tests/test_metrics_contract.py"),
    ("macro_f1_matches_reference", "tests/test_metrics_contract.py"),
    ("package_70_30_protocol_frozen", "tests/test_split_contract.py"),
    ("paper_10fold_protocol_frozen", "tests/test_split_contract.py"),
    ("trait_state_isolated", "tests/test_trait_state_isolation.py"),
    ("checkpoint_roundtrip_exact", "tests/test_checkpoint_roundtrip.py"),
]


def main():
    out_dir = os.path.join(ROOT, "results", "trainer_validation", "g4")
    os.makedirs(out_dir, exist_ok=True)
    log = []
    file_results = {}
    for key, path in KEYS:
        if path in file_results:
            continue
        log.append(f"== {path} ==")
        proc = subprocess.run([PY, os.path.join(ROOT, path)],
                              capture_output=True, text=True, cwd=ROOT)
        log.append(proc.stdout)
        if proc.stderr:
            log.append(proc.stderr)
        log.append(f"exit code: {proc.returncode}\n")
        file_results[path] = proc.returncode == 0
    with open(os.path.join(out_dir, "test.log"), "w") as f:
        f.write("\n".join(log))
    contract = {key: file_results[path] for key, path in KEYS}
    contract["all_pass"] = all(contract.values())
    with open(os.path.join(out_dir, "contract.json"), "w") as f:
        json.dump(contract, f, indent=2)
    print(json.dumps(contract, indent=2))
    sys.exit(0 if contract["all_pass"] else 1)


if __name__ == "__main__":
    main()
