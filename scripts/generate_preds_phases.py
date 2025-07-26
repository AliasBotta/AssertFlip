import os
import json
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd

load_dotenv()
DATASET = "princeton-nlp/SWE-bench_Verified"

ds = load_dataset(DATASET, split="test")
dataset = pd.DataFrame(ds)

# Configuration
# Adjust these paths as needed
RESULTS_DIR = Path()
OUT_FILE = RESULTS_DIR / ""
MODEL_NAME = "AssertFlip"
INTERESTING_INSTANCES = []

# Mapping from repo name → target path prefix
REPO_TO_PREFIX = {
    "astropy": "astropy/tests/test_assertflip_",
    "django": "tests/test_assertflip_",
    "matplotlib": "lib/matplotlib/tests/test_assertflip_",
    "pallets": "tests/test_assertflip_",
    "mwaskom_seaborn": "tests/test_assertflip_",
    "psf": "test_assertflip_",
    "pydata": "xarray/tests/test_assertflip_",
    "pylint-dev": "tests/test_assertflip_",
    "pytest-dev": "testing/test_assertflip_",
    "scikit-learn": "sklearn/tests/test_assertflip_",
    "sphinx-doc": "tests/test_assertflip_",
    "sympy": "sympy/polys/tests/test_assertflip_",
}

def rewrite_patch_as_new_file(raw_patch: str, target_path: str) -> str:
    lines = raw_patch.splitlines()
    new_lines = []

    for line in lines:
        if line.startswith("diff --git"):
            new_lines.append(f"diff --git a/dev/null b/{target_path}")
            new_lines.append("new file mode 100644")
        elif line.startswith("---"):
            new_lines.append("--- /dev/null")
        elif line.startswith("+++"):
            new_lines.append(f"+++ b/{target_path}")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)

def make_patch(test_code: str, instance_id: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        old_file = tmp / f"{instance_id}_empty.py"
        new_file = tmp / f"{instance_id}_test.py"

        old_file.write_text("")  # baseline
        new_file.write_text(test_code)

        result = subprocess.run(
            ["git", "diff", "--no-index", "--minimal", str(old_file), str(new_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()

def already_processed(instance_id: str) -> bool:
    if not OUT_FILE.exists():
        return False
    with open(OUT_FILE) as f:
        for line in f:
            entry = json.loads(line)
            if entry["instance_id"] == instance_id:
                return True
    return False

def get_final_successful_test(attempts: list) -> dict:
    last = attempts[-2]
    if last.get("outcome") == "success" and "final_test" in last:
        return last
    return None

def get_target_path(instance_id: str) -> str:
    repo_name = instance_id.split("__")[0]
    prefix = REPO_TO_PREFIX.get(repo_name, "tests/test_assertflip_")  # fallback default
    return f"{prefix}{instance_id}.py"

def main():
    preds = []

    for fpath in tqdm(list(RESULTS_DIR.glob("attempts_*.json"))):
        with open(fpath) as f:
            attempts = json.load(f)

        instance_id = attempts[0]["instance_id"]

        if INTERESTING_INSTANCES and instance_id not in INTERESTING_INSTANCES:
            continue
        if already_processed(instance_id):
            continue

        last_successful = get_final_successful_test(attempts)
        if not last_successful:
            print(f"{instance_id}: No successful final_test found. Skipping.")
            continue

        final_test = last_successful["final_test"]
        target_path = get_target_path(instance_id)
        raw_patch = make_patch(final_test, instance_id)
        patch = rewrite_patch_as_new_file(raw_patch, target_path)

        preds.append({
            "instance_id": instance_id,
            "model_name_or_path": MODEL_NAME,
            "model_patch": patch
        })

        with open(OUT_FILE, "a") as out:
            out.write(json.dumps(preds[-1]) + "\n")

    print(f"\n Wrote {len(preds)} entries to {OUT_FILE}")

if __name__ == "__main__":
    main()
