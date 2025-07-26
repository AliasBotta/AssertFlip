import os
import re
import json
import subprocess
from constants import MAP_REPO_VERSION_TO_SPECS, CUSTOM_INSTRUCTIONS
import docker
from pathlib import Path
import shlex
from unidiff import PatchSet
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    DATASET_PATH,
    DATASET,
    ASSERTFLIP_DIR,
    RESULTS_DIR,
    max_attempts,
    model,
    phase_mode,
    max_generation_retries,
)

# Parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

NUM_WORKERS = 1  # Number of parallel workers

load_dotenv()

# List of instances you want to rerun
FORCE_INSTANCES = {

}

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define repo directories mapping
repo_directories = {
    'flask': {'source_dir': 'src/flask', 'tests_dir': 'tests'},
    'django': {'source_dir': 'django', 'tests_dir': 'tests'},
    'matplotlib': {'source_dir': 'lib/matplotlib', 'tests_dir': 'lib/matplotlib/tests'},
    'pylint': {'source_dir': 'pylint', 'tests_dir': 'tests'},
    'pytest': {'source_dir': 'src/_pytest', 'tests_dir': 'testing'},
    'requests': {'source_dir': 'requests', 'tests_dir': 'tests'},
    'scikit-learn': {'source_dir': 'sklearn', 'tests_dir': 'sklearn/tests'},
    'seaborn': {'source_dir': 'seaborn', 'tests_dir': 'tests'},
    'sphinx': {'source_dir': 'sphinx', 'tests_dir': 'tests'},
    'sympy': {'source_dir': 'sympy', 'tests_dir': 'sympy/testing'},
    'astropy': {'source_dir': 'astropy', 'tests_dir': 'astropy/tests'},
    'xarray': {'source_dir': 'xarray', 'tests_dir': 'xarray/tests'}
}

NON_TEST_EXTS = [
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
]

def parse_diff(diff_text: str):
    try:
        patch_set = PatchSet(diff_text)
    except Exception:
        return []
    file_changes = []
    for patched_file in patch_set:
        if patched_file.is_added_file:
            continue
        sorted_lines = []
        file_path = patched_file.path
        if not file_path.startswith('/'):
            file_path = '/' + file_path
        file_changes.append([file_path, sorted_lines])
    return file_changes

def get_test_dir(repo_name: str, instance_id: str) -> str:
    if instance_id in ["psf__requests-1142", "psf__requests-1921", "psf__requests-2931"]:
        return "."
    if repo_name == "sympy":
        ds = load_dataset(DATASET, split="test")
        dataset = pd.DataFrame(ds)
        _test_patch = dataset[dataset["instance_id"] == instance_id]["test_patch"].values[0]
        patch = parse_diff(_test_patch)
        if patch:
            for p in patch:
                if p[0].endswith(".py"):
                    return str(Path(p[0]).parent).strip("/")
    return repo_directories[repo_name]['tests_dir']

def get_test_directives(repo: str, test_patch: str) -> list:
    if repo == "swe-bench/humaneval":
        return ["test.py"]
    diff_pat = r"diff --git a/.* b/(.*)"
    directives = re.findall(diff_pat, test_patch)
    directives = [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]
    if repo == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[:-len(".py")] if d.endswith(".py") else d
            d = d[len("tests/"):] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed
    return directives

def file_exists_in_container(container_id: str, file_path: str) -> bool:
    check_cmd = f"test -f {shlex.quote(file_path)}"
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", check_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

def process_instance(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single SWE‑bench instance inside its own Docker container.

    Returns a dict describing the outcome so the parent process can log it.
    """
    instance_id = instance["instance_id"]
    outcome = {
        "instance_id": instance_id,
        "status": "success",
        "error": "",
    }

    attempts_path = Path(RESULTS_DIR) / f"attempts_{instance_id}.json"
    if instance_id not in FORCE_INSTANCES and attempts_path.exists():
        try:
            with open(attempts_path) as f:
                if os.path.getsize(f.name):
                    data = json.load(f)
                    if data:
                        outcome["status"] = "skipped"
                        return outcome
        except Exception:
            pass  # fall through and re‑process if file is corrupt / empty

    repo_full_name = instance["repo"]  # e.g. "astropy/astropy"
    repo_name = repo_full_name.split("/")[-1]

    if repo_name not in repo_directories:
        outcome["status"] = "skipped"
        outcome["error"] = f"Unknown repo: {repo_name}"
        return outcome

    # Docker image name transformation
    id_docker_compatible = instance_id.replace("__", "_1776_")
    image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}"

    source_dir = repo_directories[repo_name]['source_dir']
    tests_dir = get_test_dir(repo_name, instance_id)

    temp_json_path = Path(RESULTS_DIR) / f"{instance_id}.json"
    try:
        with open(temp_json_path, "w") as json_file:
            json.dump(instance, json_file, indent=4)

        version = str(instance["version"])
        if version not in MAP_REPO_VERSION_TO_SPECS[repo_full_name]:
            version_alt = version + "0"
            if version_alt not in MAP_REPO_VERSION_TO_SPECS[repo_full_name]:
                outcome["status"] = "skipped"
                outcome["error"] = f"Version {version} not in specs"
                return outcome
            version = version_alt
        _spec = MAP_REPO_VERSION_TO_SPECS[repo_full_name][version]
        test_cmd = _spec["test_cmd"]
        eval_commands = _spec.get("eval_commands", [])
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]
        
        result = subprocess.run(["docker", "images", "-q", image_name], capture_output=True, text=True)
        imageStored =  bool(result.stdout.strip())
        if imageStored:
            print("Using image from cache for: ", image_name)
        else:
            print("Pulling: ", image_name)
            subprocess.run(["docker", "pull", image_name], check=True)

        docker_client = docker.from_env()
        
        env = {
            "AZURE_API_KEY": os.getenv("AZURE_API_KEY", ""),
            "AZURE_API_BASE": os.getenv("AZURE_API_BASE", ""),
            "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION", ""),
            "CUSTOM_INSTRUCTIONS": CUSTOM_INSTRUCTIONS.get(repo_name, None),
            "PROJECT_NAME": repo_name,
        }
        # Ensure RESULTS_DIR is absolute
        results_dir_abs = str(Path(RESULTS_DIR).resolve())
        container = docker_client.containers.run(
            image_name,
            command="sleep infinity",
            detach=True,
            remove=False,
            volumes={
                str(results_dir_abs): {'bind': '/results', 'mode': 'rw'},
            },
            environment=env,
        )
        container_id = container.id
        container_assertflip_path = f"/assertflip"

        # Copy into container (safe for parallel builds)
        subprocess.run(
          ["docker", "cp", ASSERTFLIP_DIR, f"{container_id}:{container_assertflip_path}"],
          check=True,
        )
        env_export = "export PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning && " if repo_name == 'sympy' else ""
        if repo_name == 'sympy':
            test_cmd = "/testbed/bin/test -C --verbose"
        quoted_test_cmd = shlex.quote(test_cmd)

        # Prepare container: basic deps + coverage
        prep_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "python -m pip install coverage"
        )
        subprocess.run(f'docker exec {container_id} bash -c "{prep_cmd}"', shell=True, check=True)

        eval_cmd_str = " && ".join(eval_commands) + " && " if eval_commands else ""

        bash_cmd = (
            f"{env_export}{eval_cmd_str}"
            "pip install /assertflip && "
            "pip install hypothesis && "
            f"assertflip --test-cmd {quoted_test_cmd} "
            f"--source-dir {source_dir} "
            f"--tests-dir {tests_dir} "
            f"--max-attempts {max_attempts} "
            f"--dataset /results/{instance_id}.json "
            f"--model {model} "
            f"--phase {phase_mode} "  # Options: pass_then_invert (default mode), direct_fail_variant 
            f"--max-generation-retries {max_generation_retries} " 
            f"--max-attempts {max_attempts} "
            # f"--no-llm-validation "  # Uncomment to disable LLM validation
        )
        subprocess.run(f'docker exec {container_id} bash -c "{bash_cmd}"', shell=True, check=True, timeout=3600)

        # Copy generated test file back to host (optional)
        generated_test_path = f"/testbed/{tests_dir}/test_assertflip_{instance_id}.py"
        local_test_path = Path(RESULTS_DIR) / f"test_assertflip_{instance_id}.py"
        if file_exists_in_container(container_id, generated_test_path):
            subprocess.run(
                ["docker", "cp", f"{container_id}:{generated_test_path}", str(local_test_path)],
                check=True,
            )
    except Exception as e:
        outcome["status"] = "error"
        outcome["error"] = str(e)
    finally:
        # Teardown container & image
        try:
            if 'container' in locals():
                container.stop()
                container.remove()
        except docker.errors.APIError:
            pass
        try:
            temp_json_path.unlink(missing_ok=True)
        except Exception:
            pass

    return outcome

def main():
    # Load dataset
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} instances from {DATASET_PATH}")

    error_counter = 0
    skipped_counter = 0

    # Submit all jobs to the process pool
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_instance, inst): inst["instance_id"] for inst in dataset}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result["status"] == "error":
                error_counter += 1
                print(f"[ERROR] {result['instance_id']}: {result['error']}")
            elif result["status"] == "skipped":
                skipped_counter += 1

    print(f"Finished. Errors: {error_counter}, Skipped: {skipped_counter}")


if __name__ == "__main__":
    main()
