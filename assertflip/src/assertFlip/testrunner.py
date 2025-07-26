from pathlib import Path
import tempfile
import subprocess
import typing as T
import json
import os
from .utils import subprocess_run


async def measure_test_coverage(*, test: str, tests_dir: Path, test_command: str, pytest_args='', test_attempt: dict = {}):
    """Runs a given test and returns the coverage obtained."""
    with tempfile.NamedTemporaryFile(prefix="test_", suffix='.py', dir=str(tests_dir), mode="w", delete=False) as t:
        try:
            t.write(test)
            t.flush()
            _executable = "/opt/miniconda3/envs/testbed/bin/python"
            _cover = "coverage run"
            if "pytest" not in test_command:
                pytest_args = ""
                is_module = False
                test_command_dir = Path(test_command.split()[0]).parent
                test_command_dir = "/testbed" / test_command_dir
                env = os.environ.copy()
                env['PYTHONPATH'] = str(test_command_dir)
            else:
                is_module = True
                env = None
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as j:
                try:
                    if os.getenv("PROJECT_NAME", "") == "django":
                        filename_only = os.path.splitext(os.path.basename(t.name))[0]
                        # should be picked up by runner as long as its in the same directory
                        _tname = filename_only
                    else:
                        _tname = t.name
                    _test_command = [*_executable.split(), '-m', *_cover.split(),
                                            *(['-m'] if is_module else []), *test_command.split(), *pytest_args.split(), _tname]
                    test_attempt["test_command"] = _test_command
                    test_attempt["test_command_env"] = env or {}
                    _erase_old_coverage_command = [*_executable.split(), '-m', 'coverage', 'erase']
                    _p = await subprocess_run(_erase_old_coverage_command,
                                            check=True)

                    # TODO :: check how to handle time out cases                
                    try:
                        p = await subprocess_run(_test_command, check=False, env=env, timeout=120)
                        rc = p.returncode
                        raw_out = p.stdout
                    except subprocess.TimeoutExpired as e:
                        # Do not run combine/json — propagate timeout to caller
                        print("Timed out")
                        raise

                    if not list(Path('.').glob('.coverage*')):
                        raise FileNotFoundError(
                            "No coverage data file produced — test may have crashed or done no work."
                        )
                    try:
                        _convert_to_json_command = [*_executable.split(), '-m', 'coverage', 'combine']
                        _p = await subprocess_run(_convert_to_json_command)
                        _convert_to_json_command = [*_executable.split(), '-m', 'coverage', 'json', '-o', j.name]
                        _p = await subprocess_run(_convert_to_json_command,
                                                check=True)
                    except Exception as e:
                        raise FileNotFoundError(f"Failed to convert coverage to JSON: {e}")

                    cov = json.load(j)
                finally:
                    j.close()
                    try:
                        os.unlink(j.name)
                    except FileNotFoundError:
                        pass
        finally:
            # pass
            t.close()
            try:
                os.unlink(t.name)
            except FileNotFoundError:
                pass
    return cov, rc, raw_out

