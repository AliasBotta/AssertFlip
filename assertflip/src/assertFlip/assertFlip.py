import asyncio
import json
import argparse
import subprocess
import re
import sys
import os
from types import SimpleNamespace
import typing as T

from pathlib import Path

from . import llm
from .segment import *
from .prompt.prompter import Prompter
from .testrunner import *
from pydantic import BaseModel, Field


def load_dataset(dataset: Path):
    with dataset.open() as f:
        return json.load(f)

def get_prompters() -> dict[str, T.Callable[[T.Any], Prompter]]:

    from .prompt.gpt_v2 import GptV2Prompter

    return {
        "gpt-v2": GptV2Prompter,
    }

prompter_registry = get_prompters()

def parse_args(args=None):
    ap = argparse.ArgumentParser(prog='AssertFlip',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--source-dir', type=Path, default='src',
                     help='directory where source code resides')

    def Path_dir(value):
        path_dir = Path(value).resolve()
        if not path_dir.is_dir(): raise argparse.ArgumentTypeError(f"\"{value}\" must be a directory")
        return path_dir

    ap.add_argument('--tests-dir', type=Path_dir, default='tests',
                    help='directory where tests reside')
    
    ap.add_argument('--dataset', type=Path, default='dataset.json',
                    help='path to the dataset file')
    ap.add_argument("--max-tokens", type=int, default=90_000, help='max. tokens above which context is truncated')
    def default_model():
        if 'OPENAI_API_KEY' in os.environ:
            return "gpt-4o"

    ap.add_argument('--model', type=str, default=default_model(),
                    help='OpenAI model to use')

    ap.add_argument('--prompt', '--prompt-family', type=str,
                    choices=list(prompter_registry.keys()),
                    default='gpt-v2',
                    help='Prompt style to use')
    
    ap.add_argument('--phase', choices=['both', 'pass_then_invert', 'direct_fail_variant'], default='pass_then_invert', help='Which phases to run')
    ap.add_argument('--no-cov-opt', action='store_true', help='disable coverage optimization')
    ap.add_argument('--no-llm-validation', action='store_true', help='disable LLM validation')
    ap.add_argument('--max-generation-retries', type=int, default= 10,
                    help='max number of retries for regeneration attempts')
    
    ap.add_argument('--no-planner', action='store_true', help='disable Planner')

    ap.add_argument('--model-temperature', type=float, default=0,
                    help='Model "temperature" to use')

    ap.add_argument('--rate-limit', type=int, default= 0,
                    help='max. tokens/minute to send in prompts')

    ap.add_argument('--max-attempts', type=int, default=10,
                    help='max. number of refinement attempts for a test')

    ap.add_argument('--max-backoff', type=int, default=64,
                    help='max. number of seconds for backoff interval')

    ap.add_argument('--pytest-args', type=str, default='',
                    help='extra arguments to pass to pytest')
    
    ap.add_argument('--test-cmd', type=str, default='pytest')

    ap.add_argument('--prefix', type=str, default='assertflip',
                    help='prefix to use for test file names')


    ap.add_argument('--repeat-tests', type=int, default=5,
                    help='number of times to repeat test execution to help detect flaky tests')
    ap.add_argument('--no-repeat-tests', action='store_const', const=0, dest='repeat_tests', help=argparse.SUPPRESS)

    ap.add_argument('--prompt-for-tests', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='prompt LLM for new tests')

    ap.add_argument('--isolate-tests', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='run tests in isolation (to work around any state pollution) when measuring suite coverage')

    ap.add_argument('--branch-coverage', default=True,
                    action=argparse.BooleanOptionalAction,
                    help=argparse.SUPPRESS)


    args = ap.parse_args(args)

    if not args.model:
        ap.error('Specify the model to use with --model')

    return args


def test_file_path(args, instance_id: str) -> Path:
    """Returns the Path for a test's file, given its sequence number."""
    return args.tests_dir / f"test_{args.prefix}_{instance_id}.py"


def new_test_file(args: argparse.Namespace, instance_id: str) -> Path:
    """Creates a new test file, returning its Path."""
    p = test_file_path(args, instance_id)
    # Check if it already exists or is disabled
    if p.exists() or (p.parent / f"disabled_{p.name}").exists():
        raise FileExistsError(f"Test file already exists: {p}")

    p.touch(exist_ok=False)
    return p


def clean_error(error: str) -> str:
    """Conservatively removes pytest-generated (and possibly other) output not needed by GPT,
       to cut down on token use.  Conservatively: if the format isn't recognized, leave it alone."""

    if (match := re.search("=====+ (?:FAILURES|ERRORS) ===+\n" +\
                           "___+ [^\n]+ _+___\n" +\
                           "\n?" +\
                           "(.*)", error,
                           re.DOTALL)):
        error = match.group(1)

    if (match := re.search("(.*\n)" +\
                           "===+ short test summary info ===+", error,
                           re.DOTALL)):
        error = match.group(1)

    return error

# ---------------------- PROGRESS TRACKING ------------------------

PROGRESS_COUNTERS=['G', 'F', 'U', 'R']  # good, failed, useless, retry
class Progress:
    """Tracks progress, showing a tqdm-based bar."""

    def __init__(self, total, initial):
        import tqdm
        from collections import OrderedDict

        self._bar = tqdm.tqdm(total=total, initial=initial)

        self._postfix = OrderedDict()
        for p in [*PROGRESS_COUNTERS, 'cost']:
            self._postfix[p] = ''  # to establish order

        self._bar.set_postfix(ordered_dict=self._postfix)

    def update_cost(self, cost: float):
        self._postfix['cost'] = f'~${cost:.02f}'
        self._bar.set_postfix(ordered_dict=self._postfix)

    def update_counters(self, counters: dict):
        """Updates the counters display."""
        for k in counters:
            self._postfix[k] = counters[k]

        self._bar.set_postfix(ordered_dict=self._postfix)

    def signal_one_completed(self):
        """Signals an item completed."""
        self._bar.update()

    def close(self):
        """Closes the underlying tqdm bar."""
        self._bar.close()


class State:
    def __init__(self): 
        """Initializes the state."""

        self._cost = 0.0
        self._counters = {k:0 for k in PROGRESS_COUNTERS}
        self._bar: Progress|None = None


    def set_progress_bar(self, bar: Progress):
        """Specifies a progress bar to update."""
        self._bar = bar
        if bar is not None:
            self._bar.update_cost(self._cost)
            self._bar.update_counters(self._counters)


    def add_cost(self, cost: float) -> None:
        self._cost += cost
        if self._bar:
            self._bar.update_cost(self._cost)


    def inc_counter(self, key: str):
        """Increments a progress counter."""
        self._counters[key] += 1

        if self._bar:
            self._bar.update_counters(self._counters)


def extract_python(response: str) -> str:
    m = re.search(r'```python\n(.*?)(?:```|\Z)', response, re.DOTALL)
    if not m: raise RuntimeError(f"Unable to extract Python code from response {response}")
    return m.group(1)

def add_to_pythonpath(dir: Path):
    import os
    os.environ['PYTHONPATH'] = str(dir) + (f":{os.environ['PYTHONPATH']}" if 'PYTHONPATH' in os.environ else "")
    sys.path.insert(0, str(dir))

state: State

def trim_messages(messages: T.List[dict]) -> T.List[dict]:
    if len(messages) < 3:
        return messages
    
    _sys = messages[0]
    _user = messages[1]
    _resp = messages[-1]
    return [_sys, _user, _resp]


# ---------------------- MAIN PIPELINE ------------------------

async def generate_bug_revealing_test(args, chatter, prompter, seg, phase_mode="pass_then_invert"):
    attempt_log = []
    attempt_log.append({
        "instance_id": seg.instance_id,
        "problem_statement": seg.problem_statement
    })

    if phase_mode in ("both", "pass_then_invert"):
        messages_passing = await new_prompt(args, seg, chatter, prompter, mode="passing_first")
        generation_retries = 0
        max_generation_retries = args.max_generation_retries

        while generation_retries < max_generation_retries:
            print(f"Phase A generation attempt {generation_retries + 1}/{max_generation_retries}...")
            attempt_log.append({"generation_attempt": generation_retries})

            result = await phase_a_cycle(args, chatter, prompter, seg, attempt_log, messages_passing, generation_retries)
            if result is not None:
                print("Phase A succeeded!")
                attempt_log.append({
                    "phase": "terminating",
                    "mode": "passing_first",
                    "final_test": result,
                    "outcome": "success",
                })
                attempt_log.append({
                    "meta": "cost_summary",
                    "total_cost": state._cost
                })
                await save_test(args, seg, result, "passing_first")
                await save_attempt_log(seg, attempt_log)
                return True

            generation_retries += 1
            # Rebuild prompt for retries
            messages_passing = await new_prompt(args, seg, chatter, prompter, mode="passing_first", retry=True)

        print("Phase A failed after max retries.")
        if phase_mode == "phase_a":
            attempt_log.append({
                "phase": "terminating",
                "mode": "passing_first",
                "outcome": "failure",
            })
            attempt_log.append({
                "meta": "cost_summary",
                "total_cost": state._cost
            })
            await save_attempt_log(seg, attempt_log)
            return False

    if phase_mode in ("both", "direct_fail_variant"):
        messages_failing = await new_prompt(args, seg, chatter, prompter, mode="failing_first")
        generation_retries = 0
        max_generation_retries = args.max_generation_retries

        while generation_retries < max_generation_retries:
            print(f"Phase B generation attempt {generation_retries + 1}/{max_generation_retries}...")
            attempt_log.append({"generation_attempt": generation_retries})

            failing_test, _ = await generate_failing_test(
                args, chatter, prompter, seg, attempt_log, messages_failing
            )
            if failing_test:
                print("Phase B succeeded!")
                attempt_log.append({
                    "phase": "terminating",
                    "mode": "failing_first",
                    "final_test": failing_test,
                    "outcome": "success",
                })
                attempt_log.append({
                    "meta": "cost_summary",
                    "total_cost": state._cost
                })
                await save_test(args, seg, failing_test, "failing_first")
                await save_attempt_log(seg, attempt_log)
                return True

            generation_retries += 1
            messages_failing = await new_prompt(args, seg, chatter, prompter, mode="failing_first", retry=True)

        print("Phase B failed after max retries.")
        attempt_log.append({
            "phase": "terminating",
            "mode": "failing_first",
            "outcome": "failure",
        })
        attempt_log.append({
            "meta": "cost_summary",
            "total_cost": state._cost
        })
        await save_attempt_log(seg, attempt_log)
        return False

# ---------------------- HELPERS ------------------------

async def phase_a_cycle(args, chatter, prompter, seg, attempt_log, messages_passing, generation_retries):
    passing_test, _ = await generate_passing_test(args, chatter, prompter, seg, attempt_log, messages_passing)
    if not passing_test:
        return None

    test_to_invert = passing_test

    retry_msg = None
    for invert_attempt in range(1, 4):
        print(f"Inversion Attempt {invert_attempt}/3...")
        inverted_test = await ask_llm_to_invert_test(chatter, seg, prompter, test_to_invert, attempt_log, invert_attempt, retry_msg)
        if not inverted_test:
            retry_msg = (
            "The previous response did not contain a valid Python code snippet. "
            "Please respond with complete Python code wrapped in triple backticks like: ```python\n...\n```."
            )
            continue

        status, coverage, error = await run_test_and_capture(args, seg, inverted_test, attempt_log, phase="invert_to_failing", attempt=invert_attempt)

        if status == "passing":
            print("Inverted test passed unexpectedly. Retrying inversion.")
            retry_msg = (
            "The inverted test passed but it should fail to reveal the bug. "
            "Please rewrite the test to ensure it fails and exposes the issue."
            )
            continue

        elif status == "failing":  
            # Skip LLM validation if disabled
            if args.no_llm_validation:
                print("Skipping LLM validation (disabled). Accepting test as bug-revealing.")
                return inverted_test
    
            validation_result = await validate_bug_with_llm(chatter, prompter, seg, inverted_test, error, attempt_log)
            if validation_result.revealing:
                print("LLM confirms the test exposes the bug.")
                return inverted_test
            

            print("Test failed validation. Triggering new generation attempt.")
            attempt_log.append({
                "phase": "generation_feedback",
                "generation_attempt": generation_retries + 1,
                "feedback": "Previous test was rejected — trying a different approach."
            })
            seg.continue_from = {
                "plan": seg.continue_from.get("plan", ""),
                "test_code": inverted_test,
                "error": error,
                "reason": validation_result.reason
            }
            return None

    # If we reach here, all inversion attempts failed
    print("All inversion attempts failed. Triggering new generation attempt.")
    attempt_log.append({
        "phase": "invert_to_failing",
        "outcome": "failure",
        "reason": "All inversion attempts produced passing tests; no failing test to validate"
    })
    seg.continue_from = {
        "plan": seg.continue_from.get("plan", ""),
        "test_code": test_to_invert,
        "error": "Inverted test never failed after 3 tries",
        "reason": "Inversion did not yield a failing test"
    }
    return None

async def generate_passing_test(args, chatter, prompter, seg, attempt_log, messages):
    test = await generate_test(args, chatter, prompter, seg, attempt_log, attempt_num=1, mode="passing_first", messages=messages)
    if not test:
        print("Initial passing test generation failed — cannot proceed.")
        return None, None

    return await fix_test_loop(
        args, chatter, prompter, seg, test, attempt_log,
        phase="generate_passing_test", goal_status="passing",
        validate_with_llm=False, error_prompt_mode="passing_first", messages=messages
    )

async def ask_llm_to_invert_test(chatter, seg, prompter, test_code, attempt_log, attempt_num, retry_message: str = None):
    inversion_prompt = prompter.invert_test_prompt(seg.problem_statement, test_code)
    if retry_message:
        inversion_prompt.append({
            "role": "user",
            "content": retry_message
        })

    response = await chatter.chat(inversion_prompt, ctx=seg)
    if not response:
        attempt_log.append({"phase": "invert_to_failing", "attempt": attempt_num, "status": "failed_to_generate"})
        return None

    response_message = response["choices"][0]["message"]

    if response_message.get("content") and '```python' in response_message["content"]:
        code = extract_python(response_message["content"])
        attempt_log.append({"phase": "invert_to_failing", "attempt": attempt_num, "status": "generated", "test_code": code})
        return code

    attempt_log.append({"phase": "invert_to_failing", "attempt": attempt_num, "status": "failed_to_generate", "error": "No Python code in response"})
    return None

# ---------------------- PHASE B HELPERS ------------------------

async def generate_failing_test(args, chatter, prompter, seg, attempt_log, messages):
    test = await generate_test(args, chatter, prompter, seg, attempt_log, attempt_num=1, mode="failing_first", messages=messages)
    if not test:
        print("Initial failing test generation failed — skipping.")
        return None, None

    return await fix_test_loop(
        args, chatter, prompter, seg, test, attempt_log,
        phase="generate_failing_test", goal_status="failing",
        validate_with_llm=not args.no_llm_validation, error_prompt_mode="failing_first", messages=messages
    )

# ---------------------- COMMON HELPERS ------------------------

async def new_prompt(args, seg, chatter, prompter, *, mode: str, retry: bool = False):
    planner = not args.no_planner
    messages, plan = await prompter.initial_prompt(seg, chatter, mode=mode, retry_generation=retry, planner= planner)
    if seg.continue_from is None:
        seg.continue_from = {"plan": plan}
    seg.continue_from["plan"] = plan
    return messages

async def generate_test(args, chatter, prompter, seg, attempt_log, attempt_num, mode, messages, max_attempts=3):
    for sub_attempt in range(1, max_attempts + 1):
        print(f"[LLM] Attempting to generate test (try {sub_attempt}/{max_attempts})...")
        response = await chatter.chat(messages, ctx=seg)
        if not response:
            attempt_log.append({
                "phase": "first_test",
                "attempt": attempt_num,
                "sub_attempt": sub_attempt,
                "status": "no_response"
            })
            print("No response from LLM.")
            await asyncio.sleep(1)
            continue

        response_message = response["choices"][0]["message"]
        content = response_message["content"]
        messages.append(response_message)
        messages = trim_messages(messages) 

        if '```python' in content:
            test_code = extract_python(content)
            if test_code:
                attempt_log.append({
                    "phase": "first_test",
                    "attempt": attempt_num,
                    "sub_attempt": sub_attempt,
                    "status": "generated",
                    "test_code": test_code
                })
                return test_code

        attempt_log.append({
            "phase": "first_test",
            "attempt": attempt_num,
            "sub_attempt": sub_attempt,
            "status": "no_python_code",
            "content": content[:500]
        })
        messages.append({
            "role": "user",
            "content": "Please provide a complete and valid Python code snippet wrapped in triple backticks like: ```python\n...\n```."
        })
        await asyncio.sleep(1)

    print("All attempts to generate initial test failed.")
    return None

async def run_test_and_capture(args, seg, test_code, attempt_log, phase, attempt):
    test_attempt = {"phase": phase, "attempt": attempt}
    try:
        pytest_args = (f"--count={args.repeat_tests} " if args.repeat_tests else "") + args.pytest_args if args.test_cmd == "pytest" else ""
        coverage, rc, raw = await measure_test_coverage(
            test=test_code, tests_dir=args.tests_dir, test_command=args.test_cmd,
            pytest_args=pytest_args,test_attempt=test_attempt
        )
        status = "failing" if rc else "passing"
        error = None
        if status == "failing":
            error = clean_error(raw.decode("utf-8", errors="ignore"))
            test_attempt["error"] = error
        test_attempt.update({"status": status, "coverage": coverage["totals"], "test_code": test_code})
        attempt_log.append(test_attempt)
        return status, coverage, error
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        status = "error"
        error_output = e.stdout.decode("utf-8", errors="ignore") if hasattr(e, "stdout") and e.stdout else str(e)
        if hasattr(e, "stderr") and e.stderr:
            error_output += "\n" + e.stderr.decode("utf-8", errors="ignore")
        test_attempt.update({"status": status, "error": error_output, "test_code": test_code})
        attempt_log.append(test_attempt)
        return status, None, error_output
    

def normalize(path: str) -> str:
    path = os.path.normpath(path).replace("\\", "/")
    path = path.lstrip("./")
    if path.startswith("/testbed/"):
        path = path[len("/testbed/"):]
    return path


async def fix_test_loop(args, chatter, prompter, seg, test, attempt_log, phase, goal_status, validate_with_llm=False, error_prompt_mode="error_fix", messages=None, max_attempts= 10):
    attempt = 1
    while attempt <= args.max_attempts:
        state.inc_counter("F")
        print(f"{phase.capitalize()} Attempt {attempt}...")
        status, coverage, error = await run_test_and_capture(args, seg, test, attempt_log, phase=phase, attempt=attempt)
        last_error = error

        if goal_status == "passing" and status == "passing":
            print("Test passed — done.")
            return test, coverage

        if goal_status == "failing" and status in ("failing", "error"):
            if validate_with_llm:
                test_for_validation = test
                error_for_validation = error

                validation = await validate_bug_with_llm(
                    chatter, prompter, seg, test_for_validation, error_for_validation, attempt_log
                    )
                
                if validation.revealing:
                    print("LLM confirms the test exposes the bug.")
                    return test_for_validation, error_for_validation

                print("LLM says failure is unrelated — fixing test ...")
                seg.continue_from = {
                    "plan": seg.continue_from.get("plan"),
                    "test_code": test_for_validation,
                    "error": error_for_validation,
                    "reason": validation.reason,
                }
                test, error = test_for_validation, error_for_validation
            else:
                # If validation is not required, just return the test and error
                print("Test failed as expected, LLM validation is disabled.")
                return test, error

        prompt_error = error or (
            "Test passed but should fail to reveal the bug." if goal_status == "failing" and status == "passing" else ""
        )
        if validate_with_llm and goal_status == "failing" and seg.continue_from.get("reason"):
            prompt_error += f"\n\nVALIDATION_FEEDBACK:\n{seg.continue_from['reason'].strip()}"


        prompts = prompter.error_prompt(seg, prompt_error, error_prompt_mode)
        if prompts and prompts[0]["content"] == "__ABANDON__":
            print("Stuck on identical traceback — abandoning this test.")
            seg.continue_from = {
                "plan": seg.continue_from.get("plan"),
                "test_code": test,
                "error": error,
                "reason": "Repeated identical traceback",
            }
            return None, None

        if not prompts:
            prompts = [{
                "role": "user",
                "content": "The test did not meet the goal. Please rewrite the test in a different way."
            }]

        messages.extend(prompts)
        response = await chatter.chat(messages, ctx=seg)
        if not response:
            print("No response from LLM — retrying.")
            attempt += 1
            continue

        response_message = response["choices"][0]["message"]
        messages.append(response_message)
        messages = trim_messages(messages)

        attempt_log.append({"phase": phase, "attempt": attempt, "response": response_message})
        if "```python" in response_message["content"]:
            test = extract_python(response_message["content"])
            attempt_log[-1]["test_code"] = test
            attempt += 1
            continue
        else:
            print("LLM produced no Python code — retrying.")
            attempt_log[-1]["error"] = "No Python code found in LLM response."
            attempt += 1
            continue

    print(f"Failed after {max_attempts} attempts.")
    seg.continue_from = {
        "plan": seg.continue_from.get("plan", ""),
        "test_code": test,
        "error": last_error or "<no error captured>",
        "reason": "Exceeded max attempts in fix_test_loop",
    }
    return None, None

async def validate_bug_with_llm(chatter, prompter, seg, test_code, error_output, attempt_log):
    validation_messages = prompter.validation_prompt(seg, error_output, test_code)
    class TestValidation(BaseModel):
        reason: str = Field(..., description="Reason for the validation")
        revealing: bool = Field(..., description="Whether the test failure reveals the bug")

    validation_ctx = SimpleNamespace(instance_id=f"{seg.instance_id}-validation")
    result = await chatter.chat(validation_messages, ctx=validation_ctx, response_format=TestValidation)

    attempt_log.append({
        "phase": "validate_bug_with_llm",
        "test_code": test_code,
        "error": error_output,
        "revealing": result.revealing if result else None,
        "reason": result.reason if result else None
    })
    return result

async def save_test(args, seg, test_code, mode):
    new_test = new_test_file(args, seg.instance_id)
    new_test.write_text(test_code)
    state.inc_counter('G')

async def save_attempt_log(seg, attempt_log):
    attempt_path = Path("/results") / f"attempts_{seg.instance_id}.json"
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(attempt_path, "w") as f:
        json.dump(attempt_log, f, indent=2)
    print(f"Saved all attempts to {attempt_path}")

def main():
    print(f"Running AssertFlip to generate bug-revealing tests for the reported issue...")
    global state
    args = parse_args()

    if args.prompt_for_tests:
        try:
            chatter = llm.Chatter(model=args.model)
            chatter.set_signal_retry(lambda: state.inc_counter('R'))
            chatter.set_model_temperature(args.model_temperature)
            chatter.set_max_backoff(args.max_backoff)
            if args.rate_limit > 0:
                chatter.set_token_rate_limit((args.rate_limit, 60))

            prompter = prompter_registry[args.prompt](cmd_args=args)
            for f in prompter.get_functions():
                chatter.add_function(f)

        except llm.ChatterError as e:
            print(e)
            return 1

         # --- Load dataset --- #   
        dataset = load_dataset(args.dataset)
        selected_files = []
        objects = dataset["line_level_localization"]
        for obj in objects:
            selected_files.append(obj["filename"])

        state = State()
        chatter.set_add_cost(state.add_cost)

        segments = []
        problem_statement = dataset["problem_statement"]    
        localized_code = dataset["localized_code"]
        line_level_localization = dataset["line_level_localization"]
        instance_id = dataset["instance_id"]
        continue_from = dataset.get("continue_from", None)
 
        while llm.count_tokens(args.model, {"messages": [{"role": "user", "content": localized_code}]} ) > args.max_tokens:
            localized_code = "\n".join(localized_code.split("\n")[:-10])
            print(f"Trimming localized code to {len(localized_code.splitlines())} lines")
        # Create CodeSegment objects correctly
        segments.append(CodeSegment(
                    filename="",
                    name="",
                    begin=0,
                    end=0,
                    lines_of_interest=set(),
                    missing_lines=set(),
                    executed_lines=set(),
                    missing_branches=set(),
                    context=[],
                    imports=[],
                    problem_statement=problem_statement,
                    buggy_files=localized_code,
                    line_level_localization=line_level_localization,
                    instance_id=instance_id,
                    continue_from=continue_from
            ))


        # --- prompt for tests ---

        print(f"Prompting {args.model} for tests that reproduce the issue reported...")
        print("(in the following, P=pass, F=failed, U=useless and R=retry)")

        async def work_segment(seg: CodeSegment) -> None:
            """Sends the buggy files to the LLM and waits for a result."""

            progress = Progress(total=1, initial=0)
            state.set_progress_bar(progress)
            await generate_bug_revealing_test(args, chatter, prompter, seg, phase_mode=args.phase)  
            progress.signal_one_completed()
            progress.close()
            
        asyncio.run(work_segment(segments[0]))

    return 0

