import typing as T
from .prompter import *
import assertFlip.codeinfo as codeinfo
import os
try:
    from rep_package.assertflip.src.assertFlip.assertFlip import llm
except ModuleNotFoundError:
    from assertFlip import llm
from pathlib import Path

class GptV2Prompter(Prompter):
    """Prompter for GPT 4."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_planning_prompt(segment, mode:str) -> T.List[dict]:
        print("MODE IS", mode)
        assert mode in ("passing_first", "failing_first"), f"Unsupported initial_prompt mode: {mode}"
        if mode == "passing_first":
            goal = "pass if the bug is present and fail once the bug is fixed"
            plan_rules = """
9. Plan how to trigger the bug while ensuring the test passes.  
10. Your goal is to plan a test that passes (does not fail during execution), but still triggers and exposes the bug as described in the issue ticket. 
11. Plan assertions that will pass while still exposing the bug.
"""
        elif mode == "failing_first":
            goal = "fail if the bug is present and pass once the bug is fixed"
            plan_rules = """
9. Plan how to trigger the bug while ensuring the test fails due to the bug
10. Plan assertions that will expose the bug.
11. You must not plan tests that fail for reasons unrelated to the reported bug. Your test must fail only by exposing the specific incorrect behavior described in the issue, not because of test mistakes, unrelated exceptions, or missing dependencies.
"""
        return [
            mk_message(f"""\
You are an expert senior Python test-driven developer tasked with planning tests that reveal reported bugs to be used an oracle by your team. Your goal is to plan out the creation of one or more test functions that pass but still reproduce and expose the reported bug.
You will be provided with a ISSUE TICKET and a set of CODE SNIPPETS which might contain the buggy logic and examples of tests in the project for reference. Your task is to analyze the described problem in detail and plan your approach to writing the test.
                       
First, carefully read the ISSUE TICKET and inspect the collected CODE SNIPPETS.
Analyze the problem reported critically and plan your approach:

1. Identify the type of bug (exception, logic error, etc.)
2. Quote relevant parts of the ISSUE TICKET and the CODE SNIPPETS
3. Determine the expected incorrect behavior
4. List potential test cases
5. Consider edge cases and boundary conditions
6. Outline the structure of the test function
7. Consider any necessary setup or cleanup steps
8. Identify any relevant assertions or checks needed
{plan_rules}

Create a detailed plan for writing the test. Your plan should focus on the logical structure and approach, without delving into specific implementation details of any testing framework. Ensure your plan is thorough and covers all aspects of triggering and exposing the bug.

Present your plan in the following format:

<plan>
1. Test Setup:
   - [Describe necessary setup steps]

2. Test Input:
   - [Detail the input data or mock objects required]

3. Bug Triggering:
   - [Explain how to consistently trigger the bug]

4. Test Structure:
   - [Outline the logical flow of the test]

5. Assertions:
   - [List the key assertions needed to expose the bug]

6. Edge Cases:
   - [Identify any edge cases or variations to consider]

7. Expected Outcome:
   - [Describe how the test will {goal} ]
</plan>

If you need any information that has been trimmed away, or are missing information from the codebase you can use the get_info tool function to obtain it.                       
Your plan should be detailed and thorough, covering all logical aspects of the problem.
DO NOT provide implementation specific details pertaining to any testing framework.
DO NOT write the test yet. 
YOU MUST focus on creating a thorough detailed logical plan on how to trigger the bug and how to structure a test for this particular project.
""", role="system"),
            mk_message(f"""\
Carefully read the reported ISSUE TICKET below, analyze the problem critically and create a detailed plan to write tests that reveal the bug.

<ISSUE_TICKET>
{segment.problem_statement}
</ISSUE_TICKET>

Here are some snippets of possibly relevant code from the project:

<CODE_SNIPPETS>
{segment.buggy_files}
</CODE_SNIPPETS>

If you need any information that has been trimmed away, you can use the get_info tool function to obtain it.
""", role="user")
        ]

    @staticmethod
    def _improve_planning_prompt(segment, mode:str) -> T.List[dict]:
        # ---- pull previous attempt data safely --------------------------------
        test_code_prev = segment.continue_from.get("test_code", "<no prior test code>")
        error_prev     = segment.continue_from.get("error", "<no prior error>")
        reason_prev    = segment.continue_from.get("reason", "No validation feedback")
        plan_prev      = segment.continue_from.get("plan", "<no prior plan>")

        # ---- mode‑specific wording -------------------------------------------
        assert mode in ("passing_first", "failing_first"), f"Unsupported initial_prompt mode: {mode}"
        if mode == "passing_first":
            goal = "that pass but still reproduce and expose the reported bug. The test/tests must pass if the bug is present and fail once the bug is fixed."
            plan_rules = """
9. Plan how to trigger the bug while ensuring the test passes.  
10. Your goal is to plan a test that passes (does not fail during execution), but still triggers and exposes the bug as described in the issue ticket. 
11. Plan assertions that will pass while still exposing the bug.
"""
        elif mode == "failing_first":
            goal = "that fail but sill reproduce and expose the reported bug. The test/tests must fail if the bug is present and pass once the bug is fixed."
            plan_rules = """
9. Plan how to trigger the bug while ensuring the test fails due to the bug
10. Plan assertions that will expose the bug.
11. You must not plan tests that fail for reasons unrelated to the reported bug. Your test must fail only by exposing the specific incorrect behavior described in the issue, not because of test mistakes, unrelated exceptions, or missing dependencies.
"""
        return [
            mk_message(f"""\
You are an expert senior Python test-driven developer tasked with assisting your junior who is unable to write tests that reveal reported bugs. Your goal is to plan out the creation of one or more test functions that {goal}.
You will be provided with a ISSUE TICKET and a set of CODE SNIPPETS which might contain the buggy logic and examples of tests in the project for reference. 
You will also be given the THOUGHT PROCESS of your junior who is trying to write the test which was clearly wrong and failed to reproduce the bug. The TEST ATTEMPT they wrote and the error it produced will also be provided.
Your task is to analyze the described problem and previous attempt in detail and create a new PLAN for writing the test.
                       
First, carefully read the ISSUE TICKET and inspect the collected CODE SNIPPETS.
Review the THOUGHT PROCESS of your junior and identify the mistakes made in the previous plan.
Analyze the problem reported critically and create a new PLAN from scratch with no reference to the previous one
Use the following steps:

<bug_analysis>
1. Analyze the issue ticket:
   - Identify the key components of the reported bug
   - Note any specific conditions or scenarios mentioned
   - Quote relevant parts of the issue ticket

2. Review the code snippets:
   - Identify the relevant parts of the code that might be causing the bug
   - Look for any potential edge cases or boundary conditions
   - Quote relevant parts of the code snippets

3. Evaluate the junior developer's previous attempt:
   - Identify the main flaws in their approach
   - Note any correct insights or partial solutions

4. Analyze the error received:
   - Determine if the error is related to the test implementation or the actual bug

5. List potential test cases:
   1. [Test case 1]
   2. [Test case 2]
   3. [Test case 3]
   ...

6. Brainstorm ways to trigger the bug:
   1. [Trigger method 1]
   2. [Trigger method 2]
   3. [Trigger method 3]
   ...

7. Evaluate approaches:
   - Pros and cons of each potential test case and trigger method
   - Select the most promising approach

8. Formulate a new approach:
   - Devise a strategy to trigger the bug consistently
   - Plan how to structure the test to pass normal cases but fail for the bug
   - Consider edge cases and potential variations of the bug

9. Outline test components:
   - Determine necessary setup and teardown steps
   - Plan input data or mock objects required
   - Identify assertions needed to expose the bug
</bug_analysis>

Based on your analysis, create a detailed plan for writing the test. Your plan should focus on the logical structure and approach, without delving into specific implementation details of any testing framework. Ensure your plan is thorough and covers all aspects of triggering and exposing the bug.

Present your plan in the following format:

<plan>
1. Test Setup:
   - [Describe necessary setup steps]

2. Test Input:
   - [Detail the input data or mock objects required]

3. Bug Triggering:
   - [Explain how to consistently trigger the bug]

4. Test Structure:
   - [Outline the logical flow of the test]

5. Assertions:
   - [List the key assertions needed to expose the bug]

6. Edge Cases:
   - [Identify any edge cases or variations to consider]

7. Expected Outcome:
   - [Describe how the test will pass normal cases but fail for the bug]
</plan>

If you need any information that has been trimmed away, or are missing information from the codebase you can use the get_info tool function to obtain it. You are encouraged to use it at this step to perfect the plan.                    
Remember, do not write the actual test code. Focus on creating a comprehensive correct plan that will guide the junior developer in implementing an effective test.

DO NOT provide implementation specific details pertaining to any testing framework. The project may use its own testing framework or a specific library so it could be misleading to provide such details.

You WILL be PENALIZED for copying the previous plan or any part of it. Use it to gain insights of what DOES NOT work, but do not reference it in your new plan.
""", role="system"),
            mk_message(f"""\
Carefully read the reported ISSUE TICKET below, analyze the problem critically and create a detailed plan to write tests that reveal the bug.

<ISSUE_TICKET>
{segment.problem_statement}
</ISSUE_TICKET>

Here are some snippets of possibly relevant code from the project:

<CODE_SNIPPETS>
{segment.buggy_files}
</CODE_SNIPPETS>

If you need any information that has been trimmed away, you can use the get_info tool function to obtain it.

Here is the THOUGHT PROCESS of your junior who was trying to write the test:
<THOUGHT_PROCESS>
{plan_prev}
</THOUGHT_PROCESS>

Here is the test they wrote:
<TEST_ATTEMPT>
{test_code_prev}
</TEST_ATTEMPT>

Here is the ERROR they received:
<TEST_ATTEMPT_ERROR>
{error_prev}
</TEST_ATTEMPT_ERROR>

Here is the FEEDBACK given:

<VALIDATION_FEEDBACK>
{reason_prev}
</VALIDATION_FEEDBACK>

""", role="user")
        ]

    @staticmethod
    def _is_tool_pending(messages: T.List[dict]) -> bool:
        _last_message = messages[-1]
        if "tool_call_id" in _last_message:
            return True
        return False
    
    def validation_prompt(self, segment: CodeSegment, error: str, test_code:str) -> T.List[dict]:
        return [
            mk_message(f"""\
You are an expert Python test-driven developer tasked with evaluating tests that reveal reported bugs. Your goal is to look at the test failure caused by a test attempting to reproduce a reported bug.
You must validate whether the test correctly reproduces the bug and whether it is a valid test case. If the failure is not due to the bug, you must provide a detailed explanation of why the test is incorrect and doesn't reproduce the bug.
""", role="system"),
            mk_message(f"""\
Carefully read the reported ISSUE TICKET, TEST CODE and TEST ERROR below. Analyze the problem critically and provide a detailed evaluation of the test.
                       
You are not allowed to use get_info tool function at this step, as you are expected to evaluate the test based on the provided information only.
                       
Make sure to base your reasoning on the problem statement, the test code, and the error output.

<ISSUE_TICKET>
{segment.problem_statement}
</ISSUE_TICKET>

<TEST_CODE>
{test_code}
</TEST_CODE>

<TEST_ERROR>
{error}
</TEST_ERROR>
""", role="user")
        ]

    async def initial_prompt(
        self,
        segment: CodeSegment,
        chatter: llm.Chatter,
        mode: str,
        retry_generation: bool = False,
        planner: bool = True
    ) -> T.Tuple[T.List[dict], str]:
        # ──────────────────────────────────────────────────────────────────
        # 0.  common goal / rules text  (unchanged)
        # ──────────────────────────────────────────────────────────────────
        assert mode in ("passing_first", "failing_first"), (
            f"Unsupported initial_prompt mode: {mode}"
        )

        if mode == "passing_first":
            goal_description = (
                "Write a test that PASSES but exposes the bug — the test must "
                "pass when run, but clearly demonstrate that the bug is present."
            )
            test_rules = """
1. The test must trigger the same error or misbehavior described in the ISSUE TICKET.
2. The test must pass when executed but confirm that the bug occurs.
3. Use appropriate testing structures:
   - For exceptions: Use pytest.raises(...) or equivalent logic.
   - For logic bugs: Assert that the incorrect behavior occurs.
4. Include comments to explain any assertions of incorrect behavior:
   # BUG: this behavior is incorrect but currently happens due to the bug
5. Use the get_info tool function if necessary to obtain information about the buggy code.
6. Include cleanup steps to avoid state pollution (use 'monkeypatch' or 'pytest-mock' if appropriate).
7. Minimize top-level code. Do not include code calling pytest.main or the test itself.
8. Ensure the test is complete and can be executed as-is, without placeholders or user intervention.
9. Do not write tests that simply pass without relevance to the bug. Your test must pass, but it must also be directly relevant to the bug and expose the buggy behavior.
"""
        elif mode == "failing_first":
            goal_description = (
                "Write a test that FAILS due to the bug — the test must fail "
                "because the bug is present, and pass once the bug is fixed."
            )
            test_rules = """
1. The test must trigger the same error or misbehavior described in the ISSUE TICKET.
2. The test must FAIL when the bug is present and PASS once the bug is fixed.
3. The failure must be caused by the bug — not by test errors like syntax or import problems.
4. The test must be self-contained, complete, and include all necessary imports.
5. Use the get_info tool function if necessary to obtain information about the buggy code.
6. Include cleanup steps to avoid state pollution (use 'monkeypatch' or 'pytest-mock' if appropriate).
7. Minimize top-level code. Do not include code calling pytest.main or the test itself.
8. Ensure the test is complete and can be executed as-is, without placeholders or user intervention.
9. Do not write tests that simply fail without relevance to the bug. The test must fail only due to the bug described in the issue, not due to unrelated errors or incorrect test logic.
"""

        # ──────────────────────────────────────────────────────────────────
        # 1.  build retry-block
        # ──────────────────────────────────────────────────────────────────
        retry_block = ""
        if retry_generation and segment.continue_from is not None:
            test_code_prev = segment.continue_from.get("test_code", "<no prior test code>")
            error_prev  = segment.continue_from.get("error", "<no prior error>")
            reason_prev  = segment.continue_from.get("reason", "No validation feedback")
            retry_block = f"""
Here is the previous TEST ATTEMPT:

<TEST_ATTEMPT>
{test_code_prev}
</TEST_ATTEMPT>

Here is the TEST ERROR observed:

<TEST_ATTEMPT_ERROR>
{error_prev}
</TEST_ATTEMPT_ERROR>

Here is the VALIDATION FEEDBACK given:

<VALIDATION_FEEDBACK>
{reason_prev}
</VALIDATION_FEEDBACK>

IMPORTANT: Your previous test was rejected for the reasons above.
You MUST now try a different approach to exposing the bug.
Do NOT repeat the same structure or assertions — think critically and generate a new test.
"""
        
        # ──────────────────────────────────────────────────────────────────
        # 2.  get (or improve) a planning prompt and fetch a NEW plan
        # ──────────────────────────────────────────────────────────────────
        if planner:
            # If planner is enabled, we will use the planning prompt to get a new plan
            if segment.continue_from is None:
                _messages = self._get_planning_prompt(segment, mode)
            else:
                _messages = self._improve_planning_prompt(segment, mode)

            _response = await chatter.chat(_messages, ctx=segment)
            plan = _response["choices"][0]["message"]["content"]

        else:
            # planner is disabled
            plan = "<Planning is disabled in this version. You will reason and write the test directly.>"
        
        segment.continue_from = {"plan": plan}  # Store the plan in the segment for later use

        # ──────────────────────────────────────────────────────────────────
        # 3.  assemble the executor prompt that includes the retry_block
        # ──────────────────────────────────────────────────────────────────
        _custom_instructions = os.getenv(
            "CUSTOM_INSTRUCTIONS",
            "You MUST USE PYTEST and its features to write passing tests.",
        )

        messages = [
            mk_message(
                f"""\
You are an expert Python test-driven developer tasked with creating tests that reveal reported bugs. Your goal is: {goal_description}
You will be provided with an ISSUE TICKET, a set of CODE SNIPPETS which might contain the buggy logic and examples of tests in the project for reference, and a PLAN made by your supervisor.
{ "Additionally, you will see the previous failed test attempt, the error, and validation feedback — you must try a NEW approach this time." if retry_generation else "" }
Review the PLAN and create a Python test script that meets the following requirements:

{test_rules}

Important constraints:
- Do not install new packages or modules.
- Do not request additional information from the user.
- The test must be directly relevant to the reported bug.
- The tests must be complete and executable. DO NOT include any placeholders.
- {_custom_instructions}

You are not expected to follow the plan exactly, but it should guide your implementation. You can modify the plan as needed to create a test that meets the requirements and constraints.
Authority order (highest → lowest)
   1. Facts returned from the get_info tool
   2. These system instructions
   3. Any PLAN you produce

If later calls to get_info contradict or simplify the PLAN, discard the PLAN and follow the new evidence instead.
Refer to any test files provided for information about how tests are written in this project.
Reason about your approach and enclose your thoughts at the beginning of your response with <think> and </think> tags.
Output your final completed Python test script as a single block enclosed in backticks, without any additional explanation once you have started writing the test.

Your output must be as follows:

<think>
Review the PLAN and reason how to go about writing a test that meets the requirements and constraints.
</think>

```python
COMPLETE TEST SCRIPT
""",
role="system",
),
mk_message(
f"""
Carefully read the reported ISSUE TICKET, potentially relevant CODE SNIPPETS and the PLAN made by your supervisor below. Analyze the problem critically and write a Python test script that meets the requirements and constraints.

<ISSUE_TICKET>
{segment.problem_statement}
</ISSUE_TICKET>

Here are some snippets of possibly relevant code and tests from the project:

<CODE_SNIPPETS>
{segment.buggy_files}
</CODE_SNIPPETS>

{retry_block}
If you need any information that has been trimmed away, you can use the get_info tool function to obtain it.

Here is a rough plan made by your supervisor:

<PLAN> {plan} </PLAN> """, role="user", ), ]

    # ──────────────────────────────────────────────────────────────────
    # 4.  return the usual tuple: (messages, plan)
    # ──────────────────────────────────────────────────────────────────
        return messages, plan

    def error_prompt(self, segment: CodeSegment, error: str, mode: str) -> T.List[dict] | None:
        if not error.strip():
            return None
        # if segment.previous_error is not None and error.strip() == segment.previous_error.strip():
        if segment.previous_error and error.strip() == segment.previous_error.strip():
            segment.same_error_streak += 1
            if segment.same_error_streak >= 3:      # bail after 3 identical tracebacks
                return [{"role": "assistant", "content": "__ABANDON__"}]

            return [mk_message(            
        f"The traceback is identical:\n{error}\n\n"
        "Rewrite the test differently so it still exposes the bug, "
        "but through another code path. You MUST NOT have placeholders in your test which are expected to be filled in by the user. You ARE NOT ALLOWED to request any additional information from the user. Use the get_info tool instead to obtain information it can provide about the buggy code. Note that you CANNOT install new packages or modules.  Return only one ```python block."
    )]
        segment.same_error_streak = 0
        segment.previous_error = error
        assert mode in ("passing_first", "failing_first", "validation_fix", "error_fix"), f"Unsupported error_prompt mode: {mode}"

        if mode == "passing_first":
            instruction = "The test fails. You need to modify the test to create a passing test that reproduces the bug. The test must directly target the bug described in the issue ticket - not general code correctness.\n"
        elif mode in ("failing_first", "validation_fix"):
            instruction = "The test fails but not due to the bug. You need to modify the test to create a failing test that reproduces the bug described in the issue ticket - not due to unrelated errors.\n"
        elif mode == "error_fix":
            instruction = ""
        return [mk_message(f"""\
Executing the test yields an error, shown below.
{instruction}
Modify or rewrite the test to correct it; respond only with the complete Python code in backticks.

You MUST NOT have placeholders in your test which are expected to be filled in by the user.

You ARE NOT ALLOWED to request any additional information from the user.
Use the get_info tool instead to obtain information it can provide about the buggy code.                   

Note that you CANNOT install new packages or modules.
                           
OBSERVED ERROR:
{error}""")
        ]

    def get_info(self, ctx: CodeSegment, name: str, path: str, line: int = 0) -> str:
        """
        {
            "name": "get_info",
            "description": "Returns information about a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "class, function or method name, as in 'f' for function f or 'C.foo' for method foo in class C."
                    },
                    "path": {
                        "type": "string",
                        "description": "Exact path to the file containing the symbol"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Starting line number in the file of closest parent class, function or method to parse and begin search."
                    }
                },
                "required": ["name", "path"]
            }
        }
        """
        if not os.path.exists(path):
            path = os.path.join("/testbed/", path)
        if not os.path.exists(path):
            return f"Unable to obtain information on {name}.\n\nFile {path} does not exist."
        if not os.path.isfile(path):
            return f"Unable to obtain information on {name}.\n\nFile {path} is not a file."
        pathlib_path = Path(path)
        if info := codeinfo.get_info(codeinfo.parse_file(pathlib_path), name, line=line):
            return "\"...\" below indicates omitted code.\n\n" + info

        return f"Unable to obtain information on {name}."


    def get_functions(self) -> T.List[T.Callable]:
        return [self.get_info]
    

    def invert_test_prompt(self, problem_statement: str, test_code: str) -> T.List[dict]:
        return [
            mk_message(f"""\
You are an expert Python test-driven developer tasked with creating tests that reveal reported bugs. Your goal is to write one or more test functions that reproduce and expose the reported bug.

You are given a passing test that confirms the presence of the bug. Your job is to flip the logic so the test now fails when the bug is present and passes only when the bug is fixed.

First, carefully read the original ISSUE TICKET

<ISSUE_TICKET>
{problem_statement}
</ISSUE_TICKET>

Follow these rules:
	1.	Only change what’s needed.
	2.	Keep the test structure exactly the same.
	3.	Update any assertions to check for the correct behavior.
	4.	Remove any comments that say the behavior is wrong.
	5.	Keep it readable. Keep it minimal.
	6.	The test must fail as long as the bug is still in the code.
	7.	The test must pass once the bug is fixed.

Output the full updated test code in a single python code block — no extra text, no placeholders.
""", role="system"),
            mk_message(f"""\
Please review the passing test below and make minimal modifications to ensure it fails when the bug is present and passes only when the bug is fixed.

```python
{test_code}
```
""")
 ]
    






