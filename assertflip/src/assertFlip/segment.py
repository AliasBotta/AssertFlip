import typing as T
from pathlib import Path
from .utils import *

class CodeSegment:
    """Represents a section of code with its associated metadata."""

    def __init__(self, filename: Path, name: str, begin: int, end: int,
                 lines_of_interest: T.Set[int],
                 missing_lines: T.Set[int],
                 executed_lines: T.Set[int],
                 missing_branches: T.Set[T.Tuple[int, int]],
                 context: T.List[T.Tuple[int, int]],
                 imports: T.List[str],
                 problem_statement: str,
                 buggy_files: str, 
                 line_level_localization: T.List[T.Dict[str, T.Union[str, T.List[int]]]],
                instance_id: str,
                continue_from: dict = None,
    ):
        # self.path = Path(filename).resolve()
        self.path = "CONSTANT_PATH"
        self.filename = filename
        self.name = name
        self.begin = begin
        self.end = end
        self.lines_of_interest = lines_of_interest
        self.missing_lines = missing_lines
        self.executed_lines = executed_lines
        self.missing_branches = missing_branches
        self.context = context
        self.imports = imports
        self.problem_statement=problem_statement,
        self.buggy_files=buggy_files,
        self.line_level_localization=line_level_localization
        self.instance_id = instance_id
        self.previous_missed_coverage = None
        self.previous_error = None
        self.continue_from = continue_from
        self.previous_test_code = None
        self.same_error_streak = 0
        self.same_error_limit = 3
     

    def __repr__(self):
        return f"CodeSegment(\"{self.filename}\", \"{self.name}\", {self.begin}, {self.end}, " + \
               f"{self.missing_lines}, {self.executed_lines}, {self.missing_branches}, {self.context})"

    def identify(self) -> str:
        return f"{self.filename}:{self.begin}-{self.end-1}"

    def __str__(self) -> str:
        return self.identify()
    
    def format_coverage_targets(self) -> str:
        """Formats the line-level localization data into readable coverage targets."""
        if not self.line_level_localization:
            return "No specific files were identified."

        formatted_files = []
        for entry in self.line_level_localization:
            filename = entry["filename"]
            suspect_lines = ", ".join(map(str, entry["suspect_lines"]))
            formatted_files.append(f"{filename} (lines: {suspect_lines})")

        return "\n".join(formatted_files)



