"""The API guides' parameter tables restate constructor defaults; this test
fails when a documented default drifts from the implementation signature.

Each ``| Parameter | Type | Default | Description |`` (or ``| Field | ... |``)
table in docs/user_guides/*.md is attributed to the nearest preceding heading,
bold name, or backticked ``fabricpc.*`` path that resolves to a class or
function in REGISTRY_MODULES. Every row's Default cell is then compared
against the actual default: ``required`` must be a parameter without a
default, ``Name(...)`` cells match the default instance's class name, and
literal cells (numbers, strings, ``None``, ``[]``) must equal the signature
value. Dataclass factory defaults are compared against ``default_factory()``.
"""

import ast
import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs" / "user_guides"

REGISTRY_MODULES = [
    "fabricpc.nodes",
    "fabricpc.models",
    "fabricpc.core.inference",
    "fabricpc.core.inference_epc",
    "fabricpc.core.activations",
    "fabricpc.core.energy",
    "fabricpc.core.initializers",
    "fabricpc.utils.data",
    "fabricpc.tuning",
    "fabricpc.utils.dashboarding.trackers",
]

TABLE_HEADER = re.compile(
    r"^\|\s*(?:Parameter|Field)\s*\|\s*Type\s*\|\s*Default\s*\|\s*Description\s*\|$"
)
SUBJECT_PATTERNS = [
    re.compile(r"^#{2,4}\s+([A-Za-z_]\w*)"),
    re.compile(r"^\*\*([A-Za-z_]\w*)\*\*"),
    re.compile(r"^`fabricpc(?:\.\w+)*\.(\w+)`$"),
]

REQUIRED = object()


def _registry():
    reg = {}
    for module_name in REGISTRY_MODULES:
        module = importlib.import_module(module_name)
        for name in dir(module):
            if not name.startswith("_"):
                reg.setdefault(name, getattr(module, name))
    return reg


REGISTRY = _registry()


def _find_subject(lines, header_index):
    """Nearest preceding heading/bold/backticked-path name found in REGISTRY."""
    for line in reversed(lines[:header_index]):
        for pattern in SUBJECT_PATTERNS:
            match = pattern.match(line.strip())
            if match and match.group(1) in REGISTRY:
                return match.group(1)
    return None


def _collect_tables():
    """(doc path, table line number, subject name or None, rows) per table;
    rows are (row line number, parameter name cell, default cell)."""
    tables = []
    for doc in sorted(DOCS_DIR.glob("*.md")):
        lines = doc.read_text().splitlines()
        i = 0
        while i < len(lines):
            if not TABLE_HEADER.match(lines[i].strip()):
                i += 1
                continue
            subject = _find_subject(lines, i)
            rows = []
            j = i + 2  # skip the separator line
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().split("|")[1:-1]]
                if len(cells) == 4:
                    rows.append((j + 1, cells[0].strip("`"), cells[2]))
                j += 1
            tables.append((doc, i + 1, subject, rows))
            i = j
    return tables


def _actual_defaults(obj):
    """Parameter name -> default value (REQUIRED when there is none)."""
    if dataclasses.is_dataclass(obj):
        out = {}
        for field in dataclasses.fields(obj):
            if field.default is not dataclasses.MISSING:
                out[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                out[field.name] = field.default_factory()
            else:
                out[field.name] = REQUIRED
        return out
    signature = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
    return {
        name: REQUIRED if p.default is inspect.Parameter.empty else p.default
        for name, p in signature.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def _cell_matches(cell, actual):
    cell = cell.strip().strip("`").strip()
    if cell == "required":
        return actual is REQUIRED
    if actual is REQUIRED:
        return False
    instance = re.fullmatch(r"([A-Za-z_]\w*)\(.*\)", cell)
    if instance:
        return type(actual).__name__ == instance.group(1)
    try:
        value = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return cell == repr(actual) or cell == str(actual)
    if isinstance(value, bool) != isinstance(actual, bool):
        return False
    return value == actual


TABLES = _collect_tables()


@pytest.mark.parametrize(
    "doc, header_line, subject, rows",
    TABLES,
    ids=[f"{doc.name}:{line}:{subject}" for doc, line, subject, _ in TABLES],
)
def test_documented_defaults_match_signatures(doc, header_line, subject, rows):
    assert subject is not None, (
        f"{doc}:{header_line}: cannot attribute this table to a class or "
        f"function; name it in a preceding heading, bold line, or backticked "
        f"fabricpc path, or add its module to REGISTRY_MODULES"
    )
    actual = _actual_defaults(REGISTRY[subject])
    problems = []
    for line, param, cell in rows:
        if param not in actual:
            problems.append(f"{doc}:{line}: `{param}` is not a parameter of {subject}")
        elif not _cell_matches(cell, actual[param]):
            shown = "required" if actual[param] is REQUIRED else repr(actual[param])
            problems.append(
                f"{doc}:{line}: {subject}.{param} documented as {cell!r}, "
                f"signature default is {shown}"
            )
    assert not problems, "\n" + "\n".join(problems)


def test_all_default_tables_collected():
    """Guards the collector itself: every Default column in the guides must
    surface as a parametrized table above."""
    header_count = sum(
        1
        for doc in DOCS_DIR.glob("*.md")
        for line in doc.read_text().splitlines()
        if TABLE_HEADER.match(line.strip())
    )
    assert header_count == len(TABLES) and header_count > 0
