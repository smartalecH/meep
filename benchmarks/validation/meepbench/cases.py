"""Case discovery: built-in cases live in ../cases/<name>.py (each defines
CASE); any other .py path defining CASE works too, and a .gds path builds an
ad-hoc case from command-line options."""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List

from .spec import Case

CASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cases")


def _load_file(path: str) -> Case:
    name = "meepbench_case_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # lets derived cases reach a base case's module
    spec.loader.exec_module(module)
    if not hasattr(module, "CASE"):
        raise AttributeError(f"{path} does not define CASE")
    return module.CASE


def names() -> List[str]:
    return sorted(f[:-3] for f in os.listdir(CASE_DIR)
                  if f.endswith(".py") and not f.startswith("_"))


def load(name_or_path: str) -> Case:
    if name_or_path.endswith(".py"):
        return _load_file(name_or_path)
    path = os.path.join(CASE_DIR, name_or_path + ".py")
    if not os.path.exists(path):
        raise KeyError(f"unknown case {name_or_path!r}; known: {', '.join(names())}")
    return _load_file(path)


def all_cases() -> Dict[str, Case]:
    return {n: load(n) for n in names()}
