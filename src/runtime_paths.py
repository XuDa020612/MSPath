from __future__ import annotations

import os
import sys
from typing import Optional


def add_repo_root_to_sys_path(repo_root: Optional[str] = None) -> str:
    """Ensure CHIEF-main repo root is on sys.path.

    This makes imports like `from models.ctran import ctranspath` work no matter
    where the script is launched from.

    If repo_root is None, it is inferred by walking up from this file.
    """
    if repo_root is None:
        # .../CHIEF-main/Downstream/WSI_Report/src/runtime_paths.py
        # repo root is 4 levels up from this file.
        here = os.path.abspath(__file__)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))

    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def add_wsi_report_root_to_sys_path() -> str:
    """Ensure Downstream/WSI_Report is on sys.path."""
    here = os.path.abspath(__file__)
    wsi_report_root = os.path.dirname(os.path.dirname(here))  # .../WSI_Report
    if wsi_report_root not in sys.path:
        sys.path.insert(0, wsi_report_root)
    return wsi_report_root
