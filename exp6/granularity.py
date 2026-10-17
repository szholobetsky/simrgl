"""
granularity.py — RQ3 multi-granularity prior filtering
(IMPLEMENTATION_DETAIL_CL.md section 6).

`result["modules"]` comes for free from every simargl search() call
(simargl/searcher.py:576-584, _aggregate_modules) — no separate "Stage 1"
call is needed. Extension-level grouping does not exist in simargl at all,
so it's computed here from the same result["files"] list, mirroring
simargl's own module_from_path() convention (simargl/utils.py:24-27).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from typing import Dict, List


def extract_extension(path: str) -> str:
    """Same never-raises contract as exp3/utils.py:extract_module_path."""
    if not path:
        return "noext"
    normalized = str(path).replace('\\', '/').strip()
    suffix = PurePosixPath(normalized).suffix.lower()
    return suffix if suffix else "noext"


def top_extensions(files: List[dict], top_m: int) -> List[dict]:
    """Aggregate files by extension the same way simargl aggregates by
    module: max(score) per extension, top-M (mirrors
    simargl/searcher.py:_aggregate_modules)."""
    ext_score: Dict[str, float] = defaultdict(float)
    for f in files:
        ext = extract_extension(f["path"])
        ext_score[ext] = max(ext_score[ext], f["score"])
    ranked = sorted(ext_score.items(), key=lambda kv: kv[1], reverse=True)
    return [{"extension": e, "score": s} for e, s in ranked[:top_m]]


def apply_prior_filter(
    files: List[dict],
    top_modules: List[str],
    top_extensions_list: List[str],
    lambda_module: float,
    lambda_ext: float,
) -> List[str]:
    """Re-rank files by penalizing (not hard-filtering) those outside the
    predicted module/extension priors:

      score'(f)  = score(f) * (1.0 if module(f) in top_modules else lambda_module)
      score''(f) = score'(f) * (1.0 if ext(f) in top_extensions else lambda_ext)

    A soft multiplicative penalty rather than a hard filter, so a wrong
    module/extension guess degrades ranking instead of zeroing recall
    outright. See IMPLEMENTATION_DETAIL_CL.md section 6 for the rationale
    and section 10 for the lambda-grid backlog.
    """
    module_set = set(top_modules)
    ext_set = set(top_extensions_list)

    rescored = []
    for f in files:
        score = f["score"]
        if f.get("module") not in module_set:
            score *= lambda_module
        if extract_extension(f["path"]) not in ext_set:
            score *= lambda_ext
        rescored.append((f["path"], score))

    rescored.sort(key=lambda kv: kv[1], reverse=True)
    return [p for p, _ in rescored]
