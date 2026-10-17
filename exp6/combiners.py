"""
combiners.py — RQ2.2 combination strategies (IMPLEMENTATION_DETAIL_CL.md 5.3).

Reciprocal Rank Fusion is NOT reimplemented here — `simargl rrf` already does
it (simargl/searcher.py:314-394, exposed as CLIRetriever.rrf_search()). These
are the strategies simargl doesn't provide: Intersection, Union, Weighted
fusion, Borda count.

Every combiner takes {mode: [path, path, ...]} (already top-K, rank order —
one path list per simargl mode for a single query) and returns a single
ranked list of paths, so results plug straight into
utils.calculate_metrics_for_query.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def intersection(mode_results: Dict[str, List[str]], **_kw) -> List[str]:
    """Files present in ALL modes' result lists, ordered by summed reciprocal rank."""
    if not mode_results:
        return []
    path_sets = [set(paths) for paths in mode_results.values()]
    common = set.intersection(*path_sets) if path_sets else set()
    return _order_by_summed_rank(common, mode_results)


def union(mode_results: Dict[str, List[str]], **_kw) -> List[str]:
    """All unique files across modes, ordered by summed reciprocal rank."""
    all_paths = {p for paths in mode_results.values() for p in paths}
    return _order_by_summed_rank(all_paths, mode_results)


def weighted_fusion(mode_results: Dict[str, List[str]], mode_weights: Dict[str, float] = None, **_kw) -> List[str]:
    """score(f) = sum( weight[mode] / rank ) across modes it appears in.

    mode_weights is typically each mode's own MAP from the RQ1 baseline, so a
    mode that individually performs better gets more say in the fusion.
    """
    mode_weights = mode_weights or {}
    scores: Dict[str, float] = defaultdict(float)
    for mode, paths in mode_results.items():
        w = mode_weights.get(mode, 1.0)
        for rank, path in enumerate(paths, start=1):
            scores[path] += w / rank
    return [p for p, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def borda_count(mode_results: Dict[str, List[str]], **_kw) -> List[str]:
    """Classic Borda count: each mode hands its top-N results (N-rank+1)
    points; points summed across modes, higher wins."""
    points: Dict[str, int] = defaultdict(int)
    for paths in mode_results.values():
        n = len(paths)
        for rank, path in enumerate(paths, start=1):
            points[path] += n - rank + 1
    return [p for p, _ in sorted(points.items(), key=lambda kv: kv[1], reverse=True)]


def _order_by_summed_rank(paths: set, mode_results: Dict[str, List[str]]) -> List[str]:
    """Shared ordering for intersection()/union(): summed reciprocal rank
    across modes (0 contribution from a mode that doesn't contain the file —
    not a penalty), so files present in more modes / at better ranks rise."""
    scores: Dict[str, float] = defaultdict(float)
    for paths_in_mode in mode_results.values():
        for rank, path in enumerate(paths_in_mode, start=1):
            if path in paths:
                scores[path] += 1.0 / rank
    return [p for p, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


COMBINERS = {
    'intersection': intersection,
    'union': union,
    'weighted': weighted_fusion,
    'borda': borda_count,
}
