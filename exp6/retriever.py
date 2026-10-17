"""
retriever.py

Thin wrapper around simargl's own Python API (`simargl.searcher.search()` /
`rrf_search()`), called IN-PROCESS rather than shelling out to the `simargl`
CLI via subprocess.

Why: measured cold-start cost of one `simargl search` CLI call on this
machine — 12.56s total, of which 12.49s (99.5%) is `import torch` +
`import sentence_transformers` + model load, and only 0.068s is the actual
encode. simargl's own embedder cache (simargl/embedder.py:_cache, keyed by
model_key) means that cost is paid ONCE per process, not once per call — so
importing in-process instead of spawning a subprocess per query turns
~12.5s/query into ~0.07s/query, roughly a 180x reduction in run_eval.py's
total wall-clock time. See the conversation / EXPERIMENT_PLANNING.md for the
full reasoning; this is why hardware (CPU vs GPU) barely mattered under the
old subprocess-per-query design but starts to matter once encode is what's
actually left on the critical path.

The JSON shape returned by search()/rrf_search() is exactly what the CLI's
--format json prints (cli.py just serializes this same dict) — no other
module (combiners.py, granularity.py, run_eval.py) needed to change.
"""
from __future__ import annotations

import logging
from typing import List

from simargl.searcher import rrf_search as _simargl_rrf_search
from simargl.searcher import search as _simargl_search

import config

logger = logging.getLogger(__name__)


class SimarglRetriever:
    """In-process retriever for one project. Reuses simargl's own cached
    embedder across every call within this process (see module docstring).
    """

    def __init__(self, project_id: str, store_dir: str = ".simargl"):
        self.project_id = project_id
        self.store_dir = store_dir

    def search(self, query: str, mode: str, top_n: int = None) -> dict:
        """Return the full simargl result dict:
        {"files": [{"path","score","module",...}], "modules": [...], "units": [...]}
        """
        top_n = top_n or config.DEFAULT_TOP_N
        try:
            return _simargl_search(
                query, mode=mode, top_n=top_n,
                project_id=self.project_id, store_dir=self.store_dir,
            )
        except Exception as e:
            logger.error(f"search failed (project={self.project_id}, mode={mode}): {e}")
            return {"files": [], "modules": [], "units": []}

    def run_search(self, query: str, mode: str, top_n: int = None) -> List[str]:
        """Return just the retrieved file paths, in rank order."""
        data = self.search(query, mode, top_n)
        return [f["path"] for f in data.get("files", [])]

    def rrf_search(self, query: str, sources: str, top_n: int = None, k: int = 60) -> List[str]:
        """`sources`: comma-separated "mode:project_id" pairs, e.g.
        "task:sonar,file:sonar,aggr:sonar" (RQ2.2)."""
        top_n = top_n or config.DEFAULT_TOP_N
        try:
            data = _simargl_rrf_search(
                query, sources=sources, top_n=top_n, k=k, store_dir=self.store_dir,
            )
            return [f["path"] for f in data.get("files", [])]
        except Exception as e:
            logger.error(f"rrf_search failed (sources={sources}): {e}")
            return []
