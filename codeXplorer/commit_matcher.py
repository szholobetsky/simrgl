"""
Commit Matcher — links unmatched commits to tasks via text similarity.

Solves the low link rate problem: projects where developers write task titles
(or paraphrases) as commit messages instead of copying the task ID.

Two implemented strategies:
  1. fuzzy   — rapidfuzz token_sort_ratio, CPU-only, fast (~10k commits/sec)
  2. difflib — stdlib fallback if rapidfuzz not installed, ~5x slower

One planned strategy (see EMBEDDING MATCHING PLAN below):
  3. embedding — bge-large cosine similarity, GPU, highest quality

Usage:
    # Fuzzy matching (rapidfuzz recommended):
    python commit_matcher.py --method fuzzy --threshold 75

    # Stdlib fallback:
    python commit_matcher.py --method difflib --threshold 0.80

    # Dry run (no DB writes):
    python commit_matcher.py --method fuzzy --threshold 75 --dry-run

    # Or from code:
        matcher = CommitMatcher(db_manager)
        matched = matcher.match_fuzzy(threshold=75)
"""

import re
import logging
import argparse
from typing import List, Tuple, Optional
from tqdm import tqdm

import config
from database import DatabaseManager

logger = logging.getLogger(__name__)


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

_NOISE = re.compile(
    r'\b(fix(es|ed)?|close[sd]?|resolve[sd]?|revert|merge|update|refactor|'
    r'add(s|ed)?|remove[sd]?|bump|wip|nit|minor|style|cleanup|typo|test[s]?)\b',
    re.IGNORECASE
)
_BRACKETS = re.compile(r'[\[\](){}<>]')
_WHITESPACE = re.compile(r'\s+')


def _normalize(text: str) -> str:
    """
    Normalize text for matching: lowercase, strip common git noise words,
    remove brackets and punctuation, collapse whitespace.
    Keeps domain-specific tokens (class names, error types, module names).
    """
    if not text:
        return ''
    text = text.split('\n')[0]         # first line only
    text = text.lower()
    text = _BRACKETS.sub(' ', text)
    text = _NOISE.sub(' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = _WHITESPACE.sub(' ', text).strip()
    return text


# =============================================================================
# FUZZY MATCHER (rapidfuzz)
# =============================================================================

class CommitMatcher:

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def match_fuzzy(self, threshold: float = 75.0, dry_run: bool = False) -> int:
        """
        Match unmatched commits to tasks using rapidfuzz token_sort_ratio.

        token_sort_ratio sorts tokens before comparing — handles reordering:
            "Fix NPE in SessionManager"  ↔  "SessionManager NPE fix"  → high score

        Args:
            threshold: Minimum score 0–100. Recommended: 75–85.
                       75 = lenient (more matches, some false positives)
                       85 = strict  (fewer matches, high precision)
            dry_run:   If True, print matches but do not write to DB.

        Returns:
            Number of newly matched commits.
        """
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            logger.error(
                "rapidfuzz not installed. Run: pip install rapidfuzz\n"
                "Or use --method difflib for stdlib fallback."
            )
            return 0

        commits = self.db.get_unmatched_commits()   # (sha, message)
        tasks   = self.db.get_all_tasks_with_titles()  # (name, title)

        if not commits or not tasks:
            logger.info("Nothing to match.")
            return 0

        logger.info(f"Matching {len(commits)} commits against {len(tasks)} tasks "
                    f"(threshold={threshold}, rapidfuzz)...")

        # Pre-normalize task titles once
        task_names   = [t[0] for t in tasks]
        task_titles  = [_normalize(t[1]) for t in tasks]

        matched = 0
        for sha, message in tqdm(commits, desc="Fuzzy matching", unit="commit"):
            norm_msg = _normalize(message)
            if len(norm_msg) < 8:   # too short — skip (high false positive risk)
                continue

            result = process.extractOne(
                norm_msg,
                task_titles,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
            )
            if result is None:
                continue

            best_title, score, idx = result
            task_name = task_names[idx]

            if dry_run:
                original_msg = message.split('\n')[0][:80]
                original_title = tasks[idx][1][:80]
                print(f"  [{score:5.1f}] commit: {original_msg}")
                print(f"           task:   {original_title}  ({task_name})")
                print()
            else:
                self.db.update_task_name_by_sha(sha, task_name)
                self.db.insert_match_log(sha, task_name, 'fuzzy', score)

            matched += 1

        logger.info(f"Fuzzy matched: {matched} commits")
        return matched

    def match_difflib(self, threshold: float = 0.80, dry_run: bool = False) -> int:
        """
        Match unmatched commits using Python stdlib difflib.SequenceMatcher.
        Slower than rapidfuzz (~5x) but requires no extra dependencies.

        Args:
            threshold: 0.0–1.0 (0.80 ≈ 80% similarity). Recommended: 0.78–0.85.
        """
        from difflib import SequenceMatcher

        commits = self.db.get_unmatched_commits()
        tasks   = self.db.get_all_tasks_with_titles()

        if not commits or not tasks:
            logger.info("Nothing to match.")
            return 0

        logger.info(f"Matching {len(commits)} commits against {len(tasks)} tasks "
                    f"(threshold={threshold:.2f}, difflib)...")

        task_names  = [t[0] for t in tasks]
        task_titles = [_normalize(t[1]) for t in tasks]

        matched = 0
        for sha, message in tqdm(commits, desc="Difflib matching", unit="commit"):
            norm_msg = _normalize(message)
            if len(norm_msg) < 8:
                continue

            best_score = 0.0
            best_idx = -1
            for i, title in enumerate(task_titles):
                score = SequenceMatcher(None, norm_msg, title).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_score < threshold or best_idx < 0:
                continue

            task_name = task_names[best_idx]

            if dry_run:
                print(f"  [{best_score:.3f}] {message.split(chr(10))[0][:80]}")
                print(f"           {tasks[best_idx][1][:80]}  ({task_name})")
            else:
                self.db.update_task_name_by_sha(sha, task_name)
                self.db.insert_match_log(sha, task_name, 'difflib', best_score * 100)

            matched += 1

        logger.info(f"Difflib matched: {matched} commits")
        return matched


# =============================================================================
# EMBEDDING MATCHING PLAN
# =============================================================================
#
# Status: NOT IMPLEMENTED — described here for future implementation.
#
# Why embeddings beat fuzzy matching:
#   fuzzy:     "Cannot connect to remote server"  vs  "Connection refused exception"
#              → token_sort_ratio ≈ 30  (no common tokens) → missed
#
#   embedding: embed("Cannot connect to remote server")
#              embed("Connection refused exception")
#              → cosine_sim ≈ 0.87  → matched  ✓
#
# Approach:
#
#   Step 1 — Encode all tasks (one-time, cache to disk):
#       from sentence_transformers import SentenceTransformer
#       model = SentenceTransformer('BAAI/bge-large-en-v1.5')
#       task_texts = [f"{title} {description[:200]}" for name, title, desc in tasks]
#       task_embeddings = model.encode(task_texts, batch_size=64, show_progress_bar=True)
#       np.save('task_embeddings.npy', task_embeddings)
#
#   Step 2 — Encode all unmatched commit messages (one-time):
#       commit_texts = [message.split('\n')[0] for sha, message in commits]
#       commit_embeddings = model.encode(commit_texts, batch_size=64, ...)
#
#   Step 3 — Cosine similarity matrix + top-1 match:
#       from sklearn.metrics.pairwise import cosine_similarity
#       # For large sets: use faiss for fast ANN instead of full matrix
#       sims = cosine_similarity(commit_embeddings, task_embeddings)
#       best_idx   = sims.argmax(axis=1)
#       best_score = sims.max(axis=1)
#       matched = [(commits[i][0], task_names[best_idx[i]], best_score[i])
#                  for i in range(len(commits))
#                  if best_score[i] >= threshold]
#
#   Step 4 — Write matches to DB:
#       for sha, task_name, score in matched:
#           db.update_task_name_by_sha(sha, task_name)
#           db.insert_match_log(sha, task_name, 'embedding', float(score))
#
# Recommended model: BAAI/bge-large-en-v1.5
#   — already used in exp2/exp3/exp4, best MAP in experiments
#   — 6GB VRAM sufficient with batch_size=32 (or =4 for safety)
#   — encode 10k commits in ~5min on GPU
#
# Recommended threshold: 0.82–0.88 (cosine)
#   — < 0.80: too many false positives
#   — > 0.90: misses valid paraphrases
#   — calibrate on a small manual sample first
#
# Scale considerations:
#   intellij-community: ~300k tasks × ~500k commits → full matrix = 150B ops
#   → must use faiss (HNSW index) for ANN search instead of full cosine matrix
#       import faiss
#       index = faiss.IndexFlatIP(embedding_dim)  # inner product = cosine if normalized
#       faiss.normalize_L2(task_embeddings)
#       index.add(task_embeddings)
#       D, I = index.search(commit_embeddings, k=1)  # top-1 per commit
#
# Libraries needed:
#   pip install sentence-transformers faiss-cpu  # or faiss-gpu
#
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description='Match unmatched commits to tasks')
    parser.add_argument('--method', choices=['fuzzy', 'difflib'], default='fuzzy')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold. Default: 75 for fuzzy, 0.80 for difflib')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print matches without writing to DB')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    db_manager = DatabaseManager(config.DB_FILE)
    db_manager.create_match_log_table()

    matcher = CommitMatcher(db_manager)

    if args.method == 'fuzzy':
        threshold = args.threshold if args.threshold is not None else 75.0
        matcher.match_fuzzy(threshold=threshold, dry_run=args.dry_run)
    else:
        threshold = args.threshold if args.threshold is not None else 0.80
        matcher.match_difflib(threshold=threshold, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
