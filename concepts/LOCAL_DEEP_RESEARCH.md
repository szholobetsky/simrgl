# Local Deep Research: Academic-Grounded Recursive Document Generation

## Abstract

This document describes a proposed extension to the `deepagent_md` flow that transforms it from a general-purpose recursive document generator into an academic research assistant. The core additions are: (1) academic-source filtering for web searches, (2) automatic reference collection and citation injection, and (3) a `--rag` mode that queries a local document index via simargl instead of (or in addition to) the web. The resulting system is a local, privacy-preserving alternative to cloud-based deep research tools — capable of producing structured, cited, multi-level documents from a combination of web sources and personal document libraries.

---

## 1. The Gap: Generation Without Grounding

The current `deepagent_md` produces rich, hierarchically structured documents. Its weakness is epistemic: the content is generated from the model's parametric knowledge, supplemented by general web search. For academic and technical writing, this is insufficient. A researcher needs:

- Sources that can be cited and verified
- Content anchored to specific papers, not to model hallucinations
- References that can be followed, not invented URLs

The `--web` flag partially addresses this, but it queries the open web — Wikipedia, blog posts, Stack Overflow. For academic work, the signal-to-noise ratio is poor.

---

## 2. Component 1: Academic Source Filtering (`--academic`)

Replace general DuckDuckGo search with queries restricted to academic sources.

**Approach A — DDG site: operators:**
```python
academic_suffix = (
    " (site:arxiv.org OR site:scholar.google.com "
    "OR site:academia.edu OR site:semanticscholar.org "
    "OR site:ncbi.nlm.nih.gov OR site:jstor.org)"
)
query = f"{title} {root_task}" + academic_suffix
```
Simple, no API key required. DDG support for complex OR site: is inconsistent.

**Approach B — Semantic Scholar API (recommended):**
```
GET https://api.semanticscholar.org/graph/v1/paper/search
    ?query=<title+root_task>
    &fields=title,authors,year,abstract,externalIds,url
    &limit=5
```
Free, no key, returns structured data: title, authors, year, abstract, DOI, URL. The abstract can serve as the "web context" injected into the prompt — clean, structured, citable.

**Output per result:**
```
[1] Smith et al. (2023). "Title of Paper." ArXiv:2301.XXXXX
    Abstract: ...first 300 words...
```

---

## 3. Component 2: Reference Collection and Citation

**Programmatic references (always works, even with 1B models):**

During `_web_research()` or `_academic_search()`, collect:
```python
refs: list[dict] = [
    {"id": 1, "title": ..., "url": ..., "authors": ..., "year": ...},
    ...
]
```

Inject into prompt:
```
Sources available for citation (use [1], [2] inline):
[1] Smith et al. (2023) — "Heat Equation Methods" — arxiv.org/...
[2] Jones (2021) — "Cauchy Problems in Java" — doi.org/...
```

After generation, append to the file:
```markdown
## References

[1] Smith, J., et al. (2023). *Heat Equation Methods for Distributed Systems*. ArXiv:2301.XXXXX. https://arxiv.org/...
[2] Jones, M. (2021). *Numerical Cauchy Problems in Java*. doi:10.1000/xyz123
```

Small models will inconsistently use `[1]` inline — but the References section is always appended programmatically regardless. Inline citations improve with larger models.

**HTML compose integration:**
When `--mode html_plain --plan` is active, references in each section become `<a href="url">[1]</a>` links. The References section at the end becomes a proper bibliography with external links.

---

## 4. Component 3: Local RAG via simargl (`--rag`)

For users with a local document library already indexed by simargl, the `--rag` flag queries it instead of (or alongside) the web.

**Invocation:**
```python
import subprocess
result = subprocess.run(
    ["simargl", "search", query, "--format", "json", "--limit", "5"],
    capture_output=True, text=True, cwd=project_dir
)
```

simargl returns ranked fragments from local documents. These fragments are injected into the prompt as "local knowledge context" — distinct from web context, with their own reference block:

```markdown
## Local Sources

[L1] internal/papers/heat_equation_study.pdf — p.14
[L2] internal/notes/cauchy_numerical.md
```

**Combined mode (`--web --rag --academic`):**
- Local simargl index → `[L1]`, `[L2]`... 
- Semantic Scholar API → `[1]`, `[2]`...
- Both injected into prompt, both collected in References

This is the full pipeline: local knowledge + academic web → grounded generation → hierarchical document with bibliography.

---

## 5. Relationship to Google DeepSearch

| Feature | Google DeepSearch | deepagent_md + this proposal |
|---|---|---|
| Multi-level document structure | ✓ | ✓ |
| Web search grounding | ✓ | ✓ |
| Academic sources | ✓ | ✓ (Semantic Scholar) |
| Citations | ✓ | ✓ (programmatic + prompted) |
| Local document RAG | ✗ | ✓ (simargl) |
| Multi-hop search | ✓ | ✗ (one level per section) |
| Privacy (runs locally) | ✗ | ✓ |
| Parallel workers | ✗ | ✓ |
| Cost | cloud API pricing | free (local models) |
| Model quality | large cloud models | 1B–8B local |

**The honest gap:** multi-hop search — DeepSearch reads a paper, finds references within it, reads those too. We search once per section. For most academic writing tasks (literature review, concept exploration, methodology planning) one-hop is sufficient. For genuine discovery of unknown sources, multi-hop matters.

**The honest advantage:** the combination of local RAG + academic web + parallel workers + hierarchical structure with a fully local, private, free pipeline is not available anywhere else at this level of integration.

---

## 6. Implementation Plan

### Phase 1 — Academic search + References (self-contained)
- Add `--academic` flag to `deepagent_md`
- Implement `_academic_search(title, root_task)` using Semantic Scholar API
- Collect refs per file during generation
- Append `## References` block to each generated file
- In `html_plain` compose: render refs as `<a href>` links

### Phase 2 — RAG integration
- Add `--rag [index_path]` flag
- Implement `_rag_search(query, index_path)` via simargl subprocess
- Merge local refs (`[L1]`) with web refs (`[1]`)
- Respect `--ctx-worker 0` — rag context stays local unless explicitly shared

### Phase 3 — Multi-hop (optional, future)
- After fetching a paper abstract, extract cited-by DOIs from Semantic Scholar response
- Fetch top 2 cited-by papers as additional context
- Adds one more level of grounding without full multi-hop complexity

---

## 7. Sample Output Structure

```markdown
# Heat Equation as a Cauchy Problem in Java

## 1. Mathematical Foundation

The heat equation, as a parabolic PDE, admits a unique solution under Cauchy
initial conditions when the initial data belongs to a suitable function space [1].
Numerical discretization via finite differences requires stability conditions
derived from the von Neumann analysis [2].

...

## References

[1] Evans, L.C. (2010). *Partial Differential Equations*. AMS Graduate Studies.
    https://doi.org/10.1090/gsm/019
[2] Strikwerda, J.C. (2004). *Finite Difference Schemes and PDEs*. SIAM.
    https://doi.org/10.1137/1.9780898717938
[L1] local/papers/heat_eq_java_impl.pdf — section 3.2 (via simargl)
```
