---
name: feedback-research-vs-hardware-constraints
description: "For exp4/exp5 research model selection, do not filter out models based on 6GB VRAM — theoretical feasibility is tested in practice regardless of local hardware fit."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5242261c-51ee-42c6-9c30-12ac80d2755a
  modified: 2026-07-21T07:12:15.791Z
---

Do not constrain experimental design (exp4/exp5 architecture-family comparisons, model selection) by the local 6GB VRAM limit. This is fundamental/dissertation research — the point is to test theoretical feasibility of an approach in practice, not to only propose what fits on the researcher's own GPU.

**Why**: User explicitly corrected this (2026-07-21): "ми не повинні орієнтуватись жодним чином на 6gb VRAM, у нас фундаментальна наука, теоретична можливість перевіряється на практиці" — research conclusions should not be pre-filtered by what's locally deployable.

**How to apply**: When proposing embedding models, architectures, or experiment configurations for exp4/exp5 (or future research experiments), do not exclude a model or downgrade a recommendation solely because it exceeds 6GB VRAM. Larger models can be run via Ollama quantization, cloud, or borrowed hardware if the research question calls for it. The 6GB VRAM constraint remains relevant only for **production/deployment** contexts (ragmcp, actual local tool usage — see [[project_academic_context]] for the research/product split), not for research experiment design itself. This appears to partially conflict with the AGENTS.md line "Hardware-aware suggestions: keep 6GB VRAM limit in mind for any model recommendations" — flagged to the user for possible clarification/update of that doc.
