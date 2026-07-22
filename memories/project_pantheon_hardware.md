---
name: project-pantheon-hardware
description: "The \"Pantheon\" mini PC hardware setup running the tool ecosystem, and the empirical finding that Vulkan/iGPU only helps LLM inference if the iGPU itself is strong enough — a weak iGPU makes things slower, not faster"
metadata: 
  node_type: memory
  type: project
  originSessionId: 59f4e9c6-779f-4f0a-9009-ff73ef6ea5a3
---

# Пантеон — mini PC hosting the tool ecosystem

User's mini PC (i5-6500T, 32GB RAM, CPU-only, no discrete GPU) is nicknamed **Пантеон** ("Pantheon"). It runs the growing tool ecosystem: [[project_vyrii_flask|vyrii]], simargl, svitovyd, yasna, radogast, and a newer tool called **syryn** (сирин).

**Why:** Pantheon was bought as a dedicated local-LLM box for the tool ecosystem, but pure CPU-only inference turned out to hit the context-length wall very early (see [[project_ctxtimer|CTXTIMER concept]]) — the same limitation that motivated building `/flow ctxtimer` in 1bcoder.

## Corrected finding (2026-07-06): Vulkan/iGPU benefit is conditional on iGPU strength, NOT universal

**Earlier claim in this memory was wrong and has been corrected.** The user enabled Vulkan on Pantheon itself (added `OLLAMA_VULKAN`/`OLLAMA_IGPU_ENABLE` to [[project_vyrii_flask|vyrii]]'s Ollama settings), but Pantheon's iGPU is an **Intel HD 530** (Skylake-era, 2015, only 24 execution units) — and with Vulkan+iGPU enabled, LLMs ran **significantly slower** than plain CPU-only inference on the same machine. The model still fully fit (HD 530 sees system RAM as unified memory, could address even 20GB+ models), so it wasn't a capacity problem — it was a raw throughput problem: HD 530 is simply too weak to beat the CPU's own AVX2 SIMD throughput, and Vulkan dispatch/driver overhead on top of that made it a net loss.

By contrast, the user's laptop has an **Intel Iris Xe** (Tiger Lake-era, 2020, up to 96 execution units) — a genuinely stronger iGPU — and that specific card is what gave the earlier-observed ~2x context/speed improvement, not "any Vulkan-capable iGPU" as a general rule.

**Why this matters:** The earlier hypothesis ("attention's quadratic term is embarrassingly parallel, so even a weak iGPU should help") was too optimistic — it ignored that a sufficiently weak GPU can have LOWER raw compute throughput than a modest desktop CPU already achieves via AVX2/SIMD across 4 cores, in which case adding Vulkan dispatch overhead on top makes things worse, not better.

**How to apply:** Do NOT recommend "enable Vulkan/iGPU" as a blanket improvement for this project's local-LLM tools. The correct recommendation is iGPU-specific: check the actual EU count / generation (HD 5xx/6xx era = likely not worth it; Iris Xe / Xe-LP or newer = likely worth it), and — more reliably — use `/flow ctxtimer` (see [[project_ctxtimer]]) to empirically A/B test Vulkan on vs off on the specific machine before assuming either direction. This is exactly the kind of assumption ctxtimer exists to replace with measurement.

## Timeline correction (corrected 2026-07-05)
In May 2025 the user was NOT working on 1bcoder or simargl yet — was exploring computational linguistics generally; simargl wasn't even conceptually researched at that point. The real turning point for 1bcoder was **2026-03-05**: that's when the user moved from simpler flow-style experiments (exp2/exp3/exp4-style scripts) to building an actual agent architecture — comparable to Claude Code, opencode, or aider, but scoped to 1B-parameter local models. Earliest known 1bcoder prototype on disk: `C:\Users\stzh\WorkFile\PyProj\1bcoder\old versions\chat - old.py` (dated 2026-03-05). Collaboration with Claude on this project started November/December 2025 — predating the "real agent" pivot, so early sessions were likely on earlier, simpler versions of 1bcoder.
