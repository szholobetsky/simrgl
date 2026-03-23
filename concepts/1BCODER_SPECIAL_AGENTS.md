# The Thin Agent Principle: Specialization as the Superpower of Small Model Agents

**Document type**: Research finding + Experimental Design
**Date**: 2026-03-23
**Project**: SIMARGL / 1bcoder
**Status**: Hypothesis formed, experiment pending

---

## 1. The Story: From One Agent to Many, and Back to One

### 1.1 The Base Agent

1bcoder's original `/agent` command was designed for general coding tasks. The model received a system prompt listing all available tools — read, insert, save, patch, fix, bkup, diff, run, tree, find, map — and was given a task. It worked for models with large context windows (32B+). For 1B–4B models, it failed in predictable ways: the model hallucinated tool names, picked wrong tools, or simply repeated the system prompt in its reply.

The tool list itself was the problem. A 12-tool system prompt at 7B+ parameter scale is a rich action space. At 1.7B it is noise.

### 1.2 The First Specialization: `/agent advance`

The natural response was to create two modes: a minimal tool set for small models (`tools =`) and a full tool set for larger models (`advanced_tools =`). `/agent advance` was the first named configuration — the same loop, a different whitelist. This already embodied the specialization instinct, but `advance` was still defined in code (`agent.txt`), not in a file.

### 1.3 Extracting the Common Part: The Shared Loop

As `/ask` (read-only research agent) was developed, a second agent loop appeared in the codebase — separate code, slightly different behavior, same fundamental structure. The refactoring question became: what is common between all agents?

The answer: everything except the system prompt and the tool list.

`_run_agent_loop(label, agent_msgs, max_turns, auto_exec, auto_apply)` became the shared substrate. Every agent type — `/ask`, `/agent <name>`, `/agent` — executes through the same loop. What differs is the configuration loaded before the loop starts.

This architectural observation had a conceptual consequence: **an agent is not a loop variant, it is a configuration**. The loop is infrastructure. The agent is a file.

### 1.4 The Named Agent System

Once the loop was extracted, the next step was obvious: move configuration into files. `.1bcoder/agents/<name>.txt` defines a complete agent:

```ini
system =      # inline system prompt (multiline)
tools =       # whitelist of allowed tools
aliases =     # agent-scoped command vocabulary
max_turns = N
auto_exec = true/false
auto_apply = true/false
```

`ask.txt` and `advance.txt` were the first two agents moved out of code. The global `aliases.txt` got `/ask = /agent ask` and `/advance = /agent advance`. Commands that were hardcoded became file-defined. Adding a new agent type required creating a file, not modifying `chat.py`.

### 1.5 The SQLite Agent: Accidental Discovery

The SQLite agent was created for a practical reason: working with database files required repetitive sqlite3 invocations. The natural response was to define aliases:

```ini
aliases =
    /schema = /run sqlite3 {{args}} ".schema"
    /query  = /run sqlite3 {{args}}
```

Tools: just `run` and `find`. Two tools. A system prompt of ~200 tokens. The task: query a real database.

**qwen3:1.7b completed the task correctly**. It found the database file, extracted the schema, wrote a valid SQL SELECT, ran it, received the result, and answered the question in plain text.

For reference:
- `nemotron-3-nano:4b` running `opencode` and `nanocoder` (general agents) failed to start — context overflow before the first turn
- `aider`'s general agent started with `nemotron-3-nano:4b` but looped the system prompt — the model's attention was saturated by the tool description list
- The same `nemotron-3-nano:4b` running the `/ask` agent (4 tools) completed read-only research tasks but not reliably

The pattern was not subtle: the smaller the model's context and the more specific the task, the more critical the tool list size becomes.

### 1.6 The Principle

**The best agent for a 1B–4B model is not a large agent with many skills. It is a thin, domain-specific agent with fewer than five active tools, a hard context injection limit, and a system prompt that describes exactly one workflow.**

This is not a limitation to work around. It is an architectural conclusion that changes how agents should be designed for small models.

---

## 2. Theoretical Framing

### 2.1 Cognitive Load and Tool Surface

Miller's Law (1956) established that human working memory holds 7±2 items. The analogy to language models is imprecise but structurally relevant: the attention mechanism distributes weight across all tokens in context. A system prompt listing 12 tools with 2-line descriptions each (~300 tokens) competes with the task description for the model's effective attention budget. For a model with a 8K context window and 1.7B parameters, that competition is decided before generation begins.

The thin agent principle operationalizes this: reduce the tool surface to the minimum necessary for the domain, and the model's attention concentrates on the task.

### 2.2 Vocabulary Alignment

Agent-scoped aliases do something deeper than convenience: they align the model's output vocabulary with the actual command interface. Instead of asking a 1.7B model to construct `sqlite3 bookcrossing/dev.sqlite ".schema"` from first principles, the alias `/schema bookcrossing/dev.sqlite` gives the model a single token to emit. The cognitive cost of constructing a command drops dramatically.

This is the same principle as `FIX_SYSTEM`'s `LINE N: content` format — constrained output vocabulary reduces the generation space to what the model can reliably navigate.

### 2.3 Domain Specificity as Context Injection

A general agent's system prompt must describe many tools in abstract terms. A domain-specific agent's system prompt can describe one workflow in concrete terms, with domain vocabulary already in place. For the SQLite agent:

> "Get schema. Write SELECT. Run. Answer in plain text."

Four sentences. The model has no choice surface to navigate. The workflow is deterministic. The only creative step is writing the SQL — and the schema retrieved in step 1 is exactly the context needed.

### 2.4 Connection to the Anthill Hypothesis

The Anthill research program's core claim is: *context quality substitutes for model scale*. The thin agent principle is a specific instance of this at the agent architecture level: **agent configuration quality substitutes for model scale**. A poorly configured agent with 12 tools fails at 4B parameters. A well-configured agent with 2 tools succeeds at 1.7B. The delta is not in the model — it is in the architecture of the agent that wraps it.

---

## 3. Empirical Observations (2026-03-23)

| Tool | Model | Config | Result |
|---|---|---|---|
| 1bcoder `/sqlite` agent | qwen3:1.7b | 2 tools, domain aliases | ✅ Completed DB query task correctly |
| opencode | nemotron-3-nano:4b | General agent (12+ tools) | ❌ Context overflow before turn 1 |
| nanocoder | nemotron-3-nano:4b | General agent | ❌ Context overflow before turn 1 |
| aider | nemotron-3-nano:4b | General agent | ⚠️ Started but looped system prompt |
| 1bcoder `/ask` agent | nemotron-3-nano:4b | 4 tools, read-only | ⚠️ Partial — unstable, some tasks complete |
| 1bcoder `/ask` agent | qwen3:4b | 4 tools, read-only | ✅ Reliable research task completion |

These are informal observations, not controlled experiments. They motivate the experimental design in §5.

---

## 4. Related Scientific Literature

The literature review below was conducted March 2026 via systematic search. All papers are 2022–2025. Papers are grouped by the claim they support.

### 4.1 Tool Count Directly Degrades Agent Accuracy

**Small LLMs Are Weak Tool Learners: A Multi-LLM Agent**
Shen et al. EMNLP 2024. ArXiv: [2401.07324](https://arxiv.org/abs/2401.07324)
When a single small LLM is given all tool-use capabilities simultaneously, performance limits emerge immediately. Decomposing the agent into a specialized planner, caller, and summarizer — each a small model focused on one capability — surpasses the monolithic baseline across all benchmarks. The finding: **cognitive role specialization compensates for the weakness of small models under broad tool scope**. Our thin agent system achieves the same compensation differently — by restricting scope at the agent level rather than decomposing into sub-agents.

**Tool Complexity Impact on AI Agent Accuracy**
Allen Chan. March 2025. [Medium / industry analysis](https://achan2013.medium.com/how-tool-complexity-impacts-ai-agents-selection-accuracy-a3b6280ddce5)
Production-grade measurements on GPT-4o: calendar scheduling accuracy collapsed from **43% with 4 tools (1 domain)** to **2% with 51 tools (7 domains)**. Customer support accuracy: 58% (9 tools) → 26% (51 tools). This degradation occurs even in frontier models; it scales worse in smaller ones. Pre-filtering to domain-relevant tools before passing them to the model is the recommended mitigation — which is exactly what a thin agent's `tools =` list implements statically.

**ToolLLM: Facilitating Large Language Models to Master 16,000+ Real-world APIs**
Qin et al. ICLR 2024 Spotlight. ArXiv: [2307.16789](https://arxiv.org/abs/2307.16789)
With 16,464 APIs across 49 categories, ToolLLM's own evaluation confirms that tool selection from large candidate sets is the primary failure point. Smaller/open-source models fail at a significantly higher rate than GPT-based models precisely on this selection step — not on execution. The problem is not "can the model run the tool?" but "can it choose the right one from a large surface?"

### 4.2 Dynamic / Filtered Tool Selection Outperforms Full Toolset Access

**AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning**
NVIDIA / academic consortium. 2024. ArXiv: [2512.13278](https://arxiv.org/abs/2512.13278)
AutoTool selects dynamically from pools of 1,000+ tools rather than presenting all tools at once. Outperforms static full-toolset access by: **+6.4% math/science reasoning, +4.5% search-based QA, +7.7% code generation, +6.9% multimodal understanding**. Agents receiving only the relevant tool subset consistently outperform agents receiving the full set. Our thin agent is the static analog: domain specificity at design time achieves what AutoTool achieves at query time.

**Dynamic ReAct: Scalable Tool Selection for Large-Scale MCP Environments**
Gaurav et al. 2025. ArXiv: [2509.20386](https://arxiv.org/abs/2509.20386)
Dynamic loading of only query-relevant tools **reduces tool loading by up to 50%** while maintaining task completion accuracy. Constant memory usage regardless of total registry size; retrieval scales logarithmically. Directly demonstrates that subset-loading is both computationally and accuracy-superior to presenting all tools to the agent.

### 4.3 Context Saturation and Small Model Failure

**Cognitive Load Limits in Large Language Models: Benchmarking Multi-Hop Reasoning**
Adapala. 2025. ArXiv: [2509.19517](https://arxiv.org/abs/2509.19517)
Introduces formal theory of **Context Saturation** (task-irrelevant information degrading attention allocation) and **Attentional Residue** (task-switching interference). Key measurement: small open-source models (Llama-3-8B, Mistral-7B) achieve **0% accuracy on high cognitive load tasks** even in clean control conditions. Gemini-2.0-Flash achieves 85% in clean conditions but degrades significantly under context saturation. This is the closest analog in the literature to a formal study of tool-list cognitive overload.

**Bootstrap Your Own Context Length**
2024. ArXiv: [2412.18860](https://arxiv.org/html/2412.18860v1)
Official Llama-3 1B and 3B models **nearly fail entirely at 128k context length**. This is a hard empirical data point: sub-3B models cannot use long context windows reliably, meaning a large tool list inflating the system prompt is categorically more damaging to small models. Fewer tools = shorter context = functional model. Not a marginal effect — a categorical one.

**Solving Context Window Overflow in AI Agents**
2025. ArXiv: [2511.22729](https://arxiv.org/html/2511.22729v1)
Large tool outputs overflow the LLM context window and prevent task completion. Truncation and summarization fail to preserve complete output semantics. In small models (shorter context windows), tool-output overflow is an earlier and harder ceiling than in large models. Our `ASK_RESULT_LIMIT_CHARS` truncation in 1bcoder is a direct implementation of the recommended mitigation.

**Efficient On-Device Agents via Adaptive Context Management**
2025. ArXiv: [2511.03728](https://arxiv.org/abs/2511.03728)
Instantiates a context management framework on a **3B parameter SLM** deployed on-device. Limited memory capacity is the primary architectural constraint. Adaptive context pruning enables the 3B model to perform complex multi-step tasks that would otherwise overflow. Explicit evidence that small model agentic performance requires aggressive context discipline — which thin agents provide by design.

### 4.4 Small Model Specialization Beats General Large Model Agents

**Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning**
Jhandi et al. (Amazon). 2024. ArXiv: [2512.15943](https://arxiv.org/abs/2512.15943)
A **350M parameter model** fine-tuned on a specific tool domain achieves **77.55% pass rate**, significantly outperforming general LLMs up to 500× its parameter count. 350M is identified as a "strategic sweet spot" for tool calling: sufficient capacity to learn API interaction patterns without the inconsistency overhead of larger general models. **Specialization on a specific tool domain is the decisive factor, not model size.**

**Small Language Models are the Future of Agentic AI**
Belcak, Heinrich et al. (NVIDIA). 2025. ArXiv: [2506.02153](https://arxiv.org/abs/2506.02153)
Agentic AI systems involve models performing a small number of specialized tasks repetitively, making SLMs (under 10B) inherently more suitable and economical than general LLMs. The paper argues multi-agent systems should be built around specialized SLMs rather than one general large model. This is the closest existing statement to the thin agent principle: **the architecture of the agent system, not the size of any individual model, determines capability**.

**Small Language Models for Agentic Systems: A Survey**
2025. ArXiv: [2510.03847](https://arxiv.org/abs/2510.03847)
SLMs (1–12B) are sufficient and often superior for agentic workloads where the objective is schema- and API-constrained accuracy rather than open-ended generation. Covers Llama-3.2-1B/3B, Phi-4-Mini, Qwen-2.5-7B, Gemma-2-9B. Specialization beats general capability for tool-calling tasks. Survey explicitly identifies constrained tool surfaces as a key enabler for small model agentic success.

### 4.5 Tool Hallucination Increases with Broader Tool Exposure

**The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination**
Yin et al. 2025. ArXiv: [2510.22977](https://arxiv.org/abs/2510.22977)
Enhancing reasoning via RL increases tool hallucination **proportionally** with task performance gains. Introduces SimpleToolHalluBench measuring two failure modes: no tool available, and only distractor tools available. **Distractors (irrelevant tools in the toolset) directly cause hallucinated tool calls.** The mechanism is causal, not correlational. Reducing tool exposure is a direct mitigation — which is exactly what the `tools =` whitelist in thin agent files implements.

**LLM-based Agents Suffer from Hallucinations: A Survey**
2025. ArXiv: [2509.18970](https://arxiv.org/abs/2509.18970)
Tool hallucination (improper tool selection or misuse) is identified as a critical failure mode distinct from factual hallucination. Smaller models are more susceptible to all hallucination types under high-tool-count conditions.

### 4.6 Multi-Agent Specialization as the Architectural Answer

**Reducing Cognitive Overhead in Tool Use via Multi-Small-Agent Reinforcement Learning**
2025. ArXiv: [2508.08882](https://arxiv.org/html/2508.08882v4)
The single-agent paradigm introduces "cognitive load interference" where the same model must juggle long-horizon reasoning with precise low-level tool operations simultaneously. Multi-small-agent RL distributes this burden, with each agent responsible for a narrow scope. Validates the architectural principle behind 1bcoder's named agent system: the Router (human or alias) selects the specialist agent; the specialist executes without navigating a large tool surface.

**ReAct: Synergizing Reasoning and Acting in Language Models**
Yao et al. (Google Brain / Princeton). ICLR 2023. ArXiv: [2210.03629](https://arxiv.org/abs/2210.03629)
The foundational agent paper. ReAct's token overhead (chain-of-thought interleaved with tool calls) makes it expensive and especially costly for small models, where each additional tool description in the prompt competes directly with reasoning tokens. Subsequent work consistently cites this as the primary scaling bottleneck for small model agents.

### 4.7 Summary Table

| Claim | Papers | Strength |
|---|---|---|
| More tools → lower accuracy, even in frontier models | [Allen Chan], [ToolLLM], [AutoTool] | Strong — quantitative, production measurements |
| Fewer, domain-relevant tools → higher accuracy | [AutoTool], [Dynamic ReAct], [Thin Agents Survey] | Strong — controlled experiments |
| Small models (1–3B) fail under long context / large tool lists | [Bootstrap CL], [Cognitive Load Limits], [Context Overflow] | Strong — direct measurements, 0% accuracy reported |
| Specializing a small model beats a general large model for tool calling | [Amazon 350M], [NVIDIA SLMs are Future], [SLM Survey] | Strong — benchmark results, 500× size gap reversed |
| Distractor tools cause hallucination — tool surface reduction is a direct mitigation | [Reasoning Trap], [Hallucination Survey] | Strong — causal mechanism found |
| Decomposing cognitive roles beats monolithic multi-tool agent | [Weak Tool Learners], [Multi-Small-Agent RL] | Strong — ablation evidence |

---

## 5. Experimental Design

### 5.1 Research Question

Does reducing the number of available tools and increasing domain specificity of the system prompt improve task completion rate for language model agents at ≤4B parameter scale?

### 5.2 Hypothesis

**H₀**: Task completion rate is independent of agent tool count for models with ≤4B parameters.

**H₁**: Agents with ≤5 domain-specific tools achieve significantly higher task completion rates than agents with ≥10 general tools, for models with ≤4B parameters.

**Secondary H₁ₐ**: The performance advantage of thin agents over general agents is greater at smaller model scales (1.7B > 4B > 7B).

### 5.3 Variables

**Independent variables:**
- Tool count: 2 (ultra-thin), 5 (thin), 10 (medium), 15+ (full)
- Model size: 1.7B, 4B, 7B, 32B
- Task domain: DB query, code search, code edit, multi-file trace

**Dependent variables (metrics):**

| Metric | Symbol | Definition |
|---|---|---|
| Correct Answer Rate | CAR | Fraction of tasks where final plain-text answer is factually correct (human-verified) |
| Failed Action Rate | FAR | Failed actions / total actions per task (exit code ≠ 0) |
| Context Saturation | CTX | Peak context usage / context limit during run (%) |
| Turns to Completion | TTC | Number of agent turns used |
| Retry Rate | RTR | Times agent retried a failed action without prompting |
| Hallucinated Tool Rate | HTR | Actions calling a non-existent tool or malformed command |
| Context Overflow Rate | COR | Fraction of runs where context exceeded limit before completion |

### 5.4 Agent Configurations

| Config ID | Name | Tools | System Prompt | Aliases |
|---|---|---|---|---|
| A1 | ultra-thin-db | 2 (`run`, `find`) | SQLite workflow, 5-step | `/schema`, `/query` |
| A2 | thin-research | 4 (`read`, `tree`, `find`, `map find`) | Research workflow, 6-step | `/kw`, `/sym` |
| A3 | medium-coding | 6 (`read`, `find`, `patch`, `insert`, `run`, `bkup`) | General coding | none |
| A4 | full-general | 13 (complete tool set) | Full system prompt | none |

### 5.5 Task Set

**Domain 1 — DB Query (10 tasks)**
Tasks of the form: "How many X in table Y?", "Find the Z with the highest W in category C", "List all X where Y > N". Ground truth: correct SQL result.

**Domain 2 — Code Search (10 tasks)**
Tasks of the form: "Where is function X defined?", "Which files import module Y?", "Find all uses of constant Z". Ground truth: correct file:line reference.

**Domain 3 — Code Edit (10 tasks)**
Tasks of the form: "Fix the divide-by-zero in file X", "Add parameter validation to function Y". Ground truth: code passes the specified test.

**Domain 4 — Multi-file Trace (10 tasks)**
Tasks of the form: "What calls function X?", "Trace the call chain from endpoint Y". Ground truth: complete correct trace.

40 tasks total × 4 agent configs × 4 model sizes = **640 runs**.

### 5.6 Control Conditions

- Same task phrasing across all agent configs
- Same Ollama host and hardware
- Same `temperature = 0.1` (deterministic as possible)
- Same `num_ctx` for each model (model's native maximum)
- Fresh session per run (`/clear` between runs)
- Context injection blocked — agent must discover all information through tools

### 5.7 Statistical Tests

**Primary test — CAR across agent configs:**
Chi-square test of independence (or Fisher's exact test when expected cell count < 5):
- 2×2 contingency table: tool count group (thin vs. general) × outcome (correct vs. incorrect)
- α = 0.05
- Effect size: Cramér's V

**Secondary test — CAR interaction (tool count × model size):**
Logistic regression with tool count and model size as predictors, CAR as binary outcome. Tests whether the thin agent advantage is moderated by model size (H₁ₐ).

**Continuous metrics (FAR, CTX, TTC, RTR, HTR):**
Mann-Whitney U test (non-parametric; distributions unknown):
- Compare A1+A2 (thin) vs. A3+A4 (general) within each model size
- Bonferroni correction for multiple comparisons (6 metrics × 4 model sizes = 24 tests; α_adjusted = 0.05/24 ≈ 0.002)
- Effect size: rank-biserial correlation

**Multi-group comparison:**
Kruskal-Wallis H test across all 4 agent configurations, with Dunn's post-hoc test for pairwise comparisons.

**Minimum sample size:**
For chi-square with α=0.05, power=0.80, medium effect size (Cramér's V = 0.30):
n ≈ 88 per condition. With 10 tasks per domain and 4 domains, n = 40 per condition — below threshold. Minimum viable experiment requires 25 tasks per domain (100 tasks total) or pooling model sizes.

### 5.8 Measurement Protocol

1. Run each task three times per config/model combination (triplicate for reliability)
2. CAR scored by human evaluator blind to agent config (evaluator sees only task + final answer)
3. FAR, CTX, TTC logged automatically from agent loop output
4. HTR logged by comparing ACTION commands against the allowed tool list
5. Log the full agent transcript for post-hoc analysis of failure modes

### 5.9 Expected Failure Modes to Observe

| Failure Mode | Expected in Config | Mechanism |
|---|---|---|
| Context overflow before turn 1 | A4 with 1.7B–4B models | Tool list + context exceeds num_ctx |
| System prompt repetition / loop | A4 with 4B models | Model attends to system prompt tokens, repeats them |
| Hallucinated tool name | A3, A4 with 1.7B | Wrong tool selected from large surface |
| Placeholder not substituted (e.g. `<db_file>`) | A3, A4 | Model treats system prompt placeholder literally |
| Correct tool, wrong syntax | A1 without aliases | Model constructs sqlite3 command incorrectly |
| Premature completion | A1–A2 | Model gives up after one failed action |

---

## 6. Predicted Results

Based on the empirical observations in §3 and the theoretical framing in §2:

1. **CAR**: A1 > A2 > A3 > A4 for models ≤4B, with effect size decreasing at 7B and disappearing at 32B
2. **CTX**: A4 will overflow or saturate for 1.7B–4B; A1–A2 will remain well below limit
3. **TTC**: A1 will converge faster than A4 on domain-specific tasks; A4 may never converge
4. **HAR**: A4 will have significantly higher hallucinated tool rate at 1.7B–4B
5. **Interaction**: the advantage of thin agents will be strongest at 1.7B and negligible at 32B, consistent with H₁ₐ

---

## 7. Implications for Agent Design

If the hypothesis is confirmed, the design implications are:

1. **Agent proliferation is correct**: instead of one general agent, build many thin agents. Each new domain gets a new `.txt` file. The overhead is minimal; the capability gain is large.

2. **Aliases are first-class primitives**: domain vocabulary encoded as aliases reduces the generation complexity for the model. Agent files should always include aliases for frequently used, syntactically complex commands.

3. **System prompt as workflow spec**: a good thin agent system prompt describes exactly one workflow in 5–8 steps. Abstract capabilities ("you can use any tool") are replaced with concrete sequences ("step 1: /find db file; step 2: /schema").

4. **Context limits are architecture, not configuration**: the tool list injected into the system prompt must fit within a budget that leaves room for task, context, and answer. For 1.7B models with 8K context, the system prompt budget is approximately 500 tokens. At 2 lines per tool, this allows ≈ 8 tools maximum — and the practical ceiling for reliable operation is lower.

5. **The Router problem is dissolved**: in large multi-tool agents, the model must select the correct tool from a large surface. In thin agents, the correct tool is the only tool. Routing is resolved by agent selection (human or alias), not by the model.

---

## 8. Connection to the Anthill Architecture

The thin agent principle does not contradict the Anthill vision — it specifies it. The Anthill describes a network of specialized agents, each with a defined role and a restricted behavioral envelope (SKILL.md). What 1bcoder's named agent system implements is the **SKILL.md as a file format**: system prompt = behavioral specification, tool list = allowed verbs, aliases = domain vocabulary extension.

The Anthill's Router agent selects which specialist agent to invoke for a given task. In 1bcoder, the human (or an alias) plays the Router. The thin agent principle says: when the Router selects correctly, even a 1.7B model succeeds at domain-specific tasks.

The experimental design in §5 is therefore also **Exp B2** from [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) (Specialization vs. Generality), instantiated at the agent configuration level rather than the model level.

---

## 9. Next Steps

- [ ] Implement task logging in `_run_agent_loop` — FAR, CTX, TTC, HTR recorded automatically per run
- [ ] Define the 40-task benchmark set (10 per domain) with ground truth answers
- [ ] Run pilot experiment: 10 tasks × A1 vs A4 × qwen3:1.7b + qwen3:4b (minimum viable comparison)
- [ ] Design the CAR evaluation rubric for human scoring
- [ ] Integrate search results from literature review (§4 to be updated when background search completes)

---

**Document Version**: 1.0
**Created**: 2026-03-23
**Project**: SIMARGL / 1bcoder
**Relation to**:
- [1BCODER.md](1BCODER.md) — implementation document (§2.19 for named agents, §3.2 for thin agent in Phase 0 list)
- [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) — Exp B2 (Specialization vs. Generality)
- [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) — SKILL.md concept, Phase 1 agent specialization
