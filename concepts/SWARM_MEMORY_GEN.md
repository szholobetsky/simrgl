# Swarm Memory with Generational GC: A Constrained Communication Protocol for Multi-Agent Code Generation

**Status**: Concept — not implemented
**Created**: 2026-03-18
**Related**: `ANTHILL_DISTRIBUTED_COGNITIVE_OS.md`, `ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md`, `1BCODER.md`

---

## 1. The Problem This Solves

The Anthill OS (see `ANTHILL_DISTRIBUTED_COGNITIVE_OS.md`) provides a rich coordination infrastructure: OKG, Object Passports, LangGraph orchestrator, role-specialized agents. But it requires significant infrastructure to deploy.

This concept explores the opposite extreme: **what is the minimum viable coordination mechanism for a swarm of small models?**

The constraint: a 1B model cannot hold a large codebase in context. It cannot plan a multi-file refactor. It cannot decompose tasks it does not understand. But it *can* read 25 lines and write one sentence that is true and useful.

The question: can 20 such models, each seeing only 25 lines of shared text, collectively move a codebase in a coherent direction?

---

## 2. The Core Mechanism

### 2.1 The Shared Memory File

A single text file: `.1bcoder/swarm/memory.txt`. Maximum 25 lines. Never exceeds this limit.

**Line 0** (immutable): The task statement. Written once by the human. Never deleted, never modified. This is the gravitational anchor of the system.

```
[TASK] implement token bucket rate limiter for the API gateway
```

**Lines 1–N**: Agent contributions. Each line is one distilled observation, decision, or constraint — written by one agent after thinking for up to 20 turns.

```
[G2] RateLimiter must wrap RequestHandler, not replace it
[G1] token bucket chosen over sliding window: simpler state, proven in Redis
[G0] tests expect interface: .allow(user_id) -> bool
agent_4: X-RateLimit-Remaining header missing from current response
agent_7: Redis key format proposal: rl:{user_id}:{minute_bucket}
agent_2: allow() must be atomic — race condition if two requests arrive simultaneously
```

### 2.2 Generational Garbage Collection

When the file reaches 25 lines, the GC cycle runs:

1. **Identify the 3 youngest non-immutable lines** (most recently added)
2. **Give them to an LLM** along with the task line and the question: *"Compress these three observations into one sentence that preserves what matters."*
3. **The LLM thinks for up to 20 turns** (plan steps, internal reasoning)
4. **A `/proc` script extracts exactly one line** from the last LLM message
5. The extracted line replaces the 3 source lines, tagged with its generation: `[G1]`, `[G2]`, etc.

**Generation tagging**: A line that has survived one GC cycle is tagged `[G1]`. If it survives the next compression as part of a merged line, the new line is tagged `[G2]`. Lines that reach `[G3]` or beyond represent **emergent consensus** — observations stable enough to survive repeated compression by different models.

**The first line is never touched by GC.** It is the only fixed point in the system.

### 2.3 The Agent Write Protocol

Each agent worker (one step in the swarm plan) does the following:

1. **Read** `memory.txt` in full (25 lines max — always fits in context)
2. **Think** for up to 20 plan steps (using 1bcoder's `/agent` loop or plan steps)
3. **Write** exactly one line to `memory.txt` via `/proc run swarm-write`

The `/proc` script:
- Reads the last LLM message
- Extracts the last sentence or the most specific factual claim
- Appends it to `memory.txt`
- Triggers GC if line count reaches 25

**An agent never reads its own output before writing.** It reads the collective memory, not its own history.

---

## 3. The Communicative Turn

This mechanism is philosophically distinct from a task queue or a shared log.

In Max Weber's sense, a **social action** is an action *oriented toward the other*. A muraха carrying food is not performing a social action — it is reacting to a chemical gradient. But when an agent writes a line knowing that *the next agent will read it and act differently because of it*, this is a communicative act in Weber's sense.

The key design constraint that forces this: **each agent sees the full memory but not its own reasoning history**. It cannot continue its own thread of thought. It can only contribute to the collective thread. This is stigmergy — indirect coordination through environment modification — but with language as the medium instead of pheromones.

The generational GC is the evolutionary pressure that makes the communication meaningful: lines that are vague, redundant, or wrong do not survive compression. Lines that are specific, non-obvious, and correct tend to survive because compressors (other models) preserve what they find useful.

**The file, over time, becomes a distillation of what the swarm collectively knows about the task.** Not what any one agent knows — what survives the compression of many different agents reading and summarizing.

---

## 4. Why Not Just Use the Anthill OKG?

The Anthill OKG is a graph database with structured triplets, schema validation, and role-specialized agents. It is the right architecture for a production system with infrastructure.

Swarm Memory GC is a text file with 25 lines. It runs on any machine that can run 1bcoder. The tradeoffs:

| | Anthill OKG | Swarm Memory GC |
|---|---|---|
| Infrastructure | Neo4j/FalkorDB, LangGraph | One text file |
| Coordination | Role-specialized agents (Notary, Librarian, Auditor) | GC compression loop |
| Memory capacity | Unbounded graph | 25 lines |
| Long-term memory | Graph grows indefinitely | Only what survives compression |
| Deployment | Multi-GPU rig | Any laptop |
| Failure mode | OKG inconsistency | Memory drift from task anchor |

Swarm Memory GC is not a replacement for Anthill — it is a **minimum viable version** of the same idea that can run with 1B models and no external infrastructure.

---

## 5. The Direction Problem and Embeddings

During the design of this concept, the question arose: does the swarm know which direction it is moving?

The initial intuition was to use **semantic embeddings** to measure "direction": embed the user request and the agent response, measure cosine distance, and use this as a discriminator signal. If distance is increasing, the swarm is drifting.

**This intuition is partially wrong.**

Measuring distance between user request and agent response does not capture correctness. The example:
- Request: "fix indentation on line 62" → Response: "Line 62:        a = b/c" → small distance (both about line 62)
- Request: "fix division by zero on line 62" → Response: "Line 62:        a = b/c" → also small distance

Same embedding distance, one correct and one wrong. The "direction" of the task is not in the user-agent pair — it is in the **code itself and its relationship to the task description**.

**The correct discriminator for code generation is structural, not semantic:**
- Does the response match the expected format (`LINE N: content` for `/fix`, SEARCH/REPLACE for `/patch`)?
- Does it reference real identifiers that exist in `map.txt`?
- Does it address the file/line mentioned in the request?

These are deterministic checks, not embedding similarity. The existing `grounding-check` proc in 1bcoder already implements the identifier check.

**Embeddings remain useful for one specific sub-problem**: detecting when the memory file has drifted from the task anchor. Embed `memory.txt` line 0 (task) and the current `memory.txt` content (concatenated). If semantic distance grows beyond a threshold over successive GC cycles, the memory has drifted — compression is generalizing away from the task. This is a valid use of semantic similarity as a **drift alarm**, not as a quality measure.

**Recommended embedding model if implemented**: `all-MiniLM-L6-v2` (22MB, CPU, handles mixed code+NL, ~200ms cold start per subprocess invocation). No Ollama dependency — loaded directly via `sentence_transformers` in the proc script. This works regardless of what backend the user has connected (LMStudio, LiteLLM, or Ollama).

---

## 6. The Conflict Mechanism

When two agents write contradictory lines, the log makes the conflict visible:

```
agent_3: allow() should raise RateLimitExceeded exception on deny
agent_6: allow() should return False on deny — exceptions are wrong for hot paths
```

Two lines, same function, opposing design decisions.

The GC compressor will see both and must produce one line. This forces resolution. The compressor (another LLM) will either:
- Pick one (implicit vote)
- Synthesize (`allow() returns bool; exception variant available as allow_or_raise()`)
- Defer (`allow() interface: decision pending — see conflict in GC log`)

The third option (explicit deferral) is interesting: the compressor can write a line that names the unresolved conflict, escalating it to the next GC cycle where a different compressor will see it. If a conflict survives 3 GC cycles without resolution, it represents a **genuine ambiguity** that requires human input — the coordinator's "ooOoO, нормально, а ну ідіть сюда обидва" moment.

---

## 7. Implementation Components

When this concept moves to implementation, it requires:

```
1. .1bcoder/swarm/memory.txt         — the shared memory file
2. .1bcoder/swarm/swarm.log          — full history (what each agent wrote before GC)
3. .1bcoder/proc/swarm-write.py      — proc: extract one line + append + trigger GC
4. .1bcoder/proc/swarm-gc.py         — proc: compress 3 lines → 1 via LLM
5. .1bcoder/proc/swarm-drift.py      — proc: semantic drift alarm (optional)
6. .1bcoder/plans/swarm-worker.txt   — plan: read memory → think N steps → swarm-write
7. .1bcoder/teams/swarm.yaml         — team: N workers with swarm-worker plan
8. System prompt for workers         — "you are one agent in a swarm; write for the next agent, not for yourself"
```

The GC proc calls the currently active LLM (via the same Ollama/LMStudio/LiteLLM backend already configured). No separate infrastructure needed.

---

## 8. Experimental Validation

### 8.1 Core Hypotheses

| ID | Hypothesis | Prediction |
|----|-----------|------------|
| SW1 | Generational stability | Lines tagged [G2]+ are more factually correct than [G0] lines (as judged by human review) |
| SW2 | Compression preserves intent | After 5 GC cycles, memory.txt line 0 distance to memory.txt content (embedding) remains below drift threshold |
| SW3 | Conflict resolution convergence | Contradictory line pairs are resolved within 3 GC cycles in ≥70% of cases |
| SW4 | Collective > individual | A swarm of 5×1B models writing to shared memory produces better code artifacts than a single 1B model working alone for the same total token budget |
| SW5 | Small model viability | qwen2.5-coder:1.5b can write useful non-trivial lines when it reads the full 25-line memory context |

### 8.2 Falsification Conditions

**SW1 falsified if**: Human reviewers find no quality difference between G0 and G2+ lines — compression is not improving signal, just shortening text.

**SW2 falsified if**: Memory drift alarm triggers on >50% of tasks — the GC process systematically drifts from the task. This would suggest the task anchor (line 0) is insufficient to maintain coherence across multiple compression cycles.

**SW3 falsified if**: Conflicts become the majority of memory content — the swarm produces more disagreements than insights. This would indicate the models are too divergent in their assumptions about the codebase to coordinate through text alone.

**SW4 falsified if**: Single model with the same token budget (but full context per turn) outperforms the swarm. This would mean the overhead of compression and coordination consumes the benefit of parallelism.

**SW5 falsified if**: 1.5B model consistently writes lines like "I think this is a good approach" or "The code looks correct" — generic text that provides no useful information to the next agent. This would confirm that 1B models cannot perform even minimal distillation tasks.

### 8.3 Minimal Experiment Design

**Experiment S1** — Single task, 5 agents, 3 GC cycles:

1. Set task: "add input validation to the create_user() function in users.py"
2. Run 5 agents sequentially, each reading and writing to memory.txt
3. Trigger GC manually when file reaches 10 lines (smaller scale for first experiment)
4. After 3 GC cycles, evaluate:
   - Does memory.txt contain actionable, specific observations?
   - Does a human developer find the memory useful as a starting context?
   - Would the task be easier with memory.txt than without?

This experiment requires no measurement infrastructure — just human judgment on the output.

**Experiment S2** — Conflict detection:

Insert two deliberately contradictory observations into memory.txt manually. Run GC. Observe whether the compressor resolves, defers, or ignores the conflict. Repeat with 3 different LLM sizes (1B, 7B, 14B) as compressor to measure resolution quality as a function of compressor scale.

---

## 9. Open Questions

1. **What is the optimal memory size?** 25 lines is a guess based on typical context window efficiency for 1B models. May need tuning.

2. **Should the GC compressor be the same model as the workers?** Using a larger model as compressor (7B) while workers are 1B may produce higher-quality consolidation — but breaks the "no infrastructure" promise.

3. **Is the task anchor sufficient?** One line describing the task may not be enough to prevent drift on complex tasks. An alternative: reserve lines 0–2 as immutable (task + scope + constraints).

4. **How does this interact with map.txt?** The most natural integration: workers read both `memory.txt` and relevant sections of `map.txt` before writing. Memory encodes the evolving understanding of the task; map encodes the stable structure of the codebase.

5. **What is the right write protocol?** Should workers write the most useful thing they know, or the most uncertain thing they need resolved? A system where workers write **questions** (what they don't know) rather than observations (what they do know) might converge faster on blockers.

---

## 10. Relationship to Existing Concepts

This concept is a **lightweight, file-based projection** of the Anthill OS ideas:

- **OKG** → `memory.txt` (external knowledge, but minimal and volatile)
- **Notary agent** → `swarm-write.py` proc (writes knowledge after each action)
- **Observer Loop** → GC cycle (compression is the update trigger)
- **Crooked Wall** → conflict lines surviving GC (adversarial pressure without explicit critic role)
- **Qualia Horizon** → unresolved conflicts after 3+ GC cycles (the things the swarm cannot figure out)

The generational GC idea has no direct analog in Anthill — it is the novel contribution of this concept. In Anthill, the OKG grows indefinitely and is managed by a Notary. Here, memory is bounded and self-organizing through compression pressure. This is a fundamentally different approach to the same problem: how does a system remember what matters?

---

*"The file, over 25 lines and many GC cycles, becomes what the swarm collectively knows. Not what any agent was told. What survived."*
