# Prompt and Context Evaluation Without an LLM
## Metrics for Measuring What You Ask, What You Got, and Where You Are

*Stanislav Zholobetsky, 2026*

---

## The Problem

When working with a small local LLM inside a bounded context window (4K–8K tokens), two evaluation questions arise simultaneously:

1. **Prompt quality**: given several candidate phrasings of the same question, which will produce the most useful reply?
2. **Context quality**: given the full conversation so far, how well does it cover the original task? What fraction of the task has been "addressed" and how much signal is left?

These are related but distinct. A great prompt in a context that has already drifted far from the task is still wasted. A mediocre prompt in a well-focused context may still work.

This document describes both dimensions using metrics that can be computed from text alone — no LLM call required.

---

## Part I: Prompt Quality Metrics

### The Core Question

Given three candidate prompts P₁, P₂, P₃ — and their corresponding LLM replies R₁, R₂, R₃ — which (P, R) pair was most valuable?

Because we are comparing against actual replies, this is a *response evaluation* problem dressed as a prompt evaluation problem. The prompt is the instrument; the reply is the output. We score the output.

### Metric 1: Term Novelty

**What it measures:** How much new vocabulary does the reply introduce relative to the existing context?

```
novel_terms(R, C) = {w ∈ words(R) : w ∉ words(C), len(w) > 3}
novelty(R, C) = |novel_terms| / |unique_words(R)|
```

Where C is the full conversation context before this turn.

High novelty means the reply added new knowledge. Low novelty means the model repeated what was already known. Best used after filtering stop words.

**Caution:** very high novelty on a technical question can also mean the model hallucinated or went off-topic. Combine with grounding.

### Metric 2: Task Grounding

**What it measures:** What fraction of words in the reply overlap with words in the original task description T?

```
grounding(R, T) = |words(R) ∩ words(T)| / |unique_words(R)|
```

High grounding = the reply stayed on-topic relative to the original task.
Low grounding = the reply may have answered the question correctly but drifted from the problem domain.

### Metric 3: Information Density

**What it measures:** How much unique content per token?

```
density(R) = |unique_words(R)| / |total_words(R)|
```

A reply that repeats phrases has low density. A reply with specific technical terms has high density. Combine with novelty: a reply can be dense but not novel (dense repetition of known facts).

### Metric 4: Token Efficiency

**What it measures:** How many new concepts were introduced per token consumed?

```
efficiency(R, C) = |novel_terms(R, C)| / token_count(R)
```

This is the key metric when operating near a context limit. You want maximum new knowledge per token spent.

### Metric 5: Actionability Signals

**What it measures:** Does the reply contain actionable content — code, steps, formulas?

```python
code_blocks   = count("```" in R)
numbered_steps = count(re.match(r'^\d+\.', line) for line in R.splitlines())
math_blocks   = count("$$" in R or "$" in R)
actionability = (code_blocks + numbered_steps + math_blocks) > 0
```

Boolean or count. For a task that requires implementation, an explanatory reply with no code and no steps scores zero here regardless of other metrics.

### Composite Prompt Score

```
score(R, C, T) = (
    0.35 * novelty(R, C) +
    0.30 * grounding(R, T) +
    0.20 * density(R) +
    0.15 * efficiency(R, C)
)
```

Weights are heuristic. The right balance depends on the task type:
- **Exploration phase** (what is X?): prioritize novelty
- **Grounding phase** (how does X apply here?): prioritize grounding
- **Implementation phase** (write the code): prioritize actionability

---

## Part II: Context Quality Metrics

### The Core Question

Given a growing conversation context C = [m₁, m₂, ..., mₙ] and an original task T, how well does C cover T? What is still uncovered? Is the context drifting?

These metrics treat the context as a kind of *dynamic summary* that should progressively cover the task.

### Metric 6: ROUGE-Recall (Coverage)

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) was designed for summarization evaluation. In our case: we treat C as the "generated summary" and T as the "reference". ROUGE-Recall measures what fraction of the reference is covered.

```
ROUGE-1-Recall = |unigrams(T) ∩ unigrams(C)| / |unigrams(T)|
ROUGE-2-Recall = |bigrams(T) ∩ bigrams(C)|   / |bigrams(T)|
```

**ROUGE-1-Recall** answers: what fraction of task words have appeared at least once in the conversation?
**ROUGE-2-Recall** answers: what fraction of task two-word phrases have appeared? This is stricter and catches whether concepts were discussed together or just mentioned in isolation.

Implementation: `pip install rouge-score` (pure Python, no LLM).

### Metric 7: Keyword Coverage

A simpler version of ROUGE that focuses only on high-value terms — nouns, technical terms, proper nouns — filtered by POS tag or by IDF weight from a domain corpus.

```
key_terms(T) = {w ∈ T : idf(w) > threshold, pos(w) in {NOUN, PROPN}}
covered(T, C) = {w ∈ key_terms(T) : w ∈ C}
coverage = |covered| / |key_terms(T)|
```

A coverage of 1.0 means every important term from the task has been mentioned. Coverage of 0.4 means the conversation has only touched on 40% of the task vocabulary.

Can be implemented with pure string matching if IDF is precomputed. The 1bcoder `/map keyword` infrastructure already provides a keyword vocabulary per project — this can be reused.

### Metric 8: Context Drift (Embedding-Based)

**What it measures:** How far has the conversation drifted from the original task over time?

For each turn k:
```
drift(k) = 1 - cosine(embed(mₖ), embed(T))
```

Plot drift over turns. A good conversation should show drift oscillating near zero. A drifting conversation shows drift monotonically increasing as the model starts discussing general concepts instead of the specific task.

This requires embeddings, but a small model (MiniLM, ~22MB) is sufficient. The metric is cheap once the model is loaded.

**Variant:** use the centroid of all messages so far instead of a single turn:
```
context_centroid(k) = mean(embed(m₁), ..., embed(mₖ))
alignment(k) = cosine(context_centroid(k), embed(T))
```

This should monotonically increase in a well-managed conversation: each turn brings the context closer to the task.

### Metric 9: Contextual Entropy (Focus)

**What it measures:** How focused or scattered is the context vocabulary?

```
p(w) = count(w in C) / total_words(C)
H(C) = -Σ p(w) * log₂(p(w))
```

Low entropy = focused context (few terms, repeated frequently — typical of a deep technical discussion).
High entropy = scattered context (many different topics, each touched briefly).

For a coding task, you want entropy to decrease as the conversation progresses: early turns are exploratory (high entropy), later turns should be implementation-focused (low entropy, concentrated on specific symbols, functions, module names).

### Metric 10: Turn-by-Turn Information Gain

**What it measures:** How much new knowledge did each turn add?

```
novel(k) = |words(mₖ) - words(m₁ ∪ ... ∪ mₖ₋₁)|
gain(k)  = novel(k) / token_count(mₖ)
```

Plot this over turns. A healthy conversation has high gain early and converges as the task is progressively covered. If gain is still high late in the conversation, the context may be drifting (new topics appearing). If gain drops to zero, the conversation is running in circles.

---

## Part III: Remaining Context Budget Analysis

### The Token Budget Problem

At any point in a session with a context limit L (e.g. 8192 tokens):

```
used(k)      = Σ token_count(mᵢ) for i = 1..k
remaining(k) = L - used(k)
coverage(k)  = ROUGE-1-Recall at turn k
```

The key question: **how much coverage can we still gain per remaining token?**

```
coverage_rate = Δcoverage / Δtokens_used  (per last N turns)
projected_coverage = coverage(k) + coverage_rate * remaining(k)
```

If projected_coverage < 1.0 at the current rate, and remaining tokens are near exhaustion, then:
- The next prompt should prioritize breadth over depth (cover uncovered task terms)
- Or the context should be compacted (1bcoder `/ctx compact`) before it gets worse

### The "What Is Still Uncovered" View

Given coverage metrics, the uncovered terms can be listed explicitly:

```python
uncovered = key_terms(T) - {w for w in words(C)}
```

The next prompt should address terms from `uncovered`. This is a direct answer to "what should I ask next?" — not from an LLM, but from a set difference.

---

## Part IV: Implementation in 1bcoder

### Minimal Implementation (Pure Python Proc)

A proc `context-eval.py` that fires after every reply:

```python
# ~/.1bcoder/proc/context-eval.py
import sys, os, re, math

STOP = {'the','a','is','of','to','and','in','it','for','that','this','with',
        'be','are','was','were','will','have','has','do','does','but','not',
        'what','how','can','use','which','their','from','an','at','by','or'}

def tokenize(text):
    return {w.lower() for w in re.findall(r'\b\w{4,}\b', text)} - STOP

task_file = os.path.join(os.environ.get("BCODER_WORKDIR", "."), ".1bcoder", "task.txt")
task_words = tokenize(open(task_file).read()) if os.path.isfile(task_file) else set()

response    = sys.stdin.read()
ctx_text    = os.environ.get("BCODER_CTX_TEXT", "")  # injected if we add this env var

r_words  = tokenize(response)
c_words  = tokenize(ctx_text)

novelty   = len(r_words - c_words) / max(len(r_words), 1)
grounding = len(r_words & task_words) / max(len(r_words), 1)
density   = len(r_words) / max(len(response.split()), 1)
coverage  = len(tokenize(ctx_text) & task_words) / max(len(task_words), 1) if task_words else 0

tokens_used      = int(os.environ.get("BCODER_CTX_USED", "0"))
tokens_max       = int(os.environ.get("BCODER_CTX_MAX", "8192"))
tokens_remaining = tokens_max - tokens_used

score = round(novelty * 0.35 + grounding * 0.30 + density * 0.20, 3)

print(f"[eval] reply score={score}  novelty={round(novelty,2)}  grounding={round(grounding,2)}")
print(f"[eval] task coverage={round(coverage*100)}%  ctx remaining≈{tokens_remaining} tok")
if task_words:
    uncovered = task_words - tokenize(ctx_text)
    if uncovered:
        sample = sorted(uncovered)[:5]
        print(f"[eval] uncovered task terms: {', '.join(sample)}")
```

Activate with:
```
/proc on context-eval
```

### Parallel Prompt Comparison

```
/parallel list: Що таке сплайн?, Куди рахують сплайн?, Які методи LP для матриці обмежень? ctx: full
/proc run context-eval     ← applied to each reply
```

With a `collect: score` mode (future extension), comparison could be automated.

### Storing the Task

The system needs to know T (the original task). Simplest approach:

```
/var set task = оптимальне рішення матриці обмежень через сплайн-метод
/var save project.var
```

Or save the full task text to `.1bcoder/task.txt` manually. The proc reads it from there.

---

## Part V: Task-Specific Decomposition and Stage Signal Words

### The False Positive Problem with Generic Plans

A generic lifecycle (formalization → classification → method analysis → implementation → validation) applies to every task — but at different depths. A simple script and a research system both have a "formalization" stage; the difference is whether it lasted one sentence or ten turns.

The problem with keyword coverage against a generic plan: the metric cannot distinguish "this stage was addressed superficially" from "this stage is complete and closed." The word "метод" appearing once in the context does not mean method analysis is done.

### Two Classes of Stage Keywords

Every stage in the task-specific decomposition should carry two keyword sets:

**Entry keywords** — signal that work on this stage has *begun*. Appear early in the stage, may appear in other stages too. Their presence means "we are here."

**Terminal signals** — signal that the stage is *closed*. Unambiguous, specific, unlikely to appear before the stage is finished. Their presence means "we passed through here."

The distinction is the **varenyky principle**: if you are making dumplings, the words *картопля, вода, почистити* mean you are in the filling stage. But *сметана* means the dish is served — you do not mention smetana while peeling potatoes. Terminal signals are like smetana: they only appear when a stage is genuinely complete.

### Structure of a Task-Specific Decomposition

Generated once by the LLM when `/target` is set with `--decompose`. Stored in `.1bcoder/target.yaml`:

```yaml
task: "знайти оптимальну кількість автомобілів автопарку для обслуговування населення"

stages:
  formalization:
    description: Define the objective function and constraint system
    entry_keywords:    [задача, умова, обмеження, функція, змінна, визначити]
    terminal_signals:  [цільова функція, система нерівностей, записати як, f(x) =, мінімізувати]

  classification:
    description: Identify problem class and applicable mathematical domain
    entry_keywords:    [клас, тип задачі, відноситься, схожа, аналог]
    terminal_signals:  [задача лінійного програмування, задача оптимізації, відноситься до класу]

  domain_and_types:
    description: Define variable domains, value ranges, data types
    entry_keywords:    [область, діапазон, тип, ціле, дійсне, від, до]
    terminal_signals:  [x ∈, цілочисельна, обмежена знизу, невід'ємна, допустима область]

  method_analysis:
    description: Survey applicable methods, justify selection
    entry_keywords:    [метод, підхід, алгоритм, симплекс, LP, порівняти]
    terminal_signals:  [обираємо метод, перевага X над Y, оптимальний вибір, використаємо]

  planning:
    description: Define implementation steps and tooling
    entry_keywords:    [план, кроки, реалізувати, бібліотека, scipy, pulp, ortools]
    terminal_signals:  [план реалізації, крок 1, структура коду, модуль]

  implementation:
    description: Write and verify the solution code
    entry_keywords:    [def, import, solve, constraint, return, функція]
    terminal_signals:  [код реалізовано, функція повертає, тест пройшов, результат]

  validation:
    description: Test against known cases, analyze sensitivity
    entry_keywords:    [перевірити, тест, крайовий, синтетичний, sensitivity]
    terminal_signals:  [тест пройшов, результат збігається, похибка, висновок валідації]

  conclusions:
    description: Aggregate findings, formulate recommendation
    entry_keywords:    [висновок, підсумок, результат, рекомендація, отже]
    terminal_signals:  [оптимальна кількість становить, рекомендується, отже відповідь]
```

### Coverage Evaluation with Two Signal Classes

Given a stage S with entry keywords E and terminal signals T, the conversation context C gives three states:

```
UNTOUCHED  — neither E nor T appear in C
IN_PROGRESS — some E appear, but no T
COMPLETED  — at least one T appears
```

Coverage score uses only COMPLETED stages:

```
stage_coverage = |{S : completed(S, C)}| / |all_stages|
```

In-progress stages contribute fractionally if needed:

```
weighted_coverage = (|completed| + 0.5 * |in_progress|) / |all_stages|
```

### Why Terminal Signals Are Harder to Fake

Entry keywords are broad and common. The word "метод" appears in natural language constantly. Terminal signals are structurally different:

- They are **specific phrases**, not single words: "цільова функція має вигляд", "відноситься до класу", "тест пройшов"
- They are **consequential**: they describe a result, not an activity
- They are **rarely used until the stage closes**: you do not write "тест пройшов" while still analyzing methods

This asymmetry makes terminal signals reliable detectors of genuine stage completion without an LLM. A ROUGE-1 metric on entry keywords gives noisy coverage; terminal signal detection gives a finite-state machine of actual progress.

### Stage Progress as a Finite State Machine

```
UNTOUCHED → IN_PROGRESS → COMPLETED
              ↑____________|
              (backtracking allowed)
```

Backtracking is valid: a completed formalization may be revisited if new constraints appear. The FSM tracks which stages have ever reached COMPLETED, not whether they are currently active.

The current FSM state of the session is readable from the context in O(n) time with simple string matching — no LLM required after the initial decomposition.

### Generating the Decomposition: The One LLM Call

The `/target "..." --decompose` command sends a single structured prompt:

```
Task: {task_text}

You are generating a coverage map for evaluating a research conversation.
For each applicable stage from the list below, output:
- entry_keywords: 4-6 words/phrases that indicate work on this stage has begun
- terminal_signals: 2-4 specific phrases that only appear when the stage is genuinely complete

Stages: formalization, classification, domain_and_types, method_analysis,
        planning, implementation, validation, conclusions

Output YAML. Omit stages that do not apply to this task.
```

Result saved to `.1bcoder/target.yaml`. All subsequent evaluation is pure string matching against this file — no further LLM calls.

### The Smetana Principle

The most important design rule for terminal signals:

> A terminal signal must not appear naturally in any earlier stage.

*Сметана* does not appear while peeling potatoes or making dough. "Тест пройшов" does not appear during method analysis. "Оптимальна кількість становить N" does not appear during formalization.

When selecting terminal signals during decomposition, prefer:
- **Result statements** over activity descriptions ("функція повертає X" vs "пишемо функцію")
- **Past tense or completion forms** ("реалізовано", "пройшов", "обрано")
- **Quantified conclusions** ("оптимальна кількість = 12", "похибка < 0.01")
- **Transitional declarations** ("переходимо до реалізації", "задачу формалізовано")

These are the smetana words of software research: they only exist at the table, never in the kitchen.

---

## Part VI: The `/target` Command — Task Anchor

### Why a Separate Command

All evaluation metrics described in Parts I–V require a reference point T — the original task. Without T, novelty and grounding have no anchor, keyword coverage has nothing to cover, and the FSM has no stages to track.

The `/target` command establishes T once per session and persists it to `.1bcoder/target.txt` and `.1bcoder/target.yaml`. All procs and eval metrics read from these files automatically.

```
/target "знайти оптимальну кількість автомобілів для обслуговування населення"
/target show          — display current target and coverage status
/target clear         — remove target
/target --decompose   — one LLM call: generate task-specific stage map with signal words
```

### Separation of Concerns

`/target` separates two things that are often conflated:

- **What we want to achieve** — the task description, stable throughout the session
- **How we explore it** — the conversation, grows and changes with each turn

The target is the fixed star. The context is the ship navigating toward it. The evaluation metrics measure the distance between ship and star.

### `/target --decompose`

Calls the LLM once to generate the stage decomposition (Part V) with entry keywords and terminal signals. The output is `.1bcoder/target.yaml`. After this single call, all subsequent coverage evaluation is pure string matching — no more LLM calls needed for monitoring.

---

## Part VII: `/deepagent` — Recursive Tree Decomposition

### A Different Kind of Agent

The standard `/agent` in 1bcoder is a **linear chain**: each turn reads the accumulated context and decides the next tool call. State lives in `self.messages`. Output is a conclusion.

`/deepagent` is a **tree builder**: each iteration takes the current tree and expands every leaf node into child nodes. State lives in a file. Output is a hierarchical plan, article, or program skeleton.

```
Standard agent:  [call] → [call] → [call] → conclusion
DeepAgent:       [tree depth 0] → [tree depth 1] → [tree depth 2] → [tree depth N]
```

The tree IS the output — not a side effect, not a summary. The agent's job is complete when the tree reaches the desired depth and all leaves are concrete.

### The Fractal Nature

A plan grown by iterative self-similar decomposition has the structural property of a fractal: the same rule applied recursively at every scale produces a complex, coherent whole. The overall shape of depth-3 plan mirrors the shape of depth-1 — more detailed, but recognizably self-similar.

This is not a metaphor. The expansion prompt is literally the same function `f(node, level_label)` applied at every level. Fixed-branching constraints break this property by imposing an artificial grid on what should be an organic structure. Giving the LLM freedom to decide how many children a node needs preserves the fractal property: some nodes are atomic at depth 2, others need 5 children at depth 4.

### Command Grammar

```
/deepagent <task_inline_or_omitted>
  target:  "<task description>"       — what we are building (or use /target set earlier)
  prompt:  <saved_name or inline>     — the self-similar expansion template
  plan:    l1, l2, l3, l4             — semantic labels for each depth level
  list:    a1, a2, a3                 — analytical lenses applied to every node
  profile: <name or host|model>       — which models expand which nodes
  file:    output.md                  — where the tree is written
```

**Full example:**
```
/deepagent
  target:  "triangulation of a surface"
  prompt:  decompose_mathematical
  plan:    abstraction, algorithm design, implementation, optimization
  list:    euclidean space, lebesgue space, hilbert space
  profile: three_gpu_rag
  file:    triangulation_plan.md
```

### Parameter Semantics

**`target:`** — the task anchor. Stable throughout all iterations. Every expansion call receives it as context so no node loses track of the overall goal.

**`prompt:`** — the self-similar function. Applied identically at every node and every depth. May reference a saved prompt from `/prompt save` (just the name) or be written inline. The `{level}` placeholder is replaced by the current `plan:` label; `{aspects}` by the `list:` items.

**`plan:` — vertical axis (depth semantics).** Each comma-separated label describes what KIND of decomposition should happen at that depth level. The same node is described differently depending on which level it lives at:

```
plan: abstraction, principles, implementation, code examples

depth 1 ("abstraction"):   "Define metric space"
depth 2 ("principles"):    "Euclidean distance properties in ℝⁿ"
depth 3 ("implementation"): "scipy.spatial.distance.euclidean"
depth 4 ("code examples"): "np.sqrt(np.sum((a-b)**2))"
```

**`list:` — horizontal axis (analytical lenses).** Each item is a perspective applied to EVERY node at EVERY depth. Not separate trees — the same tree analyzed through multiple lenses simultaneously. The LLM decides whether a lens produces a child node or an inline annotation depending on its significance for that particular node.

```
list: euclidean space, lebesgue space, hilbert space

Node "1.2 Define distance metric":
  1.2.A [Euclidean]  L2 norm, standard Delaunay
  1.2.B [Lebesgue]   Lᵖ integral metric, weighted triangulation
  1.2.C [Hilbert]    orthogonal projection, wavelet approximation
```

**`profile:`** — which models do the expansion. Standard 1bcoder profile syntax (`host|model`). Multiple workers expand different leaves in parallel. The orchestrator assembles results.

**`file:`** — output path. The tree is written here after each level completes. Human-readable numbered markdown outline. Can be loaded by `/agent file: output.md` for execution after planning.

### Stopping Criterion: Concrete Leaf Detection

A leaf is not expanded if it is already concrete. Concreteness is detected without an LLM using the Smetana Principle (Part V): a node is concrete when its text contains specific, result-oriented language — code fragments, library names, numbers, quantified statements, completion forms.

```python
CONCRETE_SIGNALS = [
    r'\b(import|def|class|return|assert)\b',   # code keywords
    r'\b\w+\.\w+\(',                            # function calls
    r'\d+\.\d+',                               # numbers
    r'O\([^)]+\)',                             # complexity notation
    r'\b(реалізовано|обрано|пройшов|готово)\b' # completion forms
]

def is_concrete(node_text: str) -> bool:
    return any(re.search(p, node_text) for p in CONCRETE_SIGNALS)
```

---

## Part VIII: Distributed Tree Assembly — Orchestration

### The Central Principle

**LLMs do not write to the file. LLMs do not number nodes. LLMs do not indent.**

Each LLM receives one node and returns a flat list of lines — nothing more. All structure (numbering, indentation, tree insertion) is assigned by the orchestrator after collecting all responses for the current level. One atomic write per level.

This eliminates race conditions entirely: parallel workers return to the orchestrator, which assembles the full level and writes once.

### Expansion Loop

```python
def run_deepagent(target, prompt_template, plan_labels, list_items, workers, output_file):
    tree = parse_outline(output_file) if exists(output_file) else init_root(target)

    for level_label in plan_labels:
        leaves = [node for node in tree.all_nodes() if node.is_leaf()
                  and not is_concrete(node.text)]

        if not leaves:
            print(f"[deepagent] all leaves concrete — stopping at '{level_label}'")
            break

        # build prompts
        jobs = []
        for leaf in leaves:
            prompt = format_prompt(prompt_template, leaf, level_label, list_items, target)
            worker = assign_worker(leaf, workers)   # round-robin
            jobs.append((worker, prompt, leaf))

        # parallel execution — wait for ALL before writing
        results = parallel_call(jobs)

        # assemble — still not writing
        for leaf, raw in zip(leaves, results):
            items = parse_llm_items(raw)
            children = [Node(id=f"{leaf.id}.{i+1}", text=item)
                        for i, item in enumerate(items)]
            tree.attach(leaf, children)

        # single atomic write
        write_outline(tree, output_file)
        print(f"[deepagent] '{level_label}': {len(leaves)} expanded → "
              f"{sum(len(parse_llm_items(r)) for r in results)} new nodes")
```

### The Flat-Output Prompt

The most important prompt engineering constraint: force the LLM to output a flat unnumbered list with no explanations.

```
You are expanding one node of a hierarchical plan.

Root task: "{target}"
Path to current node: {node_path}
Current node: "{node_text}"
Depth focus: "{level_label}"
Analytical lenses: {list_items}

OUTPUT RULES — follow strictly:
- Output ONLY the direct children of this node
- One item per line, no blank lines between items
- No numbering, no bullets, no dashes, no markdown
- No explanations, no "because", no "in order to"
- No preamble ("Here are...", "The following...")
- No postamble ("These steps ensure...")
- 2–5 items maximum
- Each item: noun phrase or verb phrase, 3–8 words

WRONG:
1. **Define coordinate system** — this ensures consistency across the dataset.
Here are the sub-steps:
- Validate input

CORRECT:
Define coordinate system
Validate boundary conditions
Select triangulation algorithm
Choose numerical precision
```

The WRONG/CORRECT pair is critical — it works better than abstract rules alone, even for small models.

### Response Parser

```python
def parse_llm_items(raw: str) -> list[str]:
    lines = raw.strip().splitlines()
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # strip list markers
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-•*#]+\s*', '', line)
        # strip markdown
        line = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', line)
        line = re.sub(r'`(.+?)`', r'\1', line)
        # cut explanations after common separators
        for sep in [' — ', ' - ', ' : ', ' (', ' because', ' in order', ' so that']:
            if sep in line:
                line = line[:line.index(sep)]
        line = line.strip().rstrip('.,;:')
        # filter garbage lines
        if len(line) < 3:
            continue
        if any(line.lower().startswith(p) for p in [
            'here are', 'the following', 'these are', 'below are',
            'note:', 'example:', 'output:', 'step', 'sub-step'
        ]):
            continue
        items.append(line)
    return items
```

### Failure Modes and Responses

| Failure | Detection | Response |
|---|---|---|
| LLM adds explanations | `len(items) < expected and len(raw) > 300` | parser strips; if < 2 items: retry with stricter prompt |
| LLM skips a level (outputs grandchildren) | items contain concrete signals at wrong depth | treat as concrete, skip expansion |
| LLM repeats parent node | `item.lower() == node.text.lower()` | filter exact match |
| LLM returns 0 items | `len(items) == 0` | retry once, then mark node as concrete |
| Different terminology across workers | — | do nothing — this is valuable information |
| LLM answers instead of listing | long raw, 0–1 parsed items | retry with `"List only. No explanation."` prepended |

The last row — "different terminology across workers" — is explicitly not a failure. If three models describe the same node differently through three analytical lenses, the variation is signal, not noise.

### Output Format

Numbered markdown outline stored in `file:`. Parseable by regex, human-readable, executable by `/agent file:`:

```markdown
# Task: triangulation of a surface

1 Define the problem space
  1.1 Identify input data format
    1.1.1 [Euclidean] point cloud in ℝ³
    1.1.2 [Lebesgue]  measurable surface function
    1.1.3 [Hilbert]   element of L²(Ω)
  1.2 Define metric
    1.2.1 [Euclidean] L2 distance
    1.2.2 [Lebesgue]  integral metric
    1.2.3 [Hilbert]   inner product ⟨f,g⟩
2 Select algorithm
  ...
```

The `[Lens]` prefix on nodes generated from `list:` items makes the analytical dimension explicit and filterable — `grep "[Euclidean]" plan.md` extracts the Euclidean subtree.

---

## Part IX: DeepAgent vs Existing Agents — Architectural Comparison

### Three Agent Types in 1bcoder

| Property | `/plan` | `/agent` | `/deepagent` |
|---|---|---|---|
| Output | flat list in `plan.txt` | tool call chain → conclusion | tree in `file:` |
| State lives in | `plan.txt` (one pass) | `self.messages` (context) | `file:` (persistent) |
| Turns structure | one shot | linear chain | iterative level expansion |
| LLM calls | 1 | N sequential | N parallel per level × M levels |
| Human in loop | after (reviews plan) | per action (Y/n) | after each level |
| Parallelism | none | none | full (all leaves at one level) |
| Scales with | task complexity | context window | tree width × depth |
| Best for | quick step list | tool-driven tasks | deep research, writing, architecture |

### Why Not Just Use `/plan` Deeper

The existing `/plan` agent (from `agents/planning.txt`) produces a flat numbered list in one LLM call. For simple tasks this is enough. It fails at depth because:

1. A single LLM call with a 7-level decomposition request produces inconsistent depth — some branches reach level 4, others stop at 2
2. The model loses coherence as the list grows: late items drift from the root task
3. No mechanism for parallel expansion — one model serializes everything

`/deepagent` solves all three: each level is a separate set of calls, depth is controlled by the `plan:` label count, and parallel workers expand independently.

### The Free Branching Principle

Fixed branching factor (`"expand each node into exactly 3 sub-items"`) produces a uniform grid, not a natural plan. Real decomposition is irregular: some nodes are concrete at depth 2, others need 5 children at depth 4.

Giving the LLM freedom to decide branching width preserves the fractal property — the same rule applied at every scale produces a structure that mirrors nature rather than a spreadsheet. The only constraint: a minimum of 2 items (to prevent degenerate single-child chains) and a soft maximum of 5 (to prevent runaway expansion).

The stopping criterion (Smetana Principle / `is_concrete`) provides the natural termination instead of a fixed depth limit.

---

## Part X: Reusable Prompt Library

### The `prompt:` Parameter as a Reusable Asset

The expansion prompt is task-independent when written at the right level of abstraction. A prompt designed for technical planning works for any technical planning task; one designed for narrative structure works for any story. These prompts are worth saving and naming.

Save with 1bcoder's existing `/prompt save`:
```
/prompt save decompose_technical
/prompt save decompose_research
/prompt save decompose_narrative
```

Then reference by name in `/deepagent`:
```
/deepagent  target: "..."  prompt: decompose_technical  plan: ...
```

### Suggested Starter Library

**`decompose_technical`** — for software architecture, system design, algorithm planning:
```
You are expanding one node of a technical plan.
Root task: {target}. Current node: "{node_text}". Focus: {level}.
{aspects}
Output only direct child items. Noun or verb phrases. No explanation. No numbering.
Prefer specific technical terms over generic ones.
If the node already describes a concrete implementation detail, output it unchanged on one line.
```

**`decompose_research`** — for scientific articles, literature reviews, academic analysis:
```
You are expanding one section of a research outline.
Topic: {target}. Current section: "{node_text}". Granularity level: {level}.
{aspects}
Output only direct sub-sections. Each as a short heading (3–7 words).
No explanation. Maintain academic register. No numbering.
```

**`decompose_narrative`** — for story structure, screenplay, historical narrative:
```
You are expanding one narrative beat.
Story: {target}. Current beat: "{node_text}". Detail level: {level}.
{aspects}
Output only the immediate sub-beats or scenes. Concrete, sensory, specific.
No meta-commentary. No explanation. No numbering.
```

**`decompose_legal`** — for requirements analysis, specification decomposition:
```
You are expanding one requirement clause.
System: {target}. Current clause: "{node_text}". Abstraction level: {level}.
{aspects}
Output only direct sub-requirements. Use shall/must/should language.
No explanation. No numbering. Each item: one clear obligation.
```

### The `{aspects}` Injection

When `list:` items are present, `{aspects}` expands to:

```
Consider the following analytical lenses for each sub-item:
- euclidean space
- lebesgue space  
- hilbert space
For each lens that produces a meaningful distinction, prefix the item with [LensName].
If a lens does not meaningfully apply to a particular item, omit it.
```

When `list:` is empty, `{aspects}` expands to an empty string — the prompt works without it.

---

## Part XI: Complete Workflow — From Task to Execution

### Phase 1: Anchor

```
/target "знайти оптимальну кількість автомобілів для обслуговування населення"
```

Saves to `.1bcoder/target.txt`. Now all evaluation metrics have a reference point.

```
/target --decompose
```

One LLM call. Generates `.1bcoder/target.yaml` with stage keywords and terminal signals. After this, `/proc on context-eval` can track FSM progress automatically.

### Phase 2: Deep Planning

```
/deepagent
  target:  "знайти оптимальну кількість автомобілів..."
  prompt:  decompose_technical
  plan:    abstraction, method selection, algorithm design, implementation, validation
  list:    LP relaxation, integer programming, simulation approach
  profile: three_gpu_rag
  file:    fleet_plan.md
```

Produces `fleet_plan.md` — a numbered markdown tree, 4–5 levels deep, each node analyzed through three mathematical lenses. Human reviews and edits the file.

### Phase 3: Evaluation During Exploration

While using the plan as a guide for a manual LLM conversation:

```
/proc on context-eval
```

After every reply, the proc prints:
```
[eval] reply score=0.71  novelty=0.68  grounding=0.74
[eval] task coverage=43%  ctx remaining≈3200 tok
[eval] stage: method_selection=COMPLETED  implementation=IN_PROGRESS
[eval] uncovered task terms: обмеження, крайові, обслуговування
```

This tells you:
- Whether the current reply was useful (score)
- How much of the task has been addressed (coverage %)
- Which stages are done vs in progress (FSM)
- What to ask next (uncovered terms)

### Phase 4: Execution

When the plan is ready, hand it to the standard `/agent` for execution:

```
/agent implement the solution  file: fleet_plan.md
```

The `/agent` reads `fleet_plan.md` as a step-by-step plan and executes each node with tool calls (`/fim`, `/run`, `/save`). The tree structure from deepagent becomes the agent's instruction list.

### Phase 5: Prompt Comparison (Optional)

When unsure which phrasing to use for the next question:

```
/parallel
  list: Що таке сплайн?, Куди рахують сплайн?, Які методи LP для матриці обмежень?
  ctx:  full
  profile: three_gpu_rag
```

Then score each reply:
```
/proc run context-eval     ← on each of the three replies
```

Choose the reply with highest score (or highest coverage gain for remaining uncovered terms).

### The Full Picture

```
/target      — anchor: what are we solving
/target --decompose  — stage map: how to know when we're done
/deepagent   — plan tree: structured path to the solution
/proc on context-eval  — live monitoring: are we on track
/parallel + /proc run  — prompt comparison: which question next
/agent file: — execution: act on the plan
```

Each command addresses a different failure mode of unstructured LLM conversation:
- Forgetting the goal → `/target`
- Not knowing what's left → `context-eval` coverage
- Shallow exploration → `/deepagent` forced depth
- Wasted tokens on bad prompts → `/parallel` comparison
- Losing plan coherence → `file:` tree as ground truth

---

## Part XII: `/deepcheck` — Latent Dependency Detection

### The Problem: Trees Hide Lateral Connections

A tree captures one relationship type: **IS-A-PART-OF**. A node is a child of its parent, a sibling of its peers. But real systems have cross-cutting dependencies that trees cannot represent: two nodes in different branches that configure, constrain, produce for, or mirror each other.

The `/deepagent` builds the tree depth-first — it thinks *down*, not *across*. When reaching node `3.1 Commits vectorizer`, the model is no longer thinking about `1.1 Create default config`. The connection — that the config should define the vectorizer model name, aggregation strategy, and vector dimensions — is lost.

This is the classical **cross-cutting concern** problem in software architecture. AOP (Aspect-Oriented Programming) was invented precisely to handle concerns that cut across the primary decomposition hierarchy. The `/deepcheck` command is the planning equivalent: a post-hoc pass that finds what the tree missed.

### False Positive Rate Analysis

The false positive rate depends critically on how the question is framed.

**Generic framing** ("how are these two nodes connected?"):
The LLM finds a connection *always* — "both use strings", "both are part of the same system", "both need error handling". False positive rate: 60–80% even for 7B models.

**Typed framing** ("does A define a parameter that B requires as input?"):
The model must be concrete and specific. A spurious connection cannot survive a typed question because there is no specific parameter to name. False positive rate: 20–35% for 1B models, 10–20% for 7B+.

The solution is not a better model — it is a better question structure.

### Six Dependency Types

Each pair of cross-branch nodes is checked against six typed relationships:

| Type | Question | Example |
|---|---|---|
| **configures** | Does A define a parameter that B needs? | `default config` → vectorizer model name for `commits vectorizer` |
| **produces** | Does A create a data structure that B consumes? | `GIT integration` → commit format consumed by `commits vectorizer` |
| **constrains** | Does A define rules or schema that B must satisfy? | `extract rules` → field weighting schema for `task vectorizer` |
| **mirrors** | Do A and B model the same real-world entity differently? | `commits vectorizer` and `task vectorizer` — both vectorize natural language text |
| **orders** | Must A complete before B can start? | `user interview` → before `GIT integration` (must know repo URL first) |
| **shares** | Do A and B access the same external resource? | both vectorizers → same vector store instance |

Each question is sent as a separate LLM call with a typed template. The model answers yes/no with a concrete explanation. No explanation = LOW confidence = discard.

### The Hidden Need

The user's requirement is not always visible before comparison. Nobody knew the config should include the vectorizer model name — until the comparison revealed that `3.1 Commits vectorizer` needs a model name and `1.1 Create default config` is exactly where it would live.

This is **latent dependency**: a connection that exists in the real system but is invisible from the decomposition structure. The LLM acts as a domain knowledge proxy — it knows that vectorization systems need configuration, that config systems expose model selection, that both facts together imply a connection. This general knowledge surfaces latent structure that structural decomposition misses.

The criterion for a real connection is not "is there a need right now?" but **"is there a potential point of divergence?"** — a place where two nodes will evolve independently but describe the same real-world entity, and that divergence will cause a bug or inconsistency later.

### The N² Problem and Three-Level Filtering

For N leaf nodes, naive pairwise comparison is N×(N-1)/2 LLM calls:
- 8 nodes → 28 pairs
- 20 nodes → 190 pairs  
- 50 nodes → 1,225 pairs

Three filtering levels reduce this to a manageable set:

**Level 1 — Structural filter**: nodes in the same branch are already hierarchically related — not hidden connections. Check only **cross-branch pairs**. For a tree with branching factor 3 and depth 3: cross-branch pairs ≈ 30–40% of all pairs.

**Level 2 — Embedding pre-filter**: compute cosine similarity between node text embeddings.
- similarity < 0.15 → almost certainly unrelated, skip
- similarity > 0.50 → relationship already obvious from text, skip (or auto-include)
- 0.15–0.50 → non-obvious but potential — send to LLM

This range (0.15–0.50) is where latent connections live. Savings: 60–70% of LLM calls.

**Level 3 — Confidence score**: for each proposed connection, the LLM assigns HIGH / MEDIUM / LOW. LOW is discarded automatically. Only HIGH and MEDIUM reach the human reviewer.

Combined, the three filters typically reduce LLM calls from N² to approximately N×log(N).

### Design Structure Matrix (DSM) Output

The formal representation of cross-cutting dependencies is the **Design Structure Matrix** — an N×N grid where rows and columns are plan nodes and cells contain dependency types. This is the standard tool in systems engineering for exactly this problem.

```
              config  interview  git   jira  commits_v  task_v
config        [    ]  [      ]   [  ]  [  ]  [CFG]      [CFG]
interview     [    ]  [      ]   [ORD] [ORD] [    ]      [    ]
git           [    ]  [      ]   [  ]  [  ]  [PRD]      [    ]
jira          [    ]  [      ]   [  ]  [  ]  [    ]      [PRD]
commits_v     [    ]  [      ]   [  ]  [  ]  [    ]      [MIR]
task_v        [    ]  [      ]   [  ]  [  ]  [MIR]      [    ]
```

A filled cell `[CFG]` means "row node configures column node." A `[MIR]` means both nodes model the same real-world entity and should be checked for consistency. The DSM can be saved as `.1bcoder/dep_matrix.yaml` for later reference.

### `/deepcheck` Command

```
/deepcheck file: plan.md  [--scope cross-branch|all]  [--embed-filter]  [profile: name]
```

Algorithm:
1. Parse tree from `file:`
2. Build candidate pairs (structural filter + optional embedding filter)
3. For each pair, send 6 typed dependency questions in parallel (via `profile:`)
4. Collect HIGH/MEDIUM answers, discard LOW
5. Output DSM + human-readable proposal list

**Output format:**
```
[deepcheck] Analyzed 12 cross-branch pairs → 4 potential hidden connections

HIGH  1.1 → 3.1  [configures]
      "default config should declare: vectorizer model name (BGE/Word2Vec/GloVe),
       aggregation strategy (mean/max/cls), vector dimensions — all required by
       commits vectorizer as initialization parameters"

HIGH  1.3 → 3.2  [constrains]
      "extracted rules define which task fields carry semantic weight — task
       vectorizer needs this schema to weight field contributions correctly"

MEDIUM  3.1 → 3.2  [mirrors]
      "both vectorize natural language text — shared tokenization and normalization
       strategy would improve retrieval consistency across commit and task spaces"

MEDIUM  1.2 → 2.1  [orders]
      "user interview should capture git repository URL and branch strategy before
       GIT integration can be configured"

Add to plan? Review each with [y]es / [n]o / [e]dit
```

Nothing is added automatically. The human decides which connections are real and worth adding to the plan.

### Connection to `/deepagent`: the `list:` Workaround

The `list:` parameter in `/deepagent` is a partial preventive solution: specifying analytical lenses like `list: configuration concerns, data flow, ordering dependencies` forces the LLM to consider cross-cutting concerns *during* decomposition. This catches some latent dependencies early.

But `list:` only helps for dimensions the user anticipated before decomposition. `/deepcheck` catches what `list:` missed — the dependencies that only become visible when you hold two nodes side by side. They are complementary:

- `list:` — preventive, runs during tree building, requires foresight
- `/deepcheck` — corrective, runs after tree building, requires no foresight

### Limitations

**1B models and latent dependencies**: small models can surface latent connections using general domain knowledge ("vectorizers need config") but will miss deep system-specific connections that require understanding the codebase. For system-specific dependency detection, provide the relevant code files as context before running `/deepcheck`.

**Emergent dependencies**: type 4 dependencies (those that only appear during implementation) cannot be detected by any static analysis of a plan. `/deepcheck` targets latent (type 3) dependencies only. Emergent dependencies require a separate post-implementation review pass.

**Subjectivity of "real"**: whether a MEDIUM connection is real depends on design decisions not in the plan. The human reviewer is the final arbiter. `/deepcheck` is a proposal engine, not a decision engine.

---

## Summary: Metrics Cheatsheet

| Metric | Type | Measures | Needs embeddings? |
|---|---|---|---|
| Term novelty | Reply | New knowledge added | No |
| Task grounding | Reply | On-topic relevance | No |
| Information density | Reply | Content per token | No |
| Token efficiency | Reply | New concepts per token | No |
| Actionability | Reply | Code/steps present | No |
| ROUGE-1-Recall | Context | Task word coverage | No |
| ROUGE-2-Recall | Context | Task phrase coverage | No |
| Keyword coverage | Context | Key term coverage | No (IDF optional) |
| Context drift | Context | Semantic distance from task | Yes (small model) |
| Contextual entropy | Context | Focus vs scatter | No |
| Information gain/turn | Context | Per-turn novelty decay | No |
| Projected coverage | Context | Will we finish the task? | No |
| Stage FSM state | Context | Which stages are complete | No |
| Terminal signal detection | Context | Stage genuinely closed | No |
| Weighted stage coverage | Context | Progress incl. in-progress | No |

The embedding-based metrics (context drift) require a small sentence-transformer model but provide the most semantically meaningful signal. All others run on pure text in milliseconds.

### DeepAgent Parameters

| Parameter | Axis | What it controls |
|---|---|---|
| `target:` | anchor | task description, reference for all metrics |
| `prompt:` | template | self-similar expansion function, reusable across tasks |
| `plan:` | vertical | semantic label per depth level — what kind of detail |
| `list:` | horizontal | analytical lenses applied to every node |
| `profile:` | execution | which models expand which leaves |
| `file:` | output | numbered markdown outline, readable by `/agent` |

---

## What This Is Not

This system does not evaluate whether the LLM's answer is *correct*. Correctness requires a ground truth, which requires a human or another LLM. What it evaluates is whether the conversation is *efficiently covering the task space* — a structural property that can be measured without understanding the content.

The analogy: a doctor can measure a patient's temperature, blood pressure, and oxygen saturation without diagnosing the disease. These proxy signals are useful for triage even without diagnosis. Context evaluation metrics are the same: useful proxies, not ground truth.

`/deepagent` is not an autonomous problem-solver. It is a **structured knowledge elicitation tool**: it uses the LLM's language capability to decompose a task into a human-navigable tree, then hands that tree to a human (or a standard `/agent`) for execution. The intelligence is in the structure, not in any single LLM call.
