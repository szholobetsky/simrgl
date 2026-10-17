# Symmetry via Jev — One Questionnaire, Two Objects

**Created**: 2026-10-14
**Status**: Concept, nothing implemented. Nothing in this document has been measured with tev1 on code.
**Related**: `../JEV.md` (decision models, rubrics, the adapter), `PASSPORT.md`, `RETRIEVAL.md`, `EMBEDDING.md`, `METRICS.md`

---

## 1. The Problem, Restated

Task-to-code retrieval fails where the task text and the code live in different spaces: the task is natural language, the file is code. That is how `RETRIEVAL.md §1` found documentation instead of `RuleFinder.java`.

**Symmetrizing the task** means bringing both sides into one space, where a match becomes a tautology (`RETRIEVAL.md §4`).

The common space today is the embedding. A 1B model writes a hypothesis of the unfixed code, and the hypothesis is compared with the code by cosine. This works, but has three weak points:
- cosine is not calibrated;
- the threshold is arbitrary;
- a garbage hypothesis is indistinguishable from a good one until it fails.

Jev decision models (`../JEV.md`) offer a second common space: **a fixed set of closed questions**. They never write text. They answer `noul` (yes/no), `choice` and `score` questions with probabilities, and Python decides by thresholds.

> LLM generates → Jev perceives → Python decides.

---

## 2. The Object Passport Is Already a Jev Rubric

`PASSPORT.md` gives each file a vector of answers to YES/NO questions (`ops:div`, `ops:db`, `layer:svc`, `flags:deprecated`, …). It was designed for a generative 1B model:
- one question per call, 4 s each;
- a text answer, parsed as YES or NO;
- no confidence.

These are exactly jev's native question types:

| Passport today | Passport on jev |
|---|---|
| one call per question | one prompt holds the chunk and a group of up to 8 questions |
| YES/NO parsed from text | `noul` with a probability |
| `layer:svc`, `layer:repo`, `layer:ctrl` as three questions | one `choice` question, probabilities over all layers |
| no confidence | `confidence`; threshold in the rubric, per question |
| the model judges its own output format | the judge never generates; format is guaranteed by the API |

The passport becomes a rubric file (`~/.1bcoder/rubrics/passport.yaml`), and `index_passport.py` (`PASSPORT.md §7`) becomes a `/loop`/`/jev` over file chunks.

---

## 3. The Symmetric Questionnaire

The main idea of this document is to ask **the same questions of the task** as of the code.

```
              questionnaire Q = {q1 … qn}
                  /                    \
   task passport T(q)              file passport F(q)
   "does the task concern          "does the code contain
    arithmetic division?"           arithmetic division?"
                  \                    /
             match(T, F) — computed by Python
```

The match is symmetric by construction: one questionnaire, one judge, two objects. The embedding aligns the two spaces approximately. The questionnaire aligns them **by definition**, because both sides are expressed in the same coordinates.

### 3.1 Every question has two forms

A question cannot be asked of both sides with the same wording. Code *contains* division; a task *concerns* division. Each question is therefore a pair, and **the pair of forms is itself the symmetry pair**:

| Key | Form about code | Form about the task |
|---|---|---|
| `ops:div` | Does this code perform arithmetic division? | Does this task concern a division, a ratio or a divide-by-zero? |
| `ops:db` | Does this code run SQL or access a database? | Does this task concern stored data, queries or the schema? |
| `flags:deprecated` | Does this code deprecate something or use something deprecated? | Does this task concern deprecation or a migration away from an API? |
| `layer` (choice) | Which layer is this code: UI, controller, service, repository, config? | Which layer would the change be in: UI, controller, service, repository, config? |

### 3.2 Matching

The two passports are probability vectors over the same keys. Candidate scores, all computed in Python:
- **dot product** of the probabilities: simple, symmetric;
- **naive Bayes**, as in gamayun: `P(file changed | task passport)`, with priors learned from task history. Here the key is no longer a word but a property;
- **hard filter:** a `choice` question with high confidence (`layer = repository`, p > 0.8) narrows the candidates before ranking.

The passport ranking is then fused with simargl's embedding ranking by RRF, as in exp10's triangulation.

### 3.3 Why this may help where embeddings fail

`PASSPORT.md §1` lists three failures of semantic search:
1. **Unknown domain vocabulary** (PPCon, EBU). The passport does not help here, because its vocabulary is technical.
2. **Syntactic bugs** (division by zero, deprecated annotation). The passport is built for this: the task says "fails for empty projects", the questionnaire maps that to `ops:div`, and the file passport has `ops:div`.
3. **Infrastructure files** (repo, config) with low similarity to the task text. Here the `layer` choice reaches them where cosine does not.

Expectation: the passport alone is weaker than embeddings and complements them on case 2 and 3 tasks. That is a hypothesis for §6, not a claim.

---

## 4. Three More Places Where Jev Fits Directly

### 4.1 Level 3 of the cascade

`RETRIEVAL.md §7`, Level 3 (`ask.py`, YES/NO per file, 4 s per file) becomes one `score` question per candidate: "how likely is it that this chunk must change for this task?". 5–10 candidates are **ranked** by a weighted score instead of being cut by a binary filter. The threshold stays in the rubric.

### 4.2 Domain hint and hypothesis filtering

`RETRIEVAL.md §6`: gemma3:1b without a domain hint writes garbage hypotheses. Jev covers both ends of this:
- **Before generation:** one `choice` question over the module names ("which module is this task about?") gives the `domain_hint` without an LLM call. This is Option B of §6 with a calibrated answer.
- **After generation:** each hypothesis is judged with `noul`: "is this code from the {domain} domain and does it show the state described in the task?". Garbage hypotheses are dropped before they are embedded, instead of polluting the cross-hypothesis score.

### 4.3 Aligned pairs for the rotation matrices

`EMBEDDING.md §5` marks the cross-layer aligned pairs as "not built", and the rotation matrices W cannot exist without them. Candidate pairs come cheaply: naming normalization (~70 %), ORM mappings, git co-changes. What is missing is a judge of each pair.
- **The question** is `noul`: "do `SALARY1` (table `EMP`) and `main_salary` (class `Employee`) denote the same concept?" That is alpha-equivalence (`EMBEDDING.md §1`), decided by a judge instead of a rule.
- **The output** is the pairs above the threshold. Procrustes runs on them.
- **IRR of the pair** comes from the same judge, as a `score` from "same value renamed" to "aggregate of many values". This decides whether the Noether or the thermodynamic framework applies (`METRICS.md §6`).

---

## 5. Where Jev Is Not Needed

The structural symmetries of `METRICS.md` are computed deterministically from `map.txt`. No model, decision or generative, belongs there:
- GHOST, ORPHAN, ARITY_MISMATCH, PHANTOM;
- ASYMMETRY_SCORE, ΔASYMMETRY;
- HARMONY@K, SYMMETRY@K.

Jev is for the pairs that cannot be seen syntactically:

| Pair | Why syntax is not enough | Question |
|---|---|---|
| `@Deprecated` ↔ migration path | the alternative is named in prose or in another file | `noul`: does this deprecation name a replacement? (MISLEADING_DEP) |
| `open` ↔ `close` | `try-with-resources`, context managers, close in a callee | `noul`: is every resource opened here released on all paths? |
| `serialize` ↔ `deserialize` | field order or versions agree only by meaning | `noul`: do these two read and write the same fields? |
| `DB field` ↔ `model field` | fields renamed across layers | §4.3 |

The division of labour is the same as everywhere else: Python decides what it can see; jev perceives what Python cannot.

---

## 6. Risks and How to Measure

**Quality on code is unknown.** tev1 was tested on prose only. `tev1:0.8b` called every word *technology* and accepted an invented fact. `tev1:4b` separated them. For code, expect `4b` as the floor.

First test: the verified examples of `PASSPORT.md §2` on tev1. `int result = total / count;` must be `ops:div = yes`, and `String path = "/usr/local/bin";` must be `no`.

**Context and cost.**
- tev1 reads at most 2 050 tokens and does not truncate. Files are cut at function boundaries from `map.txt`, as in `PASSPORT.md §5`, and the file passport is the maximum over its chunks.
- On a CPU host one question group takes 10–60 s, so the whole sonar repository (12 532 files) is tens of hours.
- Start from the top 500 files by task frequency (`PASSPORT.md §3`), or use a GPU host. The passport is incremental: unchanged files are skipped, as in the glossary's `_index.json`.

**Whether the task passport carries a signal at all.** Task texts often describe a symptom ("the page shows 0"), not an operation. The task form of each question must reach from symptom to operation, and that is where it may fail.

**Experiment (sonar, modn split, ground truth = files changed by the task's commits):**

| Run | Ranking |
|---|---|
| P | task passport ↔ file passport (dot product; naive Bayes with history priors) |
| S | simargl, embeddings |
| P+S | RRF of P and S |
| H | hypothesis search (`RETRIEVAL.md §3`), with and without jev filtering of hypotheses (§4.2) |

Measures: hit@1/5/10, MAP. They are broken down by task type, so that "syntactic" tasks (case 2 of §3.3) are reported separately. P+S beating S overall, or P beating S on syntactic tasks, would support §3. P being weak alone is expected and does not refute it.

---

## 7. Open Questions

- **Writing the two-form questions.** Can the task form be generated from the code form by a template, or does every pair need a human?
- **The questionnaire.** Fixed (the 9 keys of `PASSPORT.md`), or derived from the project? Candidates are the most frequent identifiers' operations, or gamayun's terms per module.
- **Granularity.** Is the file-level passport (max over chunks) enough, or must matching be chunk-level? The task usually points at a function.
- **Probabilities as features.** Are jev's probabilities calibrated enough to multiply, or should only thresholded bits be used?
