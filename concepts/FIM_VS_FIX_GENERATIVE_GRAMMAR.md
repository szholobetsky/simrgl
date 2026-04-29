# Why FIM Works Better Than Structured Output for Small LLMs
## On Generative Grammar, Completion Instinct, and the Monkey with a Hammer

*Stanislav Zholobetsky, 2026*

---

## The Problem

When building coding assistants backed by local LLMs, a natural instinct is to constrain the model's output to a structured format. If we need to fix line 3 of a file, we ask the model to respond with:

```
LINE 3:     if b == 0:
```

Simple, parseable, deterministic. Except it doesn't work reliably — especially on models below 7B parameters. The model forgets indentation, flips the operator back, or produces explanation text instead. The smaller the model, the less reliable the structured output becomes.

This document argues that **the failure is not a model quality problem — it is a task framing problem**. And the solution comes from understanding what capabilities persist across all scales of language models.

---

## The Scale of Cognition: From Paramecium to Whale

Consider the cognitive spectrum in biology. A blue whale and a paramecium share almost nothing in terms of cognitive complexity. Yet they share something fundamental: reactive behavior. Fear. Avoidance. The drive to move toward light, away from danger. These are not learned strategies — they are instincts wired into the organism at a level below reasoning.

LLMs exhibit a similar structure across scale. A 450B model (the "whale") can:
- Follow multi-step plans
- Track multiple constraints simultaneously
- Reason abstractly about code structure
- Produce complex structured output formats

A 1B model (the "paramecium") loses most of this. But it retains:
- **Completion instinct** — the drive to continue a sequence in the most probable direction
- **Local coherence** — keeping the immediate context consistent
- **Grammar adherence** — following the structural rules of the language being generated

The last point is critical.

---

## Generative Grammar as a Shared Foundation

Chomsky's theory of generative grammar proposes that a finite set of rules can generate an infinite number of valid sentences. Programming languages are formal grammars — far stricter than natural language. Python's indentation rules, for instance, are not stylistic preferences but structural constraints that determine meaning.

When a language model is trained on billions of lines of code, it does not merely memorize patterns. It internalizes the generative grammar of each language. This internalization is robust — it persists even at extreme compression. A model distilled to 0.6B parameters may lose the ability to plan, reason, or follow instructions reliably, but it still "knows" that an `if` block body must be indented relative to the `if` statement.

This is not a rule the model consciously applies. It is a constraint embedded in the weight distribution — a structural prior so strong that generating syntactically invalid code feels, from the model's perspective, like generating a grammatically impossible sentence.

**This grammar knowledge is the shared foundation between the whale and the paramecium.**

---

## Why `LINE N: content` Fails

The structured output approach (`LINE N: content`) requires the model to:

1. Understand the code context and identify the bug
2. Reason about what the correct fix is
3. Switch from code generation mode to text generation mode
4. Produce an artificial format it was not specifically trained on
5. While simultaneously remembering all constraints (indentation, operators, semicolons)

Steps 3-5 are where small models fail. The moment the model is asked to produce `LINE 3:` — an instruction-following format — it exits the code generation regime. The generative grammar of Python is no longer constraining the output. The model is now generating text, and in text generation mode, "four spaces" and "no spaces" are equally probable.

This is not stupidity. It is a mode switch. The model was trained overwhelmingly on code and natural language, not on `LINE N:` formats. There is no strong grammar to fall back on.

---

## Why FIM Works

Fill-In-the-Middle (FIM) is a training technique used by CodeLlama, Qwen, StarCoder, DeepSeek-Coder, and most modern code models. During training, the model learns to complete a gap between a prefix and a suffix:

```
<prefix> code before the gap
<middle> [model fills this]
<suffix> code after the gap
```

When we frame a code fix as FIM, we present the model with:

```python
def divide(a, b):
<<<    if b != 0:>>>
        return a / b
    else:
        return None
```

And ask it to replace `<<<...>>>` with the corrected version.

The model is now in **code generation mode**. The surrounding Python code activates the generative grammar. The indentation of `return a / b` four spaces after the gap tells the model, through structural constraint, that the replacement must also be indented four spaces. The model does not need to *remember* this — the grammar enforces it.

Furthermore, the model does not need to produce an artificial format. It just needs to write one line of Python. The completion instinct — the most primitive and reliable capability — is sufficient.

---

## The Monkey and the Hammer

There is an old engineering maxim: *if you give a monkey a hammer and a saw and ask it to build a bow, it will fail — but if you let it hunt in its natural way, it will succeed*.

Tool calling in LLM frameworks is the hammer. Structured output formats are the saw. They assume the model can step outside its natural mode of operation and follow precise procedural instructions. For large models, this works. For small models, it fails — not because small models are fundamentally different in kind, but because they lack the working memory to hold the procedural constraint while also doing the semantic work.

**The insight**: instead of giving the monkey tools, design the task so that the monkey's natural behavior produces the correct result.

For LLMs, "natural behavior" is completion. Structure the prompt so that the most natural completion IS the correct answer. The grammar constraints come from context, not from instructions.

---

## Empirical Validation

This hypothesis was validated in the 1bcoder project (a terminal coding assistant for local LLMs). Two commands were compared:

- `/fix` — asks the model to produce `LINE N: corrected content`
- `/fim` — marks the target line with `<<<...>>>`, asks for the corrected file

Results across models:
- **qwen3:0.6b (500MB, Q4)** — `/fix` fails (loses indentation, flips operators). `/fim` succeeds.
- **gemma3:1b** — `/fix` unreliable. `/fim` reliable.
- **nemotron-3:4b** — previously failed with `/fix`. Succeeds with `/fim`.
- **All 1B+ models tested** — reliable with `/fim`.

The conclusion: the failure with `/fix` was not a model capability problem. It was a task framing problem. `/fim` aligns the task with the model's fundamental capability — completion within a grammatically constrained context.

---

## Implications

1. **For coding tool designers**: Prefer completion-based prompts over structured output formats when targeting small models. The `SEARCH/REPLACE` format used by aider, cursor, and opencode is better than `LINE N:` precisely because it appears in training data and requires the model to copy-then-modify, activating grammar constraints.

2. **For researchers**: The boundary between "model knows how to do X" and "model fails to do X" may often be a framing boundary, not a capability boundary. The capability exists but requires activation through appropriate context structure.

3. **For agent design**: Agents backed by small models should be designed so that each individual action is a primitive completion task. The complexity should emerge from the structure of the agent workflow, not from the cognitive complexity of individual model calls.

4. **For LLM evaluation**: Benchmarks that measure structured output compliance may systematically underestimate the capabilities of small models in tasks where framing can be adjusted.

---

## The Universal Primitive

Every code edit can be expressed as a Fill-In-the-Middle operation:

```
eml(before, after, hint) = model_completion(before + MARKER + after, hint)
```

Where `MARKER` indicates what needs to be replaced, `before` and `after` provide grammatical context from both sides, and `hint` guides the semantic direction of the completion.

- **Replace a line**: before = lines above, after = lines below, fill = corrected line
- **Fix indentation**: same structure — grammar constraints from `after` enforce the correct indentation
- **Refactor a function**: before = code above function, after = code below function, fill = refactored version

This single operation subsumes all local code editing tasks. It requires only completion instinct and grammar adherence — capabilities that persist from 0.5B to 450B.

---

## Conclusion

Small language models are not broken large language models. They are compressed versions of the same system, retaining the deep structural knowledge (grammar, completion instinct, common patterns) while losing the higher-order cognitive functions (multi-step planning, constraint tracking, format compliance).

Effective use of small models requires designing tasks that leverage what is retained, not what is lost. FIM-based code editing is one such design: it activates the model's most fundamental capability — completing a sequence within a grammatically constrained context — and produces reliable results even at 0.6B scale.

The whale and the paramecium both know how to swim. Design for swimming.
