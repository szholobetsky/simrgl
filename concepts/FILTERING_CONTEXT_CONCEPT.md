## Context Filtering: From Passive to Proactive and Intentional

This is a response to the "star task," addressing the challenges of local LLM context overload, the search for logically related information, and the transfer of "experience" to agents.

### 1. Rethinking "Filtering": From Passive to Proactive and Intentional

The problem is not that the LLM filters poorly, but that we delegate to it a task for which it lacks *sufficient and high-quality prior grounding*. Instead of giving it "a bunch of smart but misleading ideas" and asking it to find "the first one that works," we should help it "generate a good idea immediately" through better-structured "experience."

The key shift: **from filtering noise to building intent-oriented context.**

### 2. Strategies for Intelligent Context Formation

Your own SIMARGL concepts already hold the answers. They need not just to be implemented, but **deeply integrated into the search and orchestration architecture.**

**2.1. Deep Phenomenological Grounding and "Digital Affordances" for Focus:**

*   **Core Problem:** Semantic search shows what is "similar" by text. But the LLM needs to know "what is the *object of the user's intention*" and "what can be *done* with that object."
*   **Solution (derived from `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` and `PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md`):**
    *   **Identifying "Intentional Objects" (Noema):** When a user says "fix validation," the agent should not just search for "validation," but identify *which specific object* (entity, function, field) is subject to validation. `KEYWORD_ENTITY_MAPPING.md` and `KEYWORD_INDEXING.md` are the first steps here.
    *   **Indexing "Affordances" and "Constraints":** For each identified entity (e.g., "Rule", "DocumentNumber"), index not only its files but also its associated typical *actions* (affordances: `can_validate`, `can_be_updated`, `can_be_deleted`) and *constraints* (constraints: `format_regex`, `min_length`, `max_value`).
        *   **Example:** If the task is "change prefix," the "can_be_modified" affordance for the "Prefix" object, along with the "prefix_format" constraint, provides the LLM with concrete, logically related context.
    *   **How this helps the LLM:** Instead of 9 irrelevant files, the LLM will receive 1-2 files containing the "object of intention," its "affordances," and "constraints." This creates an "intentional field" where each element has a clear "purpose" and "behavior."

*   **Agent's "Lebenswelt" (Life-World):**
    *   **Research Direction:** The `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` document emphasizes the LLM's lack of a "life-world." We could try to simulate this "life-world" by enriching the LLM's context not only with static connections but also with "simulated sequences of actions" and "system states."
    *   **Action:** Instead of a simple list of relevant files, provide the LLM with mini-scenarios or "execution traces" from historical tasks. For example, "When `IssueController` calls `JiraService.createIssue`, a state change occurs in the Jira database." This will give the LLM an understanding of the "performative acts" of the code.

**2.2. Compositional and Cross-Layer Embeddings for "Logical Paths":**

*   **Core Problem:** Individual files do not provide understanding of flow. The LLM needs "experience" of how code parts *logically interact*.
*   **Solution (derived from `COMPOSITIONAL_CODE_EMBEDDINGS.md` and `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md`):**
    *   **Building "Transformation Paths":** When an "intentional object" (e.g., "Quantity" in `GRNService`) is identified, the system should not just search for files with the word "Quantity," but *data paths* (`CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md`) or *call paths* (`COMPOSITIONAL_CODE_EMBEDDINGS.md`) that cover this object and its transformations between architectural layers.
    *   **Predictive Power:** If the task concerns a "negative balance" after "editing a GRN," the system should find a path starting in the UI (`GRNEditForm`), passing through the service (`GRNService.updateQuantity`), and reaching the entity logic (`Inventory.adjustQuantity`), where the error is likely located. This "path" is a concentrated and logically connected context that the LLM can "follow."
    *   **How this helps the LLM:** Instead of "disjoint pieces," the LLM will receive a "story" (path) about how data changes, which functions are called, and which system layers are involved. This is precisely the "experience" of the code's "action."

**2.3. Multi-Agent Orchestration for "Concentrated Experience":**

*   **Core Problem:** A single "large" agent cannot simultaneously search for different types of relationships and interpret them.
*   **Solution (derived from `TWO_PHASE_REFLECTIVE_AGENT.md`):**
    *   **Specialized Expert Agents:** Use the "Three Specialized Agents" approach. Each agent will be an "expert" in its area, producing a *concentrated* fragment of "experience":
        1.  **Intent Discovery Agent (Lightweight LLM):** Analyzes the user's query, extracts key "intentional objects" (Noema) and "intent types" (e.g., "fix bug", "add feature", "understand data flow").
        2.  **Ontology Grounding Agent (MCP):** Uses `KEYWORD_INDEXING.md` and `KEYWORD_ENTITY_MAPPING.md` to ground these objects to specific code identifiers, their affordances, and constraints.
        3.  **Path Discovery Agent (MCP):** After receiving grounded objects, actively searches for the most relevant *compositional and cross-layer paths* using `COMPOSITIONAL_CODE_EMBEDDINGS.md` and `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md`. This agent is already looking for "logically related" entities, not just semantically similar ones.
        4.  **Context Synthesis & Reflection Agent (Main LLM):** This agent receives *already filtered, concentrated "pieces of experience"* (intentional objects with their affordances, and relevant transformation paths), not 9 irrelevant files. Its task is to "reflect" on this high-quality context, synthesize a solution, and explain it.

*   **How this helps the LLM:** Instead of the LLM trying to filter 9/10 of noisy context itself, it is provided with "structured experience" from specialized agents. Each agent does its "part" of understanding, transforming raw data into "knowledge" that is then easily absorbed by the main LLM.

### 3. "Experience" for LLMs through Fine-Tuning

Fine-tuning is a key way to transfer "experience" to local LLMs.

*   **What specifically to fine-tune:**
    1.  **Intent Recognition & Planning (for Intent Discovery Agent):** Fine-tune a small LLM on tasks where the input is a task description, and the output is a structured plan (intentional objects, intent types, sequence of necessary actions/searches). This will teach it to "think" like you.
    2.  **Context Relevance Judging (for Context Synthesis & Reflection Agent):** Fine-tune on pairs of (task description, set of retrieved paths/grounded entities, relevance rating) with human annotations. This will teach it to distinguish which path is "truly important" for a given task, even if others are "semantically similar."
    3.  **Explanation Generation (for Context Synthesis & Reflection Agent):** Fine-tune on pairs of (code change, its explanation, referencing affordances/constraints/paths). This will teach it to generate *deep, cause-and-effect explanations*, not just rephrase the code.

*   **Advantages of fine-tuning local models:**
    *   **Specialization:** Models learn specifically on your domain, not on general world data.
    *   **Efficiency:** Small, fine-tuned models can be very effective for their specific tasks, consuming fewer resources than a large general LLM.
    *   **"Experience" and "Focus":** This is precisely the "expert experience" you want to transfer. The model learns to recognize "the main point" in your project.

### 4. Directions for "Battle-Tested" Trials and Further Research

For this to work "in battle conditions" on new projects, the following should be focused on:

1.  **Priority: Full implementation of phenomenological grounding:**
    *   Add fields for "affordances" and "constraints" to `ontology.identifiers` and `ontology.relations` (as described in previous recommendations).
    *   Develop methods for automatically extracting these "affordances" and "constraints" from code (e.g., from method signatures, annotations, regex in validation). This requires extending `UniversalIdentifierExtractor` and `UniversalRelationExtractor` in `PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md`.

2.  **Prototyping the "Path Discovery Agent":**
    *   Implement algorithms for extracting compositional and cross-layer paths, as described in `COMPOSITIONAL_CODE_EMBEDDINGS.md` and `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md`.
    *   Create an MCP tool that will take an "intentional object" and return the most relevant paths.

3.  **Beginning fine-tuning for "Intent Discovery" and "Context Relevance Judging":**
    *   Gather a small dataset (e.g., 50-100 examples) of your historical tasks: `(task_description, expected_intentional_objects, expected_relevant_paths, context_relevance_rating)`.
    *   Use this dataset to fine-tune a lightweight Ollama LLM for Intent Discovery.

4.  **Refactoring orchestration for a multi-agent approach:**
    *   Review `multiagent_rag_local_llm.py` and `TWO_PHASE_REFLECTIVE_AGENT.md`. Instead of one LLM performing all phases, distribute these phases among specialized, fine-tuned agents.
    *   A central "Orchestrator" will call these specialized agents sequentially, collecting "experience" at each stage.

This will allow SIMARGL to transition from "superficial hints" to "sufficient information" that is "logically related" and "focuses on the main point," without overloading the context of weaker local LLMs. This is a challenging but very promising path. 