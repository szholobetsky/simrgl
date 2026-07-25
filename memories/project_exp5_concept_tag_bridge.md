---
name: project-exp5-concept-tag-bridge
description: "Real-world precedent (past job, Oracle PL/SQL packages) refining exp5 RQ2 cross-vocabulary co-occurrence — the bridge needs a canonical concept-tag mediation layer, not direct term-to-identifier lexical extraction."
metadata: 
  node_type: memory
  type: project
  originSessionId: 5242261c-51ee-42c6-9c30-12ac80d2755a
  modified: 2026-07-21T08:37:46.248Z
---

User has direct prior experience with the exp5 RQ2 cross-vocabulary co-occurrence idea, from a previous job with a large Oracle PL/SQL codebase (>1GB of packages, confusingly named — e.g. `FINANCE_PAYMENT_UTILS`, `PAYMENT_PKG`, `PAYMENT_FINANCE_PKG` all touching payment logic).

**The manual workflow that worked**: for each task (ABC-123), tag it with a business keyword (e.g. "Payment Allocation", "Payment Due", "Amortisation") chosen from a controlled vocabulary — independent of the task's own language/text. After the task shipped, record which packages were actually changed. Aggregate: keyword → package change-frequency table, e.g. "Payment Allocation" → `PAYMENT_FINANCE_PKG`: 213, `FINANCE_PAYMENT_UTILS`: 5, `PAYMENT_PKG`: 2. This reliably pinpointed where "Payment Allocation" logic actually lived, with zero lexical/string overlap between the keyword and the identifier.

**Why: (2026-07-21 conversation)** User explicitly corrected an over-elaborate multi-vector/transliteration proposal I made — the mechanism they want is *purely statistical co-occurrence*, deliberately not tied to any lexical, semantic, or transliteration similarity, because source tasks can be in any language (Ukrainian, Arabic, Chinese) and code identifiers can be obfuscated. The bridge must work even when there is no string or phonetic relationship at all between term and identifier.

**Refinement this adds to exp5 RQ2** (`../exp5/README.md`'s cross-vocabulary co-occurrence matrix M[term][identifier]): the current exp5 draft extracts key terms directly from raw task TEXT via exp1's HHI filter. The user's proven-in-practice version instead mediates through a **canonical concept-tag layer** — a human (or classifier) assigns a language-agnostic concept label to the task first, and co-occurrence is computed between concept-tag ↔ identifier, never between raw multilingual task text and identifier strings. This is what makes it robust to language and obfuscation. Automating the "assign concept tag" step (previously manual) is the real open problem — options: closed-set multilingual few-shot classification if the concept vocabulary is small/known, or unsupervised cross-lingual clustering of tasks into concept groups.

**Statistical pitfall to fix before using raw counts**: 213 vs 5 vs 2 raw co-occurrence counts can be misleading if `PAYMENT_FINANCE_PKG` is simply a package that gets touched a lot for many unrelated reasons (same popularity-bias problem exp5 already flags for SonarQube's `server/` module = 70% of all changes). Should normalize with PMI or lift (co-occurrence rate relative to the identifier's overall base-rate change frequency), not raw counts.

**Literature to add to exp5's references** (currently cites Ye et al. 2016, Harnad 1990, etc. but not this specific subfield):
- Antoniol, G., et al. (2002). *Recovering traceability links between code and documentation*. IEEE TSE — foundational paper in software traceability recovery, exactly this term↔artifact linking problem.
- Marcus, A., & Maletic, J. I. (2003). *Recovering documentation-to-source-code traceability links using latent semantic indexing*. ICSE — LSI applied to the same term-to-code linking task.
- Structurally identical math to item-based collaborative filtering (Sarwar et al., 2001, "Item-based collaborative filtering recommendation algorithms") and association-rule mining / Apriori (Agrawal & Srikant, 1994) — same co-occurrence-matrix math, different application domain.

Related: [[feedback_research_vs_hardware_constraints]] (same conversation, prior turn — hardware constraints shouldn't limit exp4/exp5 research design).
