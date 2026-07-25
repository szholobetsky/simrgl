---
name: 1bcoder agent model capabilities
description: Which local models work well as 1bcoder /agent workers and which fail on ACTION: format
type: feedback
---

Models tested with /agent (action-required gate enabled):
- qwen:4b and nemotron:4b — follow instructions and ACTION: format correctly
- qwen:1.7b — can follow instructions but less reliably
- lfm2.5-thinking — understands the task but cannot produce correct ACTION: lines (thinking models at small size struggle with strict output format)

**Why:** Small thinking-focused models reason well but fail on structured output constraints like `ACTION: /command`.
**How to apply:** For /agent tasks requiring reliable ACTION: output, recommend qwen:4b or nemotron:4b minimum. Avoid lfm2.5-thinking as a primary agent worker; it may work as a /parallel worker for reasoning but not for action loops.
