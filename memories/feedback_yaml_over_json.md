---
name: feedback-yaml-over-json
description: User prefers YAML over JSON for config/metadata files across all projects
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 59f4e9c6-779f-4f0a-9009-ff73ef6ea5a3
---

User explicitly stated a standing preference: "yaml everywhere" — prefer YAML over JSON for config/metadata files (e.g. persisted run-parameters, small structured sidecar files), not just a one-off choice for a single feature.

**Why:** No specific incident cited — stated as a general taste/readability preference.

**How to apply:** When adding a new small structured file (metadata, config, persisted settings) in 1bcoder/vyrii/simargl or related tools, default to YAML (`yaml.safe_dump`/`yaml.safe_load`, `import yaml as _yaml` local-import convention already used in `_bcoder_data/flows/webcrawl.py`, `obfuscate.py`, `deobfuscate.py`, `autocheck.py`, `external_help.py`) instead of `json`, unless the consumer specifically requires JSON (e.g. an external API contract, or interop with a tool that only reads JSON). Don't ask each time — just default to YAML for new internal sidecar files. Applied concretely: `deepagent_md.py`'s `_deepagent_meta.yaml` (run-parameter persistence for the `continue` subcommand) was switched from `.json` to `.yaml` on request — see [[project-deepagent-spec]].
