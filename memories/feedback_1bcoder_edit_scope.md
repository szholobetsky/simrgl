---
name: feedback-1bcoder-edit-scope
description: "Edit only _bcoder_data files, never ~/.1bcoder/ directly"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 81f99d6f-5159-485d-b1aa-700a61904c86
---

Edit 1bcoder flow/proc/script files only in `C:\Project\1bcoder\_bcoder_data\`, never in `C:\Users\stzh\.1bcoder\`.

**Why:** User wants changes to stay in the project source, not the global user directory.

**How to apply:** When fixing bugs or adding features to flows/procs, always target `_bcoder_data\flows\`, `_bcoder_data\proc\` etc. Do not touch `~/.1bcoder/` even if that's the file currently being executed.
