# Phase 1 Timing Analysis Guide

## Overview

The two-phase agent now displays detailed timing information for Phase 1 to help you understand where time is spent.

## Sample Output

```
================================================================================
PHASE 1: REASONING & FILE SELECTION
Started at: 14:32:15
================================================================================

[1.1] Searching for similar tasks (RECENT: 3, ALL: 2)...
      ⏱️  Started at: 14:32:15.123
      ✓ Completed in 0.52s

[1.1+] Enhancing tasks with details (files=True, diffs=True)...
       ⏱️  Started at: 14:32:15.643
       → Fetching details for 5 tasks...
       → Task 1/5: SONAR-23206 Remove design-system build... (1.23s)
       → Task 2/5: SONAR-23105 Add OAuth authentication... (1.45s)
       → Task 3/5: SONAR-22987 Fix memory leak in buffer... (1.12s)
       → Task 4/5: SONAR-22856 Optimize database queries... (0.98s)
       → Task 5/5: SONAR-22745 Implement rate limiting... (1.34s)
       ✓ Completed in 6.12s

[1.2] Searching for relevant modules (RECENT: 5, ALL: 5)...
      ⏱️  Started at: 14:32:21.763
      ✓ Completed in 0.78s

[1.3] Searching for relevant files (RECENT: 10, ALL: 10)...
      ⏱️  Started at: 14:32:22.543
      ✓ Completed in 1.23s

[1.4] LLM reasoning and file selection...
      ⏱️  Started at: 14:32:23.773
      ✓ Completed in 5.67s

[✓] Selected 8 files for detailed analysis

================================================================================
PHASE 1 COMPLETE - Total time: 14.32s
Finished at: 14:32:29
================================================================================
```

## Timing Breakdown

### Step 1.1: Task Search (0.5-1s)
**What it does:**
- Searches RECENT collection (last 100 tasks)
- Searches ALL collection (complete history)
- Returns top-k most similar tasks

**Expected time:** 0.3-1.5 seconds
- Fast: 0.3-0.6s
- Normal: 0.6-1.0s
- Slow: 1.0-1.5s

**Factors:**
- Number of tasks requested (top_k_recent + top_k_all)
- Vector collection size
- PostgreSQL performance

### Step 1.1+: Task Enhancement (varies greatly!)
**What it does:**
- For each similar task found:
  - Fetches list of changed files (0.1-0.3s per task)
  - If diffs enabled: Fetches diff for each file (0.3-0.5s per file, up to 3 files per task)

**Expected time:**
- **Diffs OFF**: 0.5-1.5s (only fetches file lists)
- **Diffs ON**: 5-15s (fetches file lists + diffs)

**Formula with diffs enabled:**
```
Time ≈ (tasks × 0.2s) + (tasks × files_per_task × 0.4s)
     ≈ (5 × 0.2s) + (5 × 3 × 0.4s)
     ≈ 1s + 6s = 7s
```

**Optimization:**
- Disable "Show File Diffs" checkbox → saves 5-10 seconds!
- Reduce number of tasks from RECENT/ALL collections

### Step 1.2: Module Search (0.5-1s)
**What it does:**
- Searches RECENT modules collection
- Searches ALL modules collection
- Returns top-k most similar modules

**Expected time:** 0.5-1.2 seconds

**Factors:**
- Number of modules requested (top_k_recent + top_k_all)
- Module collection size

### Step 1.3: File Search (0.8-1.5s)
**What it does:**
- Searches RECENT files collection
- Searches ALL files collection
- Returns top-k most similar files

**Expected time:** 0.8-2.0 seconds

**Factors:**
- Number of files requested (top_k_recent + top_k_all)
- File collection size (usually larger than modules)

### Step 1.4: LLM Reasoning (4-8s)
**What it does:**
- Sends all search results to Ollama
- LLM analyzes context and selects most important files
- Returns JSON with selected files

**Expected time:** 3-10 seconds

**Factors:**
- Ollama model (qwen2.5-coder:latest)
- CPU/GPU performance
- Prompt length (longer with task details)
- Temperature (0.3 = faster, more deterministic)

## Performance Profiles

### Fast Configuration (Total: ~6-8 seconds)
```python
# In Web UI Configuration:
Tasks RECENT: 2
Tasks ALL: 1
Modules RECENT: 3
Modules ALL: 3
Files RECENT: 5
Files ALL: 5
Show Task Diffs: ❌ OFF

Expected breakdown:
- Task search: 0.4s
- No task enhancement
- Module search: 0.6s
- File search: 0.9s
- LLM reasoning: 4s
Total: ~6s
```

### Balanced Configuration (Total: ~10-12 seconds) - **Default**
```python
Tasks RECENT: 3
Tasks ALL: 2
Modules RECENT: 5
Modules ALL: 5
Files RECENT: 10
Files ALL: 10
Show Task Files: ✅ ON
Show Task Diffs: ❌ OFF

Expected breakdown:
- Task search: 0.5s
- Task enhancement (files only): 1s
- Module search: 0.8s
- File search: 1.2s
- LLM reasoning: 6s
Total: ~10s
```

### Comprehensive Configuration (Total: ~18-25 seconds)
```python
Tasks RECENT: 5
Tasks ALL: 5
Modules RECENT: 10
Modules ALL: 10
Files RECENT: 20
Files ALL: 20
Show Task Files: ✅ ON
Show Task Diffs: ✅ ON

Expected breakdown:
- Task search: 0.8s
- Task enhancement (with diffs): 8-12s
- Module search: 1.2s
- File search: 2s
- LLM reasoning: 8s
Total: ~20-24s
```

## Optimization Tips

### To Save 5-10 Seconds: Disable Diffs
The **biggest time saver** is disabling "Show File Diffs":
- Reduces Phase 1 from ~15s to ~8s
- You still get file names, just not the actual code changes
- Best for quick lookups

### To Save 2-3 Seconds: Reduce Collection Queries
Reduce the number of results from each collection:
```
Instead of:     Use:
Files: 10+10    Files: 5+5   (saves ~0.5s)
Modules: 5+5    Modules: 3+3 (saves ~0.3s)
Tasks: 3+2      Tasks: 2+1   (saves ~0.2s + less enhancement time)
```

### To Save 1-2 Seconds: Faster LLM
Use a smaller/faster Ollama model:
```bash
# Instead of qwen2.5-coder:latest (8B parameters)
# Use a smaller model:
ollama pull qwen2.5-coder:1.5b
```

Update `two_phase_agent.py`:
```python
self.model = "qwen2.5-coder:1.5b"  # Faster but less capable
```

## Troubleshooting

### Phase 1 Takes >30 Seconds
**Possible causes:**
1. **Task diffs enabled with many tasks**
   - Solution: Disable "Show File Diffs" checkbox

2. **PostgreSQL slow**
   - Check: Run `check_active_queries.sql` to see if other queries are blocking
   - Solution: Restart PostgreSQL or close other connections

3. **Ollama slow**
   - Check: Run `ollama ps` to see model load status
   - Solution: Use smaller model or add GPU support

4. **Network/disk I/O bottleneck**
   - Check: Task manager for disk/CPU usage
   - Solution: Close other applications

### Task Enhancement Step Missing
If you don't see `[1.1+] Enhancing tasks with details...`:
- All detail checkboxes are OFF
- This is normal and expected - saves time!

### Individual Task Taking >5 Seconds
If you see `Task 1/5: ... (8.45s)`:
- That task has many changed files
- Diffs are being fetched for 3 files
- Each diff query takes ~0.5-1s
- Solution: Disable diffs or reduce tasks

## Comparing to Simple Agent

| Metric | Simple Agent | Two-Phase Agent |
|--------|-------------|-----------------|
| Vector searches | 3 | 6 (dual) |
| Task details | None | Optional (5-15s) |
| LLM calls | 1 (shorter) | 1 (longer) |
| **Total time** | **2-3s** | **8-20s** |

The two-phase agent is slower because:
1. ✅ Searches both RECENT and ALL (2x searches)
2. ✅ Optional task detail fetching (big time cost)
3. ✅ Longer prompt to LLM (more context)
4. ✅ More thorough = more time

But you get:
- 📊 Better coverage (dual collections)
- 📝 Task details with actual diffs
- 🎯 More informed file selection
- 💡 Higher quality analysis

## Summary

Phase 1 timing is now fully visible! Use this information to:
1. **Understand** where time is spent
2. **Optimize** by disabling task diffs if you need speed
3. **Balance** between speed and thoroughness
4. **Debug** if Phase 1 is unusually slow

The default configuration (10-12s) provides a good balance of speed and quality.
