# Configurable Diffs Per Task

## Overview

The `max_diffs_per_task` parameter allows users to control how many file diffs are fetched and displayed for each similar historical task. This makes the system flexible for different use cases - from quick overviews (1-2 diffs) to comprehensive analysis (up to 10 diffs).

## What Changed

### Before
- Hardcoded limit: 3 file diffs per task
- No way to adjust without editing code
- Message: "📝 File Diffs (showing top 3):"

### After
- Configurable via UI slider: 1-10 diffs
- Default: 3 (same as before)
- Message: "📝 File Diffs (showing top 3):" (number updates dynamically)
- Changes apply immediately to next query

## Implementation

### 1. Backend Parameter (two_phase_agent.py)

#### Added to __init__ (line 109)
```python
def __init__(
    self,
    mcp_server_path: str = "mcp_server_postgres.py",
    ollama_url: str = None,
    model: str = None,
    use_dual_search: bool = True,
    top_k_tasks_recent: int = 3,
    top_k_tasks_all: int = 2,
    top_k_modules_recent: int = 5,
    top_k_modules_all: int = 5,
    top_k_files_recent: int = 10,
    top_k_files_all: int = 10,
    show_task_details: bool = True,
    show_task_files: bool = True,
    show_task_diffs: bool = True,
    max_diffs_per_task: int = 3  # NEW PARAMETER
):
    # ... initialization ...
    self.max_diffs_per_task = max_diffs_per_task
```

#### Used in Task Enhancement (lines 901-902)
```python
# Inside _enhance_tasks_with_details()
if file_list:
    enhanced_text += f"\n📝 File Diffs (showing top {self.max_diffs_per_task}):\n"
    for file_info in file_list[:self.max_diffs_per_task]:
        # Fetch and display diff for each file...
```

**Key Code Flow:**
1. User adjusts slider to 5
2. Backend stores: `self.max_diffs_per_task = 5`
3. During task enhancement: `file_list[:5]` fetches top 5 diffs
4. Message displays: "📝 File Diffs (showing top 5):"

### 2. Web Configuration (two_phase_agent_web.py)

#### Default Config (line 33)
```python
self.config = {
    'top_k_tasks_recent': 3,
    'top_k_tasks_all': 2,
    'top_k_modules_recent': 5,
    'top_k_modules_all': 5,
    'top_k_files_recent': 10,
    'top_k_files_all': 10,
    'show_task_details': True,
    'show_task_files': True,
    'show_task_diffs': True,
    'max_diffs_per_task': 3  # DEFAULT VALUE
}
```

#### Agent Initialization (line 54)
```python
self.agent = TwoPhaseRAGAgent(
    mcp_server_path=self.mcp_server_path,
    use_dual_search=True,
    top_k_tasks_recent=self.config['top_k_tasks_recent'],
    top_k_tasks_all=self.config['top_k_tasks_all'],
    top_k_modules_recent=self.config['top_k_modules_recent'],
    top_k_modules_all=self.config['top_k_modules_all'],
    top_k_files_recent=self.config['top_k_files_recent'],
    top_k_files_all=self.config['top_k_files_all'],
    show_task_details=self.config['show_task_details'],
    show_task_files=self.config['show_task_files'],
    show_task_diffs=self.config['show_task_diffs'],
    max_diffs_per_task=self.config['max_diffs_per_task']  # PASSED TO AGENT
)
```

#### Update Config Method (lines 75-109)
```python
def update_config(
    self,
    top_k_tasks_recent, top_k_tasks_all,
    top_k_modules_recent, top_k_modules_all,
    top_k_files_recent, top_k_files_all,
    show_task_details, show_task_files, show_task_diffs,
    max_diffs_per_task  # NEW PARAMETER
):
    """Update agent configuration"""
    self.config = {
        # ... other configs ...
        'max_diffs_per_task': int(max_diffs_per_task)
    }

    # Reinitialize agent with new config if already initialized
    if self.agent:
        # ... other updates ...
        self.agent.max_diffs_per_task = self.config['max_diffs_per_task']

    return f"✅ Configuration updated!\n\nDiffs per task: {self.config['max_diffs_per_task']}"
```

### 3. UI Component (two_phase_agent_web.py lines 460-470)

#### Gradio Slider
```python
gr.Markdown("### Diff Display Settings")
gr.Markdown("*Control how many file diffs to fetch per historical task*")

max_diffs_slider = gr.Slider(
    minimum=1,
    maximum=10,
    value=3,
    step=1,
    label="Diffs Per Task",
    info="Number of file diffs to fetch for each similar historical task"
)
```

**Slider Properties:**
- **Range**: 1-10 diffs
- **Default**: 3 (matches original hardcoded value)
- **Step**: 1 (integer values only)
- **Label**: "Diffs Per Task" (concise)
- **Info**: Explains what it does

#### Button Event Handler (lines 531-541)
```python
update_config_btn.click(
    fn=web_agent.update_config,
    inputs=[
        top_k_tasks_recent, top_k_tasks_all,
        top_k_modules_recent, top_k_modules_all,
        top_k_files_recent, top_k_files_all,
        show_task_details, show_task_files, show_task_diffs,
        max_diffs_slider  # CONNECTED TO UPDATE FUNCTION
    ],
    outputs=[config_status]
)
```

## Usage

### Step 1: Launch Web UI
```bash
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp
launch_two_phase_web.bat
```

### Step 2: Adjust Configuration
1. Click "⚙️ Configuration" accordion to expand
2. Scroll to "Diff Display Settings" section
3. Move slider to desired value (1-10)
4. Click "💾 Apply Configuration"
5. See confirmation: "✅ Configuration updated! Diffs per task: 5"

### Step 3: Run Query
Enter a task description and click "🚀 Analyze with Two-Phase Agent"

### Step 4: See Results
In Phase 1 output, similar tasks will show:
```
📝 File Diffs (showing top 5):

1. server/sonar-webserver-auth/.../OAuth2ContextFactory.java
   Message: Add OAuth2 context creation
   Diff:
   +package org.sonar.server.authentication;
   +
   +public class OAuth2ContextFactory {
   ...

2. server/sonar-webserver-auth/.../JwtHttpHandler.java
   ...

[continues for up to 5 files based on slider value]
```

## Use Cases

### Quick Overview (1-2 diffs)
**Scenario**: Just want to see the most important file changed
**Setting**: max_diffs_per_task = 1 or 2
**Benefit**: Faster processing, less information overload

### Balanced Analysis (3-5 diffs)
**Scenario**: Understand main implementation pattern
**Setting**: max_diffs_per_task = 3-5 (default is 3)
**Benefit**: Good balance between detail and speed

### Comprehensive Deep Dive (7-10 diffs)
**Scenario**: Need to see extensive changes across multiple files
**Setting**: max_diffs_per_task = 7-10
**Benefit**: Maximum context for complex tasks

## Example: Different Settings

### Query: "Add OAuth authentication support"

**Setting: 1 diff per task**
```
Similar Tasks Found:
1. SONAR-23105 (similarity: 0.92)
   📝 File Diffs (showing top 1):
   1. OAuth2ContextFactory.java
      [diff shown]
```

**Setting: 3 diffs per task (default)**
```
Similar Tasks Found:
1. SONAR-23105 (similarity: 0.92)
   📝 File Diffs (showing top 3):
   1. OAuth2ContextFactory.java
      [diff shown]
   2. JwtHttpHandler.java
      [diff shown]
   3. SecurityProperties.java
      [diff shown]
```

**Setting: 10 diffs per task**
```
Similar Tasks Found:
1. SONAR-23105 (similarity: 0.92)
   📝 File Diffs (showing top 10):
   1. OAuth2ContextFactory.java
      [diff shown]
   2. JwtHttpHandler.java
      [diff shown]
   ...
   10. AuthenticationEventImpl.java
      [diff shown]
```

## Performance Considerations

### Processing Time Impact

| Diffs Per Task | Approx. Time* | Use Case |
|----------------|---------------|----------|
| 1 | ~2-3s | Quick scan |
| 3 (default) | ~5-7s | Standard analysis |
| 5 | ~8-10s | Detailed review |
| 10 | ~15-20s | Comprehensive study |

*Times are estimates for 3 similar tasks. Actual time depends on diff size and MCP performance.

### Database Query Impact
Each diff requires:
1. Query to RAWDATA table
2. Fetch diff content (can be large)
3. Format for display

**Recommendation**: Use lower values (1-3) for initial exploration, increase (5-10) when deeper analysis is needed.

## Technical Details

### Where Diffs Come From
```sql
-- Executed for each file in file_list[:max_diffs_per_task]
SELECT diff
FROM vectors.rawdata
WHERE task_name = %s AND path = %s
LIMIT 1
```

### Data Flow
```
User adjusts slider
     ↓
update_config() called
     ↓
self.agent.max_diffs_per_task = value
     ↓
User submits query
     ↓
_enhance_tasks_with_details() runs
     ↓
For each similar task:
    file_list[:max_diffs_per_task]
     ↓
Fetch diff for each selected file
     ↓
Display in Phase 1 output
```

### Memory Usage
- **1 diff**: ~5-10 KB per task
- **3 diffs**: ~15-30 KB per task
- **10 diffs**: ~50-100 KB per task

For 3 similar tasks with 10 diffs each: ~300 KB total

## Configuration Persistence

### Session-Based
Configuration is stored in `self.config` and persists:
- ✅ For the entire session
- ✅ Across multiple queries
- ❌ NOT saved to disk (resets on restart)

### Reset to Defaults
To reset:
1. Restart the web UI: `launch_two_phase_web.bat`
2. Or manually set slider back to 3 and click "Apply Configuration"

## Related Settings

### show_task_diffs Checkbox
Controls whether diffs are shown at all:
- `True`: Show diffs (up to max_diffs_per_task)
- `False`: Hide all diffs (max_diffs_per_task ignored)

**Interaction:**
```python
if show_task_diffs and max_diffs_per_task > 0:
    # Fetch and display diffs
else:
    # Skip diff fetching
```

### show_task_files Checkbox
Controls whether file list is shown:
- `True`: Show changed files (diffs expandable if show_task_diffs=True)
- `False`: Hide file list entirely

**Hierarchy:**
1. `show_task_details` - Show task titles/descriptions
2. `show_task_files` - Show which files changed
3. `show_task_diffs` - Show actual diff content
4. `max_diffs_per_task` - How many diffs to fetch

## Troubleshooting

### Issue: Slider not working

**Symptoms**: Moving slider and clicking "Apply" doesn't change output

**Possible causes**:
1. Forgot to click "💾 Apply Configuration" button
2. `show_task_diffs` is unchecked
3. Similar tasks don't have any files in RAWDATA

**Solution**:
- Always click "Apply Configuration" after adjusting slider
- Ensure "Show File Diffs" checkbox is enabled
- Check RAWDATA table has data: `SELECT COUNT(*) FROM vectors.rawdata`

### Issue: Still showing 3 diffs after changing to 5

**Cause**: Task only has 3 files total in RAWDATA table

**Solution**: This is expected behavior - can't show more diffs than files exist. The limit is MIN(max_diffs_per_task, available_files)

### Issue: Performance slow with 10 diffs

**Cause**: Fetching 10 diffs × 3 tasks = 30 database queries + large content transfer

**Solution**:
- Use lower value (3-5) for faster results
- Only increase to 10 when comprehensive analysis is needed
- Check PostgreSQL performance

## Summary

**Feature**: Configurable diffs per task via UI slider

**Benefits**:
- ✅ User control over detail level
- ✅ Performance tuning (1 = fast, 10 = thorough)
- ✅ Easy to adjust without code changes
- ✅ Clear feedback on what's being shown

**Default**: 3 diffs per task (matches original hardcoded behavior)

**Range**: 1-10 diffs

**Files Modified**:
1. `ragmcp/two_phase_agent.py` - Added parameter and logic
2. `ragmcp/two_phase_agent_web.py` - Added config, UI, and handler

**Usage**: Adjust slider in "Diff Display Settings" section, click "Apply Configuration", run query
