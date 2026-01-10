# Enhanced Two-Phase Agent Features Guide

## Overview

The Two-Phase RAG Agent has been significantly enhanced with new configuration controls, visibility features, and task detail retrieval capabilities.

## New Features

### 1. **Configurable Collection Search Parameters**

You can now control exactly how many results to retrieve from each collection:

#### Tasks
- **RECENT collection**: 1-10 results (default: 3)
- **ALL collection**: 1-10 results (default: 2)

#### Modules
- **RECENT collection**: 1-20 results (default: 5)
- **ALL collection**: 1-20 results (default: 5)

#### Files
- **RECENT collection**: 1-50 results (default: 10)
- **ALL collection**: 1-50 results (default: 10)

**Why this matters:**
- RECENT collections provide high precision for current work
- ALL collections provide comprehensive coverage for finding older functionality
- You can balance speed vs. thoroughness by adjusting these values

### 2. **Task Detail Visibility Controls**

Three toggleable options to control what information is shown about similar tasks:

#### Show Task Titles & Descriptions
- Displays the full title and description of similar tasks
- Helps understand the context of past work

#### Show Changed Files
- Lists all files that were modified in each similar task
- Useful for understanding the scope of past changes

#### Show File Diffs
- Shows the actual code changes (diffs) from similar tasks
- Provides concrete examples of how problems were solved before
- Limited to first 3 files per task and 1000 characters per diff to avoid overwhelming output

**Use cases:**
- Enable all for maximum context (slower but most comprehensive)
- Disable diffs for faster execution when you just need file names
- Disable all for minimal output when you trust the LLM's file selection

### 3. **LLM Input Visibility**

A new tab shows **exactly what was sent to the LLM** in each phase:

#### What You'll See:
- **System Prompt**: The instructions given to the LLM about its role
- **User Prompt**: The full context including search results, file contents, etc.
- **Temperature**: The creativity parameter used for each phase

#### Why This Matters:
- **Transparency**: See exactly what information the LLM had when making decisions
- **Debugging**: Identify if the LLM is missing critical context
- **Learning**: Understand how the prompts are structured
- **Trust**: Verify that relevant information from searches made it to the LLM

#### Available for All Three Phases:
1. **Phase 1 (File Selection)**: See similar tasks, module scores, file scores
2. **Phase 2 (Deep Analysis)**: See the actual file contents provided
3. **Phase 3 (Reflection)**: See the analysis being reflected upon

## Web Interface Usage

### Configuration Section

1. **Open the Configuration Accordion** (⚙️ Configuration)
2. **Adjust Sliders** for collection search parameters
3. **Toggle Checkboxes** for task detail visibility
4. **Click "Apply Configuration"** to save changes

### LLM Inputs Tab

1. **Run a Query** using the main interface
2. **Navigate to "LLM Inputs" Tab**
3. **Expand Details** sections to see full prompts (they can be long)
4. **Review** what context was provided in each phase

## CLI Usage

When using the CLI agent, you can configure these options programmatically:

```python
from two_phase_agent import TwoPhaseRAGAgent

agent = TwoPhaseRAGAgent(
    use_dual_search=True,

    # Collection search configuration
    top_k_tasks_recent=3,
    top_k_tasks_all=2,
    top_k_modules_recent=5,
    top_k_modules_all=5,
    top_k_files_recent=10,
    top_k_files_all=10,

    # Task detail visibility
    show_task_details=True,
    show_task_files=True,
    show_task_diffs=True
)

await agent.initialize()
result = await agent.process_query("Your query here")

# Access LLM inputs
print(result['llm_inputs']['phase1'])
print(result['llm_inputs']['phase2'])
print(result['llm_inputs']['phase3'])
```

## Performance Considerations

### Speed vs. Completeness Trade-offs

**Faster Execution:**
- Lower top_k values (fewer results from each collection)
- Disable task diffs (most expensive feature)
- Keep task files and details for reasonable context

**Most Complete Analysis:**
- Higher top_k values (more results from each collection)
- Enable all task detail features
- Accept longer processing times

### Recommended Configurations

#### Quick Lookup (Fast)
```
Tasks: RECENT=2, ALL=1
Modules: RECENT=3, ALL=3
Files: RECENT=5, ALL=5
Details: All disabled
```

#### Balanced (Default)
```
Tasks: RECENT=3, ALL=2
Modules: RECENT=5, ALL=5
Files: RECENT=10, ALL=10
Details: Titles + Files enabled, Diffs disabled
```

#### Comprehensive (Thorough)
```
Tasks: RECENT=5, ALL=5
Modules: RECENT=10, ALL=10
Files: RECENT=20, ALL=20
Details: All enabled
```

## Technical Details

### MCP Server Updates

The `mcp_server_dual.py` now accepts separate parameters:
- `top_k_recent`: Results from RECENT collection
- `top_k_all`: Results from ALL collection

Previously used `top_k_each` for both, now more flexible.

### Agent Architecture

The `TwoPhaseRAGAgent` class now:
1. Tracks all LLM inputs in `self.llm_inputs` dictionary
2. Passes configuration parameters through the entire pipeline
3. Enhances task results with details when toggles are enabled
4. Includes LLM inputs in execution records

### Task Enhancement Pipeline

When task details are enabled, the agent:
1. Extracts task names from search results
2. Calls `get_task_files` for each task
3. If diffs enabled, calls `get_file_diff` for each file (limited to 3 per task)
4. Appends enhanced information to the search context
5. This enhanced context is then shown in LLM Inputs tab

## Troubleshooting

### Configuration Not Taking Effect
- Make sure to click "Apply Configuration" button
- If agent is already initialized, it will update dynamically
- Check console logs for configuration update messages

### LLM Inputs Show Truncated Content
- Prompts longer than 5000 characters are truncated in display
- Full prompts are still sent to the LLM
- This is just for UI readability

### Task Details Not Showing
- Verify that MCP server has access to RAWDATA in PostgreSQL
- Check that `get_task_files` and `get_file_diff` tools are working
- Enable debug logging to see tool call results

## Example Workflows

### Workflow 1: Finding Similar Bug Fixes

1. **Configure** for comprehensive search:
   - Tasks: RECENT=5, ALL=3
   - Enable all task details including diffs
2. **Query**: "Fix null pointer exception in cache manager"
3. **Review LLM Inputs Tab**: See what similar bugs were found
4. **Check Phase 1**: Review file selection reasoning with full context

### Workflow 2: Quick Feature Lookup

1. **Configure** for speed:
   - Reduce all top_k values
   - Disable diffs
2. **Query**: "Where is OAuth token validation implemented?"
3. **Review Summary**: Get quick answer
4. **LLM Inputs**: Verify search results if needed

### Workflow 3: Understanding Past Solutions

1. **Enable** show_task_diffs
2. **Query**: "How was rate limiting added before?"
3. **Phase 1 Tab**: See the actual code changes from similar tasks
4. **LLM Inputs**: Verify the diffs were provided to LLM

## Benefits

### For Users
- **Control**: Fine-tune agent behavior for your specific needs
- **Transparency**: See exactly what the LLM "knows"
- **Learning**: Understand how the system works
- **Debugging**: Identify issues with context or relevance

### For Development
- **Observability**: Track what information flows through the system
- **Optimization**: Identify bottlenecks and unnecessary work
- **Validation**: Verify search results are relevant
- **Testing**: Ensure prompts are correctly constructed

## Next Steps

- Experiment with different configurations for your use cases
- Use LLM Inputs tab to understand agent decisions
- Provide feedback on which features are most valuable
- Suggest additional visibility or control features

## Related Documentation

- `TWO_PHASE_AGENT_README.md` - Core agent documentation
- `DUAL_INDEXING_GUIDE.md` - Understanding dual collections
- `WEB_INTERFACE_GUIDE.md` - Web UI usage guide
- `QUICKSTART_DUAL_INDEXING.md` - Setup instructions
