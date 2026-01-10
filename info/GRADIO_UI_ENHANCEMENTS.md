# Gradio UI Enhancements

## Overview

The Gradio UI (`gradio_ui.py`) has been enhanced with two major features:
1. **Collection Mode Selector** - Switch between RECENT (w100) and ALL collections
2. **Expandable File Diffs** - View changed files and their diffs for each historical task

## Feature 1: Collection Mode Selector

### What It Does

Allows users to choose which collection set to search:
- **RECENT**: Last 100 tasks (w100 collections)
- **ALL**: Complete history (all collections)

### Where It Appears

Added to multiple tabs with appropriate defaults:

**Search Tab:**
- Default: RECENT
- Affects: Module and File searches

**Task Search Tab:**
- Default: ALL
- Affects: Historical task searches

### Implementation

#### Updated Functions

**1. `search_modules()` - Added `collection_mode` parameter**
```python
def search_modules(task_description: str, top_k: int, target: str, collection_mode: str = "RECENT"):
    # Select collection based on target and mode
    if target == "Module":
        collection_name = config.COLLECTION_MODULE_RECENT if collection_mode == "RECENT" else config.COLLECTION_MODULE_ALL
    else:
        collection_name = config.COLLECTION_FILE_RECENT if collection_mode == "RECENT" else config.COLLECTION_FILE_ALL
```

**2. `search_tasks()` - Added `collection_mode` parameter**
```python
def search_tasks(task_description: str, top_k: int, collection_mode: str = "ALL"):
    # Select collection based on mode
    collection_name = config.COLLECTION_TASK_RECENT if collection_mode == "RECENT" else config.COLLECTION_TASK_ALL
```

#### UI Components

**Radio Button:**
```python
collection_mode_radio = gr.Radio(
    choices=["RECENT", "ALL"],
    value="RECENT",  # or "ALL" for tasks
    label="Collection Mode",
    info="RECENT: Last 100 tasks | ALL: Complete history"
)
```

**Connected to Search:**
```python
search_btn.click(
    fn=search_modules,
    inputs=[task_input, top_k_slider, target_radio, collection_mode_radio],
    outputs=results_output
)
```

### Usage

1. User enters task description
2. User selects collection mode (RECENT or ALL)
3. User clicks search
4. Results come from the selected collection

**Example:**
- Query: "Fix memory leak"
- Mode: RECENT → Searches last 100 modules/files
- Mode: ALL → Searches complete history

### Benefits

- **Faster searches** with RECENT (smaller collections)
- **Comprehensive coverage** with ALL (complete history)
- **User control** over precision vs coverage trade-off

## Feature 2: Expandable File Diffs

### What It Does

For each historical task found, displays:
1. List of files that were changed in that task
2. Expandable diff viewer for each file
3. Click 🔍 Diff button to toggle diff visibility

### Where It Appears

**Task Search Tab** - Shows changed files for each similar task

### Implementation

#### New Function: `get_task_files_and_diffs()`

```python
def get_task_files_and_diffs(task_name: str):
    """
    Fetch changed files and their diffs for a given task from RAWDATA table

    Returns:
        List of dicts with 'path', 'message', 'diff'
    """
    conn = psycopg2.connect(...)
    cursor.execute("""
        SELECT path, message, diff
        FROM vectors.rawdata
        WHERE task_name = %s
        ORDER BY id
        LIMIT 50
    """, (task_name,))

    files = []
    for row in cursor.fetchall():
        files.append({
            'path': row[0],
            'message': row[1],
            'diff': row[2]
        })

    return files
```

#### Updated `search_tasks()` HTML Generation

For each task result:
1. Fetch changed files from RAWDATA table
2. Generate HTML with file list
3. Add clickable diff buttons
4. Include hidden diff containers
5. Add JavaScript toggle function

**HTML Structure:**
```html
<div>
  <h3>Task SONAR-12345</h3>
  <div>Similarity: 0.87</div>

  <div>📁 Changed Files (3):</div>

  <!-- File 1 -->
  <div>
    <button onclick="toggleDiff('diff_1_0')">🔍 Diff</button>
    <code>src/main/java/LoginService.java</code>

    <div id="diff_1_0" style="display: none;">
      <pre>
        +public void authenticate() {
        +  // New authentication logic
        +}
      </pre>
    </div>
  </div>

  <!-- File 2 -->
  ...
</div>

<script>
function toggleDiff(id) {
    var element = document.getElementById(id);
    if (element.style.display === 'none') {
        element.style.display = 'block';
    } else {
        element.style.display = 'none';
    }
}
</script>
```

#### Styling

**File Container:**
- Background: Light gray (#f9f9f9)
- Border: 1px solid #ddd
- Padding: 8px
- Rounded corners

**Diff Button:**
- Blue background (#007bff)
- White text
- Clickable, changes cursor to pointer
- Emoji icon: 🔍

**Diff Viewer:**
- Dark theme (#282c34 background)
- Light text (#abb2bf)
- Monospace font (Courier New)
- Max height: 400px with scroll
- Horizontal scroll for long lines
- Pre-wrap for text wrapping

**No Files Message:**
- Yellow background (#fff3cd)
- Left border: 3px solid #ffc107
- Italic text

### Data Flow

```
User searches tasks
     ↓
search_tasks() executes vector search
     ↓
For each task result:
     ↓
get_task_files_and_diffs(task_name)
     ↓
Query RAWDATA table in PostgreSQL
     ↓
Fetch: path, message, diff
     ↓
Generate HTML with expandable sections
     ↓
Return HTML to Gradio
     ↓
User clicks 🔍 Diff button
     ↓
JavaScript toggleDiff() shows/hides diff
```

### Database Query

**Table:** `vectors.rawdata`

**Columns Used:**
- `task_name`: Task identifier (e.g., "SONAR-12345")
- `path`: File path that was changed
- `message`: Commit message
- `diff`: Git diff content

**Query:**
```sql
SELECT path, message, diff
FROM vectors.rawdata
WHERE task_name = %s
ORDER BY id
LIMIT 50
```

**Limit:** 50 files per task (prevents UI overload)

### Security

**HTML Escaping:**
All user/database content is escaped to prevent XSS:
```python
file_path = html.escape(file_info['path'])
diff_content = html.escape(file_info['diff'] or 'No diff available')
file_message = html.escape(file_info['message'] or '')
```

### Usage Example

**1. User searches:**
```
Task Description: "Add OAuth authentication"
Collection Mode: ALL
Number of Results: 5
```

**2. Results show:**
```
📋 Similar Historical Tasks

1. Task SONAR-23105: Add OAuth authentication
   Similarity: 0.92

   📁 Changed Files (3):

   [🔍 Diff] server/sonar-webserver-auth/.../OAuth2ContextFactory.java
   [🔍 Diff] server/sonar-webserver-auth/.../JwtHttpHandler.java
   [🔍 Diff] sonar-core/.../SecurityProperties.java

2. Task SONAR-22987: Fix authentication bug
   Similarity: 0.85
   ...
```

**3. User clicks "🔍 Diff" on first file:**
```
[🔍 Diff] server/sonar-webserver-auth/.../OAuth2ContextFactory.java

+package org.sonar.server.authentication;
+
+public class OAuth2ContextFactory {
+  public OAuth2Context create(HttpServletRequest request) {
+    // New OAuth2 context creation logic
+    return new OAuth2Context(request);
+  }
+}
```

**4. User clicks again to hide diff**

## Benefits of Enhancements

### Collection Mode Selector

✅ **Performance Control**
- RECENT = faster searches (smaller index)
- ALL = comprehensive searches (complete history)

✅ **Precision vs Coverage**
- RECENT = high precision for current work
- ALL = maximum coverage for rare cases

✅ **User Flexibility**
- Easy to switch between modes
- Clear labels explain the difference

### Expandable File Diffs

✅ **Context-Rich Results**
- See exactly what changed in similar tasks
- Understand implementation patterns
- Learn from actual code changes

✅ **On-Demand Details**
- Collapsed by default (clean UI)
- Expand only files of interest
- Avoid information overload

✅ **Better Decision Making**
- Compare multiple approaches
- See full diff context
- Identify relevant patterns

✅ **Historical Learning**
- Learn how similar problems were solved
- See actual code examples
- Understand change patterns

## Configuration

### Collection Names (config.py)

```python
# RECENT collections (w100)
COLLECTION_MODULE_RECENT = 'rag_exp_desc_module_w100_modn_bge-small'
COLLECTION_FILE_RECENT = 'rag_exp_desc_file_w100_modn_bge-small'
COLLECTION_TASK_RECENT = 'task_embeddings_w100_bge-small'

# ALL collections
COLLECTION_MODULE_ALL = 'rag_exp_desc_module_all_modn_bge-small'
COLLECTION_FILE_ALL = 'rag_exp_desc_file_all_modn_bge-small'
COLLECTION_TASK_ALL = 'task_embeddings_all_bge-small'
```

### Database Connection

```python
conn = psycopg2.connect(
    host=config.POSTGRES_HOST,
    port=config.POSTGRES_PORT,
    database='semantic_vectors',
    user=config.POSTGRES_USER,
    password=config.POSTGRES_PASSWORD
)
```

## Files Modified

**Main File:**
- `ragmcp/gradio_ui.py` - Complete UI enhancement

**Dependencies Added:**
```python
import psycopg2  # For database queries
import html     # For HTML escaping
```

**Functions Modified:**
- `search_modules()` - Added collection_mode parameter
- `search_tasks()` - Added collection_mode parameter + file diffs

**Functions Added:**
- `get_task_files_and_diffs()` - Fetch files and diffs from RAWDATA

**UI Components Added:**
- Collection mode radio buttons (2x)
- File list sections with expandable diffs
- JavaScript toggle function

## Testing

### Test Collection Mode Selector

1. Launch UI: `launch_ui.bat`
2. Go to "Search" tab
3. Enter query: "Fix memory leak"
4. Select "RECENT" → Click Search → Note results
5. Select "ALL" → Click Search → Note results (may differ)

### Test File Diffs

1. Launch UI: `launch_ui.bat`
2. Go to "Task Search" tab
3. Enter query: "Add OAuth authentication"
4. Select "ALL" mode
5. Click "🔍 Find Similar Tasks"
6. Look for "📁 Changed Files" section
7. Click "🔍 Diff" button next to a file
8. Verify diff appears in dark theme box
9. Click again to hide diff

### Expected Behavior

**Collection Mode:**
- RECENT searches show recent tasks/files/modules
- ALL searches show comprehensive results
- Radio button state persists during session

**File Diffs:**
- Each task shows "📁 Changed Files (N)" if files exist
- Blue "🔍 Diff" buttons appear for each file
- Clicking button toggles diff visibility
- Diff appears in dark theme with syntax
- Multiple diffs can be open simultaneously

## Troubleshooting

### Issue: No changed files shown

**Possible causes:**
1. RAWDATA table not populated
2. Task name mismatch between vectors and RAWDATA
3. Database connection issue

**Solution:**
- Check RAWDATA table has data: `SELECT COUNT(*) FROM vectors.rawdata`
- Verify task_name matches between collections and RAWDATA
- Check PostgreSQL connection

### Issue: Diff doesn't toggle

**Possible causes:**
1. JavaScript not loading in Gradio
2. Unique ID collision
3. Browser JavaScript disabled

**Solution:**
- Reload page
- Check browser console for errors
- Try different browser

### Issue: Collection mode has no effect

**Possible causes:**
1. Only one collection exists (not both RECENT and ALL)
2. Collections have identical data
3. Backend not connected

**Solution:**
- Run: `create_all_task_embeddings.bat` to create missing collections
- Verify collections exist: Check "Collections" tab
- Check backend connection

## Summary

**Enhancements:**
1. ✅ Collection mode selector (RECENT vs ALL)
2. ✅ Expandable file diffs in task results
3. ✅ Clean, intuitive UI design
4. ✅ Security (HTML escaping)
5. ✅ Performance (limit 50 files per task)

**Impact:**
- Better user control over search scope
- Richer historical context
- Learning from past implementations
- Improved decision-making

**User Experience:**
- Toggle between recent and historical data
- View actual code changes from similar tasks
- On-demand detail expansion
- Clean, uncluttered interface
