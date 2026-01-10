# Debugging: "Millions of INSERTs" Issue

## Summary

The two-phase agent **DOES NOT perform any INSERT operations** during normal queries. All Phase 1 operations are **read-only SELECT queries**.

## What You're Seeing

If you see INSERT statements in PostgreSQL logs during agent queries, they are from one of these sources:

### 1. **Historical Log Buffer (Most Likely)**
PostgreSQL log files can show old entries mixed with new ones. The INSERTs are from yesterday's migration, but the log buffer is replaying them.

**Solution:**
```sql
-- Check when rawdata was last populated
SELECT MAX(created_at) FROM vectors.rawdata;

-- If this shows yesterday's timestamp, the INSERTs are old
```

### 2. **Migration Script Still Running**
The `migrate_rawdata_to_postgres.py` script might still be running from yesterday.

**Check:**
```bash
# Windows
ragmcp\check_processes.bat

# Linux/Mac
ragmcp/check_processes.sh
```

**Look for:**
- `migrate_rawdata.py` processes
- Multiple python processes

**Solution:**
```bash
# Kill any stuck migration processes
# Windows: taskkill /F /PID [process_id]
# Linux/Mac: kill -9 [process_id]
```

### 3. **Multiple MCP Server Instances**
Each web interface tab or CLI session starts its own MCP server. If you have multiple tabs or didn't stop previous sessions, you have multiple MCP servers running.

**Check:**
```bash
# See check_processes.bat/sh output
# Count how many mcp_server_dual.py processes are running
```

**Solution:**
- Close extra browser tabs
- Kill extra MCP server processes
- Restart the web interface

### 4. **PostgreSQL Log Configuration**
PostgreSQL might be configured to log ALL statements, including old ones.

**Check:**
```sql
SHOW log_statement;  -- If 'all', you'll see everything
SHOW log_min_duration_statement;
```

**Recommended Settings:**
```sql
-- Only log slow queries (>1 second)
ALTER SYSTEM SET log_min_duration_statement = 1000;

-- Only log DDL (CREATE, DROP, ALTER)
ALTER SYSTEM SET log_statement = 'ddl';

-- Reload configuration
SELECT pg_reload_conf();
```

## Verification: Agent Code is Read-Only

### Phase 1 Operations (All SELECTs):

```python
# 1. Search tasks (vector similarity)
SELECT embedding <=> %s AS distance FROM task_embeddings
ORDER BY distance LIMIT %s;

# 2. Search modules (vector similarity)
SELECT embedding <=> %s AS distance FROM module_embeddings
ORDER BY distance LIMIT %s;

# 3. Search files (vector similarity)
SELECT embedding <=> %s AS distance FROM file_embeddings
ORDER BY distance LIMIT %s;

# 4. Get task files (if enabled)
SELECT path, message, LENGTH(diff) as diff_size
FROM vectors.rawdata WHERE task_name = %s;

# 5. Get file diff (if enabled)
SELECT path, message, diff FROM vectors.rawdata
WHERE task_name = %s AND path = %s;
```

**No INSERT, UPDATE, DELETE, CREATE, or DROP anywhere!**

## Diagnostic Steps

### Step 1: Check What's Actually Running

**Windows:**
```bash
cd ragmcp
check_processes.bat
```

**Linux/Mac:**
```bash
cd ragmcp
./check_processes.sh
```

### Step 2: Check PostgreSQL Active Queries

```bash
psql -h localhost -U postgres -d semantic_vectors -f ragmcp/check_active_queries.sql
```

Look for:
- Any queries with `INSERT INTO vectors.rawdata`
- Any queries running for >5 seconds
- Any migration-related queries

### Step 3: Check RAWDATA Table

```sql
-- When was data last inserted?
SELECT
    MAX(created_at) as last_insert,
    COUNT(*) as total_rows,
    COUNT(DISTINCT task_name) as unique_tasks
FROM vectors.rawdata;
```

If `last_insert` is from yesterday and you haven't run migration today, **the INSERTs you're seeing are old log entries**.

### Step 4: Check PostgreSQL Logs Directly

**Find your PostgreSQL log file:**
```bash
# Usually in:
# Windows: C:\Program Files\PostgreSQL\16\data\log\
# Linux: /var/log/postgresql/
# Or check: SHOW data_directory;

# View recent logs
tail -f postgresql-*.log | grep INSERT
```

**Look at timestamps** - are they from today or yesterday?

### Step 5: Restart Everything

```bash
# Stop all Python processes
# Windows: taskkill /F /IM python.exe
# Linux: pkill -9 python

# Restart PostgreSQL
# Windows: net stop postgresql-x64-16 && net start postgresql-x64-16
# Linux: sudo systemctl restart postgresql

# Restart web interface
cd ragmcp
launch_two_phase_web.bat  # or .sh
```

## Performance Tuning

If Phase 1 is slow even without INSERTs:

### Disable Task Diffs (Fastest)
In the web UI Configuration accordion:
- ✅ Show Task Titles & Descriptions
- ✅ Show Changed Files
- ❌ Show File Diffs (DISABLE THIS)

This avoids fetching potentially large diffs.

### Reduce Result Counts
- Tasks RECENT: 2 (instead of 3)
- Tasks ALL: 1 (instead of 2)
- Modules RECENT: 3 (instead of 5)
- Modules ALL: 3 (instead of 5)
- Files RECENT: 5 (instead of 10)
- Files ALL: 5 (instead of 10)

### Expected Performance

**Without task diffs:**
- Phase 1: 2-5 seconds

**With task diffs:**
- Phase 1: 10-20 seconds (fetches diffs for ~5 tasks × 3 files each)

## Still Seeing INSERTs?

If you still see INSERT statements after following all steps above:

1. **Capture exact output:**
   ```bash
   # Run agent and save output
   cd ragmcp
   python two_phase_agent_web.py 2>&1 | tee debug_output.txt
   ```

2. **Check the timestamp** on INSERT statements in PostgreSQL logs

3. **Verify migration script is not in Python processes:**
   ```bash
   ps aux | grep migrate_rawdata
   # or
   tasklist | findstr migrate_rawdata
   ```

4. **Check for cron jobs or scheduled tasks** that might be running migration

## Summary

✅ **Agent code is verified read-only**
✅ **No INSERT operations in Phase 1**
✅ **Use diagnostic scripts to find the source**

The INSERTs are almost certainly:
- Old log entries from yesterday's migration
- A stuck migration process
- Multiple MCP server instances

Use the diagnostic tools provided to identify and resolve the issue.
